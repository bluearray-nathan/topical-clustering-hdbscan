# app.py: Keyword Topic Clustering (lean core)
# ---------------------------------------------------------------------------
# Groups a keyword list into topic clusters by meaning, using
# text-embedding-3-large and agglomerative clustering on cosine distance.
#
# This is the validated foundation. It captures TOPIC, not intent, so a topic
# such as "comprehensive insurance" will hold both informational and commercial
# keywords. Splitting each topic into separate PAGES by intent is the next layer,
# which sits on top of this.

import time
import json
import base64
import hashlib
import threading
import concurrent.futures as cf

import numpy as np
import pandas as pd
import streamlit as st
import openai
import requests
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Keyword Clustering and Page Mapping", layout="wide")
st.title("Keyword Clustering and Page Mapping")

with st.expander("What this does"):
    st.markdown("""
Groups a keyword list into **topic clusters** by meaning, then splits each topic into **pages by
intent**. Topic = pillar, page = topic plus intent. Every keyword gets a home; there is no "noise" bucket.

Clustering uses text-embedding-3-large with agglomerative clustering on cosine distance. Intent is
currently read from the **keyword text**; SERP-grounded intent is the planned next upgrade.
""")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error('Missing OpenAI API key. Add it in Streamlit secrets:\n\n[openai]\napi_key = "sk-..."')
    st.stop()

EMBED_MODEL = "text-embedding-3-large"
LABEL_MODEL = "gpt-4o-mini-2024-07-18"

if "label_cache" not in st.session_state:
    st.session_state.label_cache = {}

# DataForSEO credentials (optional, Basic auth). Used for SERP-based intent.
try:
    _dfs = st.secrets["dataforseo"]
    DFS_AUTH = base64.b64encode(f"{_dfs['login']}:{_dfs['password']}".encode()).decode()
except Exception:
    DFS_AUTH = None

DFS_BASE = "https://api.dataforseo.com/v3/serp/google/organic"


def _intent_from_item_types(types):
    """Map a DataForSEO result's item_types to an intent, or None to keep the text label."""
    t = {str(x).lower() for x in (types or [])}
    if t & {"local_pack", "map", "local_services", "hotels_pack", "google_hotels"}:
        return "Local"
    if t & {"shopping", "popular_products", "google_flights"}:
        return "Transactional"
    info = bool(t & {"featured_snippet", "people_also_ask", "answer_box", "knowledge_graph",
                     "questions_and_answers", "discussions_and_forums"})
    ads = bool(t & {"paid", "commercial_units"})
    if ads and not info:
        return "Transactional"
    if ads:
        return "Commercial"
    if info:
        return "Informational"
    return None


def dfs_live(keywords, location, auth, progress_cb=None):
    """Live Advanced: instant but pricier per call. Returns (intent_map, info)."""
    headers = {"Authorization": "Basic " + auth, "Content-Type": "application/json"}
    out, cost, failed, done = {}, 0.0, 0, 0
    chunks = [keywords[i:i + 20] for i in range(0, len(keywords), 20)]

    def fetch(chunk):
        payload = [{"keyword": k, "location_name": location, "language_code": "en", "device": "desktop"}
                   for k in chunk]
        try:
            r = requests.post(DFS_BASE + "/live/advanced", headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            j = r.json()
        except Exception:
            return {k: None for k in chunk}, 0.0, len(chunk)
        res, f = {}, 0
        for k, task in zip(chunk, j.get("tasks") or []):
            if (task.get("status_code") or 0) >= 40000:
                f += 1
            types = []
            for rr in (task.get("result") or []):
                types = rr.get("item_types") or []
                break
            res[k] = _intent_from_item_types(types)
        return res, float(j.get("cost") or 0.0), f

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, ch): ch for ch in chunks}
        for fut in cf.as_completed(futs):
            res, c, f = fut.result()
            out.update(res)
            cost += c
            failed += f
            done += 1
            if progress_cb:
                progress_cb(done, len(chunks))
    info = {"submitted": len(keywords), "resolved": sum(1 for v in out.values() if v),
            "failed": failed, "cost": cost}
    return out, info


def dfs_post(keywords, location, auth):
    """Submit queued Standard tasks (priority 1, the cheapest queue). Returns (id_to_kw, cost)."""
    headers = {"Authorization": "Basic " + auth, "Content-Type": "application/json"}
    id_to_kw, cost = {}, 0.0
    for i in range(0, len(keywords), 100):
        chunk = keywords[i:i + 100]
        payload = [{"keyword": k, "location_name": location, "language_code": "en",
                    "device": "desktop", "priority": 1} for k in chunk]
        try:
            r = requests.post(DFS_BASE + "/task_post", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception:
            j = {}
        cost += float(j.get("cost") or 0.0)
        for k, task in zip(chunk, j.get("tasks") or []):
            if task.get("id"):
                id_to_kw[task["id"]] = k
    return id_to_kw, cost


def dfs_collect(id_to_kw, auth, progress_cb=None, timeout_s=1200, poll_s=8):
    """Poll tasks_ready and fetch advanced results. Returns (intent_map, timed_out_count)."""
    headers = {"Authorization": "Basic " + auth}
    pending, out = dict(id_to_kw), {}
    deadline = time.time() + timeout_s
    while pending and time.time() < deadline:
        ready = []
        try:
            r = requests.get(DFS_BASE + "/tasks_ready", headers=headers, timeout=60)
            r.raise_for_status()
            for task in (r.json().get("tasks") or []):
                for res in (task.get("result") or []):
                    if res.get("id") in pending:
                        ready.append(res["id"])
        except Exception:
            ready = []
        for rid in ready:
            types = []
            try:
                rr = requests.get(f"{DFS_BASE}/task_get/advanced/{rid}", headers=headers, timeout=60)
                rr.raise_for_status()
                for t in (rr.json().get("tasks") or []):
                    for res in (t.get("result") or []):
                        types = res.get("item_types") or []
                        break
                out[pending[rid]] = _intent_from_item_types(types)
            except Exception:
                out[pending[rid]] = None
            pending.pop(rid, None)
            if progress_cb:
                progress_cb(len(out), len(id_to_kw))
        if pending:
            time.sleep(poll_s)
    for rid, kw in pending.items():
        out.setdefault(kw, None)
    return out, len(pending)


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Setup")
    st.caption("Embedding model")
    st.code(EMBED_MODEL, language="text")

    st.header("Cluster tightness")
    threshold = st.slider(
        "Cosine distance threshold",
        min_value=0.10, max_value=0.45, value=0.25, step=0.01,
        help="Lower = tighter, more granular topics. Higher = broader, fewer topics. "
             "0.25 is the validated default.",
    )

    st.header("Intent")
    intent_source = st.radio(
        "Intent source",
        ["Text (fast, free)", "DataForSEO SERP (reads the live SERP, costs credits)"],
        index=1,
        help="Text reads intent from the keyword wording. DataForSEO reads each keyword's live SERP "
             "features, which is closer to how Google treats it. Text is the fallback either way.",
    )
    use_serp = intent_source.startswith("DataForSEO")
    dfs_standard = st.radio(
        "DataForSEO mode",
        ["Standard queue (cheapest, slower)", "Live (instant, pricier)"],
        index=0,
    ).startswith("Standard")
    serp_cap = st.number_input("Max SERP lookups (0 = no limit)", min_value=0, value=2000, step=100)
    serp_location = st.text_input("SERP location", value="United Kingdom").strip() or "United Kingdom"

# ----------------------------- 1. Upload -----------------------------
st.subheader("1. Upload your keyword list")
file = st.file_uploader("CSV with a keyword column (volume optional).", type=["csv"])
if not file:
    st.info("Upload a CSV to begin.")
    st.stop()

df_in = pd.read_csv(file)
df_in.columns = [str(c).strip() for c in df_in.columns]


def find_col(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


kcol = find_col(df_in.columns, ["keyword", "keywords", "query", "term"])
if not kcol:
    st.error(f"No keyword column found. Columns seen: {list(df_in.columns)}")
    st.stop()
vcol = find_col(df_in.columns, ["search volume", "search_volume", "volume", "vol", "searches"])

df = pd.DataFrame()
df["keyword"] = df_in[kcol].astype(str).str.strip()
df["volume"] = pd.to_numeric(df_in[vcol], errors="coerce") if vcol else np.nan
df = df[df["keyword"].str.len() > 0].drop_duplicates("keyword").reset_index(drop=True)

if len(df) < 5:
    st.error("Need at least 5 distinct keywords to cluster.")
    st.stop()
st.success(f"Loaded {len(df)} distinct keywords"
           f"{' with volume.' if vcol else '. No volume column found, that is fine.'}")

# ----------------------------- 2. Embed -----------------------------
st.subheader("2. Embed")


@st.cache_data(show_spinner=False)
def embed(keywords, model):
    vecs = []
    for i in range(0, len(keywords), 200):
        resp = openai.embeddings.create(model=model, input=keywords[i:i + 200])
        vecs.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        time.sleep(0.05)
    return normalize(np.array(vecs, dtype=np.float32))


try:
    with st.spinner(f"Embedding {len(df)} keywords with {EMBED_MODEL}..."):
        X = embed(df["keyword"].tolist(), EMBED_MODEL)
    st.success(f"Embedded {len(X)} keywords.")
except Exception as e:
    st.error(f"Embedding failed: {e}")
    st.stop()

# ----------------------------- 3. Cluster -----------------------------
st.subheader("3. Cluster into topics")
labels = AgglomerativeClustering(
    n_clusters=None, distance_threshold=float(threshold),
    metric="cosine", linkage="average",
).fit_predict(X)
df["topic_id"] = labels
n_topics = int(df["topic_id"].nunique())
n_singletons = int((df["topic_id"].value_counts() == 1).sum())
st.success(f"{n_topics} topics. {n_singletons} are single keywords "
           f"({100 * n_singletons / len(df):.0f}%). Adjust the tightness slider to taste.")

# ----------------------------- 4. Label -----------------------------
st.subheader("4. Label topics")


def label_topic(keywords, cache, lock):
    # cache is a plain dict handed in from the main thread; never touch
    # st.session_state from a worker thread.
    key = hashlib.md5("|".join(sorted(keywords)).encode()).hexdigest()
    with lock:
        if key in cache:
            return cache[key]
    sample = keywords[:25]
    prompt = ("These keywords belong to one topic:\n" + ", ".join(sample) +
              "\n\nGive a concise topic name in 1 to 4 words (noun phrase, minimal punctuation). "
              "Reply with only the name.")
    try:
        resp = openai.chat.completions.create(
            model=LABEL_MODEL, temperature=0.2,
            messages=[{"role": "system", "content": "You are an SEO content analyst. Reply with only the label."},
                      {"role": "user", "content": prompt}],
        )
        name = " ".join(resp.choices[0].message.content.strip().strip('"').strip("'").split()[:5]) or "Unlabelled"
    except Exception:
        name = sample[0] if sample else "Unlabelled"
    with lock:
        cache[key] = name
    return name


ids = sorted(df["topic_id"].unique())
topic_kws = {
    tid: df[df["topic_id"] == tid].sort_values("volume", ascending=False, na_position="last")["keyword"].tolist()
    for tid in ids
}
labels_map = {}
label_cache = st.session_state.label_cache   # grab the dict on the main thread
label_lock = threading.Lock()
prog = st.progress(0.0)
done = 0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(label_topic, topic_kws[tid], label_cache, label_lock): tid for tid in ids}
    for fut in cf.as_completed(futs):
        labels_map[futs[fut]] = fut.result()
        done += 1
        prog.progress(done / len(ids))
df["topic"] = df["topic_id"].map(labels_map)
st.success("Topics labelled.")

# ----------------------------- 5. Intent + pages -----------------------------
st.subheader("5. Split topics into pages by intent")

INTENTS = ["Transactional", "Commercial", "Informational", "Navigational", "Local"]


@st.cache_data(show_spinner=False)
def classify_intents(keywords, model):
    """Classify each keyword's intent from its text. Cached, so it runs once per keyword set."""
    out = {}
    batch_size = 40
    sys_msg = ("You are an SEO search-intent classifier. For each keyword choose exactly one intent "
               "from: " + ", ".join(INTENTS) + ". "
               'Reply only as JSON: {"results":[{"keyword":"...","intent":"..."}]}.')
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        user_msg = "Classify these keywords:\n" + "\n".join(f"- {k}" for k in batch)
        try:
            resp = openai.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            )
            for r in json.loads(resp.choices[0].message.content).get("results", []):
                intent = str(r.get("intent", "")).strip()
                out[str(r.get("keyword", "")).strip()] = intent if intent in INTENTS else "Informational"
        except Exception:
            for k in batch:
                out[k] = "Informational"
    for k in keywords:                       # guarantee every keyword is covered
        out.setdefault(k, "Informational")
    return out


with st.spinner("Classifying intent from keyword text..."):
    intent_map = classify_intents(df["keyword"].tolist(), LABEL_MODEL)
df["intent"] = df["keyword"].map(intent_map).fillna("Informational")
df["intent_source"] = "text"

# Override with SERP-derived intent where selected and available.
if use_serp:
    if not DFS_AUTH:
        st.warning("DataForSEO SERP selected, but no credentials in secrets "
                   "([dataforseo] login / password). Keeping the text intent.")
    else:
        cand = df.sort_values("volume", ascending=False, na_position="last")
        if serp_cap and int(serp_cap) > 0:
            cand = cand.head(int(serp_cap))
        cand_kws = cand["keyword"].tolist()
        prog = st.progress(0.0)
        status = st.empty()
        if dfs_standard:
            status.caption("Submitting queued tasks; the Standard queue usually clears in a few minutes.")
            id_to_kw, cost = dfs_post(cand_kws, serp_location, DFS_AUTH)
            if not id_to_kw:
                st.warning("DataForSEO accepted no tasks. Check the location name and credentials. Keeping text intent.")
                serp_map, info = {}, {"submitted": len(cand_kws), "resolved": 0, "cost": cost, "timed_out": 0}
            else:
                serp_map, timed_out = dfs_collect(
                    id_to_kw, DFS_AUTH,
                    progress_cb=lambda d, t: (prog.progress(min(d / t, 1.0)),
                                              status.caption(f"Collected {d}/{t} SERPs from the queue")))
                info = {"submitted": len(cand_kws), "resolved": sum(1 for v in serp_map.values() if v),
                        "cost": cost, "timed_out": timed_out}
        else:
            serp_map, info = dfs_live(
                cand_kws, serp_location, DFS_AUTH,
                progress_cb=lambda d, t: (prog.progress(min(d / t, 1.0)),
                                          status.caption(f"Read SERPs: chunk {d}/{t}")))
        for kw, it in serp_map.items():
            if it:
                df.loc[df["keyword"] == kw, "intent"] = it
                df.loc[df["keyword"] == kw, "intent_source"] = "serp"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intent from SERP", int((df["intent_source"] == "serp").sum()))
        c2.metric("Intent from text", int((df["intent_source"] == "text").sum()))
        c3.metric("SERP resolved", f"{info.get('resolved', 0)}/{info.get('submitted', 0)}")
        c4.metric("DataForSEO cost", f"${info.get('cost', 0):.2f}")
        if info.get("timed_out"):
            st.caption(f"{info['timed_out']} tasks did not return before the timeout and kept their text intent.")

df["page"] = df["topic"] + " (" + df["intent"] + ")"
n_pages = df.groupby(["topic_id", "intent"]).ngroups
dist = ", ".join(f"{k}: {v}" for k, v in df["intent"].value_counts().items())
src = "SERP, with text as fallback" if (use_serp and DFS_AUTH) else "keyword text"
st.success(f"{n_pages} pages across {n_topics} topics. Intent source: {src}.")
st.caption(f"Intent distribution: {dist}.")

# ----------------------------- 6. Topics and pages -----------------------------
st.subheader("6. Topics and pages")

topic_rows = []
for tid in ids:
    sub = df[df["topic_id"] == tid].sort_values("volume", ascending=False, na_position="last")
    vol = sub["volume"].sum(min_count=1)
    topic_rows.append({
        "Topic": labels_map[tid],
        "Pages": int(sub["intent"].nunique()),
        "Keywords": len(sub),
        "Volume": (int(vol) if pd.notna(vol) else None),
        "Head term": sub["keyword"].iloc[0],
    })
topics_tbl = pd.DataFrame(topic_rows).sort_values("Volume", ascending=False, na_position="last")
st.markdown("**Topics (pillars)**")
st.dataframe(topics_tbl, use_container_width=True, height=280)

page_rows = []
for (tid, intent), sub in df.groupby(["topic_id", "intent"]):
    sub = sub.sort_values("volume", ascending=False, na_position="last")
    vol = sub["volume"].sum(min_count=1)
    page_rows.append({
        "Page": f"{labels_map[tid]} ({intent})",
        "Topic": labels_map[tid],
        "Intent": intent,
        "Keywords": len(sub),
        "Volume": (int(vol) if pd.notna(vol) else None),
        "Head term": sub["keyword"].iloc[0],
    })
pages_tbl = pd.DataFrame(page_rows).sort_values("Volume", ascending=False, na_position="last")
st.markdown("**Pages (each topic split by intent)**")
st.dataframe(pages_tbl, use_container_width=True, height=420)

# 2D map, colour by topic or intent
try:
    coords = PCA(n_components=2).fit_transform(X)
    df["x"], df["y"] = coords[:, 0], coords[:, 1]
    colour_by = st.radio("Colour the map by", ["Topic", "Intent"], horizontal=True)
    fig = px.scatter(df, x="x", y="y", color=colour_by.lower(),
                     hover_data=["keyword", "topic", "intent", "volume"], height=650,
                     title=f"Keywords by {colour_by.lower()}")
    if colour_by == "Topic":
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    pass

# ----------------------------- 7. Export -----------------------------
st.subheader("7. Export")
kw_export = df[["keyword", "volume", "topic", "intent", "intent_source", "page"]].rename(
    columns={"keyword": "Keyword", "volume": "Volume", "topic": "Topic", "intent": "Intent",
             "intent_source": "Intent source", "page": "Page"})
st.download_button("Download keyword mapping (CSV)",
                   kw_export.to_csv(index=False).encode("utf-8"),
                   "keyword_page_mapping.csv", "text/csv")
st.download_button("Download page summary (CSV)",
                   pages_tbl.to_csv(index=False).encode("utf-8"),
                   "page_summary.csv", "text/csv")
st.caption("Topic = pillar. Page = topic split by intent. Intent is text-based for now.")
