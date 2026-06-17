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
import hashlib
import threading
import concurrent.futures as cf

import numpy as np
import pandas as pd
import streamlit as st
import openai
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
df["page"] = df["topic"] + " (" + df["intent"] + ")"
n_pages = df.groupby(["topic_id", "intent"]).ngroups
dist = ", ".join(f"{k}: {v}" for k, v in df["intent"].value_counts().items())
st.success(f"{n_pages} pages across {n_topics} topics. Intent read from text ({LABEL_MODEL}).")
st.caption(f"Intent distribution: {dist}. SERP-grounded intent is the planned next upgrade.")

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
kw_export = df[["keyword", "volume", "topic", "intent", "page"]].rename(
    columns={"keyword": "Keyword", "volume": "Volume", "topic": "Topic", "intent": "Intent", "page": "Page"})
st.download_button("Download keyword mapping (CSV)",
                   kw_export.to_csv(index=False).encode("utf-8"),
                   "keyword_page_mapping.csv", "text/csv")
st.download_button("Download page summary (CSV)",
                   pages_tbl.to_csv(index=False).encode("utf-8"),
                   "page_summary.csv", "text/csv")
st.caption("Topic = pillar. Page = topic split by intent. Intent is text-based for now.")
