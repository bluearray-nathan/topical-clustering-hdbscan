# app.py — Keyword Page Mapping Tool
# ----------------------------------------------------------------------------
# Replaces a SERP-overlap clustering tool (e.g. Keyword Insights) with an
# intent-aware, embeddings-based approach.
#
# Pipeline (bottom-up):
#   raw keywords (any source)
#       -> embed (OpenAI text-embedding-3-small)
#       -> classify intent from the text (OpenAI), source-agnostic
#          (broad, signal-free head terms optionally resolved via SERP features)
#       -> tight semantic micro-clusters (UMAP + HDBSCAN)
#       -> split each micro-cluster by intent  => candidate PAGES (one page = one URL)
#       -> group pages into PILLARS above (1 or 2 levels)
#       -> score each page and recommend: Standalone / Section / Merge
#       -> visualise + export
#
# Only the OpenAI key is required. The SerpApi key is optional and only used
# when the broad-head SERP check is switched on.

import time
import json
import math
import hashlib
import traceback
import random
import concurrent.futures as cf
from collections import deque, Counter

import numpy as np
import pandas as pd
import streamlit as st
import openai
import hdbscan
import requests

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

random.seed(42)
np.random.seed(42)

# ----------------------------- Page config & title -----------------------------
st.set_page_config(page_title="Keyword Page Mapping Tool", layout="wide")
st.title("Keyword Page Mapping Tool")

with st.expander("What this tool does and how it works"):
    st.markdown("""
**What it does**
Takes a raw keyword list from any source (Ahrefs, SEMrush, Search Console, Reddit, a manual list)
and works out the **page architecture**: which keywords should sit on the same page, how those pages
group into topical pillars, and which clusters are strong enough to **warrant their own page**.

It is meant to replace SERP-overlap clustering tools rather than sit on top of them.

---

### How it works
1. **Upload one or more CSVs.** Only a keyword column is required. Volume, position and source are optional.
2. **Embed** every keyword so similar phrasings sit close together.
3. **Classify intent** from the keyword text (transactional, commercial, informational, and so on).
   Broad head terms that carry no intent in the words (for example "running trainers") can optionally
   be resolved by reading the live SERP features, since that is the only honest signal for those.
4. **Cluster** into tight semantic groups, then **split each group by intent** so a single theme that
   mixes "buy" and "what is" becomes two candidate pages, not one.
5. **Group pages into pillars** above.
6. **Recommend** Standalone / Section / Merge for each page, using intent distinctness, breadth,
   cross-source corroboration and volume (where you have it).
7. **Visualise and export.**

---

### Honest limitations
- Intent from text is a model's read, not Google's behaviour. The optional SERP check is the truth test for broad heads.
- Keywords with no search volume (for example Reddit) are scored on intent and breadth, so those pages are **opportunities to validate**, not guaranteed traffic.
- Labels and recommendations are suggestions. Sense-check before handing to a client.
""")

# ----------------------------- API keys -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("Missing OpenAI API key. Add it in Streamlit secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

# SerpApi key is optional. Only needed if the SERP head-check is switched on.
try:
    SERPAPI_KEY = st.secrets["serpapi"]["api_key"]
except Exception:
    SERPAPI_KEY = None

# ----------------------------- Constants -----------------------------
EMBED_MODEL = "text-embedding-3-small"   # cheaper embedding model, per spec
LLM_MODEL = "gpt-4o-mini-2024-07-18"     # used for intent, labels and recommendations
LLM_TEMP = 0.1

# Internal intent set. Deliberately fixed and hidden: the user never configures this.
INTENTS = [
    "Transactional",            # buy, price, quote, sign up, for sale
    "Commercial",               # best, top, review, vs, alternatives
    "Informational-Howto",      # how to, guide, fix, tutorial
    "Informational-Definitional",  # what is, meaning, examples, ideas
    "Navigational",             # a brand or specific site
    "Local",                    # near me, in <place>
]
# Map a resolved intent to the kind of page it implies (for grouping/merging logic).
COMMERCIAL_INTENTS = {"Transactional", "Commercial", "Local"}

# ----------------------------- Session init -----------------------------
for key, default in [
    ("intent_map", {}),
    ("last_kw_hash", None),
    ("page_labels_map", {}),
    ("pillar_labels_map", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Setup")
    st.caption("Embedding model")
    st.code(EMBED_MODEL, language="text")

    st.header("Page granularity")
    granularity = st.radio(
        "How finely should keywords be grouped into pages?",
        options=["Fewer, broader pages", "Balanced (recommended)", "More, finer pages"],
        index=1,
        help="Controls the tightness of the bottom-level semantic clustering before the intent split.",
    )

    st.header("Pillars above pages")
    pillar_levels = st.radio(
        "How many grouping levels above the page?",
        options=[1, 2],
        index=0,
        help="1 = Pillar > Page. 2 = Super-pillar > Pillar > Page.",
    )

    st.header("Warrants its own page")
    volume_floor = st.number_input(
        "Minimum monthly volume for a standalone page",
        min_value=0, value=100, step=50,
        help="Pages whose keywords have volume below this fold into a parent as a section. "
             "Pages with no volume data are judged on intent and breadth instead.",
    )
    min_keywords = st.number_input(
        "Minimum keywords for a standalone page",
        min_value=1, value=3, step=1,
        help="A handful of near-identical keywords is a heading, not a page.",
    )

    st.header("Broad head terms")
    use_serp = st.checkbox(
        "Resolve broad head terms with live SERP features (uses SerpApi credits)",
        value=False,
        help="Broad terms like 'running trainers' carry no intent in the words. "
             "When on, the tool reads the live SERP (ads, shopping, snippets) to classify just those.",
    )
    serp_cap = st.number_input(
        "Max SERP lookups per run",
        min_value=0, value=100, step=25,
        help="Hard cap so a single run cannot eat your monthly SerpApi budget.",
    )
    serp_country = st.text_input("SERP country (ISO-2)", value="gb", max_chars=2).strip().lower() or "gb"

    # Granularity presets for the bottom-level micro-clustering.
    GRAN_PRESETS = {
        "Fewer, broader pages": {"neighbors": 30, "components": 10, "mcs": 8, "ms": 2, "eps": 0.06},
        "Balanced (recommended)": {"neighbors": 15, "components": 10, "mcs": 4, "ms": 1, "eps": 0.04},
        "More, finer pages": {"neighbors": 8, "components": 8, "mcs": 3, "ms": 1, "eps": 0.02},
    }
    G = GRAN_PRESETS[granularity]

# ----------------------------- Step 1: Upload + column mapping -----------------------------
st.subheader("1. Upload your keyword list")
files = st.file_uploader(
    "Upload one or more CSVs. Only a keyword column is required.",
    type=["csv"], accept_multiple_files=True,
)
if not files:
    st.info("Upload at least one CSV to begin.")
    st.stop()

frames = []
for f in files:
    try:
        part = pd.read_csv(f)
    except Exception:
        st.error(f"Could not read {f.name}. Check the encoding/format.")
        st.code(traceback.format_exc())
        st.stop()
    part.columns = [str(c).strip() for c in part.columns]
    part["__source_file__"] = f.name
    frames.append(part)

df_in = pd.concat(frames, ignore_index=True, sort=False)
all_cols = [c for c in df_in.columns if c != "__source_file__"]

st.markdown("**Map your columns** (we only need the keyword; the rest are optional but improve the output).")
c1, c2, c3, c4 = st.columns(4)


def _guess(cands, options):
    low = {o.lower(): o for o in options}
    for cand in cands:
        if cand in low:
            return low[cand]
    return None


with c1:
    kw_col = st.selectbox(
        "Keyword column (required)", options=all_cols,
        index=(all_cols.index(_guess(["keyword", "keywords", "query", "term"], all_cols))
               if _guess(["keyword", "keywords", "query", "term"], all_cols) else 0),
    )
with c2:
    vol_opts = ["(none)"] + all_cols
    vguess = _guess(["search volume", "volume", "vol", "avg. monthly searches", "searches"], all_cols)
    vol_col = st.selectbox("Volume column (optional)", options=vol_opts,
                           index=(vol_opts.index(vguess) if vguess else 0))
with c3:
    pos_opts = ["(none)"] + all_cols
    pguess = _guess(["position", "rank", "current position"], all_cols)
    pos_col = st.selectbox("Position column (optional)", options=pos_opts,
                           index=(pos_opts.index(pguess) if pguess else 0))
with c4:
    src_opts = ["(use file name)"] + all_cols
    sguess = _guess(["source", "origin"], all_cols)
    src_col = st.selectbox("Source column (optional)", options=src_opts,
                           index=(src_opts.index(sguess) if sguess else 0))

# Build a clean working frame.
work = pd.DataFrame()
work["keyword"] = df_in[kw_col].astype(str).str.strip()
work["volume"] = (pd.to_numeric(df_in[vol_col], errors="coerce") if vol_col != "(none)" else np.nan)
work["position"] = (pd.to_numeric(df_in[pos_col], errors="coerce") if pos_col != "(none)" else np.nan)
work["source"] = (df_in[src_col].astype(str) if src_col != "(use file name)" else df_in["__source_file__"])

work = work[work["keyword"].str.len() > 0].copy()

# Deduplicate the same keyword across sources: keep max known volume, join sources.
if len(work):
    agg = (
        work.groupby("keyword")
        .agg(
            volume=("volume", "max"),
            position=("position", "min"),
            source=("source", lambda s: ", ".join(sorted(set(map(str, s))))),
            n_sources=("source", lambda s: len(set(map(str, s)))),
        )
        .reset_index()
    )
else:
    agg = work

if len(agg) < 5:
    st.error("Need at least 5 distinct keywords to cluster. Add more rows.")
    st.stop()

has_volume = agg["volume"].notna().any()
st.success(f"Loaded {len(agg)} distinct keywords from {len(files)} file(s). "
           f"{'Volume present.' if has_volume else 'No volume data — pages will be judged on intent and breadth.'}")

# ----------------------------- Step 2: Embeddings -----------------------------
st.subheader("2. Embed keywords")


@st.cache_data(show_spinner=False)
def embed_texts(texts, model):
    texts = [t if isinstance(t, str) else "" for t in texts]
    vecs = []
    for i in range(0, len(texts), 200):
        batch = texts[i:i + 200]
        resp = openai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.05)
    return np.array(vecs, dtype=np.float32)


try:
    with st.spinner("Creating embeddings..."):
        embeddings = embed_texts(agg["keyword"].tolist(), EMBED_MODEL)
    embeddings = np.nan_to_num(embeddings, posinf=0.0, neginf=0.0)
    embeddings = normalize(embeddings)
    st.success(f"Created {len(embeddings)} embeddings.")
except Exception:
    st.error("Error while creating embeddings.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Step 3: Intent classification -----------------------------
st.subheader("3. Classify intent")


def _heuristic_intent(kw: str) -> str:
    """Cheap modifier-based fallback when the model is unavailable."""
    k = f" {kw.lower()} "
    if any(t in k for t in [" near me ", " nearby ", " in london ", " in manchester "]):
        return "Local"
    if any(t in k for t in [" buy ", " price ", " cost ", " for sale ", " cheap ", " quote ", " deal ", " discount "]):
        return "Transactional"
    if any(t in k for t in [" best ", " top ", " review ", " reviews ", " vs ", " versus ", " alternative ", " compare "]):
        return "Commercial"
    if any(t in k for t in [" how to ", " how do ", " guide ", " tutorial ", " fix ", " repair ", " steps "]):
        return "Informational-Howto"
    if any(t in k for t in [" what is ", " what are ", " meaning ", " definition ", " examples ", " ideas ", " why "]):
        return "Informational-Definitional"
    return "Unknown"


@st.cache_data(show_spinner=False)
def classify_intents_llm(keywords, model, temp):
    """
    Batched intent classification straight from the keyword text.
    Returns a dict keyword -> {"intent": str, "ambiguous": bool}.
    """
    out = {}
    BATCH = 40
    sys = (
        "You are an SEO search-intent classifier. For each keyword decide the dominant search intent "
        "a searcher has, choosing exactly one label from this set: "
        + ", ".join(INTENTS) + ". "
        "Also set ambiguous=true when the keyword is a short, broad head term whose intent cannot be told "
        "from the words alone (for example a bare product noun like 'running trainers'); otherwise false. "
        'Reply only as JSON: {"results":[{"keyword":"...","intent":"...","ambiguous":true|false}]}.'
    )
    for i in range(0, len(keywords), BATCH):
        batch = keywords[i:i + BATCH]
        user = "Classify these keywords:\n" + "\n".join(f"- {k}" for k in batch)
        try:
            resp = openai.chat.completions.create(
                model=model, temperature=temp,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
            data = json.loads(resp.choices[0].message.content)
            for row in data.get("results", []):
                kw = str(row.get("keyword", "")).strip()
                intent = str(row.get("intent", "")).strip()
                if intent not in INTENTS:
                    intent = _heuristic_intent(kw)
                out[kw] = {"intent": intent, "ambiguous": bool(row.get("ambiguous", False))}
        except Exception:
            for k in batch:
                out[k] = {"intent": _heuristic_intent(k), "ambiguous": len(k.split()) <= 2}
    # Guarantee every keyword is covered.
    for k in keywords:
        if k not in out:
            out[k] = {"intent": _heuristic_intent(k), "ambiguous": len(k.split()) <= 2}
    return out


try:
    with st.spinner("Classifying intent from keyword text..."):
        intent_info = classify_intents_llm(agg["keyword"].tolist(), LLM_MODEL, LLM_TEMP)
    agg["intent"] = agg["keyword"].map(lambda k: intent_info[k]["intent"])
    agg["intent_ambiguous"] = agg["keyword"].map(lambda k: intent_info[k]["ambiguous"])
    st.success("Intent classified from text.")
except Exception:
    st.error("Error during intent classification.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Step 3b: Optional SERP feature resolution for broad heads -----------------------------
def serp_intent_from_features(keyword, country, api_key):
    """
    Read the live SERP and infer intent from its feature set.
    ads + shopping  -> Transactional ; shopping only / many ads -> Commercial
    local pack      -> Local
    answer box / PAA / no commercial features -> Informational
    Returns an intent label or None on failure.
    """
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": keyword, "gl": country, "hl": "en", "api_key": api_key},
            timeout=20,
        )
        r.raise_for_status()
        j = r.json()
    except Exception:
        return None

    has_shopping = bool(j.get("shopping_results") or j.get("immersive_products"))
    has_ads = bool(j.get("ads") or j.get("shopping_ads"))
    has_local = bool(j.get("local_results") or j.get("local_map"))
    has_answer = bool(j.get("answer_box") or j.get("knowledge_graph"))
    has_paa = bool(j.get("related_questions"))

    if has_local:
        return "Local"
    if has_shopping and has_ads:
        return "Transactional"
    if has_shopping or has_ads:
        return "Commercial"
    if has_answer or has_paa:
        return "Informational-Definitional"
    return None


if use_serp:
    st.subheader("3b. Resolve broad head terms via SERP")
    if not SERPAPI_KEY:
        st.warning("SERP check is on but no SerpApi key is set. Add [serpapi] api_key in secrets. Skipping.")
    else:
        # Route only the broad/ambiguous heads, prioritised by volume, capped.
        cand = agg[agg["intent_ambiguous"]].copy()
        if "volume" in cand:
            cand = cand.sort_values("volume", ascending=False, na_position="last")
        cand = cand.head(int(serp_cap))
        if len(cand) == 0:
            st.info("No broad head terms needed a SERP check.")
        else:
            prog = st.progress(0.0)
            done = 0
            for kw in cand["keyword"].tolist():
                resolved = serp_intent_from_features(kw, serp_country, SERPAPI_KEY)
                if resolved:
                    agg.loc[agg["keyword"] == kw, "intent"] = resolved
                done += 1
                prog.progress(done / len(cand))
            st.success(f"Resolved {len(cand)} broad head term(s) from live SERP features.")

# ----------------------------- Step 4: Semantic micro-clustering -----------------------------
st.subheader("4. Cluster keywords semantically")
try:
    from umap import UMAP
    n_samples = embeddings.shape[0]
    n_neighbors = int(max(2, min(G["neighbors"], n_samples - 1)))
    n_components = int(max(2, min(G["components"], min(embeddings.shape[1], n_samples - 1))))
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=n_components,
                   metric="cosine", random_state=42)
    X = reducer.fit_transform(embeddings)

    micro = hdbscan.HDBSCAN(
        min_cluster_size=int(G["mcs"]), min_samples=int(G["ms"]),
        cluster_selection_method="eom", cluster_selection_epsilon=float(G["eps"]),
        metric="euclidean",
    ).fit_predict(X)
    agg["micro_id"] = micro
    n_micro = len(set(micro)) - (1 if -1 in micro else 0)
    st.success(f"Found {n_micro} semantic micro-clusters "
               f"({(micro == -1).mean() * 100:.0f}% sat outside a cluster and are handled individually).")
except Exception:
    st.error("Error during semantic clustering.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Step 5: Split micro-clusters by intent -> candidate pages -----------------------------
st.subheader("5. Split by intent into candidate pages")
try:
    # A page = a micro-cluster restricted to one intent. Noise keywords (micro_id -1)
    # each become their own single-keyword candidate page so nothing is silently dropped.
    page_ids = np.full(len(agg), -1, dtype=int)
    next_page = 0
    micro_vals = agg["micro_id"].values
    intent_vals = agg["intent"].values

    for mid in sorted(set(micro_vals)):
        idx = np.where(micro_vals == mid)[0]
        if mid == -1:
            for j in idx:               # each noise keyword stands alone
                page_ids[j] = next_page
                next_page += 1
            continue
        for intent in sorted(set(intent_vals[idx])):
            sub = idx[intent_vals[idx] == intent]
            page_ids[sub] = next_page
            next_page += 1

    agg["page_id"] = page_ids
    st.success(f"Formed {agg['page_id'].nunique()} candidate pages "
               f"(intent split turned {n_micro} themes into more precise page targets).")
except Exception:
    st.error("Error while splitting by intent.")
    st.code(traceback.format_exc())
    st.stop()

# Page centroids (for pillar grouping and merge detection).
page_index = sorted(agg["page_id"].unique())
centroids = np.vstack([
    normalize(embeddings[np.where(agg["page_id"].values == pid)[0]].mean(axis=0, keepdims=True))[0]
    for pid in page_index
])
centroid_of = {pid: centroids[i] for i, pid in enumerate(page_index)}

# ----------------------------- Step 6: Group pages into pillars -----------------------------
st.subheader("6. Group pages into pillars")
try:
    pillar_of = {}
    superpillar_of = {}
    if len(page_index) <= 2:
        for pid in page_index:
            pillar_of[pid] = 0
            superpillar_of[pid] = 0
    else:
        p_neighbors = int(max(2, min(15, len(page_index) - 1)))
        p_components = int(max(2, min(8, len(page_index) - 1)))
        Xp = UMAP(n_neighbors=p_neighbors, min_dist=0.0, n_components=p_components,
                  metric="cosine", random_state=42).fit_transform(centroids)
        plabels = hdbscan.HDBSCAN(
            min_cluster_size=2, min_samples=1, cluster_selection_method="eom", metric="euclidean",
        ).fit_predict(Xp)
        # Give pillar-noise pages their own singleton pillars.
        nextp = (plabels.max() + 1) if len(plabels) and plabels.max() >= 0 else 0
        fixed = []
        for lab in plabels:
            if lab == -1:
                fixed.append(nextp); nextp += 1
            else:
                fixed.append(lab)
        for i, pid in enumerate(page_index):
            pillar_of[pid] = int(fixed[i])

        if pillar_levels == 2:
            # Super-pillars: cluster the pillar centroids.
            pillar_ids = sorted(set(pillar_of.values()))
            pcent = np.vstack([
                normalize(np.mean([centroid_of[pid] for pid in page_index if pillar_of[pid] == plid],
                                  axis=0, keepdims=True))[0]
                for plid in pillar_ids
            ])
            if len(pillar_ids) <= 2:
                sp_map = {plid: 0 for plid in pillar_ids}
            else:
                sp_labels = hdbscan.HDBSCAN(
                    min_cluster_size=2, min_samples=1, cluster_selection_method="eom", metric="euclidean",
                ).fit_predict(pcent)
                nsp = (sp_labels.max() + 1) if len(sp_labels) and sp_labels.max() >= 0 else 0
                sp_fixed = []
                for lab in sp_labels:
                    if lab == -1:
                        sp_fixed.append(nsp); nsp += 1
                    else:
                        sp_fixed.append(lab)
                sp_map = {plid: int(sp_fixed[i]) for i, plid in enumerate(pillar_ids)}
            for pid in page_index:
                superpillar_of[pid] = sp_map[pillar_of[pid]]

    agg["pillar_id"] = agg["page_id"].map(pillar_of)
    if pillar_levels == 2:
        agg["superpillar_id"] = agg["page_id"].map(superpillar_of)
    st.success(f"Grouped {len(page_index)} pages into {len(set(pillar_of.values()))} pillar(s).")
except Exception:
    st.error("Error while grouping pages into pillars.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Labelling helpers (reused, OpenAI) -----------------------------
def top_facets(texts, top_k=10):
    if not texts:
        return []
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
    try:
        Xc = cv.fit_transform(texts)
    except ValueError:
        return []
    counts = np.asarray(Xc.sum(axis=0)).ravel()
    vocab = np.array(cv.get_feature_names_out())
    order = counts.argsort()[::-1]
    return [(vocab[i], int(counts[i])) for i in order[:top_k] if counts[i] > 0]


def diversified_examples(kws, vecs, total_max=25):
    n = len(kws)
    if n <= total_max:
        return list(kws)
    k = int(np.clip(int(np.sqrt(n)), 5, min(15, total_max)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(vecs)
    picked = []
    for c in range(k):
        ci = np.where(labels == c)[0]
        if ci.size:
            d = ((vecs[ci] - km.cluster_centers_[c]) ** 2).sum(axis=1)
            picked.append(kws[ci[d.argmin()]])
    return picked[:total_max]


def label_one(kws, vecs, kind="page"):
    sample = diversified_examples(list(kws), vecs, total_max=25)
    facets = top_facets(list(kws))
    fac = ", ".join(f"{p}" for p, _ in facets[:8])
    sys = "You are an SEO content analyst. Reply with only the label, nothing else."
    user = (
        f"These keywords belong to one {kind}:\n{', '.join(sample)}\n\n"
        f"Common phrases: {fac}\n\n"
        f"Give a concise, human-friendly {kind} name in 1-4 words (noun phrase, minimal punctuation)."
    )
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL, temperature=0.2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        )
        txt = resp.choices[0].message.content.strip().strip('"').strip("'")
        return " ".join(txt.split()[:5]) or "Unlabelled"
    except Exception:
        return facets[0][0].title() if facets else "Unlabelled"


def label_groups(group_col):
    ids = sorted(agg[group_col].unique())
    out = {}
    prog = st.progress(0.0)
    done = 0

    def work_one(gid):
        idx = np.where(agg[group_col].values == gid)[0]
        kind = "page" if group_col == "page_id" else "pillar"
        return gid, label_one(agg["keyword"].values[idx], embeddings[idx], kind=kind)

    with cf.ThreadPoolExecutor(max_workers=min(12, max(1, len(ids)))) as ex:
        for fut in cf.as_completed([ex.submit(work_one, g) for g in ids]):
            gid, name = fut.result()
            out[gid] = name
            done += 1
            prog.progress(done / len(ids))
    return out


# ----------------------------- Step 7: Label pages and pillars -----------------------------
st.subheader("7. Label pages and pillars")
try:
    page_labels = label_groups("page_id")
    agg["page_label"] = agg["page_id"].map(page_labels)
    pillar_labels = label_groups("pillar_id")
    agg["pillar_label"] = agg["pillar_id"].map(pillar_labels)
    if pillar_levels == 2:
        sp_labels_map = label_groups("superpillar_id")
        agg["superpillar_label"] = agg["superpillar_id"].map(sp_labels_map)
    st.success("Labelling complete.")
except Exception:
    st.error("Error during labelling.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Step 8: Page-worthiness recommendation -----------------------------
st.subheader("8. Recommend Standalone / Section / Merge")


def dominant_intent(series):
    c = Counter(series)
    return c.most_common(1)[0][0] if c else "Unknown"


# Build a per-page feature table.
page_rows = []
for pid in page_index:
    idx = np.where(agg["page_id"].values == pid)[0]
    sub = agg.iloc[idx]
    vol = sub["volume"].sum(min_count=1)
    page_rows.append({
        "page_id": pid,
        "page_label": page_labels.get(pid, str(pid)),
        "pillar_id": pillar_of[pid],
        "pillar_label": pillar_labels.get(pillar_of[pid], str(pillar_of[pid])),
        "intent": dominant_intent(sub["intent"]),
        "n_keywords": int(len(sub)),
        "total_volume": (float(vol) if pd.notna(vol) else np.nan),
        "n_sources": int(sub["n_sources"].max() if "n_sources" in sub else 1),
        "head_term": (sub.sort_values("volume", ascending=False, na_position="last")["keyword"].iloc[0]
                      if len(sub) else ""),
    })
pages_df = pd.DataFrame(page_rows)


def merge_target(pid):
    """A sibling page in the same pillar with the same intent and very close centroid is a merge candidate."""
    me = centroid_of[pid]
    my_pillar = pillar_of[pid]
    my_intent = pages_df.loc[pages_df["page_id"] == pid, "intent"].iloc[0]
    best, best_sim = None, 0.0
    for other in page_index:
        if other == pid or pillar_of[other] != my_pillar:
            continue
        if pages_df.loc[pages_df["page_id"] == other, "intent"].iloc[0] != my_intent:
            continue
        sim = float(np.dot(me, centroid_of[other]))
        if sim > best_sim:
            best, best_sim = other, sim
    return (best, best_sim)


def heuristic_reco(row):
    """Deterministic fallback recommendation, also used as a prior for the model."""
    tgt, sim = merge_target(row["page_id"])
    if tgt is not None and sim >= 0.93:
        return "Merge", f"Near-duplicate of '{page_labels.get(tgt)}' (same pillar and intent, similarity {sim:.2f})."
    if row["n_keywords"] < min_keywords:
        return "Section", f"Only {row['n_keywords']} keyword(s); too thin to sustain a page."
    if pd.notna(row["total_volume"]):
        if row["total_volume"] >= volume_floor:
            return "Standalone", f"Distinct {row['intent'].lower()} intent with {int(row['total_volume'])} monthly searches."
        return "Section", f"Distinct intent but only {int(row['total_volume'])} searches, below the {int(volume_floor)} floor."
    # No volume: lean on breadth and corroboration.
    if row["n_keywords"] >= max(min_keywords + 2, 5) or row["n_sources"] >= 2:
        return "Standalone (validate)", "No volume data, but breadth and/or multiple sources suggest real demand to validate."
    return "Section", "No volume data and limited breadth; treat as a section until demand is confirmed."


# Heuristic prior for every page.
prior = pages_df.apply(heuristic_reco, axis=1, result_type="expand")
pages_df["recommendation"] = prior[0]
pages_df["reason"] = prior[1]

# Optional: let the model weigh the signals (honours the "Claude/GPT weighs signals" choice).
try:
    feat = pages_df[["page_id", "page_label", "pillar_label", "intent",
                     "n_keywords", "total_volume", "n_sources", "recommendation"]].copy()
    feat["total_volume"] = feat["total_volume"].where(pd.notna(feat["total_volume"]), None)
    sys = (
        "You map keyword clusters to a site's page architecture. For each candidate page decide one of: "
        "'Standalone' (warrants its own URL), 'Section' (fold into its pillar as a section), or "
        "'Merge' (duplicate of a sibling). Weigh distinct search intent first, then breadth (n_keywords), "
        "cross-source corroboration (n_sources), and volume where present. "
        f"The user's standalone volume floor is {int(volume_floor)} monthly searches, but treat it as guidance "
        "not an absolute gate, especially where volume is missing (those are opportunities to validate). "
        "A 'recommendation' field is provided as a prior; override it only with good reason. "
        'Reply only as JSON: {"results":[{"page_id":int,"recommendation":"...","reason":"one short sentence"}]}.'
    )
    recos = {}
    records = feat.to_dict(orient="records")
    BATCH = 50
    with st.spinner("Weighing page-worthiness..."):
        for i in range(0, len(records), BATCH):
            chunk = records[i:i + BATCH]
            user = "Decide for these candidate pages:\n" + json.dumps(chunk, default=str)
            resp = openai.chat.completions.create(
                model=LLM_MODEL, temperature=LLM_TEMP,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
            for r in json.loads(resp.choices[0].message.content).get("results", []):
                recos[int(r["page_id"])] = (str(r.get("recommendation", "")).strip(),
                                            str(r.get("reason", "")).strip())
    if recos:
        pages_df["recommendation"] = pages_df["page_id"].map(
            lambda p: recos.get(p, (None, None))[0] or pages_df.loc[pages_df["page_id"] == p, "recommendation"].iloc[0])
        pages_df["reason"] = pages_df["page_id"].map(
            lambda p: recos.get(p, (None, None))[1] or pages_df.loc[pages_df["page_id"] == p, "reason"].iloc[0])
    st.success("Recommendations ready.")
except Exception:
    st.warning("Model weighting failed; using the deterministic recommendations instead.")
    st.code(traceback.format_exc())

# Map page recommendation back onto keyword rows.
agg["page_recommendation"] = agg["page_id"].map(dict(zip(pages_df["page_id"], pages_df["recommendation"])))
agg["page_reason"] = agg["page_id"].map(dict(zip(pages_df["page_id"], pages_df["reason"])))

# ----------------------------- Step 9: Headline + which warrant a page -----------------------------
st.subheader("9. Which clusters warrant their own page")
standalone = pages_df[pages_df["recommendation"].str.startswith("Standalone")].copy()
n_standalone = len(standalone)
n_section = int((pages_df["recommendation"] == "Section").sum())
n_merge = int((pages_df["recommendation"] == "Merge").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidate pages", len(pages_df))
m2.metric("Warrant a page", n_standalone)
m3.metric("Fold in as section", n_section)
m4.metric("Merge (duplicate)", n_merge)

show_cols = ["page_label", "pillar_label", "intent", "n_keywords", "total_volume", "n_sources", "reason"]
st.markdown("**Pages that warrant their own URL**")
st.dataframe(
    standalone.sort_values("total_volume", ascending=False, na_position="last")[["page_label", "pillar_label", "intent", "n_keywords", "total_volume", "head_term"]],
    use_container_width=True, height=320,
)

with st.expander("Full page table (all recommendations)"):
    st.dataframe(
        pages_df.sort_values(["pillar_label", "recommendation", "total_volume"],
                             ascending=[True, True, False], na_position="last")[
            ["page_label", "pillar_label", "recommendation", "intent", "n_keywords", "total_volume", "n_sources", "reason", "head_term"]
        ],
        use_container_width=True, height=420,
    )

# ----------------------------- Step 10: Visualise -----------------------------
st.subheader("10. Visualise")
try:
    coords = PCA(n_components=2).fit_transform(embeddings)
    agg["x"], agg["y"] = coords[:, 0], coords[:, 1]
    colour_by = st.radio("Colour points by", ["Pillar", "Intent", "Recommendation"], horizontal=True)
    colour_col = {"Pillar": "pillar_label", "Intent": "intent", "Recommendation": "page_recommendation"}[colour_by]
    fig = px.scatter(
        agg, x="x", y="y", color=colour_col,
        hover_data=["keyword", "page_label", "pillar_label", "intent", "volume", "source"],
        title=f"Keywords coloured by {colour_by}", width=1100, height=700,
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.error("Error while rendering the scatter plot.")
    st.code(traceback.format_exc())

# ----------------------------- Step 11: Export -----------------------------
st.subheader("11. Export")
try:
    cols = ["keyword", "volume", "position", "source", "intent",
            "page_label", "page_recommendation", "pillar_label"]
    rename = {
        "keyword": "Keyword", "volume": "Volume", "position": "Position", "source": "Source",
        "intent": "Intent", "page_label": "Page", "page_recommendation": "Page recommendation",
        "pillar_label": "Pillar",
    }
    if pillar_levels == 2:
        cols.append("superpillar_label")
        rename["superpillar_label"] = "Super-pillar"
    cols.append("page_reason")
    rename["page_reason"] = "Why"

    export_df = agg[cols].rename(columns=rename)
    st.download_button(
        "Download keyword-level mapping (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        "keyword_page_mapping.csv", "text/csv",
    )

    page_export = pages_df.rename(columns={
        "page_label": "Page", "pillar_label": "Pillar", "intent": "Intent",
        "n_keywords": "Keywords", "total_volume": "Total volume", "n_sources": "Sources",
        "recommendation": "Recommendation", "reason": "Why", "head_term": "Head term",
    })[["Page", "Pillar", "Recommendation", "Intent", "Keywords", "Total volume", "Sources", "Head term", "Why"]]
    st.download_button(
        "Download page-level summary (CSV)",
        page_export.to_csv(index=False).encode("utf-8"),
        "page_summary.csv", "text/csv",
    )
    st.success("Done. Keyword-level mapping and page-level summary are ready to download.")
except Exception:
    st.error("Error while preparing the export.")
    st.code(traceback.format_exc())
