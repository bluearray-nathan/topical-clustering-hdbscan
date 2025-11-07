# app.py — Topical Clustering Tool
# ----------------------------------------------------------------------------
# Adds a hierarchical topical structure to the page-level clustering output
# from Keyword Insights. Groups similar clusters into Parent Topics and Child
# Subtopics to help map out a potential information architecture.

import time
import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import openai
import hdbscan
import concurrent.futures as cf
import traceback
import random
from collections import deque

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

random.seed(42)

# ----------------------------- Page config & title -----------------------------
st.set_page_config(page_title="Topical Clustering Tool", layout="wide")
st.title("Topical Clustering Tool")

# ----------------------------- Explanation (expandable) -----------------------------
with st.expander("What this tool does and how it works"):
    st.markdown("""
**What it does**  
Adds a **hierarchical topical structure** to the page-level clustering output from Keyword Insights.  
It groups similar clusters into **Parent Topics** and **Child Subtopics**, helping you **map out a potential information architecture** and identify content themes or silos.

---

### How it works (simple)
1. **Upload your CSV** — Uses your `cluster`, `keyword`, and `search volume` columns.  
2. **Generate embeddings** — Converts each cluster into a vector so similar ones sit close together.  
3. **Group into Parent Topics** — Finds broader themes using HDBSCAN clustering.  
4. **Split into Child Subtopics** — Refines each parent group into smaller related topics.  
5. **Label topics (GPT)** — Suggests short, descriptive names based on examples and common phrases.  
6. **Visualise & export** — Displays Parent Topics on a 2D scatter plot and exports to CSV.

---

### HDBSCAN selection methods (plain-English)
- **EOM (Excess of Mass)** → *Fewer, bigger subtopics.* Think: "**merge similar things together**" to keep groups stable.
- **Leaf** → *More, finer subtopics.* Think: "**split to the smallest stable pieces**" for extra detail.

You don't need to know the maths—just choose whether you want **broader** or **more granular** subtopics.
""")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Add it in Streamlit Cloud Secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

# ----------------------------- Session init for label caches -----------------------------
if "parent_labels_map" not in st.session_state:
    st.session_state.parent_labels_map = {}
if "child_labels_map" not in st.session_state:
    st.session_state.child_labels_map = {}
if "last_parent_hash" not in st.session_state:
    st.session_state.last_parent_hash = None
if "last_child_hash" not in st.session_state:
    st.session_state.last_child_hash = None

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Setup")
    st.text("Embedding model")
    st.code("text-embedding-3-large", language="text")

    st.header("Topic granularity (Parent)")
    parent_granularity = st.radio(
        "How broad should PARENT topics be?",
        options=["Fewer, broader topics", "Balanced (recommended)", "More, finer subtopics"],
        index=1,
        help="Controls the high-level grouping. Broader = fewer parent clusters; Finer = more parent clusters."
    )

    st.header("Topic granularity (Child)")
    child_granularity = st.radio(
        "How detailed should CHILD topics be?",
        options=["Fewer, broader subtopics", "Balanced (recommended)", "More, finer subtopics"],
        index=1,
        help="Controls how aggressively we split each parent into child subtopics."
    )
    st.caption("**EOM** = fewer, bigger subtopics • **Leaf** = more, finer subtopics")

    st.header("Labelling")
    # Fixed 1–4 word labels (no slider)
    MAX_LABEL_WORDS = 4
    st.caption("Labels use **1–4 words** automatically (1 word if sufficient).")

    auto_label_topics = st.checkbox("Auto-label topics with GPT", value=True,
                                    help="Automatically request GPT labels after clustering.")
    relabel_now = st.button("🔁 Re-label now", help="Force relabelling even if clusters haven't changed.")
    label_model = "gpt-4o-mini-2024-07-18"
    label_temp = st.slider("Labelling creativity", 0.0, 1.0, 0.2, 0.05,
                           help="Higher = more creative names; lower = more conservative names.")

    # ------------------------- CHANGE 1: Stronger child presets + selection method -------------------------
    PARENT_PRESETS = {
        "Fewer, broader topics": {
            "umap": {"neighbors": 60, "components": 8},
            "parent": {"min_cluster_size": 24, "min_samples": 2, "epsilon": 0.08},
        },
        "Balanced (recommended)": {
            "umap": {"neighbors": 40, "components": 10},
            "parent": {"min_cluster_size": 16, "min_samples": 2, "epsilon": 0.06},
        },
        "More, finer subtopics": {
            "umap": {"neighbors": 20, "components": 15},
            "parent": {"min_cluster_size": 10, "min_samples": 3, "epsilon": 0.04},
        },
    }

    CHILD_PRESETS = {
        # Fewer, broader → bigger clusters, more merging; keep EOM
        "Fewer, broader subtopics": {
            "child_base": {"mcs": 20, "ms": 5, "eps": 0.08},
            "adaptive": {
                "k_divisor": 28, "alpha_eps": 1.10,
                "eps_low": 0.03, "eps_high": 0.12
            },
            "method": "eom"
        },
        # Balanced → similar to original
        "Balanced (recommended)": {
            "child_base": {"mcs": 8, "ms": 2, "eps": 0.04},
            "adaptive": {
                "k_divisor": 12, "alpha_eps": 0.90,
                "eps_low": 0.01, "eps_high": 0.08
            },
            "method": "eom"
        },
        # More, finer → smaller clusters, tighter eps, Leaf mode for finer splits
        "More, finer subtopics": {
            "child_base": {"mcs": 3, "ms": 1, "eps": 0.02},
            "adaptive": {
                "k_divisor": 8, "alpha_eps": 0.75,
                "eps_low": 0.005, "eps_high": 0.05
            },
            "method": "leaf"
        },
    }

    p_parent = PARENT_PRESETS[parent_granularity]
    p_child = CHILD_PRESETS[child_granularity]

    UMAP_NEIGHBORS = p_parent["umap"]["neighbors"]
    UMAP_COMPONENTS = p_parent["umap"]["components"]
    min_cluster_size_parent = p_parent["parent"]["min_cluster_size"]
    min_samples_parent = p_parent["parent"]["min_samples"]
    epsilon_parent = p_parent["parent"]["epsilon"]

    min_cluster_size_child_base = p_child["child_base"]["mcs"]
    min_samples_child_base = p_child["child_base"]["ms"]
    epsilon_child_base = p_child["child_base"]["eps"]
    k_divisor = p_child["adaptive"]["k_divisor"]
    alpha_eps = p_child["adaptive"]["alpha_eps"]
    eps_low_bound = p_child["adaptive"]["eps_low"]
    eps_high_bound = p_child["adaptive"]["eps_high"]
    child_selection_method = p_child.get("method", "eom")  # "eom" or "leaf"

# ----------------------------- Upload data -----------------------------
st.subheader("1️⃣ Upload your CSV")
file = st.file_uploader("Upload a CSV with columns: cluster, keyword, search volume", type=["csv"])
if not file:
    st.info("Upload your CSV to begin.")
    st.stop()

try:
    df_raw = pd.read_csv(file)
except Exception:
    st.error("Could not read CSV. Please check the file encoding/format.")
    st.code(traceback.format_exc())
    st.stop()

df_raw.columns = [c.strip().lower() for c in df_raw.columns]
required_cols = {"cluster", "keyword", "search volume"}
if not required_cols.issubset(df_raw.columns):
    st.error("CSV must include columns: cluster, keyword, search volume (case-insensitive).")
    st.stop()

if len(df_raw) == 0:
    st.warning("The CSV has no rows.")
    st.stop()

st.success(f"✅ Loaded {len(df_raw)} page-level clusters.")
df_raw["descriptive_name"] = df_raw["cluster"].astype(str)

# ----------------------------- Embeddings from `cluster` -----------------------------
st.subheader("2️⃣ Generate embeddings (from `cluster`)")

@st.cache_data(show_spinner=False)
def embed_texts(texts):
    model = "text-embedding-3-large"
    vecs = []
    texts = [t if isinstance(t, str) else "" for t in texts]
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.10)  # simple pacing
    return np.array(vecs, dtype=np.float32)

try:
    clusters_list = df_raw["cluster"].fillna("").astype(str).tolist()
    if all(s.strip() == "" for s in clusters_list):
        st.error("All `cluster` values are empty. Please provide non-empty cluster strings.")
        st.stop()

    embeddings = embed_texts(clusters_list)
    if embeddings.size == 0 or embeddings.shape[0] != len(df_raw):
        st.error("Failed to create embeddings. Please verify input data.")
        st.stop()

    embeddings = np.nan_to_num(embeddings, posinf=0.0, neginf=0.0)
    embeddings = normalize(embeddings)  # cosine-friendly
except Exception:
    st.error("Error while creating embeddings.")
    st.code(traceback.format_exc())
    st.stop()

st.success(f"✅ Created {len(embeddings)} embeddings.")

# ----------------------------- UMAP (always on; set by PARENT granularity) -----------------------------
st.subheader("3️⃣ Smoothing (UMAP)")
try:
    from umap import UMAP
    n_samples = embeddings.shape[0]
    n_neighbors = int(max(2, min(int(UMAP_NEIGHBORS), max(2, n_samples - 1))))
    n_components = int(max(2, min(int(UMAP_COMPONENTS), min(embeddings.shape[1], max(2, n_samples - 1)))))

    umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.0,
        n_components=n_components,
        metric="cosine",
        random_state=42
    )
    X_for_cluster = umap.fit_transform(embeddings)
    st.success(f"✅ UMAP applied (neighbors={n_neighbors}, components={n_components}).")
except Exception:
    st.error("Error during UMAP smoothing.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- HDBSCAN Pass A — Parents -----------------------------
st.subheader("4️⃣ HDBSCAN (Parents — coarse)")
hdb_parent = dict(
    min_cluster_size=int(min_cluster_size_parent),
    min_samples=int(min_samples_parent),
    cluster_selection_method="eom",
    cluster_selection_epsilon=float(epsilon_parent),
    metric="euclidean"
)

try:
    parenter = hdbscan.HDBSCAN(**hdb_parent)
    labels_parent = parenter.fit_predict(X_for_cluster)

    df = df_raw.copy()
    df["parent_id"] = labels_parent
    n_parents = len(set(labels_parent)) - (1 if -1 in labels_parent else 0)
    noise_parent_pct = (labels_parent == -1).mean() * 100 if len(labels_parent) else 0.0
    st.success(f"✅ Parent clusters: {n_parents} • Parent noise: {noise_parent_pct:.1f}%")
except Exception:
    st.error("Error during HDBSCAN (parents).")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Adaptive helpers for child params -----------------------------
def _median_knn_distance(X, k=10):
    """
    Median distance to the k-th nearest neighbour (euclidean) for X.
    """
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, metric="euclidean")
    knn_k = min(k + 1, D.shape[0])  # include self; take last among first k+1 sorted
    sorted_rows = np.sort(D, axis=1)[:, :knn_k]
    kth = sorted_rows[:, -1]
    return float(np.median(kth))

def derive_child_params_for_parent(parent_indices, *,
                                   base_low_mcs=5, base_high_mcs=50,
                                   k_divisor=12,
                                   X_for_cluster=None,
                                   alpha=0.9,
                                   eps_low=0.01, eps_high=0.08):
    """
    Auto-scales child HDBSCAN params for a *single* parent (UMAP euclidean space).
    Returns: (min_cluster_size_child_i, min_samples_child_i, epsilon_child_i, do_children)
    """
    n = len(parent_indices)
    # 1) min_cluster_size scales with parent size
    mcs = int(np.clip(round(n / k_divisor), base_low_mcs, base_high_mcs))

    # 2) density proxy -> min_samples via median kNN distance
    X_sub = X_for_cluster[parent_indices]
    med_knn = _median_knn_distance(X_sub, k=min(10, max(2, n - 1)))

    if med_knn <= 0.15:
        min_samples = 5
    elif med_knn <= 0.25:
        min_samples = 4
    elif med_knn <= 0.35:
        min_samples = 3
    elif med_knn <= 0.50:
        min_samples = 2
    else:
        min_samples = 1

    # 3) epsilon from local spacing with bounds
    epsilon = float(np.clip(alpha * med_knn, eps_low, eps_high))

    # 4) stability floor
    if n < 2 * mcs:
        return mcs, min_samples, epsilon, False  # skip child clustering

    return mcs, min_samples, epsilon, True

# ----------------------------- HDBSCAN Pass B — Children per Parent (Adaptive) -----------------------------
st.subheader("5️⃣ HDBSCAN (Children — adaptive per parent)")
child_ids = np.full(len(df), -1, dtype=int)

try:
    next_child_base = 0  # ensures globally unique child ids
    unique_parents = sorted(set(labels_parent))

    for pid in unique_parents:
        if pid == -1:
            continue
        idx = np.where(labels_parent == pid)[0]
        if len(idx) == 0:
            continue

        # Derive per-parent child params; fallback baseline if anything odd happens
        try:
            mcs_i, ms_i, eps_i, do_children = derive_child_params_for_parent(
                idx,
                base_low_mcs=5, base_high_mcs=50,
                k_divisor=int(k_divisor),        # from CHILD granularity
                X_for_cluster=X_for_cluster,
                alpha=float(alpha_eps),          # from CHILD granularity
                eps_low=float(eps_low_bound),    # from CHILD granularity
                eps_high=float(eps_high_bound)   # from CHILD granularity
            )
        except Exception:
            mcs_i, ms_i, eps_i, do_children = (
                int(min_cluster_size_child_base),
                int(min_samples_child_base),
                float(epsilon_child_base),
                len(idx) >= 2 * int(min_cluster_size_child_base),
            )

        if not do_children:
            # Too small to split reliably -> bucket as a single child
            child_ids[idx] = next_child_base
            next_child_base += 1
            continue

        # ------------------------- CHANGE 2: Use preset-provided selection method -------------------------
        child_params_local = dict(
            min_cluster_size=int(mcs_i),
            min_samples=int(ms_i),
            cluster_selection_method=child_selection_method,  # "eom" (fewer, bigger) or "leaf" (more, finer)
            cluster_selection_epsilon=float(eps_i),
            metric="euclidean"
        )

        ch_local = hdbscan.HDBSCAN(**child_params_local).fit_predict(X_for_cluster[idx])

        # Map local child labels to global ids
        unique_local = [c for c in sorted(set(ch_local)) if c != -1]
        mapping = {c: (next_child_base + i) for i, c in enumerate(unique_local)}
        child_ids[idx] = np.array([mapping.get(c, -1) for c in ch_local])
        next_child_base += len(unique_local)

    df["child_id"] = child_ids
    n_children = len(set(child_ids)) - (1 if -1 in child_ids else 0)
    noise_child_pct = (child_ids == -1).mean() * 100 if len(child_ids) else 0.0
    st.success(f"✅ Child clusters: {n_children} • Child noise: {noise_child_pct:.1f}% (adaptive, {child_selection_method.upper()} mode)")
except Exception:
    st.error("Error during HDBSCAN (children, adaptive).")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Scalable facet-aware labelling helpers -----------------------------
def top_facets(texts, top_k=10, ngram_range=(1,2), min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"):
    """
    Returns a list of (phrase, count) for salient unigrams/bigrams in `texts`.
    """
    if not texts:
        return []
    cv = CountVectorizer(stop_words="english", ngram_range=ngram_range, min_df=min_df, token_pattern=token_pattern)
    try:
        X = cv.fit_transform(texts)
    except ValueError:
        return []
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(cv.get_feature_names_out())
    order = counts.argsort()[::-1]
    pairs = [(vocab[i], int(counts[i])) for i in order[:top_k] if counts[i] > 0]
    return pairs

def facets_block(facets, max_lines=8):
    if not facets:
        return "Facet distribution: (no salient facets detected)"
    total = sum(c for _, c in facets) or 1
    lines = []
    for phrase, cnt in facets[:max_lines]:
        pct = 100.0 * cnt / total
        lines.append(f"- {phrase}: {cnt} ({pct:.1f}%)")
    return "Facet distribution:\n" + "\n".join(lines)

def diversified_examples(indices, titles_all, embeddings, total_max=40):
    """
    Select representative examples covering sub-modes using KMeans over the cluster slice.
    """
    n = len(indices)
    if n <= total_max:
        return [titles_all[i] for i in indices]

    # k ~ sqrt(n), bounded between 5 and total_max (cap 20 to avoid tiny clusters exploding)
    k = int(np.clip(int(np.sqrt(n)), 5, min(20, total_max)))
    vecs = embeddings[indices]
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(vecs)
    centers = km.cluster_centers_

    picked = []
    used = set()
    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        sub = vecs[idx_c]
        d = ((sub - centers[c])**2).sum(axis=1)
        best_local = indices[idx_c[d.argmin()]]
        if best_local not in used:
            picked.append(titles_all[best_local])
            used.add(best_local)

    # top up to total_max with random remaining examples
    remaining = [i for i in indices if i not in used]
    if remaining:
        need = max(0, total_max - len(picked))
        picked.extend([titles_all[i] for i in random.sample(remaining, min(need, len(remaining)))])
    return picked[:total_max]

def label_topic_short(titles_all, indices, embeddings, model_name, temp, max_words,
                      total_max_examples=40, facets_top_k=10):
    """
    titles_all: list[str] for the whole dataset (indexable by absolute row index)
    indices: np.array/list of absolute row indices belonging to the cluster to label
    embeddings: np.ndarray of all row embeddings (already normalized)
    """
    # 1) Diverse sample
    sample = diversified_examples(indices, titles_all, embeddings, total_max=total_max_examples)

    # 2) Facet distribution
    cluster_texts = [titles_all[i] for i in indices]
    facets = top_facets(cluster_texts, top_k=facets_top_k)
    facts = facets_block(facets)

    # 3) Guardrails
    dominance_rule = (
        "Name the topic to reflect the overall cluster. "
        "If multiple distinct facets (e.g., locations, formats, levels, audiences, industries) are present, "
        "avoid naming a single facet unless it clearly dominates the cluster. "
        "Prefer inclusive or general phrasing in those cases."
    )

    joined = ", ".join(sample)
    prompt = (
        f"{facts}\n\n"
        "These page titles are about a similar topic:\n"
        f"{joined}\n\n"
        f"{dominance_rule}\n"
        "Return a concise, human-friendly topic name in 1–4 words (noun phrase; minimal punctuation). "
        "Use 1 word if it fully captures the theme; otherwise 2–4 words for clarity."
    )

    resp = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an SEO content analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
    )
    text = resp.choices[0].message.content.strip()

    # enforce word cap (hard backstop)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text

def label_groups(df_in, id_col, label_col_name, label_model, label_temp, max_words,
                 embeddings, titles_all, total_max_examples=40, facets_top_k=10):
    unique_ids = [i for i in sorted(df_in[id_col].unique()) if i != -1]
    labels_map = {-1: "Other"}  # renamed from "Noise / Misc" to "Other"
    MAX_WORKERS = min(12, max(1, len(unique_ids)))
    progress = st.progress(0.0)
    status = st.empty()
    done, total = 0, len(unique_ids)

    timings = deque(maxlen=20)

    def one(gid):
        idx = np.where(df_in[id_col].values == gid)[0]  # absolute row indices
        start = time.time()
        name = label_topic_short(
            titles_all, idx, embeddings,
            model_name=label_model, temp=label_temp, max_words=max_words,
            total_max_examples=total_max_examples, facets_top_k=facets_top_k
        )
        dur = time.time() - start
        return gid, name, dur

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(one, gid) for gid in unique_ids]
        for f in cf.as_completed(futures):
            gid, name, dur = f.result()
            labels_map[gid] = name
            timings.append(dur)
            done += 1
            progress.progress(done / max(1, total))
            spc = float(np.mean(timings)) if len(timings) else 0.0
            eta_sec = max(total - done, 0) * spc
            status.text(f"Labeled {done}/{total} • avg {spc:.2f}s • ETA ~{int(eta_sec//60)}m {int(eta_sec%60)}s")

    df_in[label_col_name] = df_in[id_col].map(labels_map).astype(str)
    return labels_map

# ----------------------------- Topic labelling (parents then children) -----------------------------
try:
    # Hashes for cache invalidation
    parent_hash = hashlib.md5(np.array(df["parent_id"], dtype=np.int64).tobytes()).hexdigest()
    child_hash  = hashlib.md5(np.array(df[["parent_id","child_id"]], dtype=np.int64).tobytes()).hexdigest()

    should_label_parents = auto_label_topics and (st.session_state.last_parent_hash != parent_hash or relabel_now)
    should_label_children = auto_label_topics and (st.session_state.last_child_hash != child_hash or relabel_now)

    titles_all = df["descriptive_name"].tolist()

    if auto_label_topics:
        if should_label_parents:
            st.subheader("6️⃣ Labelling parents")
            st.session_state.parent_labels_map = label_groups(
                df, "parent_id", "parent_label", label_model, label_temp, MAX_LABEL_WORDS,
                embeddings=embeddings, titles_all=titles_all, total_max_examples=40, facets_top_k=10
            )
            st.session_state.last_parent_hash = parent_hash
        else:
            if "parent_label" not in df.columns and st.session_state.parent_labels_map:
                df["parent_label"] = df["parent_id"].map(st.session_state.parent_labels_map).astype(str)

        if should_label_children:
            st.subheader("7️⃣ Labelling children")
            st.session_state.child_labels_map = label_groups(
                df, "child_id", "child_label", label_model, label_temp, MAX_LABEL_WORDS,
                embeddings=embeddings, titles_all=titles_all, total_max_examples=40, facets_top_k=10
            )
            st.session_state.last_child_hash = child_hash
        else:
            if "child_label" not in df.columns and st.session_state.child_labels_map:
                df["child_label"] = df["child_id"].map(st.session_state.child_labels_map).astype(str)
    else:
        df["parent_label"] = df["parent_id"].astype(str)
        df["child_label"]  = df["child_id"].astype(str)

    if "parent_label" not in df.columns:
        df["parent_label"] = df["parent_id"].map(st.session_state.parent_labels_map or {-1: "Other"}).astype(str)
    if "child_label" not in df.columns:
        df["child_label"] = df["child_id"].map(st.session_state.child_labels_map or {-1: "Other"}).astype(str)

    st.success("✅ Labelling step complete.")
except Exception:
    st.error("Error during topic labelling.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Visualisation -----------------------------
st.subheader("8️⃣ Visualise hierarchy (2D PCA)")
try:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # Parent label only (no child symbol)
    fig = px.scatter(
        df, x="x", y="y",
        color="parent_label",
        hover_data=["parent_label", "descriptive_name", "cluster", "keyword", "search volume"],
        title="Parent Topical Clusters (HDBSCAN selection: EOM)",
        width=1100, height=720
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.error("Error while rendering the PCA scatter plot.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Summaries -----------------------------
st.subheader("9️⃣ Parent summary")
try:
    parent_summary = (
        df[df["parent_id"] != -1]
        .groupby(["parent_id", "parent_label"])
        .agg(size=("descriptive_name", "size"))
        .reset_index()
        .sort_values("size", ascending=False)
    )
    st.dataframe(parent_summary, use_container_width=True, height=300)
except Exception:
    st.error("Error while building the parent summary.")
    st.code(traceback.format_exc())

st.subheader("🔟 Child summary (per parent)")
try:
    child_summary = (
        df[df["child_id"] != -1]
        .groupby(["parent_id", "parent_label", "child_id", "child_label"])
        .agg(size=("descriptive_name", "size"))
        .reset_index()
        .sort_values(["parent_id", "size"], ascending=[True, False])
    )
    st.dataframe(child_summary, use_container_width=True, height=420)
except Exception:
    st.error("Error while building the child summary.")
    st.code(traceback.format_exc())

# ----------------------------- Export -----------------------------
st.subheader("1️⃣1️⃣ Export")
try:
    # Export excludes "Cluster (descriptive name)"
    export_df = df[[
        "parent_label", "child_label", "cluster", "keyword", "search volume"
    ]].rename(columns={
        "parent_label": "Parent Topic",
        "child_label": "Child Topic"
    })

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Hierarchical Topics CSV", csv, "hierarchical_topics_simple.csv", "text/csv")
    st.success("✅ Done! Export excludes 'Cluster (descriptive name)'. Noise groups are labelled 'Other'. Visualisation shows parent topics only.")
except Exception:
    st.error("Error while preparing the CSV export.")
    st.code(traceback.format_exc())
    st.stop()






















