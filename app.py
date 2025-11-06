# app.py — Fast Hierarchical Topic Clustering (Parent & Child)
# ----------------------------------------------------------------------------
# Input CSV columns (required):
#   - cluster (main page-level cluster keyword)
#   - keyword (all keywords within that page-level cluster, e.g., comma-separated)
#   - search volume (total search volume of the page-level cluster)
#
# Flow:
# 1) Embeds ONLY the `cluster` text (OpenAI text-embedding-3-large, fixed)
# 2) Optional UMAP smoothing via presets (Off/Broad/Balanced/Detailed)
# 3) HDBSCAN pass A → Parent clusters (coarse)
# 4) HDBSCAN pass B → Child clusters within each parent (fine)
# 5) GPT topic labels (parents then children) with concurrency + progress
# 6) Visualisation (PCA scatter)
# 7) Summaries (parents, then children)
# 8) Export: Parent Topic, Child Topic, Cluster (descriptive name), cluster, keyword, search volume
#
# Notes:
# - Streamlit deprecation: use width="stretch" instead of use_container_width.
# - Robust error surfacing with visible tracebacks.

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
from collections import deque

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import plotly.express as px

st.set_page_config(page_title="Topical Clustering (Fast • Hierarchical)", layout="wide")
st.title("🧩 Topical Clustering (Fast • Hierarchical)")
st.caption("Embeds the page-level cluster keyword • UMAP presets • HDBSCAN Parents→Children • GPT labels with progress")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Add it in Streamlit Cloud Secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

# ----------------------------- Preset defaults -----------------------------
PRESET_DEFAULTS = {
    "Off":      {"min_cluster_size": 10, "min_samples": 2, "epsilon": 0.05, "umap_neighbors": None, "umap_components": None},
    "Broad":    {"min_cluster_size": 12, "min_samples": 1, "epsilon": 0.06, "umap_neighbors": 50, "umap_components": 8},
    "Balanced": {"min_cluster_size": 10, "min_samples": 2, "epsilon": 0.05, "umap_neighbors": 40, "umap_components": 10},
    "Detailed": {"min_cluster_size": 8,  "min_samples": 3, "epsilon": 0.03, "umap_neighbors": 20, "umap_components": 15},
}

# Initialize session state
if "umap_preset" not in st.session_state:
    st.session_state.umap_preset = "Balanced"
if "last_preset_applied" not in st.session_state:
    st.session_state.last_preset_applied = None
if "min_cluster_size" not in st.session_state:
    st.session_state.min_cluster_size = PRESET_DEFAULTS["Balanced"]["min_cluster_size"]
if "min_samples" not in st.session_state:
    st.session_state.min_samples = PRESET_DEFAULTS["Balanced"]["min_samples"]
if "cluster_selection_epsilon" not in st.session_state:
    st.session_state.cluster_selection_epsilon = PRESET_DEFAULTS["Balanced"]["epsilon"]

# Label maps & hashes
if "parent_labels_map" not in st.session_state:
    st.session_state.parent_labels_map = {}
if "child_labels_map" not in st.session_state:
    st.session_state.child_labels_map = {}
if "last_parent_hash" not in st.session_state:
    st.session_state.last_parent_hash = None
if "last_child_hash" not in st.session_state:
    st.session_state.last_child_hash = None

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Embedding")
    st.text("Embedding model")
    st.code("text-embedding-3-large", language="text")

    st.header("UMAP Smoothing")
    umap_preset = st.selectbox(
        "Preset",
        options=["Off", "Broad", "Balanced", "Detailed"],
        index=["Off", "Broad", "Balanced", "Detailed"].index(st.session_state.umap_preset),
        help=(
            "UMAP compresses the embedding space so related items sit closer together.\n"
            "• Broad: fewer, bigger topics, least noise (size=12, samples=1, ε=0.06)\n"
            "• Balanced: general use (size=10, samples=2, ε=0.05)\n"
            "• Detailed: more subtopics, possibly more noise (size=8, samples=3, ε=0.03)"
        )
    )
    if umap_preset != st.session_state.last_preset_applied:
        st.session_state.umap_preset = umap_preset
        d = PRESET_DEFAULTS[umap_preset]
        st.session_state.min_cluster_size = d["min_cluster_size"]
        st.session_state.min_samples = d["min_samples"]
        st.session_state.cluster_selection_epsilon = d["epsilon"]
        st.session_state.last_preset_applied = umap_preset

    st.header("Hierarchy Density")
    st.subheader("Parents (coarse)")
    min_cluster_size_parent = st.slider("Parent min size", 2, 200, 20)
    min_samples_parent      = st.slider("Parent min samples", 1, 10, 2)
    epsilon_parent          = st.slider("Parent ε (EoM gap-bridge)", 0.00, 0.20, 0.07, 0.01)

    st.subheader("Children (fine)")
    min_cluster_size_child = st.slider("Child min size", 2, 100, 8)
    min_samples_child      = st.slider("Child min samples", 1, 10, 2)
    epsilon_child          = st.slider("Child ε (EoM gap-bridge)", 0.00, 0.20, 0.04, 0.01)

    st.header("Labelling")
    auto_label_topics = st.checkbox("Auto-label parents & children with GPT", value=True)
    relabel_now = st.button("🔁 Re-label now")
    label_model = "gpt-4o-mini-2024-07-18"
    label_temp = st.slider("Labelling creativity", 0.0, 1.0, 0.2, 0.05)

    with st.expander("Throughput helper (what-if)"):
        n_est = st.number_input("Groups to label", 1, 100000, value=1000, step=1)
        conc_est = st.slider("Concurrency (workers)", 1, 32, value=12)
        avg_est = st.number_input("Avg seconds per label (observed)", 0.1, 60.0, value=1.8, step=0.1)
        batches = math.ceil(n_est / conc_est)
        est_sec = batches * avg_est
        st.caption(f"Naive parallel wall-time ≈ {int(est_sec//60)}m {int(est_sec%60)}s (depends on rate limits/retries)")

# ----------------------------- File upload -----------------------------
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

# No LLM naming — descriptive_name == cluster (kept for downstream consistency/export)
df_raw["descriptive_name"] = df_raw["cluster"].astype(str)

# ----------------------------- Embeddings from `cluster` -----------------------------
st.subheader("2️⃣ Generate embeddings (from `cluster`)")

@st.cache_data(show_spinner=False)
def embed_texts(texts):
    model = "text-embedding-3-large"  # locked
    vecs = []
    texts = [t if isinstance(t, str) else "" for t in texts]
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.10)
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

    # Normalize and basic NaN/Inf guard
    embeddings = np.nan_to_num(embeddings, posinf=0.0, neginf=0.0)
    embeddings = normalize(embeddings)
except Exception:
    st.error("Error while creating embeddings.")
    st.code(traceback.format_exc())
    st.stop()

st.success(f"✅ Created {len(embeddings)} embeddings.")

# ----------------------------- UMAP (presets) -----------------------------
st.subheader("3️⃣ Smoothing (optional)")
use_umap = (st.session_state.umap_preset != "Off")

X_for_cluster = None
distance_matrix = None

try:
    if use_umap:
        from umap import UMAP
        n_neighbors = PRESET_DEFAULTS[st.session_state.umap_preset]["umap_neighbors"]
        n_components = PRESET_DEFAULTS[st.session_state.umap_preset]["umap_components"]

        # Guards for tiny datasets: ensure sensible sizes
        n_samples = embeddings.shape[0]
        if n_neighbors is not None:
            n_neighbors = max(2, min(n_neighbors, max(2, n_samples - 1)))
        if n_components is not None:
            n_components = max(2, min(n_components, min(embeddings.shape[1], n_samples - 1)))

        umap = UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=n_components,
            metric="cosine",
            random_state=42
        )
        X_for_cluster = umap.fit_transform(embeddings)
        st.success(f"✅ UMAP applied (n_neighbors={n_neighbors}, n_components={n_components}).")
    else:
        distance_matrix = cosine_distances(embeddings)
        st.info("UMAP Off — using cosine distances directly.")
except Exception:
    st.error("Error during UMAP smoothing.")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- HDBSCAN Pass A — Parents -----------------------------
st.subheader("4️⃣ HDBSCAN (Parents — coarse)")
hdb_parent = dict(
    min_cluster_size=min_cluster_size_parent,
    min_samples=min_samples_parent,
    cluster_selection_method="eom",
    cluster_selection_epsilon=float(epsilon_parent),
)

try:
    if use_umap:
        hdb_parent["metric"] = "euclidean"
        parenter = hdbscan.HDBSCAN(**hdb_parent)
        labels_parent = parenter.fit_predict(X_for_cluster)
    else:
        hdb_parent["metric"] = "precomputed"
        parenter = hdbscan.HDBSCAN(**hdb_parent)
        labels_parent = parenter.fit_predict(distance_matrix)

    df = df_raw.copy()
    df["parent_id"] = labels_parent
    n_parents = len(set(labels_parent)) - (1 if -1 in labels_parent else 0)
    noise_parent_pct = (labels_parent == -1).mean() * 100 if len(labels_parent) else 0.0
    st.success(f"✅ Parent clusters: {n_parents} • Parent noise: {noise_parent_pct:.1f}%")
except Exception:
    st.error("Error during HDBSCAN (parents).")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- HDBSCAN Pass B — Children per Parent -----------------------------
st.subheader("5️⃣ HDBSCAN (Children — fine within each parent)")
child_ids = np.full(len(df), -1, dtype=int)
hdb_child_base_params = dict(
    min_cluster_size=min_cluster_size_child,
    min_samples=min_samples_child,
    cluster_selection_method="eom",
    cluster_selection_epsilon=float(epsilon_child),
)

try:
    next_child_base = 0  # ensures globally unique child ids (single int col)
    unique_parents = sorted(set(labels_parent))
    for pid in unique_parents:
        if pid == -1:
            continue
        idx = np.where(labels_parent == pid)[0]
        if len(idx) == 0:
            continue

        # If too small for a second clustering, bucket as one child
        if len(idx) < hdb_child_base_params["min_cluster_size"]:
            child_ids[idx] = next_child_base
            next_child_base += 1
            continue

        if use_umap:
            X_sub = X_for_cluster[idx]
            child_params = {**hdb_child_base_params, "metric": "euclidean"}
            ch_local = hdbscan.HDBSCAN(**child_params).fit_predict(X_sub)
        else:
            D_sub = distance_matrix[np.ix_(idx, idx)]
            child_params = {**hdb_child_base_params, "metric": "precomputed"}
            ch_local = hdbscan.HDBSCAN(**child_params).fit_predict(D_sub)

        # map local child labels to global unique ids
        unique_local = [c for c in sorted(set(ch_local)) if c != -1]
        mapping = {c: (next_child_base + i) for i, c in enumerate(unique_local)}
        child_ids[idx] = np.array([mapping.get(c, -1) for c in ch_local])
        next_child_base += len(unique_local)

    df["child_id"] = child_ids
    n_children = len(set(child_ids)) - (1 if -1 in child_ids else 0)
    noise_child_pct = (child_ids == -1).mean() * 100 if len(child_ids) else 0.0
    st.success(f"✅ Child clusters: {n_children} • Child noise: {noise_child_pct:.1f}%")
except Exception:
    st.error("Error during HDBSCAN (children).")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------- Topic labelling (parents then children) -----------------------------
def label_topic_short(titles, model_name, temp):
    joined = ", ".join(titles[:40])
    prompt = (
        "These page titles are about a similar topic:\n"
        f"{joined}\n\n"
        "Return ONLY a concise topic name (2–4 words, noun phrase, no punctuation)."
    )
    resp = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an SEO content analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
    )
    return resp.choices[0].message.content.strip()

def label_groups(df_in, id_col, label_col_name, label_model, label_temp):
    unique_ids = [i for i in sorted(df_in[id_col].unique()) if i != -1]
    labels_map = {-1: "Noise / Misc"}
    MAX_WORKERS = min(12, max(1, len(unique_ids)))
    progress = st.progress(0.0)
    status = st.empty()
    done, total = 0, len(unique_ids)

    def one(gid):
        titles = df_in.loc[df_in[id_col] == gid, "descriptive_name"].head(40).tolist()
        start = time.time()
        name = label_topic_short(titles, label_model, label_temp)
        dur = time.time() - start
        return gid, name, dur

    timings = deque(maxlen=20)
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

try:
    # Hashes for cache invalidation
    parent_hash = hashlib.md5(np.array(df["parent_id"], dtype=np.int64).tobytes()).hexdigest()
    child_hash  = hashlib.md5(np.array(df[["parent_id","child_id"]], dtype=np.int64).tobytes()).hexdigest()

    should_label_parents = auto_label_topics and (st.session_state.last_parent_hash != parent_hash or relabel_now)
    should_label_children = auto_label_topics and (st.session_state.last_child_hash != child_hash or relabel_now)

    if auto_label_topics:
        if should_label_parents:
            st.subheader("6️⃣ Labelling parents")
            st.session_state.parent_labels_map = label_groups(
                df, "parent_id", "parent_label", label_model, label_temp
            )
            st.session_state.last_parent_hash = parent_hash
        else:
            # Ensure column exists from previous run
            if "parent_label" not in df.columns and st.session_state.parent_labels_map:
                df["parent_label"] = df["parent_id"].map(st.session_state.parent_labels_map).astype(str)

        if should_label_children:
            st.subheader("7️⃣ Labelling children")
            st.session_state.child_labels_map = label_groups(
                df, "child_id", "child_label", label_model, label_temp
            )
            st.session_state.last_child_hash = child_hash
        else:
            if "child_label" not in df.columns and st.session_state.child_labels_map:
                df["child_label"] = df["child_id"].map(st.session_state.child_labels_map).astype(str)
    else:
        df["parent_label"] = df["parent_id"].astype(str)
        df["child_label"]  = df["child_id"].astype(str)

    # If labels just created by label_groups, columns are already set.
    if "parent_label" not in df.columns:
        df["parent_label"] = df["parent_id"].map(st.session_state.parent_labels_map or {}).astype(str)
    if "child_label" not in df.columns:
        df["child_label"] = df["child_id"].map(st.session_state.child_labels_map or {}).astype(str)

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

    fig = px.scatter(
        df, x="x", y="y",
        color="parent_label",      # high-level color
        symbol="child_label",      # shape differentiates children
        hover_data=["parent_label", "child_label", "descriptive_name", "cluster", "keyword", "search volume"],
        title="Parent & Child Topical Clusters (HDBSCAN EoM)",
        width=1100, height=720
    )
    st.plotly_chart(fig, width="stretch")
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
    st.dataframe(parent_summary, width="stretch", height=300)
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
    st.dataframe(child_summary, width="stretch", height=420)
except Exception:
    st.error("Error while building the child summary.")
    st.code(traceback.format_exc())

# ----------------------------- Export -----------------------------
st.subheader("1️⃣1️⃣ Export")
try:
    export_df = df.rename(columns={"descriptive_name": "Cluster (descriptive name)"})
    export_df = export_df[[
        "parent_label", "child_label", "Cluster (descriptive name)",
        "cluster", "keyword", "search volume"
    ]].rename(columns={
        "parent_label": "Parent Topic",
        "child_label": "Child Topic"
    })

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Hierarchical Topics CSV", csv, "hierarchical_topics.csv", "text/csv")
    st.success("✅ Done! Use UMAP+parent params for structure; child params to surface subtopics.")
except Exception:
    st.error("Error while preparing the CSV export.")
    st.code(traceback.format_exc())
    st.stop()

















