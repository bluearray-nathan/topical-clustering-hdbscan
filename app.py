# app.py — Topic Clustering with upfront LLM naming + preset-driven defaults
# -------------------------------------------------------------------------
# Input CSV columns (required):
#   - cluster (main page-level cluster keyword)
#   - keyword (all keywords within that page-level cluster, e.g., comma-separated)
#   - search volume (total search volume of the page-level cluster)
#
# Flow:
# 1) LLM generates a 'descriptive_name' from (cluster + keyword)
# 2) Embeds ONLY 'descriptive_name' (OpenAI text-embedding-3-large, fixed)
# 3) Optional UMAP presets (Off/Broad/Balanced/Detailed) — also auto-set HDBSCAN defaults
# 4) HDBSCAN (EoM) with easy epsilon slider
# 5) GPT topic labels + visualisation + topics table (below viz)
# 6) Export: Topic, Cluster (descriptive name), cluster, keyword, search volume

import time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import openai
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import plotly.express as px

st.set_page_config(page_title="Topical Clustering", layout="wide")
st.title("🧩 Topical Clustering")
st.caption("Upfront LLM naming • Name-only embeddings • UMAP presets • HDBSCAN (EoM) • GPT labels")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Add it in Streamlit Cloud Secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

# ----------------------------- Session state defaults -----------------------------
PRESET_DEFAULTS = {
    "Off":      {"min_cluster_size": 10, "min_samples": 2, "epsilon": 0.05, "umap_neighbors": None, "umap_components": None},
    "Broad":    {"min_cluster_size": 12, "min_samples": 1, "epsilon": 0.06, "umap_neighbors": 50, "umap_components": 8},
    "Balanced": {"min_cluster_size": 10, "min_samples": 2, "epsilon": 0.05, "umap_neighbors": 40, "umap_components": 10},
    "Detailed": {"min_cluster_size": 8,  "min_samples": 3, "epsilon": 0.03, "umap_neighbors": 20, "umap_components": 15},
}

if "umap_preset" not in st.session_state:
    st.session_state.umap_preset = "Balanced"
if "last_preset_applied" not in st.session_state:
    st.session_state.last_preset_applied = None
# Bind sliders to keys so we can update from preset automatically
if "min_cluster_size" not in st.session_state:
    st.session_state.min_cluster_size = PRESET_DEFAULTS["Balanced"]["min_cluster_size"]
if "min_samples" not in st.session_state:
    st.session_state.min_samples = PRESET_DEFAULTS["Balanced"]["min_samples"]
if "cluster_selection_epsilon" not in st.session_state:
    st.session_state.cluster_selection_epsilon = PRESET_DEFAULTS["Balanced"]["epsilon"]

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Clustering Settings")

    # Embedding model locked
    st.text("Embedding model")
    st.code("text-embedding-3-large", language="text")

    # UMAP presets (applies defaults for min_cluster_size / min_samples / epsilon)
    st.header("UMAP Smoothing")
    umap_preset = st.selectbox(
        "Preset",
        options=["Off", "Broad", "Balanced", "Detailed"],
        index=["Off", "Broad", "Balanced", "Detailed"].index(st.session_state.umap_preset),
        help=(
            "UMAP compresses the embedding space so related items sit closer together.\n"
            "• Broad: fewer, bigger topics, least noise (defaults: size=12, samples=1, ε=0.06)\n"
            "• Balanced: general use (size=10, samples=2, ε=0.05)\n"
            "• Detailed: more subtopics, possibly more noise (size=8, samples=3, ε=0.03)"
        )
    )

    # Auto-apply preset defaults when changed
    if umap_preset != st.session_state.last_preset_applied:
        st.session_state.umap_preset = umap_preset
        defaults = PRESET_DEFAULTS[umap_preset]
        # Update sliders’ session_state values
        if defaults["min_cluster_size"] is not None:
            st.session_state.min_cluster_size = defaults["min_cluster_size"]
        if defaults["min_samples"] is not None:
            st.session_state.min_samples = defaults["min_samples"]
        if defaults["epsilon"] is not None:
            st.session_state.cluster_selection_epsilon = defaults["epsilon"]
        st.session_state.last_preset_applied = umap_preset

    # min_cluster_size with plain-English help
    min_cluster_size = st.slider(
        "Minimum topic size",
        min_value=2, max_value=60, value=st.session_state.min_cluster_size, key="min_cluster_size",
        help=(
            "The smallest group HDBSCAN will call a topic. "
            "Higher = fewer, broader topics (and potentially less noise). "
            "Lower = more, smaller topics (can increase noise)."
        )
    )

    # min_samples with plain-English help
    min_samples = st.slider(
        "Strictness (min samples)",
        min_value=1, max_value=10, value=st.session_state.min_samples, key="min_samples",
        help=(
            "How picky HDBSCAN is about density. "
            "Lower values accept looser groups (less noise). "
            "Higher values require tighter groups (more noise but higher confidence)."
        )
    )

    # EoM locked; epsilon made user-friendly
    st.text("Topic merging sensitivity (EoM)")
    cluster_selection_epsilon = st.slider(
        "Bridge tiny gaps (ε)",
        min_value=0.00, max_value=0.20, value=st.session_state.cluster_selection_epsilon,
        step=0.01, key="cluster_selection_epsilon",
        help=(
            "Small values (e.g., 0.03–0.10) let HDBSCAN bridge tiny gaps between near-identical sub-groups, "
            "reducing noise. Set to 0.00 if you want stricter separation."
        )
    )

    st.header("Labelling")
    auto_label_topics = st.checkbox("Auto-label topics with GPT", value=True)
    relabel_now = st.button("🔁 Re-label topics now")
    label_model = "gpt-4o-mini-2024-07-18"
    label_temp = st.slider("Labelling creativity", 0.0, 1.0, 0.2, 0.05)

# ----------------------------- File upload -----------------------------
st.subheader("1️⃣ Upload your CSV")
file = st.file_uploader(
    "Upload a CSV with columns: cluster, keyword, search volume",
    type=["csv"]
)
if not file:
    st.info("Upload your CSV to begin.")
    st.stop()

df_raw = pd.read_csv(file)
# Normalise column names to lower for robustness
df_raw.columns = [c.strip().lower() for c in df_raw.columns]
required_cols = {"cluster", "keyword", "search volume"}
if not required_cols.issubset(df_raw.columns):
    st.error("CSV must include columns: cluster, keyword, search volume (case-insensitive).")
    st.stop()

st.success(f"✅ Loaded {len(df_raw)} page-level clusters.")

# ----------------------------- Step: LLM descriptive name ------------------
st.subheader("2️⃣ Generate a descriptive name per page-level cluster")

SYSTEM_NAMER = (
    "You are an SEO analyst. Given a page-level cluster keyword and the keywords inside that cluster, "
    "write a concise, human-readable descriptive name (2–6 words, noun phrase, no punctuation) that "
    "best represents the page you would create to target this cluster."
)

@st.cache_data(show_spinner=False)
def name_clusters_with_llm(rows: pd.DataFrame) -> pd.Series:
    out = []
    for _, r in rows.iterrows():
        head = str(r.get("cluster", "")).strip()
        kws = str(r.get("keyword", "")).strip()
        user_prompt = (
            f"Main cluster keyword: {head}\n"
            f"Keywords in this cluster: {kws}\n\n"
            "Return ONLY the descriptive name."
        )
        resp = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": SYSTEM_NAMER},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        name = resp.choices[0].message.content.strip()
        out.append(name)
        time.sleep(0.03)
    return pd.Series(out)

df_raw["descriptive_name"] = name_clusters_with_llm(df_raw)

st.success("✅ Descriptive names generated.")
with st.expander("Preview first 10 (descriptive_name)"):
    st.dataframe(df_raw[["descriptive_name", "cluster", "keyword", "search volume"]].head(10), use_container_width=True)

# ----------------------------- Embeddings (name-only) ---------------------
st.subheader("3️⃣ Generate embeddings")
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    model = "text-embedding-3-large"  # locked
    vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.12)
    return np.array(vecs)

embeddings = embed_texts(df_raw["descriptive_name"].fillna("").astype(str).tolist())
embeddings = normalize(embeddings)
st.success(f"✅ Created {len(embeddings)} embeddings.")

# ----------------------------- UMAP (presets) -----------------------------
st.subheader("4️⃣ Smoothing (optional)")
use_umap = (st.session_state.umap_preset != "Off")
if use_umap:
    from umap import UMAP
    n_neighbors = PRESET_DEFAULTS[st.session_state.umap_preset]["umap_neighbors"]
    n_components = PRESET_DEFAULTS[st.session_state.umap_preset]["umap_components"]
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
    X_for_cluster = None
    st.info("UMAP Off — using cosine distances directly.")

# ----------------------------- HDBSCAN (EoM locked) -----------------------
st.subheader("5️⃣ HDBSCAN clustering")
hdb_params = dict(
    min_cluster_size=st.session_state.min_cluster_size,
    min_samples=st.session_state.min_samples,
    cluster_selection_method="eom",
    cluster_selection_epsilon=float(st.session_state.cluster_selection_epsilon)
)

if use_umap:
    hdb_params["metric"] = "euclidean"
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(X_for_cluster)
else:
    hdb_params["metric"] = "precomputed"
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(distance_matrix)

df = df_raw.copy()
df["topic_id"] = labels

n_topics = len(set(labels)) - (1 if -1 in labels else 0)
noise_pct = (labels == -1).mean() * 100 if len(labels) else 0.0
st.success(f"✅ Topics found: {n_topics} • Noise: {noise_pct:.1f}%")

# ----------------------------- Labelling utils ---------------------------
def label_topic_short(texts, model_name, temp):
    joined = ", ".join(texts[:20])
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

# Track topic state for label caching
topic_hash = hashlib.md5(np.array(labels, dtype=np.int64).tobytes()).hexdigest()
if "last_topic_hash" not in st.session_state:
    st.session_state.last_topic_hash = None
if "topic_labels_map" not in st.session_state:
    st.session_state.topic_labels_map = {}

# Decide to (re)label
should_label = False
if auto_label_topics:
    if st.session_state.last_topic_hash != topic_hash or relabel_now:
        should_label = True

if should_label:
    labels_map = {}
    for tid in sorted(set(labels)):
        if tid == -1:
            labels_map[tid] = "Noise / Misc"
            continue
        titles = df.loc[df["topic_id"] == tid, "descriptive_name"].tolist()
        labels_map[tid] = label_topic_short(titles, label_model, label_temp)
        time.sleep(0.04)
    st.session_state.topic_labels_map = labels_map
    st.session_state.last_topic_hash = topic_hash

# Apply labels
if st.session_state.topic_labels_map:
    df["topic_label"] = df["topic_id"].map(st.session_state.topic_labels_map).astype(str)
else:
    df["topic_label"] = df["topic_id"].astype(str)

# ----------------------------- Visualisation -----------------------------
st.subheader("6️⃣ Visualise topics (2D PCA)")
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

fig = px.scatter(
    df, x="x", y="y",
    color="topic_label",
    hover_data=["descriptive_name", "cluster", "keyword", "search volume"],
    title="Topical Clusters (HDBSCAN EoM)",
    width=1100, height=720
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Topics table (below viz) -------------------
st.subheader("7️⃣ Topics table")
def topics_summary_table(frame):
    rows = []
    for tid in sorted(frame["topic_id"].unique()):
        subset = frame[frame["topic_id"] == tid]
        size = len(subset)
        label = subset["topic_label"].iloc[0]
        examples = subset["descriptive_name"].head(3).tolist()
        rows.append({
            "topic_id": int(tid),
            "size": size,
            "topic_label": label,
            "example_titles": " | ".join(examples)
        })
    return pd.DataFrame(rows).sort_values(["topic_id"])

summary_df = topics_summary_table(df)
st.dataframe(summary_df, use_container_width=True, height=420)

# ----------------------------- Export -----------------------------
st.subheader("8️⃣ Export")
# Output columns: Topic, Cluster (descriptive name), then the original 3 columns
export_df = df.rename(columns={"descriptive_name": "Cluster (descriptive name)"})
export_df = export_df[[
    "topic_label", "Cluster (descriptive name)", "cluster", "keyword", "search volume"
]].rename(columns={"topic_label": "Topic"})

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Topics CSV", csv, "clustered_topics_with_clusters.csv", "text/csv")

st.success("✅ Done! Change the UMAP preset to auto-apply sensible defaults, tweak sliders if needed, and click “Re-label topics now” any time.")













