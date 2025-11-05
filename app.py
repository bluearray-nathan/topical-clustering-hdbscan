# app.py — Topic Clustering (simple, relabel button + topics table)
# ----------------------------------------------------------------
# - Embeds ONLY descriptive_name (OpenAI text-embedding-3-large)
# - HDBSCAN with EoM selection (epsilon slider explained)
# - Optional UMAP smoothing presets (Off / Broad / Balanced / Detailed)
# - No topic merge, no noise rescue
# - GPT labelling with "Re-label now" button
# - Topics table (size + label + example titles)

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

st.set_page_config(page_title="Topical Clustering (Simple)", layout="wide")
st.title("🧩 Topical Clustering")
st.caption("Embeds **descriptive_name** only • HDBSCAN (EoM) • optional UMAP smoothing • GPT labels")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Add it in Streamlit Cloud Secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Clustering Settings")

    # Embedding model locked
    st.text("Embedding model")
    st.code("text-embedding-3-large", language="text")

    # min_cluster_size with plain-English help
    min_cluster_size = st.slider(
        "Minimum topic size",
        min_value=2, max_value=60, value=10,
        help=(
            "The smallest group HDBSCAN will call a topic. "
            "Higher = fewer, broader topics (and potentially less noise). "
            "Lower = more, smaller topics (can increase noise)."
        )
    )

    # min_samples with plain-English help
    min_samples = st.slider(
        "Strictness (min samples)",
        min_value=1, max_value=10, value=2,
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
        min_value=0.00, max_value=0.20, value=0.05, step=0.01,
        help=(
            "Small values (e.g., 0.03–0.10) let HDBSCAN bridge tiny gaps between near-identical sub-groups, "
            "reducing noise. Set to 0.00 if you want stricter separation."
        )
    )

    # UMAP presets
    st.header("UMAP Smoothing")
    umap_preset = st.selectbox(
        "Preset",
        options=["Off", "Broad", "Balanced", "Detailed"],
        index=1,
        help=(
            "UMAP compresses the embedding space so related items sit closer together.\n"
            "• Broad: fewer, bigger topics, least noise\n"
            "• Balanced: general use\n"
            "• Detailed: more subtopics, possibly more noise"
        )
    )

    st.header("Labelling")
    auto_label_topics = st.checkbox("Auto-label topics with GPT", value=True)
    label_model = "gpt-4o-mini-2024-07-18"
    label_temp = st.slider("Labelling creativity", 0.0, 1.0, 0.2, 0.05)

# ----------------------------- File upload -----------------------------
file = st.file_uploader("Upload `cluster_descriptions.csv` (needs columns: descriptive_name, explanation)", type=["csv"])
if not file:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(file)
required = {"descriptive_name", "explanation"}
if not required.issubset(df.columns):
    st.error("CSV must include 'descriptive_name' and 'explanation' columns.")
    st.stop()

st.success(f"✅ Loaded {len(df)} rows.")

# ----------------------------- Embedding text (NAME ONLY) -------------------
df["text_for_embedding"] = df["descriptive_name"].fillna("").astype(str)

# ----------------------------- Embeddings -----------------------------
st.subheader("1️⃣ Generating embeddings (OpenAI)")
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

embeddings = embed_texts(df["text_for_embedding"].tolist())
embeddings = normalize(embeddings)  # unit-length for cosine behaviour
st.success(f"✅ Created {len(embeddings)} embeddings.")

# ----------------------------- UMAP (presets) -----------------------------
use_umap = umap_preset != "Off"
if use_umap:
    from umap import UMAP

    if umap_preset == "Broad":
        n_neighbors, n_components = 50, 8
    elif umap_preset == "Detailed":
        n_neighbors, n_components = 20, 15
    else:  # Balanced
        n_neighbors, n_components = 40, 10

    st.subheader("2️⃣ UMAP smoothing")
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
    st.subheader("2️⃣ Computing cosine distance matrix")
    distance_matrix = cosine_distances(embeddings)
    X_for_cluster = None
    st.success("✅ Cosine distance matrix computed.")

# ----------------------------- HDBSCAN (EoM locked) -----------------------
st.subheader("3️⃣ HDBSCAN clustering")
hdb_params = dict(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_method="eom",
    cluster_selection_epsilon=float(cluster_selection_epsilon)
)

if use_umap:
    # cluster in reduced (UMAP) space with Euclidean metric
    hdb_params["metric"] = "euclidean"
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(X_for_cluster)
else:
    # cluster on precomputed cosine distances
    hdb_params["metric"] = "precomputed"
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(distance_matrix)

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

def topics_summary_table(frame, label_col):
    # Build a compact table: topic_id, size, label, example titles
    rows = []
    for tid in sorted(frame["topic_id"].unique()):
        subset = frame[frame["topic_id"] == tid]
        size = len(subset)
        label = subset[label_col].iloc[0] if label_col in subset.columns else str(tid)
        examples = subset["descriptive_name"].head(3).tolist()
        rows.append({
            "topic_id": int(tid),
            "size": size,
            "topic_label": label,
            "example_titles": " | ".join(examples)
        })
    return pd.DataFrame(rows).sort_values(["topic_id"])

# ----------------------------- Label state & button -----------------------
# Hash current topic assignments for caching labelling results
topic_hash = hashlib.md5(np.array(labels, dtype=np.int64).tobytes()).hexdigest()

# Prepare session state
if "last_topic_hash" not in st.session_state:
    st.session_state["last_topic_hash"] = None
if "topic_labels_map" not in st.session_state:
    st.session_state["topic_labels_map"] = {}

# Button to re-label on demand
st.subheader("4️⃣ Topic labels")
relabel_now = st.button("🔁 Re-label topics now")

# Decide whether to (re)label:
should_label = False
if auto_label_topics:
    # Auto-label on first run, or when topics changed, or when user forces relabel
    if st.session_state["last_topic_hash"] != topic_hash or relabel_now:
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
    st.session_state["topic_labels_map"] = labels_map
    st.session_state["last_topic_hash"] = topic_hash

# Apply labels (if we have them), else fall back to string of topic_id
if st.session_state["topic_labels_map"]:
    df["topic_label"] = df["topic_id"].map(st.session_state["topic_labels_map"]).astype(str)
else:
    df["topic_label"] = df["topic_id"].astype(str)

# ----------------------------- Topics table ------------------------------
st.subheader("5️⃣ Topics table")
summary_df = topics_summary_table(df, "topic_label")
st.dataframe(summary_df, use_container_width=True, height=420)

# ----------------------------- Visualisation -----------------------------
st.subheader("6️⃣ Visualise topics (2D PCA)")
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

fig = px.scatter(
    df, x="x", y="y",
    color="topic_label",
    hover_data=["descriptive_name", "explanation"],
    title="Topical Clusters (HDBSCAN EoM)",
    width=1100, height=720
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Diagnostics -----------------------------
st.subheader("Diagnostics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Topics", value=n_topics)
with col2:
    st.metric("Noise (%)", value=f"{noise_pct:.1f}%")

# ----------------------------- Export -----------------------------
st.subheader("7️⃣ Export")
export_cols = ["descriptive_name", "explanation", "topic_id", "topic_label", "x", "y"]
csv = df[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Clustered Topics", csv, "clustered_topics_simple.csv", "text/csv")

st.success("✅ Tip: Use UMAP 'Broad' and small ε (e.g., 0.05) to reduce noise while keeping sensible topics.")












