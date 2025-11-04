# app.py — Topical Clustering (Cosine + Optional Merge)
# -----------------------------------------------------
# Upload cluster_descriptions.csv (columns: descriptive_name, explanation)
# Embeds with OpenAI -> precomputed cosine distances -> HDBSCAN
# Optional centroid-based topic merging and LLM topic labels

import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import openai
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import plotly.express as px

# ----------------------------- Streamlit setup -----------------------------
st.set_page_config(page_title="Topical Clustering (Cosine)", layout="wide")
st.title("🧩 Topical Clustering — Cosine Distance")
st.caption("Upload your `cluster_descriptions.csv` (columns: descriptive_name, explanation).")

# ----------------------------- API key -----------------------------
try:
    api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

openai.api_key = api_key

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Clustering Settings")
    embedding_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-large", "text-embedding-3-small"],
        index=0
    )
    min_cluster_size = st.slider("HDBSCAN: min_cluster_size", 2, 40, 5)
    min_samples = st.slider("HDBSCAN: min_samples", 1, 10, 1)

    st.header("Topic Merge (Optional)")
    merge_clusters = st.checkbox("Auto-merge similar clusters (centroid cosine)", value=True)
    merge_threshold = st.slider("Merge threshold (cosine)", 0.50, 0.95, 0.80, 0.01)

    st.header("Labelling")
    auto_label_topics = st.checkbox("Auto-label topics (LLM)", value=True)
    label_model = st.selectbox("Labelling model", ["gpt-4o-mini-2024-07-18"], index=0)
    label_temp = st.slider("Labelling temperature", 0.0, 1.0, 0.2, 0.05)

    st.caption("💡 Tips: Lower thresholds merge more; increase min_cluster_size for fewer, larger topics.")

# ----------------------------- File upload -----------------------------
file = st.file_uploader("Upload `cluster_descriptions.csv`", type=["csv"])
if not file:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(file)
required_cols = {"descriptive_name", "explanation"}
if not required_cols.issubset(df.columns):
    st.error("CSV must include 'descriptive_name' and 'explanation' columns.")
    st.stop()

st.success(f"✅ Loaded {len(df)} page-level clusters.")

# ----------------------------- Prepare text -----------------------------
df["text_for_embedding"] = df["descriptive_name"].fillna("") + ". " + df["explanation"].fillna("")

# ----------------------------- Embeddings -----------------------------
st.subheader("1️⃣ Generating embeddings (OpenAI)")
@st.cache_data(show_spinner=False)
def embed_texts(texts, model):
    vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        time.sleep(0.2)  # gentle pacing
    return np.array(vectors)

embeddings = embed_texts(df["text_for_embedding"].tolist(), embedding_model)
embeddings = normalize(embeddings)  # unit length -> cosine-friendly
st.success(f"✅ Created {len(embeddings)} embeddings with {embedding_model}.")

# ----------------------------- Cosine distance matrix -----------------------------
st.subheader("2️⃣ Computing cosine distance matrix")
distance_matrix = cosine_distances(embeddings)
st.success("✅ Cosine distance matrix computed.")

# ----------------------------- HDBSCAN clustering -----------------------------
st.subheader("3️⃣ HDBSCAN clustering (precomputed cosine)")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="precomputed"
)
df["topic_id"] = clusterer.fit_predict(distance_matrix)

n_topics = len(set(df["topic_id"])) - (1 if -1 in df["topic_id"].values else 0)
noise_pct = (df["topic_id"].eq(-1).mean() * 100.0) if len(df) else 0.0
st.success(f"✅ Found {n_topics} initial topics (+ noise: {noise_pct:.1f}%).")

# ----------------------------- Optional centroid merge -----------------------------
if merge_clusters and n_topics > 1:
    st.subheader("4️⃣ Merging semantically similar clusters (centroid cosine)")
    centroids = {
        cid: embeddings[df["topic_id"] == cid].mean(axis=0)
        for cid in df["topic_id"].unique() if cid != -1
    }
    ids = list(centroids.keys())
    if ids:
        centroids_matrix = np.vstack([centroids[cid] for cid in ids])
        sim_matrix = cosine_similarity(centroids_matrix)
        merged_labels = {}
        visited = set()
        group_id = 0
        for i, cid in enumerate(ids):
            if cid in visited:
                continue
            group = [cid]
            for j, cid2 in enumerate(ids):
                if i != j and sim_matrix[i, j] > merge_threshold:
                    group.append(cid2)
                    visited.add(cid2)
            for g in group:
                merged_labels[g] = group_id
            visited.add(cid)
            group_id += 1
        df["merged_topic_id"] = df["topic_id"].map(merged_labels).fillna(-1).astype(int)
        n_merged = len(set(df["merged_topic_id"])) - (1 if -1 in df["merged_topic_id"].values else 0)
        st.success(f"✅ Merged to {n_merged} higher-level topics at threshold {merge_threshold:.2f}.")
    else:
        df["merged_topic_id"] = df["topic_id"]
        st.info("No non-noise clusters to merge.")
else:
    df["merged_topic_id"] = df["topic_id"]

# ----------------------------- Auto-label topics via LLM (merged) -----------------------------
def label_topic_short(texts, model_name, temp):
    joined = ", ".join(texts[:20])
    prompt = (
        "These page titles/descriptions are about a similar topic:\n"
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

if auto_label_topics:
    st.subheader("5️⃣ Auto-labelling topics via GPT")
    topic_labels = {}
    for tid in sorted(df["merged_topic_id"].unique()):
        if tid == -1:
            topic_labels[tid] = "Noise / Misc"
            continue
        texts = df.loc[df["merged_topic_id"] == tid, "descriptive_name"].tolist()
        topic_labels[tid] = label_topic_short(texts, label_model, label_temp)
        time.sleep(0.1)
    df["topic_label"] = df["merged_topic_id"].map(topic_labels)
else:
    df["topic_label"] = df["merged_topic_id"].astype(str)

# ----------------------------- Visualisation -----------------------------
st.subheader("6️⃣ Visualising topics (2D PCA projection)")
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="topic_label",
    hover_data=["descriptive_name", "explanation"],
    title="Topical Clusters (HDBSCAN + Cosine)",
    width=1100,
    height=720
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Diagnostics -----------------------------
st.subheader("Diagnostics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Initial topics", value=n_topics)
with col2:
    st.metric("Noise (%)", value=f"{noise_pct:.1f}%")

st.dataframe(
    df.groupby("topic_label").size().reset_index(name="count").sort_values("count", ascending=False).head(15),
    use_container_width=True
)

# ----------------------------- Export -----------------------------
st.subheader("7️⃣ Export")
export_cols = ["descriptive_name", "explanation", "topic_id", "merged_topic_id", "topic_label", "x", "y"]
csv = df[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Clustered Topics CSV", csv, "clustered_topics.csv", "text/csv")

st.success("✅ Done! Adjust thresholds to shape topics.")








