# app_cluster_topics_v3.py
# --------------------------------------------------------
# Topical clustering with cosine distances (precomputed)
# Works perfectly with text-embedding-3-large and avoids metric errors.

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

# --------------------------------------------------------
# Streamlit setup
# --------------------------------------------------------
st.set_page_config(page_title="Topical Clustering v3", layout="wide")
st.title("🧩 Topical Clustering — Cosine Distance Version")
st.caption("Group your page-level clusters into higher-level topics using precomputed cosine distances.")

# --------------------------------------------------------
# API key setup
# --------------------------------------------------------
try:
    api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing API key. Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()

openai.api_key = api_key

# --------------------------------------------------------
# Parameters
# --------------------------------------------------------
embedding_model = "text-embedding-3-large"
min_cluster_size = st.sidebar.slider("Min cluster size", 2, 40, 5)
min_samples = st.sidebar.slider("Min samples", 1, 10, 1)
merge_clusters = st.sidebar.checkbox("Auto-merge similar topics (cosine > 0.8)", value=True)
auto_label = st.sidebar.checkbox("Auto-label merged topics with LLM", value=True)
temperature = 0.2

st.sidebar.caption("💡 Tip: Smaller values = more clusters; larger values = fewer, broader topics.")

# --------------------------------------------------------
# Upload CSV
# --------------------------------------------------------
file = st.file_uploader("Upload your `cluster_descriptions.csv`", type=["csv"])
if not file:
    st.info("Upload a CSV to start.")
    st.stop()

df = pd.read_csv(file)
if not {"descriptive_name", "explanation"}.issubset(df.columns):
    st.error("CSV must include 'descriptive_name' and 'explanation' columns.")
    st.stop()

st.success(f"✅ Loaded {len(df)} page-level clusters.")

# --------------------------------------------------------
# Prepare text for embeddings
# --------------------------------------------------------
df["text_for_embedding"] = df["descriptive_name"].fillna("") + ". " + df["explanation"].fillna("")

# --------------------------------------------------------
# Generate embeddings
# --------------------------------------------------------
st.subheader("1️⃣ Generating embeddings...")

@st.cache_data(show_spinner=False)
def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = openai.embeddings.create(model=embedding_model, input=batch)
        batch_vectors = [d.embedding for d in response.data]
        vectors.extend(batch_vectors)
        time.sleep(0.2)
    return np.array(vectors)

embeddings = embed_texts(df["text_for_embedding"].tolist())
embeddings = normalize(embeddings)
st.success(f"✅ Created {len(embeddings)} embeddings using {embedding_model}.")

# --------------------------------------------------------
# Compute cosine distance matrix
# --------------------------------------------------------
st.subheader("2️⃣ Computing cosine distance matrix...")
distance_matrix = cosine_distances(embeddings)
st.success("✅ Cosine distance matrix computed.")

# --------------------------------------------------------
# Cluster with HDBSCAN (precomputed metric)
# --------------------------------------------------------
st.subheader("3️⃣ Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="precomputed"
)
df["topic_id"] = clusterer.fit_predict(distance_matrix)

n_clusters = len(set(df["topic_id"])) - (1 if -1 in df["topic_id"] else 0)
st.success(f"✅ Found {n_clusters} initial topic clusters (+ noise).")

# --------------------------------------------------------
# Merge similar clusters by centroid cosine similarity
# --------------------------------------------------------
if merge_clusters and n_clusters > 1:
    st.subheader("4️⃣ Merging semantically similar clusters...")
    centroids = {
        cid: embeddings[df["topic_id"] == cid].mean(axis=0)
        for cid in df["topic_id"].unique() if cid != -1
    }

    ids = list(centroids.keys())
    centroids_matrix = np.vstack([centroids[cid] for cid in ids])
    sim_matrix = cosine_similarity(centroids_matrix)

    merge_threshold = 0.8
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
    n_merged = len(set(df["merged_topic_id"])) - (1 if -1 in df["merged_topic_id"] else 0)
    st.success(f"✅ Merged into {n_merged} higher-level topics.")
else:
    df["merged_topic_id"] = df["topic_id"]

# --------------------------------------------------------
# Auto-label topics via GPT
# --------------------------------------------------------
if auto_label:
    st.subheader("5️⃣ Auto-labelling topics via GPT...")

    def label_topic(texts):
        joined = ", ".join(texts[:15])
        prompt = (
            f"These are short titles and summaries of pages about a similar topic:\n{joined}\n\n"
            "Return ONLY a concise topic name (2–4 words, noun phrase, no punctuation):"
        )
        resp = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an SEO content analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    topic_labels = {}
    for topic_id in sorted(df["merged_topic_id"].unique()):
        if topic_id == -1:
            topic_labels[topic_id] = "Noise / Miscellaneous"
            continue
        texts = df.loc[df["merged_topic_id"] == topic_id, "descriptive_name"].tolist()
        topic_labels[topic_id] = label_topic(texts)
        time.sleep(0.2)

    df["topic_label"] = df["merged_topic_id"].map(topic_labels)
    st.success("✅ Topic labels generated.")
else:
    df["topic_label"] = df["merged_topic_id"].astype(str)

# --------------------------------------------------------
# Visualise clusters
# --------------------------------------------------------
st.subheader("6️⃣ Visualising topics (2D PCA projection)...")

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
    title="Topical Clusters (HDBSCAN + Cosine Distance)",
    width=1000,
    height=700
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# Export
# --------------------------------------------------------
st.subheader("7️⃣ Export clustered topics")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Clustered Topics CSV", csv, "clustered_topics_v3.csv", "text/csv")

st.success("✅ Done! Explore clusters above or download for further analysis.")






