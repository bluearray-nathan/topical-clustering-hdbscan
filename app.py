# app_cluster_topics.py — Streamlit app for topical clustering
# -------------------------------------------------------------
# Upload cluster_descriptions.csv and group page-level clusters into higher-level topics.

import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import hdbscan
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import plotly.express as px

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------
st.set_page_config(page_title="Topical Clustering", layout="wide")
st.title("🧩 Topical Clustering — Group page-level clusters into broader topics")
st.caption("Upload your `cluster_descriptions.csv` output file")

# --------------------------------------------------
# API key setup
# --------------------------------------------------
try:
    api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Please add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()

openai.api_key = api_key

# --------------------------------------------------
# Parameters
# --------------------------------------------------
embedding_model = "text-embedding-3-large"
min_cluster_size = st.sidebar.slider("Min cluster size (HDBSCAN)", 3, 50, 8)
min_samples = st.sidebar.slider("Min samples (HDBSCAN)", 1, 20, 3)
generate_labels = st.sidebar.checkbox("Auto-label topics with LLM", value=True)
temperature = 0.2

# --------------------------------------------------
# File upload
# --------------------------------------------------
file = st.file_uploader("Upload `cluster_descriptions.csv`", type=["csv"])
if not file:
    st.info("Upload your CSV to start.")
    st.stop()

df = pd.read_csv(file)
if not {"descriptive_name", "explanation"}.issubset(df.columns):
    st.error("CSV must contain 'descriptive_name' and 'explanation' columns.")
    st.stop()

st.success(f"Loaded {len(df)} page-level clusters.")

# --------------------------------------------------
# Create text to embed
# --------------------------------------------------
df["text_for_embedding"] = df["descriptive_name"].fillna("") + ". " + df["explanation"].fillna("")

# --------------------------------------------------
# Embed all clusters
# --------------------------------------------------
st.subheader("1️⃣ Generating embeddings...")
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 100):  # batch in 100s
        batch = texts[i:i+100]
        response = openai.embeddings.create(
            model=embedding_model,
            input=batch
        )
        batch_embeds = [d.embedding for d in response.data]
        embeddings.extend(batch_embeds)
        time.sleep(0.2)
    return np.array(embeddings)

embeddings = embed_texts(df["text_for_embedding"].tolist())
st.success(f"✅ Created {len(embeddings)} embeddings using {embedding_model}.")

# --------------------------------------------------
# 2️⃣ Cluster with HDBSCAN
# --------------------------------------------------
st.subheader("2️⃣ Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean"
)
df["topic_id"] = clusterer.fit_predict(embeddings)

n_clusters = len(set(df["topic_id"])) - (1 if -1 in df["topic_id"].values else 0)
st.success(f"✅ Found {n_clusters} topics (plus noise).")

# --------------------------------------------------
# 3️⃣ Optional: Label topics via LLM
# --------------------------------------------------
if generate_labels and n_clusters > 0:
    st.subheader("3️⃣ Generating topic labels (via LLM)...")

    def summarize_cluster(texts):
        joined = ", ".join(texts[:15])
        prompt = (
            f"These are titles and summaries of pages about a similar topic:\n{joined}\n\n"
            "Provide a concise, human-readable topic name (2–4 words, noun phrase only, no punctuation):"
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
    for topic_id in sorted(df["topic_id"].unique()):
        if topic_id == -1:
            topic_labels[topic_id] = "Noise / Miscellaneous"
            continue
        texts = df.loc[df["topic_id"] == topic_id, "descriptive_name"].tolist()
        topic_labels[topic_id] = summarize_cluster(texts)
        time.sleep(0.2)

    df["topic_label"] = df["topic_id"].map(topic_labels)
    st.success("✅ Topic labels generated.")
else:
    df["topic_label"] = df["topic_id"].astype(str)

# --------------------------------------------------
# 4️⃣ Visualize clusters (PCA → 2D)
# --------------------------------------------------
st.subheader("4️⃣ Topic visualization (2D projection)")
pca = PCA(n_components=2)
coords = pca.fit_transform(normalize(embeddings))
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="topic_label",
    hover_data=["descriptive_name", "explanation"],
    title="Topic Clusters (HDBSCAN)",
    width=1000,
    height=700
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 5️⃣ Export results
# --------------------------------------------------
st.subheader("5️⃣ Export clustered data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Clustered Topics CSV", csv, "clustered_topics.csv", "text/csv")

st.success("✅ Done! You can now review your topic clusters and download the output.")




