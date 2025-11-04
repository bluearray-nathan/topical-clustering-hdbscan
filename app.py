# app.py — Topical Clustering (Cosine) + EoM/epsilon + UMAP + Noise Rescue
# -----------------------------------------------------------------------
# Upload cluster_descriptions.csv (columns: descriptive_name, explanation)
# Embeds ONLY descriptive_name -> (optional UMAP) -> HDBSCAN
# Options: EoM vs Leaf, epsilon, Noise Rescue, centroid merge, LLM labels

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
st.set_page_config(page_title="Topical Clustering (Noise-Reduced)", layout="wide")
st.title("🧩 Topical Clustering — Less Noise, Cleaner Topics")
st.caption("Name-only embeddings • EoM/epsilon • optional UMAP • optional Noise Rescue")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing OpenAI API key. Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets:\n\n[openai]\napi_key = \"sk-...\"")
    st.stop()

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Embeddings")
    embedding_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-large", "text-embedding-3-small"],
        index=0
    )

    st.header("HDBSCAN")
    min_cluster_size = st.slider("min_cluster_size", 2, 40, 10)
    min_samples = st.slider("min_samples", 1, 10, 2)
    selection_method = st.radio("cluster_selection_method", ["eom (broader, less noise)", "leaf (finer)"], index=0)
    use_epsilon = selection_method.startswith("eom")
    cluster_selection_method = "eom" if use_epsilon else "leaf"
    epsilon = st.slider("cluster_selection_epsilon (EoM only)", 0.00, 0.20, 0.05, 0.01)

    st.header("Smoothing")
    use_umap = st.checkbox("Use UMAP pre-reduction (reduces noise)", value=False,
                           help="Compresses small gaps so HDBSCAN finds denser clusters.")
    if use_umap:
        try:
            from umap import UMAP  # import only when used
            n_neighbors = st.slider("UMAP n_neighbors", 10, 80, 40)
            n_components = st.slider("UMAP components", 5, 20, 10)
        except Exception:
            st.warning("UMAP not installed. Add `umap-learn>=0.5.5` to requirements.txt.")
            use_umap = False

    st.header("Topic Merge (Optional)")
    merge_clusters = st.checkbox("Auto-merge similar clusters (centroid cosine)", value=True)
    merge_threshold = st.slider("Merge threshold (cosine)", 0.50, 0.95, 0.70, 0.01)

    st.header("Noise Rescue (Optional)")
    enable_rescue = st.checkbox("Reassign noise to nearest topic", value=True)
    rescue_threshold = st.slider("Rescue threshold (cosine)", 0.65, 0.85, 0.72, 0.01)

    st.header("Labelling")
    auto_label_topics = st.checkbox("Auto-label topics (LLM)", value=True)
    label_model = st.selectbox("Labelling model", ["gpt-4o-mini-2024-07-18"], index=0)
    label_temp = st.slider("Labelling temperature", 0.0, 1.0, 0.2, 0.05)

    st.caption("Tips: Lower min_samples, use EoM+epsilon, optionally UMAP & Noise Rescue to cut noise.")

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

# ----------------------------- Embedding text (NAME ONLY) -----------------------------
df["text_for_embedding"] = df["descriptive_name"].fillna("").astype(str)

# ----------------------------- Embeddings -----------------------------
st.subheader("1️⃣ Generating embeddings (OpenAI)")
@st.cache_data(show_spinner=False)
def embed_texts(texts, model):
    vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.15)  # gentle pacing
    return np.array(vecs)

embeddings = embed_texts(df["text_for_embedding"].tolist(), embedding_model)
embeddings = normalize(embeddings)  # unit length
st.success(f"✅ Created {len(embeddings)} embeddings with {embedding_model}.")

# ----------------------------- Optional UMAP -----------------------------
if use_umap:
    st.subheader("2️⃣ UMAP smoothing (cosine → Euclidean in reduced space)")
    umap = UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=n_components,
                metric="cosine", random_state=42)
    Xr = umap.fit_transform(embeddings)
    st.success(f"✅ UMAP reduced to {n_components} dimensions.")
else:
    st.subheader("2️⃣ Computing cosine distance matrix")
    distance_matrix = cosine_distances(embeddings)
    st.success("✅ Cosine distance matrix computed.")

# ----------------------------- HDBSCAN clustering -----------------------------
st.subheader("3️⃣ HDBSCAN clustering")
hdb_params = dict(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_method=cluster_selection_method
)
if use_epsilon:
    hdb_params["cluster_selection_epsilon"] = float(epsilon)

if use_umap:
    # cluster reduced space with Euclidean metric
    hdb_params.update(metric="euclidean")
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(Xr)
else:
    # cluster precomputed cosine distances
    hdb_params.update(metric="precomputed")
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(distance_matrix)

df["topic_id"] = labels

n_topics = len(set(labels)) - (1 if -1 in labels else 0)
noise_pct = (labels == -1).mean() * 100.0 if len(labels) else 0.0
st.success(f"✅ Found {n_topics} initial topics (+ noise: {noise_pct:.1f}%).")

# ----------------------------- Noise Rescue (optional) -----------------------------
if enable_rescue and n_topics > 0 and (labels == -1).any():
    st.subheader("4️⃣ Noise Rescue — reassigning reasonable strays")
    # Compute centroids in ORIGINAL embedding space (cosine)
    centroids = {
        cid: embeddings[labels == cid].mean(axis=0)
        for cid in np.unique(labels) if cid != -1 and (labels == cid).any()
    }
    if len(centroids) > 0:
        cid_list = list(centroids.keys())
        C = np.vstack([centroids[c] for c in cid_list])

        noise_idx = np.where(labels == -1)[0]
        sims = cosine_similarity(embeddings[noise_idx], C)
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(noise_idx)), best]

        take = best_sim >= float(rescue_threshold)
        if np.any(take):
            reassigned_ids = np.array(cid_list)[best[take]]
            labels[noise_idx[take]] = reassigned_ids
            df.loc[noise_idx[take], "topic_id"] = reassigned_ids
            st.success(f"✨ Rescued {int(take.sum())} noise points at ≥ {rescue_threshold:.2f} cosine.")
        else:
            st.info("No noise points met the rescue similarity threshold.")

    # Recompute counts after rescue
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100.0 if len(labels) else 0.0

# ----------------------------- Optional centroid merge -----------------------------
if st.sidebar.checkbox("Show centroid merge settings", value=False):
    st.write("Centroid merge is controlled from the sidebar (above).")

if merge_clusters and n_topics > 1:
    st.subheader("5️⃣ Merging semantically similar clusters (centroid cosine)")
    centroids = {
        cid: embeddings[labels == cid].mean(axis=0)
        for cid in np.unique(labels) if cid != -1 and (labels == cid).any()
    }
    ids = list(centroids.keys())
    if ids:
        C = np.vstack([centroids[cid] for cid in ids])
        sim = cosine_similarity(C)
        merged_labels = {}
        visited, group_id = set(), 0
        for i, cid in enumerate(ids):
            if cid in visited:
                continue
            group = [cid]
            for j, cid2 in enumerate(ids):
                if i != j and sim[i, j] > float(merge_threshold):
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

if auto_label_topics:
    st.subheader("6️⃣ Auto-labelling topics via GPT")
    topic_labels = {}
    for tid in sorted(df["merged_topic_id"].unique()):
        if tid == -1:
            topic_labels[tid] = "Noise / Misc"
            continue
        texts = df.loc[df["merged_topic_id"] == tid, "descriptive_name"].tolist()
        topic_labels[tid] = label_topic_short(texts, label_model, label_temp)
        time.sleep(0.05)
    df["topic_label"] = df["merged_topic_id"].map(topic_labels)
else:
    df["topic_label"] = df["merged_topic_id"].astype(str)

# ----------------------------- Visualisation -----------------------------
st.subheader("7️⃣ Visualising topics (2D PCA projection)")
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
    title="Topical Clusters (HDBSCAN)",
    width=1100,
    height=720
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Diagnostics -----------------------------
st.subheader("Diagnostics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Topics", value=n_topics)
with col2:
    st.metric("Noise (%)", value=f"{noise_pct:.1f}%")

st.dataframe(
    df.groupby("topic_label").size().reset_index(name="count").sort_values("count", ascending=False).head(15),
    use_container_width=True
)

# ----------------------------- Export -----------------------------
st.subheader("8️⃣ Export")
export_cols = ["descriptive_name", "explanation", "topic_id", "merged_topic_id", "topic_label", "x", "y"]
csv = df[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Clustered Topics CSV", csv, "clustered_topics_noise_reduced.csv", "text/csv")

st.success("✅ Done! Tweak EoM/epsilon, UMAP, and Noise Rescue to reduce noise while keeping good clusters.")










