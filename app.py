# app.py — Topic Clustering with "keep original name unless needed" + progress labelling
# --------------------------------------------------------------------------------------
# Input CSV columns (required):
#   - cluster (main page-level cluster keyword)
#   - keyword (all keywords within that page-level cluster, e.g., comma-separated)
#   - search volume (total search volume of the page-level cluster)
#
# Flow:
# 1) LLM generates 'descriptive_name' per row using rules that KEEP the original cluster name unless it's inadequate.
# 2) Embeds ONLY 'descriptive_name' (OpenAI text-embedding-3-large, fixed)
# 3) Optional UMAP smoothing via presets (Off/Broad/Balanced/Detailed) — presets also auto-set HDBSCAN defaults
# 4) HDBSCAN (EoM) with easy epsilon slider
# 5) GPT topic labels with concurrency + progress bar + rolling ETA, plus "Re-label topics now" button in the sidebar
# 6) Visualisation + Topics table (table is shown BELOW the viz)
# 7) Export: Topic, Cluster (descriptive name), cluster, keyword, search volume

import time
import re
import json
import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import openai
import hdbscan
import concurrent.futures as cf
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity  # (not used, but handy if you want centroid QA)
import plotly.express as px

st.set_page_config(page_title="Topical Clustering", layout="wide")
st.title("🧩 Topical Clustering")
st.caption("Upfront LLM naming (keeps original unless needed) • Name-only embeddings • UMAP presets • HDBSCAN (EoM) • GPT labels with progress")

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
if "topic_labels_map" not in st.session_state:
    st.session_state.topic_labels_map = {}
if "last_topic_hash" not in st.session_state:
    st.session_state.last_topic_hash = None

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Clustering Settings")

    # Embedding model locked
    st.text("Embedding model")
    st.code("text-embedding-3-large", language="text")

    # UMAP presets (auto-apply defaults)
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

    # Sliders (values are controlled by preset, but can be tweaked after)
    min_cluster_size = st.slider(
        "Minimum topic size",
        min_value=2, max_value=60, value=st.session_state.min_cluster_size, key="min_cluster_size",
        help=(
            "The smallest group HDBSCAN will call a topic. "
            "Higher = fewer, broader topics (and potentially less noise). "
            "Lower = more, smaller topics (can increase noise)."
        )
    )
    min_samples = st.slider(
        "Strictness (min samples)",
        min_value=1, max_value=10, value=st.session_state.min_samples, key="min_samples",
        help=(
            "How picky HDBSCAN is about density. "
            "Lower values accept looser groups (less noise). "
            "Higher values require tighter groups (more noise but higher confidence)."
        )
    )

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
    relabel_now = st.button("🔁 Re-label topics now")  # in sidebar as requested
    label_model = "gpt-4o-mini-2024-07-18"
    label_temp = st.slider("Labelling creativity", 0.0, 1.0, 0.2, 0.05)

    with st.expander("Throughput helper (what-if)"):
        n_est = st.number_input("Clusters to label", 1, 100000, value=1000, step=1)
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

df_raw = pd.read_csv(file)
df_raw.columns = [c.strip().lower() for c in df_raw.columns]
required_cols = {"cluster", "keyword", "search volume"}
if not required_cols.issubset(df_raw.columns):
    st.error("CSV must include columns: cluster, keyword, search volume (case-insensitive).")
    st.stop()

st.success(f"✅ Loaded {len(df_raw)} page-level clusters.")

# ----------------------------- Step 2: LLM descriptive name (keep original unless needed) -----------------------------
st.subheader("2️⃣ Generate a descriptive name per page-level cluster")

SYSTEM_NAMER = (
    "You are an SEO analyst. Decide whether the given cluster name already represents ALL the keywords in that cluster.\n"
    "Rules:\n"
    "- If the cluster name already describes the overall set of keywords (including their main modifiers/intents), KEEP IT EXACTLY as the descriptive name.\n"
    "- Only create a NEW descriptive name if the cluster name is too narrow, misleading, or misses obvious common modifiers across the keywords.\n"
    "- The descriptive name must be a short noun phrase in Title Case (2–6 words), no punctuation. Avoid adjectives that add no meaning.\n"
    "- Prefer specificity only when needed to cover the whole cluster; otherwise keep it concise.\n"
    "Return JSON with fields: \"descriptive_name\" (string), \"used_original\" (true|false), \"reason\" (short string)."
)

@st.cache_data(show_spinner=False)
def name_clusters_with_llm(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, r in rows.iterrows():
        head = str(r.get("cluster", "")).strip()
        kws = str(r.get("keyword", "")).strip()
        user_prompt = (
            f"Main cluster name: {head}\n"
            f"Keywords in this cluster: {kws}\n\n"
            "Decide: does the main cluster name already describe all keywords?\n"
            "If yes, RETURN the cluster name unchanged as \"descriptive_name\".\n"
            "If not, RETURN a better descriptive name that covers the full set.\n\n"
            "Return ONLY valid JSON with keys descriptive_name, used_original, reason."
        )
        resp = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": SYSTEM_NAMER},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        try:
            m = re.search(r"\{.*\}", text, flags=re.S)
            data = json.loads(m.group(0) if m else text)
        except Exception:
            data = {"descriptive_name": head, "used_original": True, "reason": "Fallback to original (JSON parse failed)."}

        out_rows.append({
            "descriptive_name": str(data.get("descriptive_name", head)).strip(),
            "used_original": bool(data.get("used_original", True)),
            "naming_reason": str(data.get("reason", "")).strip()
        })
        time.sleep(0.02)
    return pd.DataFrame(out_rows)

named = name_clusters_with_llm(df_raw[["cluster", "keyword", "search volume"]])
df_raw = pd.concat([df_raw.reset_index(drop=True), named.reset_index(drop=True)], axis=1)

with st.expander("Preview first 10 (descriptive_name + reason)"):
    st.dataframe(df_raw[["descriptive_name", "used_original", "naming_reason", "cluster", "keyword", "search volume"]].head(10), use_container_width=True)

# ----------------------------- Step 3: Embeddings (name-only) -----------------------------
st.subheader("3️⃣ Generate embeddings")
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    model = "text-embedding-3-large"  # locked
    vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = openai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.10)
    return np.array(vecs)

embeddings = embed_texts(df_raw["descriptive_name"].fillna("").astype(str).tolist())
embeddings = normalize(embeddings)
st.success(f"✅ Created {len(embeddings)} embeddings.")

# ----------------------------- Step 4: UMAP (presets) -----------------------------
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

# ----------------------------- Step 5: HDBSCAN (EoM) -----------------------------
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

# ----------------------------- Step 6: Labelling (with concurrency + progress) -----------------------------
def label_topic_short(titles, model_name, temp):
    joined = ", ".join(titles[:20])
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

# Build hash of current assignments
topic_hash = hashlib.md5(np.array(labels, dtype=np.int64).tobytes()).hexdigest()

should_label = False
if auto_label_topics:
    if st.session_state.last_topic_hash != topic_hash or relabel_now:
        should_label = True

if should_label:
    st.subheader("6️⃣ Topic labels (running)")
    unique_topic_ids = sorted(set(df["topic_id"]))
    if -1 in unique_topic_ids:
        unique_topic_ids.remove(-1)

    MAX_WORKERS = 12  # tune as needed
    ROLL_N = 20
    progress = st.progress(0)
    status = st.empty()
    timings = deque(maxlen=ROLL_N)
    labels_map = { -1: "Noise / Misc" }
    done = 0
    total = len(unique_topic_ids)

    def label_one_cluster(tid):
        start = time.time()
        titles = df.loc[df["topic_id"] == tid, "descriptive_name"].tolist()
        name = label_topic_short(titles, label_model, label_temp)
        dur = time.time() - start
        return tid, name, dur

    start_all = time.time()
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(label_one_cluster, tid) for tid in unique_topic_ids]
        for fut in cf.as_completed(futures):
            tid, name, dur = fut.result()
            labels_map[tid] = name
            timings.append(dur)
            done += 1
            progress.progress(done / max(total, 1))
            spc = float(np.mean(timings)) if len(timings) else 0.0
            eta_sec = max(total - done, 0) * spc
            status.text(f"Labelled {done}/{total} • avg {spc:.2f}s/label • ~ETA {int(eta_sec//60)}m {int(eta_sec%60)}s")

    st.session_state.topic_labels_map = labels_map
    st.session_state.last_topic_hash = topic_hash
    st.success("✅ Labelling complete.")

# Apply labels (cached if available)
if st.session_state.topic_labels_map:
    df["topic_label"] = df["topic_id"].map(st.session_state.topic_labels_map).astype(str)
else:
    df["topic_label"] = df["topic_id"].astype(str)

# ----------------------------- Step 7: Visualisation -----------------------------
st.subheader("7️⃣ Visualise topics (2D PCA)")
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
st.subheader("8️⃣ Topics table")
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
st.subheader("9️⃣ Export")
# Output columns: Topic, Cluster (descriptive name), then the original 3 columns
export_df = df.rename(columns={"descriptive_name": "Cluster (descriptive name)"})
export_df = export_df[[
    "topic_label", "Cluster (descriptive name)", "cluster", "keyword", "search volume"
]].rename(columns={"topic_label": "Topic"})

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Topics CSV", csv, "clustered_topics_with_clusters.csv", "text/csv")

st.success("✅ Done! Change the preset (auto-sets sensible defaults), tweak sliders if needed, and click “Re-label topics now” any time.")














