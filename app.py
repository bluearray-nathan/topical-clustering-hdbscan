# app.py: Keyword Topic Clustering (lean core)
# ---------------------------------------------------------------------------
# Groups a keyword list into topic clusters by meaning, using
# text-embedding-3-large and agglomerative clustering on cosine distance.
#
# This is the validated foundation. It captures TOPIC, not intent, so a topic
# such as "comprehensive insurance" will hold both informational and commercial
# keywords. Splitting each topic into separate PAGES by intent is the next layer,
# which sits on top of this.

import time
import hashlib
import concurrent.futures as cf

import numpy as np
import pandas as pd
import streamlit as st
import openai
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Keyword Topic Clustering", layout="wide")
st.title("Keyword Topic Clustering")

with st.expander("What this does"):
    st.markdown("""
Groups a keyword list into **topic clusters** by meaning. Every keyword gets a home; there is no
"noise" bucket. Built on text-embedding-3-large with agglomerative clustering on cosine distance.

It captures **topic, not intent**, so a topic like "comprehensive insurance" will contain both
informational keywords ("what is comprehensive cover") and commercial ones ("cheapest comprehensive
car insurance"). Splitting each topic into separate **pages by intent** is the next layer, coming on
top of this.
""")

# ----------------------------- API key -----------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error('Missing OpenAI API key. Add it in Streamlit secrets:\n\n[openai]\napi_key = "sk-..."')
    st.stop()

EMBED_MODEL = "text-embedding-3-large"
LABEL_MODEL = "gpt-4o-mini-2024-07-18"

if "label_cache" not in st.session_state:
    st.session_state.label_cache = {}

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Setup")
    st.caption("Embedding model")
    st.code(EMBED_MODEL, language="text")

    st.header("Cluster tightness")
    threshold = st.slider(
        "Cosine distance threshold",
        min_value=0.10, max_value=0.45, value=0.25, step=0.01,
        help="Lower = tighter, more granular topics. Higher = broader, fewer topics. "
             "0.25 is the validated default.",
    )

# ----------------------------- 1. Upload -----------------------------
st.subheader("1. Upload your keyword list")
file = st.file_uploader("CSV with a keyword column (volume optional).", type=["csv"])
if not file:
    st.info("Upload a CSV to begin.")
    st.stop()

df_in = pd.read_csv(file)
df_in.columns = [str(c).strip() for c in df_in.columns]


def find_col(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


kcol = find_col(df_in.columns, ["keyword", "keywords", "query", "term"])
if not kcol:
    st.error(f"No keyword column found. Columns seen: {list(df_in.columns)}")
    st.stop()
vcol = find_col(df_in.columns, ["search volume", "search_volume", "volume", "vol", "searches"])

df = pd.DataFrame()
df["keyword"] = df_in[kcol].astype(str).str.strip()
df["volume"] = pd.to_numeric(df_in[vcol], errors="coerce") if vcol else np.nan
df = df[df["keyword"].str.len() > 0].drop_duplicates("keyword").reset_index(drop=True)

if len(df) < 5:
    st.error("Need at least 5 distinct keywords to cluster.")
    st.stop()
st.success(f"Loaded {len(df)} distinct keywords"
           f"{' with volume.' if vcol else '. No volume column found, that is fine.'}")

# ----------------------------- 2. Embed -----------------------------
st.subheader("2. Embed")


@st.cache_data(show_spinner=False)
def embed(keywords, model):
    vecs = []
    for i in range(0, len(keywords), 200):
        resp = openai.embeddings.create(model=model, input=keywords[i:i + 200])
        vecs.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        time.sleep(0.05)
    return normalize(np.array(vecs, dtype=np.float32))


try:
    with st.spinner(f"Embedding {len(df)} keywords with {EMBED_MODEL}..."):
        X = embed(df["keyword"].tolist(), EMBED_MODEL)
    st.success(f"Embedded {len(X)} keywords.")
except Exception as e:
    st.error(f"Embedding failed: {e}")
    st.stop()

# ----------------------------- 3. Cluster -----------------------------
st.subheader("3. Cluster into topics")
labels = AgglomerativeClustering(
    n_clusters=None, distance_threshold=float(threshold),
    metric="cosine", linkage="average",
).fit_predict(X)
df["topic_id"] = labels
n_topics = int(df["topic_id"].nunique())
n_singletons = int((df["topic_id"].value_counts() == 1).sum())
st.success(f"{n_topics} topics. {n_singletons} are single keywords "
           f"({100 * n_singletons / len(df):.0f}%). Adjust the tightness slider to taste.")

# ----------------------------- 4. Label -----------------------------
st.subheader("4. Label topics")


def label_topic(keywords):
    key = hashlib.md5("|".join(sorted(keywords)).encode()).hexdigest()
    if key in st.session_state.label_cache:
        return st.session_state.label_cache[key]
    sample = keywords[:25]
    prompt = ("These keywords belong to one topic:\n" + ", ".join(sample) +
              "\n\nGive a concise topic name in 1 to 4 words (noun phrase, minimal punctuation). "
              "Reply with only the name.")
    try:
        resp = openai.chat.completions.create(
            model=LABEL_MODEL, temperature=0.2,
            messages=[{"role": "system", "content": "You are an SEO content analyst. Reply with only the label."},
                      {"role": "user", "content": prompt}],
        )
        name = " ".join(resp.choices[0].message.content.strip().strip('"').strip("'").split()[:5]) or "Unlabelled"
    except Exception:
        name = sample[0] if sample else "Unlabelled"
    st.session_state.label_cache[key] = name
    return name


ids = sorted(df["topic_id"].unique())
topic_kws = {
    tid: df[df["topic_id"] == tid].sort_values("volume", ascending=False, na_position="last")["keyword"].tolist()
    for tid in ids
}
labels_map = {}
prog = st.progress(0.0)
done = 0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(label_topic, topic_kws[tid]): tid for tid in ids}
    for fut in cf.as_completed(futs):
        labels_map[futs[fut]] = fut.result()
        done += 1
        prog.progress(done / len(ids))
df["topic"] = df["topic_id"].map(labels_map)
st.success("Topics labelled.")

# ----------------------------- 5. Topics table -----------------------------
st.subheader("5. Topics")
rows = []
for tid in ids:
    sub = df[df["topic_id"] == tid].sort_values("volume", ascending=False, na_position="last")
    vol = sub["volume"].sum(min_count=1)
    rows.append({
        "Topic": labels_map[tid],
        "Keywords": len(sub),
        "Volume": (int(vol) if pd.notna(vol) else None),
        "Head term": sub["keyword"].iloc[0],
    })
summary = pd.DataFrame(rows).sort_values("Volume", ascending=False, na_position="last")
st.dataframe(summary, use_container_width=True, height=420)

# 2D map (coloured by topic; legend hidden as there can be many)
try:
    coords = PCA(n_components=2).fit_transform(X)
    df["x"], df["y"] = coords[:, 0], coords[:, 1]
    fig = px.scatter(df, x="x", y="y", color="topic",
                     hover_data=["keyword", "topic", "volume"], height=650,
                     title="Keywords by topic")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    pass

# ----------------------------- 6. Export -----------------------------
st.subheader("6. Export")
export = df[["keyword", "volume", "topic", "topic_id"]].rename(
    columns={"keyword": "Keyword", "volume": "Volume", "topic": "Topic", "topic_id": "Topic ID"})
st.download_button("Download topic clustering (CSV)",
                   export.to_csv(index=False).encode("utf-8"),
                   "topic_clustering.csv", "text/csv")
st.caption("Next layer: split each topic into individual pages by intent.")
