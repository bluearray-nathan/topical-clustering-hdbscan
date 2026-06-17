#!/usr/bin/env python3
"""
cluster_lab.py: a focused bench for getting the keyword clustering right.

It embeds a keyword list once per embedding model (cached to disk), runs several
clustering approaches side by side, and for each one writes the clusters to a
readable markdown file so you can judge them by eye. A summary table prints the
cluster count, how many keywords were left as singletons, the size spread, and
the agreement with a reference clustering (e.g. Keyword Insights) shown only as a
reference, not a target.

No chat-model labelling and no SERP calls. Only embeddings, so it is fast and cheap.

Usage:
    export OPENAI_API_KEY=sk-...        # or leave it in .streamlit/secrets.toml
    python cluster_lab.py \
        --input "keywords.csv" \
        --reference "cluster_overview_results (4).csv" \
        --outdir cluster_lab_out

The Keyword Insights export already has keyword, search_volume and cluster columns,
so you can pass it as BOTH --input and --reference if you don't have a separate list.
"""

import argparse
import os
import sys
import hashlib
from math import comb

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def ensure_openai_key():
    """Use OPENAI_API_KEY if set, else lift it from .streamlit/secrets.toml."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        import tomllib
        with open(os.path.join(".streamlit", "secrets.toml"), "rb") as f:
            os.environ["OPENAI_API_KEY"] = tomllib.load(f)["openai"]["api_key"]
    except Exception:
        print("No OPENAI_API_KEY found. Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)


def _find_col(columns, candidates):
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def load_keywords(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    kcol = _find_col(df.columns, ["keyword", "keywords", "query", "term"])
    if not kcol:
        print(f"No keyword column found in {path}. Columns: {list(df.columns)}")
        sys.exit(1)
    vcol = _find_col(df.columns, ["search volume", "search_volume", "volume", "vol", "searches"])
    out = pd.DataFrame()
    out["keyword"] = df[kcol].astype(str).str.strip()
    out["volume"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else np.nan
    out = out[out["keyword"].str.len() > 0].drop_duplicates("keyword").reset_index(drop=True)
    return out


def load_reference(path):
    """Return {keyword_lower: ref_cluster} from a reference clustering export, or {}."""
    if not path:
        return {}
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    kcol = _find_col(df.columns, ["keyword", "keywords", "query", "term"])
    ccol = _find_col(df.columns, ["cluster", "cluster_id", "group", "page", "spoke"])
    if not kcol or not ccol:
        print(f"Reference needs keyword and cluster columns; found {list(df.columns)}. Skipping reference.")
        return {}
    return {str(k).strip().lower(): str(c) for k, c in zip(df[kcol], df[ccol])}


# --------------------------------------------------------------------------- #
# Embeddings (cached per model)
# --------------------------------------------------------------------------- #
def get_embeddings(keywords, model, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5((model + "|" + "\n".join(keywords)).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"emb_{model.replace('/', '_')}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    from openai import OpenAI
    client = OpenAI()
    vecs = []
    for i in range(0, len(keywords), 200):
        batch = keywords[i:i + 200]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        print(f"    embedded {min(i + 200, len(keywords))}/{len(keywords)} with {model}")
    arr = normalize(np.array(vecs, dtype=np.float32))
    np.save(path, arr)
    return arr


# --------------------------------------------------------------------------- #
# Clustering approaches
# --------------------------------------------------------------------------- #
def run_agglomerative(vecs, threshold):
    """Average-linkage agglomerative on cosine distance; every keyword gets a home."""
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=float(threshold),
        metric="cosine", linkage="average",
    )
    return model.fit_predict(vecs)


def _assign_noise_to_nearest(vecs, labels):
    labels = labels.copy()
    ids = sorted(c for c in set(labels) if c != -1)
    if not ids:
        return labels
    cents = np.vstack([normalize(vecs[labels == c].mean(axis=0, keepdims=True))[0] for c in ids])
    noise = np.where(labels == -1)[0]
    if len(noise):
        nearest = (vecs[noise] @ cents.T).argmax(axis=1)
        for j, ni in enumerate(noise):
            labels[ni] = ids[nearest[j]]
    return labels


def run_hdbscan(vecs, umap_on, min_cluster_size, assign_noise):
    import hdbscan
    X = vecs
    if umap_on:
        import umap
        n = len(vecs)
        X = umap.UMAP(
            n_neighbors=int(max(2, min(15, n - 1))),
            n_components=int(max(2, min(10, n - 1))),
            min_dist=0.0, metric="cosine", random_state=42,
        ).fit_transform(vecs)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=1,
        metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(X)
    if assign_noise:
        labels = _assign_noise_to_nearest(vecs, labels)
    return np.asarray(labels)


# --------------------------------------------------------------------------- #
# Agreement metrics (reference only)
# --------------------------------------------------------------------------- #
def _pairs(counts):
    return sum(comb(int(v), 2) for v in counts if v > 1)


def agreement(labels, ref_labels):
    """ARI and pair precision/recall of labels vs ref_labels (aligned arrays)."""
    ct = pd.crosstab(pd.Series(labels), pd.Series(ref_labels))
    both = _pairs(ct.values.flatten())
    sa = _pairs(ct.sum(axis=1).values)
    sb = _pairs(ct.sum(axis=0).values)
    n = int(ct.values.sum())
    if n < 2:
        return 0.0, 0.0, 0.0
    exp = sa * sb / comb(n, 2)
    mx = 0.5 * (sa + sb)
    ari = (both - exp) / (mx - exp) if mx != exp else 1.0
    precision = both / sb if sb else 0.0
    recall = both / sa if sa else 0.0
    return ari, precision, recall


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_clusters_md(path, df, label_col, config_name):
    lines = [f"# Clusters: {config_name}\n"]
    sizes = df[label_col].value_counts()
    for cid in sizes.index:
        sub = df[df[label_col] == cid].sort_values("volume", ascending=False, na_position="last")
        vol = sub["volume"].sum(min_count=1)
        vol_txt = f", {int(vol)} vol" if pd.notna(vol) else ""
        lines.append(f"\n## Cluster {cid} ({len(sub)} keywords{vol_txt})")
        for _, row in sub.head(50).iterrows():
            v = f" ({int(row['volume'])})" if pd.notna(row["volume"]) else ""
            lines.append(f"- {row['keyword']}{v}")
        if len(sub) > 50:
            lines.append(f"- ...and {len(sub) - 50} more")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Keyword clustering bench")
    ap.add_argument("--input", required=True, help="CSV with a keyword column (volume optional)")
    ap.add_argument("--reference", default=None, help="Reference clustering CSV (e.g. Keyword Insights)")
    ap.add_argument("--outdir", default="cluster_lab_out")
    ap.add_argument("--models", default="text-embedding-3-large,text-embedding-3-small")
    ap.add_argument("--agglom-thresholds", default="0.25,0.35",
                    help="Cosine distance thresholds for agglomerative (lower = tighter clusters)")
    args = ap.parse_args()

    ensure_openai_key()
    os.makedirs(args.outdir, exist_ok=True)
    cache_dir = os.path.join(args.outdir, "_emb_cache")

    kw = load_keywords(args.input)
    ref = load_reference(args.reference)
    keywords = kw["keyword"].tolist()
    print(f"Loaded {len(keywords)} keywords. Reference clustering: "
          f"{'yes' if ref else 'none'}.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    thresholds = [float(t) for t in args.agglom_thresholds.split(",") if t.strip()]

    # Build the configuration list.
    def short(m):
        return "large" if "large" in m else "small" if "small" in m else m
    configs = []
    for m in models:
        for t in thresholds:
            configs.append({"name": f"{short(m)}_agglom_{t}", "model": m, "method": "agglom", "threshold": t})
        configs.append({"name": f"{short(m)}_hdbscan_assign", "model": m, "method": "hdbscan",
                        "umap": True, "mcs": 3, "assign_noise": True})
    configs.append({"name": f"{short(models[0])}_hdbscan_noise", "model": models[0], "method": "hdbscan",
                    "umap": True, "mcs": 3, "assign_noise": False})

    # Embed once per model.
    emb = {}
    for m in models:
        print(f"Embedding with {m} ...")
        emb[m] = get_embeddings(keywords, m, cache_dir)

    # Reference labels aligned to our keyword order (only where the reference knows the keyword).
    ref_aligned, ref_mask = None, None
    if ref:
        ref_aligned = np.array([ref.get(k.lower(), "__missing__") for k in keywords])
        ref_mask = ref_aligned != "__missing__"

    rows = []
    for cfg in configs:
        vecs = emb[cfg["model"]]
        if cfg["method"] == "agglom":
            labels = run_agglomerative(vecs, cfg["threshold"])
        else:
            labels = run_hdbscan(vecs, cfg["umap"], cfg["mcs"], cfg["assign_noise"])
        labels = np.asarray(labels)

        out_df = kw.copy()
        out_df["cluster"] = labels
        out_df.to_csv(os.path.join(args.outdir, f"{cfg['name']}.csv"), index=False)
        write_clusters_md(os.path.join(args.outdir, f"{cfg['name']}.md"), out_df, "cluster", cfg["name"])

        sizes = pd.Series(labels).value_counts()
        non_noise = sizes.drop(index=-1, errors="ignore")
        n_noise = int((labels == -1).sum())
        n_singletons = int((non_noise == 1).sum())
        total = len(labels)
        ari = prec = rec = float("nan")
        if ref_aligned is not None and ref_mask.any():
            ari, prec, rec = agreement(labels[ref_mask], ref_aligned[ref_mask])

        rows.append({
            "config": cfg["name"],
            "clusters": int(non_noise.shape[0]),
            "noise": n_noise,
            "singletons": n_singletons,
            "%singletons": round(100 * n_singletons / total, 1),
            "median_size": int(non_noise.median()) if len(non_noise) else 0,
            "max_size": int(non_noise.max()) if len(non_noise) else 0,
            "ARI_vs_ref": round(ari, 3) if ari == ari else "-",
            "pairP": round(prec, 2) if prec == prec else "-",
            "pairR": round(rec, 2) if rec == rec else "-",
        })
        print(f"  {cfg['name']:28s} clusters={rows[-1]['clusters']:4d} "
              f"singletons={n_singletons:4d} noise={n_noise:4d} max={rows[-1]['max_size']:4d}")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.outdir, "_summary.csv"), index=False)
    print("\n================ SUMMARY (ARI/pairs are a REFERENCE, not a target) ================")
    print(summary.to_string(index=False))
    print(f"\nReadable clusters per config are in: {args.outdir}/<config>.md")
    print("Open a few and tell me which one's groupings look right. That becomes the locked core.")


if __name__ == "__main__":
    main()
