#!/usr/bin/env python3
"""
pipeline.py: the headless engine. Runs the full keyword -> page map and writes results.

Designed to run unattended (e.g. inside a GitHub Action), so it reads credentials from
the environment and takes input/output as arguments. It is the same logic the Streamlit
front-end shows, just with no UI.

Flow:
  load keywords
    -> embed (text-embedding-3-large)
    -> cluster into topics (agglomerative, cosine, threshold 0.25)
    -> label each topic (model)
    -> SERP every keyword via DataForSEO Standard queue (cheapest), collect the ranking results
    -> classify intent by reading those ranking results (model); text intent as the fallback
    -> split each topic into pages by intent
    -> write result.json (+ CSVs)

Env: OPENAI_API_KEY, DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD
Usage:
  python pipeline.py --input runs/<id>/input.csv --outdir runs/<id> \
      --threshold 0.25 --location "United Kingdom" --run-id <id>
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import datetime as dt
import concurrent.futures as cf
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
INTENTS = ["Transactional", "Commercial", "Informational", "Navigational", "Local"]
PAGE_TYPES = ["Informational", "Commercial", "Transactional", "Local"]
NAV_DOMINANCE = 0.4      # one domain holding this share of the top results => navigational
MIXED_BELOW = 0.6        # if the top page type is below this share, the SERP is Mixed
UBIQUITOUS_DF = 0.25     # domains ranking for this share of a pillar's keywords are dropped (non-discriminating)
BLEND_ALPHA = 0.65       # weight on semantic vs specific-SERP overlap when forming pages
PAGE_THRESHOLD = 0.50    # agglomerative distance threshold for the page blend
DFS_BASE = "https://api.dataforseo.com/v3/serp/google/organic"

client = OpenAI()


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Load / embed / cluster / label
# --------------------------------------------------------------------------- #
def load_keywords(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    kcol = next((low[c] for c in ["keyword", "keywords", "query", "term"] if c in low), None)
    if not kcol:
        log(f"No keyword column in {path}; columns {list(df.columns)}")
        sys.exit(1)
    vcol = next((low[c] for c in ["search volume", "search_volume", "volume", "vol", "searches"] if c in low), None)
    out = pd.DataFrame()
    out["keyword"] = df[kcol].astype(str).str.strip()
    out["volume"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else np.nan
    return out[out["keyword"].str.len() > 0].drop_duplicates("keyword").reset_index(drop=True)


def embed(keywords):
    vecs = []
    for i in range(0, len(keywords), 200):
        r = client.embeddings.create(model=EMBED_MODEL, input=keywords[i:i + 200])
        vecs.extend([d.embedding for d in sorted(r.data, key=lambda d: d.index)])
    return normalize(np.array(vecs, dtype=np.float32))


def cluster(X, threshold):
    return AgglomerativeClustering(n_clusters=None, distance_threshold=float(threshold),
                                   metric="cosine", linkage="average").fit_predict(X)


def _chat_json(system, user):
    r = client.chat.completions.create(model=LLM_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)


def label_groups(df, col, kind="topic"):
    labels = {}

    def one(gid):
        kws = df[df[col] == gid].sort_values("volume", ascending=False, na_position="last")["keyword"].tolist()
        sysmsg = "You are an SEO content analyst. Reply with only the label."
        user = (f"These keywords are one {kind}:\n" + ", ".join(kws[:25]) +
                f"\n\nGive a concise {kind} name in 1 to 4 words (noun phrase). Reply with only the name.")
        try:
            r = client.chat.completions.create(model=LLM_MODEL, temperature=0.2,
                messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": user}])
            return gid, " ".join(r.choices[0].message.content.strip().strip('"').split()[:5]) or kws[0]
        except Exception:
            return gid, kws[0] if kws else str(gid)

    ids = sorted(df[col].unique())
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for fut in cf.as_completed([ex.submit(one, g) for g in ids]):
            gid, name = fut.result()
            labels[gid] = name
    return labels


# --------------------------------------------------------------------------- #
# Intent: text base, then SERP (domain-based) override
# --------------------------------------------------------------------------- #
def text_intent(keywords):
    out = {}
    sysmsg = ("You are an SEO search-intent classifier. For each keyword choose exactly one intent "
              "from: " + ", ".join(INTENTS) + '. Reply only as JSON: {"results":[{"keyword":"...","intent":"..."}]}.')
    for i in range(0, len(keywords), 40):
        b = keywords[i:i + 40]
        try:
            data = _chat_json(sysmsg, "Classify:\n" + "\n".join("- " + k for k in b))
            for x in data.get("results", []):
                it = str(x.get("intent", "")).strip()
                out[str(x.get("keyword", "")).strip()] = it if it in INTENTS else "Informational"
        except Exception:
            for k in b:
                out[k] = "Informational"
    for k in keywords:
        out.setdefault(k, "Informational")
    return out


def dfs_post(keywords, location, auth):
    """Submit Standard-queue tasks (priority 1, cheapest). Returns (id_to_kw, cost)."""
    headers = {"Authorization": "Basic " + auth, "Content-Type": "application/json"}
    id_to_kw, cost = {}, 0.0
    for i in range(0, len(keywords), 100):
        chunk = keywords[i:i + 100]
        payload = [{"keyword": k, "location_name": location, "language_code": "en",
                    "device": "desktop", "priority": 1} for k in chunk]
        try:
            r = requests.post(DFS_BASE + "/task_post", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception:
            j = {}
        cost += float(j.get("cost") or 0.0)
        for k, task in zip(chunk, j.get("tasks") or []):
            if task.get("id"):
                id_to_kw[task["id"]] = k
    return id_to_kw, cost


def dfs_collect_results(id_to_kw, auth, timeout_s=2400, poll_s=10):
    """Poll tasks_ready, fetch advanced results, keep the top organic results per keyword.
    Returns (kw_to_results, timed_out_count)."""
    headers = {"Authorization": "Basic " + auth}
    pending, out = dict(id_to_kw), {}
    deadline = time.time() + timeout_s
    while pending and time.time() < deadline:
        ready = []
        try:
            r = requests.get(DFS_BASE + "/tasks_ready", headers=headers, timeout=60)
            r.raise_for_status()
            for task in (r.json().get("tasks") or []):
                for res in (task.get("result") or []):
                    if res.get("id") in pending:
                        ready.append(res["id"])
        except Exception:
            ready = []
        for rid in ready:
            try:
                jj = requests.get(f"{DFS_BASE}/task_get/advanced/{rid}", headers=headers, timeout=60).json()
                items = ((jj.get("tasks") or [{}])[0].get("result") or [{}])[0].get("items") or []
                org = [{"domain": i.get("domain"), "url": i.get("url"), "title": i.get("title")}
                       for i in items if i.get("type") == "organic"]
                out[pending[rid]] = org[:10]
            except Exception:
                out[pending[rid]] = []
            pending.pop(rid, None)
        log(f"  collected {len(out)}/{len(id_to_kw)} SERPs")
        if pending:
            time.sleep(poll_s)
    return out, len(pending)


def nav_check(kw, results):
    """Deterministic navigational signal: a brand-dominated SERP, the keyword naming the
    top-ranking brand, or a URL in the keyword itself."""
    domains = [d["domain"] for d in results if d.get("domain")]
    if not domains:
        return False
    _, cnt = Counter(domains).most_common(1)[0]
    if cnt / len(domains) >= NAV_DOMINANCE:
        return True
    brand = re.sub(r"^www\.", "", domains[0]).split(".")[0]
    if len(brand) >= 4 and brand in re.sub(r"[^a-z0-9]", "", kw.lower()):
        return True
    if re.search(r"\w+\.(com|co\.uk|net|org)", kw.lower()):
        return True
    return False


def serp_type_counts(kw_to_results):
    """Model labels each of the top 10 ranking pages and returns the count of each page type
    per keyword. Navigational SERPs are skipped (handled by nav_check)."""
    out = {}
    sysmsg = ("For each keyword you are given its top 10 organic results (domain, url, title). Classify "
              "each result's page type as one of: Informational (guides, explainers, definitions), "
              "Commercial (comparison, review or aggregator pages), Transactional (product, quote, buy or "
              "sign-up pages), Local (local or map listings). Return the COUNT of each type across the "
              'results, per keyword. Reply only as JSON: {"results":[{"keyword":"...","Informational":int,'
              '"Commercial":int,"Transactional":int,"Local":int}]}.')
    have = [k for k, v in kw_to_results.items() if v and not nav_check(k, v)]
    for i in range(0, len(have), 8):
        batch = have[i:i + 8]
        payload = [{"keyword": k, "ranking_results": kw_to_results[k]} for k in batch]
        try:
            data = _chat_json(sysmsg, "Count page types:\n" + json.dumps(payload))
            for x in data.get("results", []):
                k = str(x.get("keyword", "")).strip()
                out[k] = {t: int(x.get(t, 0) or 0) for t in PAGE_TYPES}
        except Exception:
            pass
    return out


def decide_intent(kw, kw_to_results, counts):
    """Navigational by rule; else the dominant page type across the top 10, or Mixed when they
    are split. Returns None when there is no usable SERP (caller falls back to text)."""
    res = kw_to_results.get(kw)
    if not res:
        return None
    if nav_check(kw, res):
        return "Navigational"
    c = counts.get(kw)
    if not c or sum(c.values()) == 0:
        return None
    dom, n = max(c.items(), key=lambda kv: kv[1])
    return dom if n / sum(c.values()) >= MIXED_BELOW else "Mixed"


def _norm_url(u):
    if not u:
        return ""
    u = u.lower().split("?")[0].split("#")[0]
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def cluster_pages_in_pillar(idx, X, keywords, kw_to_results):
    """Within one pillar, split keywords into pages by blending semantic similarity with
    specific-page SERP overlap (ubiquitous domains dropped). Returns local cluster labels."""
    n = len(idx)
    if n == 1:
        return np.array([0])
    sub_kw = [keywords[i] for i in idx]
    urlsets, dom_kw = {}, defaultdict(set)
    for k in sub_kw:
        us = {_norm_url(r.get("url")) for r in (kw_to_results.get(k) or []) if r.get("url")}
        urlsets[k] = us
        for u in us:
            dom_kw[u.split("/")[0]].add(k)
    ubiq = {d for d, ks in dom_kw.items() if len(ks) / n >= UBIQUITOUS_DF}
    filt = {k: {u for u in urlsets[k] if u.split("/")[0] not in ubiq} for k in sub_kw}
    sub = X[idx]
    Ssem = np.clip(np.nan_to_num(sub @ sub.T), 0.0, 1.0)
    D = np.ones((n, n))
    for a in range(n):
        D[a, a] = 0.0
        for b in range(a + 1, n):
            A, B = filt[sub_kw[a]], filt[sub_kw[b]]
            jac = len(A & B) / len(A | B) if (A or B) else 0.0
            sim = BLEND_ALPHA * float(Ssem[a, b]) + (1 - BLEND_ALPHA) * jac
            D[a, b] = D[b, a] = 1.0 - sim
    D = np.clip(D, 0.0, 1.0)
    return AgglomerativeClustering(metric="precomputed", linkage="average",
                                   distance_threshold=PAGE_THRESHOLD, n_clusters=None).fit_predict(D)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topic-threshold", type=float, default=0.45)
    ap.add_argument("--pillar-threshold", type=float, default=0.25)
    ap.add_argument("--cache", default=".serp_cache.json")
    ap.add_argument("--location", default="United Kingdom")
    ap.add_argument("--run-id", default="local")
    args = ap.parse_args()

    for var in ["OPENAI_API_KEY", "DATAFORSEO_LOGIN", "DATAFORSEO_PASSWORD"]:
        if not os.environ.get(var):
            log(f"Missing env var {var}")
            sys.exit(1)
    auth = base64.b64encode(f"{os.environ['DATAFORSEO_LOGIN']}:{os.environ['DATAFORSEO_PASSWORD']}".encode()).decode()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_keywords(args.input)
    log(f"loaded {len(df)} keywords")

    X = embed(df["keyword"].tolist())
    df["topic_id"] = cluster(X, args.topic_threshold)     # coarse: topics
    df["pillar_id"] = cluster(X, args.pillar_threshold)   # finer: pillars (nest inside topics)
    log(f"{df.topic_id.nunique()} topics, {df.pillar_id.nunique()} pillars")
    df["topic"] = df.topic_id.map(label_groups(df, "topic_id", "topic"))
    df["pillar"] = df.pillar_id.map(label_groups(df, "pillar_id", "pillar"))

    # ---- SERP via Standard queue, cached on disk by (location, keyword) ----
    cache = {}
    if os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache))
        except Exception:
            cache = {}

    def ckey(kw):
        return f"{args.location}|||{kw}"

    keywords = df["keyword"].tolist()
    to_fetch = [k for k in keywords if ckey(k) not in cache]
    cost, timed_out = 0.0, 0
    if to_fetch:
        log(f"SERP (Standard queue): {len(to_fetch)} to fetch, {len(keywords) - len(to_fetch)} cached")
        id_to_kw, cost = dfs_post(to_fetch, args.location, auth)
        log(f"submitted {len(id_to_kw)} tasks, cost ${cost:.2f}")
        fetched, timed_out = dfs_collect_results(id_to_kw, auth)
        for k, res in fetched.items():
            cache[ckey(k)] = res
        try:
            json.dump(cache, open(args.cache, "w"))
        except Exception:
            pass
    else:
        log("all SERPs served from cache")
    kw_to_results = {k: (cache.get(ckey(k)) or []) for k in keywords}

    counts = serp_type_counts(kw_to_results)
    df["intent"] = df["keyword"].map(lambda k: decide_intent(k, kw_to_results, counts))
    df["intent_source"] = df["intent"].apply(lambda x: "serp" if x else None)
    mask = df["intent"].isna()
    if mask.any():
        log(f"{int(mask.sum())} keywords had no usable SERP; using text intent as fallback")
        ti = text_intent(df.loc[mask, "keyword"].tolist())
        df.loc[mask, "intent"] = df.loc[mask, "keyword"].map(lambda k: ti.get(k, "Informational"))
        df.loc[mask, "intent_source"] = "text"
    log(f"intent: {int((df.intent_source == 'serp').sum())} from SERP, "
        f"{int((df.intent_source == 'text').sum())} text; {timed_out} timed out")

    # ---- Pages: blend semantic + specific-SERP overlap within each pillar ----
    page_ids = np.full(len(df), -1, dtype=int)
    next_id = 0
    for pid in sorted(df["pillar_id"].unique()):
        idx = np.where(df["pillar_id"].values == pid)[0]
        local = cluster_pages_in_pillar(idx, X, keywords, kw_to_results)
        remap = {}
        for j, lab in zip(idx, local):
            lab = int(lab)
            if lab not in remap:
                remap[lab] = next_id
                next_id += 1
            page_ids[j] = remap[lab]
    df["page_id"] = page_ids
    log(f"formed {df.page_id.nunique()} pages")

    # Page-level intent (dominant member intent, else Mixed); head term names the page
    p_intent, p_head = {}, {}
    for pgid, grp in df.groupby("page_id"):
        ic = grp["intent"].value_counts()
        p_intent[pgid] = ic.index[0] if ic.iloc[0] / len(grp) >= MIXED_BELOW else "Mixed"
        p_head[pgid] = grp.sort_values("volume", ascending=False, na_position="last")["keyword"].iloc[0]
    df["page_intent"] = df["page_id"].map(p_intent)
    df["page"] = df["page_id"].map(p_head)

    # ---- Outputs: Topic > Pillar > Page ----
    df[["keyword", "volume", "topic", "pillar", "page", "page_intent", "intent", "intent_source"]].rename(
        columns={"keyword": "Keyword", "volume": "Volume", "topic": "Topic", "pillar": "Pillar",
                 "page": "Page", "page_intent": "Page intent", "intent": "Keyword intent",
                 "intent_source": "Intent source"}
    ).to_csv(os.path.join(args.outdir, "keyword_mapping.csv"), index=False)

    pages = (df.groupby("page_id").agg(
                Topic=("topic", "first"), Pillar=("pillar", "first"), Page=("page", "first"),
                Intent=("page_intent", "first"), Keywords=("keyword", "size"), Volume=("volume", "sum"))
             .reset_index(drop=True).sort_values(["Topic", "Volume"], ascending=[True, False], na_position="last"))
    pages.to_csv(os.path.join(args.outdir, "page_summary.csv"), index=False)

    meta = {
        "run_id": args.run_id,
        "finished_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_keywords": int(len(df)),
        "n_topics": int(df.topic_id.nunique()),
        "n_pillars": int(df.pillar_id.nunique()),
        "n_pages": int(df.page_id.nunique()),
        "intent_from_serp": int((df.intent_source == "serp").sum()),
        "intent_from_text": int((df.intent_source == "text").sum()),
        "serp_cost_usd": round(cost, 4),
        "timed_out": int(timed_out),
        "page_intent_distribution": {k: int(v) for k, v in
                                     df.drop_duplicates("page_id")["page_intent"].value_counts().items()},
    }
    result = {"meta": meta, "keywords": df[["keyword", "volume", "topic", "pillar", "page",
              "page_intent", "intent_source"]].to_dict(orient="records")}
    with open(os.path.join(args.outdir, "result.json"), "w") as f:
        json.dump(result, f)
    with open(os.path.join(args.outdir, "status.json"), "w") as f:
        json.dump({"status": "done", **meta}, f)
    log(f"done: {meta['n_topics']} topics, {meta['n_pillars']} pillars, {meta['n_pages']} pages, "
        f"${meta['serp_cost_usd']} SERP cost")


if __name__ == "__main__":
    main()
