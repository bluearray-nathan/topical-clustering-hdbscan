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
PAGE_MERGE_SIM = 0.85    # within a pillar, merge same-intent pages whose centroids are at least this similar
PILLAR_CAP = 400         # pillars over this many keywords are split into sub-pillars
PILLAR_SUBTHRESHOLD = 0.40  # tighter threshold used when splitting an oversized pillar
KEEP_FLOOR = 100         # a single-keyword page with at least this volume stays its own page
FOLD_SIM = 0.55          # fold a singleton into the nearest page only if at least this similar
UNGROUPED = "Ungrouped (review)"
TOPIC_CAP = 10           # consolidate pillar sections into at most this many broad topics
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


def group_pillars_into_topics(pillar_labels):
    """Group pillars into a small set of broad topics. The model first assigns each pillar a section;
    if that yields more than TOPIC_CAP sections, a consolidation pass merges them into at most
    TOPIC_CAP broad topics. Returns {pillar_label: topic_name}."""
    if len(pillar_labels) <= 1:
        return {p: p for p in pillar_labels}
    sysmsg = ("You are an SEO information architect. Group these content pillars into broad topical "
              "sections. Every pillar goes in exactly one section. "
              'Reply only as JSON: {"results":[{"pillar":"<pillar label>","section":"<section name>"}]}.')
    sec = {}
    try:
        data = _chat_json(sysmsg, "Pillars:\n" + "\n".join(f"- {p}" for p in pillar_labels))
        for r in data.get("results", []):
            p = str(r.get("pillar", "")).strip()
            s = str(r.get("section", "")).strip()
            if p:
                sec[p] = s or "Other"
    except Exception:
        pass
    for p in pillar_labels:
        sec.setdefault(p, "Other")

    sections = sorted(set(sec.values()))
    if len(sections) <= TOPIC_CAP:
        return sec
    # too many sections: consolidate them into a few broad topics
    cons = {}
    sys2 = (f"You are an SEO information architect. Merge these content sections into AT MOST {TOPIC_CAP} "
            "broad topics. Every section maps to exactly one topic. "
            'Reply only as JSON: {"results":[{"section":"<section name>","topic":"<topic name>"}]}.')
    try:
        data = _chat_json(sys2, "Sections:\n" + "\n".join(f"- {s}" for s in sections))
        for r in data.get("results", []):
            s = str(r.get("section", "")).strip()
            t = str(r.get("topic", "")).strip()
            if s:
                cons[s] = t or s
    except Exception:
        pass
    for s in sections:
        cons.setdefault(s, s)
    return {p: cons.get(sec[p], sec[p]) for p in pillar_labels}


def name_pages(df, page_intent):
    """Name pages with the model, one pillar at a time so it sees the sibling pages and gives
    each a distinct, descriptive title (a guide and a comparison page on the same subject get
    clearly different names). Returns {page_id: name}; falls back to the head term if missed."""
    names = {}
    sysmsg = ("You are an SEO content strategist. You are given the pages within one content pillar, "
              "each as a list of keywords and a dominant intent. Give every page a short, descriptive "
              "title (2 to 6 words, title case) reflecting both its subject and its angle, so a guide "
              "and a comparison page on the same subject get clearly different titles. Every title must "
              "be DISTINCT. Do not use quotation marks or the word 'page'. "
              'Reply only as JSON: {"results":[{"id":<id>,"title":"..."}]}.')

    def one_pillar(pid):
        sub = df[df["pillar_id"] == pid]
        pages = []
        for pgid, grp in sub.groupby("page_id"):
            kws = grp.sort_values("volume", ascending=False, na_position="last")["keyword"].tolist()
            pages.append({"id": int(pgid), "intent": page_intent.get(pgid, "Mixed"), "keywords": kws[:15]})
        try:
            data = _chat_json(sysmsg, "Pages in this pillar:\n" + json.dumps(pages))
            out = {}
            for r in data.get("results", []):
                try:
                    out[int(r["id"])] = " ".join(str(r.get("title", "")).strip().strip('"').split()[:8])
                except Exception:
                    pass
            return out
        except Exception:
            return {}

    pids = sorted(df["pillar_id"].unique())
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for fut in cf.as_completed([ex.submit(one_pillar, p) for p in pids]):
            names.update(fut.result())
    for pgid, grp in df.groupby("page_id"):                       # fallback: head term
        if not names.get(pgid):
            names[pgid] = grp.sort_values("volume", ascending=False, na_position="last")["keyword"].iloc[0]
    return names


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


def dfs_collect_results(id_to_kw, auth, timeout_s=14400, poll_s=10, on_progress=None, flush_s=60):
    """Poll tasks_ready, fetch advanced results, keep the top organic results per keyword.
    on_progress(out), if given, is called at most every flush_s seconds and once at the end,
    so results can be cached as they arrive and a long or interrupted run never loses the
    SERPs it has already paid for. Returns (kw_to_results, timed_out_count)."""
    headers = {"Authorization": "Basic " + auth}
    pending, out = dict(id_to_kw), {}
    deadline = time.time() + timeout_s
    last_flush = time.time()

    def fetch_one(rid):
        try:
            jj = requests.get(f"{DFS_BASE}/task_get/advanced/{rid}", headers=headers, timeout=60).json()
            items = ((jj.get("tasks") or [{}])[0].get("result") or [{}])[0].get("items") or []
            org = [{"domain": i.get("domain"), "url": i.get("url"), "title": i.get("title")}
                   for i in items if i.get("type") == "organic"]
            return rid, org[:10]
        except Exception:
            return rid, []

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
        if ready:
            with cf.ThreadPoolExecutor(max_workers=10) as ex:
                for rid, org in ex.map(fetch_one, ready):
                    out[pending[rid]] = org
                    pending.pop(rid, None)
        log(f"  collected {len(out)}/{len(id_to_kw)} SERPs")
        if on_progress and ready and time.time() - last_flush >= flush_s:
            on_progress(out)
            last_flush = time.time()
        if pending:
            time.sleep(poll_s)
    if on_progress:
        on_progress(out)
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
    per keyword. Navigational SERPs are skipped (handled by nav_check). Batches run in parallel,
    which is the difference between minutes and a couple of hours on a large keyword set."""
    sysmsg = ("For each keyword you are given its top 10 organic results (domain, url, title). Classify "
              "each result's page type as one of: Informational (guides, explainers, definitions), "
              "Commercial (comparison, review or aggregator pages), Transactional (product, quote, buy or "
              "sign-up pages), Local (local or map listings). Return the COUNT of each type across the "
              'results, per keyword. Reply only as JSON: {"results":[{"keyword":"...","Informational":int,'
              '"Commercial":int,"Transactional":int,"Local":int}]}.')
    have = [k for k, v in kw_to_results.items() if v and not nav_check(k, v)]

    def one_batch(batch):
        payload = [{"keyword": k, "ranking_results": kw_to_results[k]} for k in batch]
        res = {}
        try:
            data = _chat_json(sysmsg, "Count page types:\n" + json.dumps(payload))
            for x in data.get("results", []):
                k = str(x.get("keyword", "")).strip()
                res[k] = {t: int(x.get(t, 0) or 0) for t in PAGE_TYPES}
        except Exception:
            pass
        return res

    batches = [have[i:i + 8] for i in range(0, len(have), 8)]
    out = {}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for fut in cf.as_completed([ex.submit(one_batch, b) for b in batches]):
            out.update(fut.result())
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
    with np.errstate(all="ignore"):
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


def split_oversized_pillars(pillar_ids, X, cap=PILLAR_CAP, sub_threshold=PILLAR_SUBTHRESHOLD):
    """Recursively re-cluster any pillar over the size cap, tightening the threshold each level until
    every pillar is within the cap. A dense head-term core gets broken up as far as needed while sparse
    pillars are left coarse. Returns relabelled, contiguous pillar ids."""
    new = np.array(pillar_ids, dtype=int)
    nxt = [int(new.max()) + 1]

    def recurse(idx, thr):
        if len(idx) <= cap or thr < 0.15:
            new[idx] = nxt[0]
            nxt[0] += 1
            return
        sub = cluster(X[idx], thr)
        groups = {}
        for j, lab in zip(idx, sub):
            groups.setdefault(int(lab), []).append(int(j))
        if len(groups) <= 1:                      # threshold did not separate it; go tighter
            recurse(idx, round(thr - 0.05, 2))
            return
        for g in groups.values():
            recurse(np.array(g), round(thr - 0.05, 2))

    over = [pid for pid in sorted(set(int(p) for p in pillar_ids)) if int((new == pid).sum()) > cap]
    for pid in over:
        recurse(np.where(new == pid)[0], sub_threshold)
    order = {v: i for i, v in enumerate(sorted(set(int(x) for x in new)))}
    new = np.array([order[int(x)] for x in new], dtype=int)
    if over:
        log(f"split {len(over)} oversized pillar(s) (> {cap} kw), recursing from threshold {sub_threshold}")
    return new


def fold_singletons(df, X, keep_floor=KEEP_FLOOR, fold_sim=FOLD_SIM):
    """Single-keyword pages: keep high-volume ones, fold relevant ones into the nearest real page,
    quarantine the rest into an ungrouped bucket (never merged into a real page, never deleted).
    Edits df columns in place; returns (kept, folded, quarantined)."""
    pid = df["page_id"].values.copy()
    counts = pd.Series(pid).value_counts()
    sizes = pd.Series(pid).map(counts).values
    multi_ids = sorted(set(int(p) for p in pid[sizes > 1]))
    if not multi_ids:
        return 0, 0, 0
    pos_by_page = {p: np.where(pid == p)[0] for p in multi_ids}
    cent = np.vstack([X[pos_by_page[p]].mean(0) for p in multi_ids])
    cent = cent / np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9, None)
    meta_cols = ["pillar_id", "pillar", "topic", "topic_id"]
    pmeta = {p: {c: df[c].values[pos_by_page[p][0]] for c in meta_cols} for p in multi_ids}
    cols = {c: df[c].values.copy() for c in meta_cols}
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0).values
    nk = nf = nq = 0
    for i in np.where(sizes == 1)[0]:
        if vol[i] >= keep_floor:
            nk += 1
            continue
        with np.errstate(all="ignore"):
            sims = cent @ X[i]
        j = int(np.argmax(sims))
        if sims[j] >= fold_sim:
            tgt = multi_ids[j]
            pid[i] = tgt
            for c in meta_cols:
                cols[c][i] = pmeta[tgt][c]
            nf += 1
        else:
            pid[i] = -1
            cols["pillar_id"][i] = -1
            cols["pillar"][i] = UNGROUPED
            cols["topic"][i] = UNGROUPED
            cols["topic_id"][i] = UNGROUPED
            nq += 1
    df["page_id"] = pid
    for c in meta_cols:
        df[c] = cols[c]
    log(f"singletons: kept {nk}, folded {nf}, quarantined {nq}")
    return nk, nf, nq


def merge_pages(df, X, sim=PAGE_MERGE_SIM):
    """Within each pillar, merge near-duplicate pages: those whose keyword centroids are at least
    `sim` similar AND share the same page intent, so a commercial comparison page is never folded
    into an informational guide on the same subject. The highest-volume page in each merged group
    keeps its id and name; the names it absorbs are recorded in a 'merged_from' column so every
    combine stays visible and reversible. Edits df in place; returns the number of pages absorbed."""
    pid = df["page_id"].values.copy()
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0).values
    name = df["page"].values.copy()
    intent_col = df["page_intent"].values
    merged_from = defaultdict(list)
    absorbed = 0
    for plr in sorted(set(int(p) for p in df["pillar_id"].values if p >= 0)):
        pages = sorted(set(int(p) for p in pid[(df["pillar_id"].values == plr) & (pid >= 0)]))
        if len(pages) < 2:
            continue
        pos = {p: np.where(pid == p)[0] for p in pages}
        cent = np.vstack([X[pos[p]].mean(0) for p in pages])
        cent = cent / np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9, None)
        with np.errstate(all="ignore"):
            S = np.clip(np.nan_to_num(cent @ cent.T), 0.0, 1.0)
        intent = {p: intent_col[pos[p][0]] for p in pages}
        vols = {p: float(vol[pos[p]].sum()) for p in pages}
        parent = {p: p for p in pages}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(pages)):
            for b in range(a + 1, len(pages)):
                if S[a, b] >= sim and intent[pages[a]] == intent[pages[b]]:
                    parent[find(pages[a])] = find(pages[b])
        comp = defaultdict(list)
        for p in pages:
            comp[find(p)].append(p)
        for members in comp.values():
            if len(members) < 2:
                continue
            keep = max(members, key=lambda p: vols[p])
            keep_name = name[pos[keep][0]]
            for p in members:
                if p == keep:
                    continue
                merged_from[keep].append(name[pos[p][0]])
                pid[pos[p]] = keep
                name[pos[p]] = keep_name
                absorbed += 1
    df["page_id"] = pid
    df["page"] = name
    df["merged_from"] = ["; ".join(sorted(set(merged_from.get(int(p), [])))) if int(p) >= 0 else ""
                         for p in pid]
    return absorbed


def designate_hub_pages(df, max_candidates=15):
    """For each pillar, pick the hub (pillar) page the supporting pages link up to. The model chooses
    from the pillar's top pages, preferring breadth and overview intent; falls back to the highest-volume
    page. Returns {page_id: 'Pillar page' | 'Supporting'} for real pages."""
    real = df[df["page_id"] >= 0]
    sysmsg = ("You are an SEO information architect. Given the pages in one content pillar (each with top "
              "keywords, total search volume and intent), choose the single PILLAR (hub) page: the broad "
              "overview page targeting the head term that the others link up to. Prefer breadth and "
              "overview intent over a narrow sub-topic, even if a narrow page has more volume. "
              'Reply only as JSON: {"hub_page_id": <id>}.')

    def one(pid):
        sub = real[real["pillar_id"] == pid]
        pg = (sub.groupby("page_id").agg(page=("page", "first"), vol=("volume", "sum"),
              intent=("page_intent", "first")).reset_index().sort_values("vol", ascending=False))
        ids = [int(x) for x in pg["page_id"]]
        if len(ids) == 1:
            return {ids[0]: "Pillar page"}
        cand = pg.head(max_candidates)
        candset = {int(x) for x in cand["page_id"]}
        hub = int(cand.iloc[0]["page_id"])
        lines = []
        for _, r in cand.iterrows():
            top = sub[sub["page_id"] == r["page_id"]].sort_values("volume", ascending=False)["keyword"].head(5).tolist()
            lines.append(f'id {int(r["page_id"])}: {r["page"]} (vol {int(r["vol"])}, {r["intent"]}; {", ".join(top)})')
        try:
            h = int(_chat_json(sysmsg, "Pages:\n" + "\n".join(lines)).get("hub_page_id"))
            if h in candset:
                hub = h
        except Exception:
            pass
        return {i: ("Pillar page" if i == hub else "Supporting") for i in ids}

    out = {}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for r in cf.as_completed([ex.submit(one, int(p)) for p in sorted(real["pillar_id"].unique())]):
            out.update(r.result())
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pillar-threshold", type=float, default=0.45)
    ap.add_argument("--cache", default=".serp_cache.json")
    ap.add_argument("--location", default="United Kingdom")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--pillar-cap", type=int, default=PILLAR_CAP)
    ap.add_argument("--pillar-subthreshold", type=float, default=PILLAR_SUBTHRESHOLD)
    ap.add_argument("--keep-floor", type=float, default=KEEP_FLOOR)
    ap.add_argument("--fold-sim", type=float, default=FOLD_SIM)
    ap.add_argument("--split-oversized", action="store_true",
                    help="optionally break pillars over --pillar-cap into sub-pillars (off by default)")
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--merge-sim", type=float, default=PAGE_MERGE_SIM,
                    help="merge same-intent pages within a pillar whose centroids are at least this similar")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip merging near-duplicate pages (on by default)")
    ap.add_argument("--no-hub", action="store_true")
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
    df["pillar_id"] = cluster(X, args.pillar_threshold)   # pillars (themes)
    if args.split_oversized:
        df["pillar_id"] = split_oversized_pillars(df["pillar_id"].values, X,
                                                  args.pillar_cap, args.pillar_subthreshold)
    pillar_labels = label_groups(df, "pillar_id", "pillar")
    df["pillar"] = df["pillar_id"].map(pillar_labels)
    # Merge pillars the labeller named identically (same theme) so each pillar is one group with one hub.
    _name_id = {}
    df["pillar_id"] = df["pillar"].map(lambda nm: _name_id.setdefault(nm, len(_name_id)))
    # Topics: the model groups the pillars into broad editorial sections.
    topic_of_label = group_pillars_into_topics(sorted(set(pillar_labels.values())))
    df["topic"] = df["pillar"].map(topic_of_label).fillna("Other")
    df["topic_id"] = df["topic"]
    log(f"{df['topic'].nunique()} topics, {df['pillar_id'].nunique()} pillars")

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

        def checkpoint(fetched):
            """Persist SERPs as they arrive, so a long or interrupted run keeps what it paid for."""
            for k, res in fetched.items():
                cache[ckey(k)] = res
            try:
                json.dump(cache, open(args.cache, "w"))
            except Exception:
                pass

        _, timed_out = dfs_collect_results(id_to_kw, auth, on_progress=checkpoint)
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

    # Single-keyword pages: keep high-volume ones, fold relevant ones into the nearest page,
    # quarantine the rest into an ungrouped bucket (never merged into a real page, never deleted).
    if not args.no_fold:
        fold_singletons(df, X, args.keep_floor, args.fold_sim)
        log(f"after folding: {df.loc[df.page_id >= 0, 'page_id'].nunique()} pages, "
            f"{int((df.page_id < 0).sum())} keywords ungrouped")

    # Page-level intent (dominant member intent, else Mixed)
    p_intent = {}
    for pgid, grp in df.groupby("page_id"):
        ic = grp["intent"].value_counts()
        p_intent[pgid] = ic.index[0] if ic.iloc[0] / len(grp) >= MIXED_BELOW else "Mixed"
    df["page_intent"] = df["page_id"].map(p_intent)
    # The model names each page from its keywords and intent, so pages that differ mainly
    # by intent (a guide versus a comparison page) get clearly distinct names.
    page_names = name_pages(df, p_intent)
    df["page"] = df["page_id"].map(page_names)
    df.loc[df["page_id"] < 0, "page"] = UNGROUPED
    df.loc[df["page_id"] < 0, "page_intent"] = "Mixed"
    log(f"named {df.loc[df.page_id >= 0, 'page_id'].nunique()} pages")

    # Merge near-duplicate pages within a pillar (same intent, very similar centroid). The
    # highest-volume page keeps its name; the names it absorbs are kept in 'merged_from'.
    merged = 0
    if not args.no_merge:
        merged = merge_pages(df, X, args.merge_sim)
        log(f"merged {merged} near-duplicate page(s); "
            f"{df.loc[df.page_id >= 0, 'page_id'].nunique()} pages remain")
    else:
        df["merged_from"] = ""

    # Designate the hub (pillar) page within each pillar; the rest are supporting pages.
    if not args.no_hub:
        df["page_role"] = df["page_id"].map(designate_hub_pages(df))
    else:
        df["page_role"] = "Supporting"
    df.loc[df["page_id"] < 0, "page_role"] = "Ungrouped"
    df["page_role"] = df["page_role"].fillna("Supporting")
    log(f"{int((df.drop_duplicates('page_id')['page_role'] == 'Pillar page').sum())} pillar pages designated")

    # ---- Outputs: Topic > Pillar > Page ----
    df[["keyword", "volume", "topic", "pillar", "page", "page_role", "page_intent", "intent", "intent_source"]].rename(
        columns={"keyword": "Keyword", "volume": "Volume", "topic": "Topic", "pillar": "Pillar",
                 "page": "Page", "page_role": "Page role", "page_intent": "Page intent",
                 "intent": "Keyword intent", "intent_source": "Intent source"}
    ).to_csv(os.path.join(args.outdir, "keyword_mapping.csv"), index=False)

    pages = (df.groupby("page_id").agg(
                Topic=("topic", "first"), Pillar=("pillar", "first"), Page=("page", "first"),
                Role=("page_role", "first"), Intent=("page_intent", "first"),
                Keywords=("keyword", "size"), Volume=("volume", "sum"),
                MergedFrom=("merged_from", "first"))
             .reset_index(drop=True).rename(columns={"MergedFrom": "Merged from"})
             .sort_values(["Topic", "Volume"], ascending=[True, False], na_position="last"))
    pages.to_csv(os.path.join(args.outdir, "page_summary.csv"), index=False)

    meta = {
        "run_id": args.run_id,
        "finished_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_keywords": int(len(df)),
        "n_topics": int(df.loc[df.page_id >= 0, "topic_id"].nunique()),
        "n_pillars": int(df.loc[df.page_id >= 0, "pillar_id"].nunique()),
        "n_pages": int(df.loc[df.page_id >= 0, "page_id"].nunique()),
        "n_pages_merged": int(merged),
        "n_ungrouped": int((df.page_id < 0).sum()),
        "intent_from_serp": int((df.intent_source == "serp").sum()),
        "intent_from_text": int((df.intent_source == "text").sum()),
        "serp_cost_usd": round(cost, 4),
        "timed_out": int(timed_out),
        "page_intent_distribution": {k: int(v) for k, v in
                                     df[df.page_id >= 0].drop_duplicates("page_id")["page_intent"].value_counts().items()},
    }
    result = {"meta": meta, "keywords": df[["keyword", "volume", "topic", "pillar", "page",
              "page_role", "page_intent", "intent_source"]].to_dict(orient="records")}
    with open(os.path.join(args.outdir, "result.json"), "w") as f:
        json.dump(result, f)
    with open(os.path.join(args.outdir, "status.json"), "w") as f:
        json.dump({"status": "done", **meta}, f)
    log(f"done: {meta['n_topics']} topics, {meta['n_pillars']} pillars, {meta['n_pages']} pages, "
        f"${meta['serp_cost_usd']} SERP cost")


if __name__ == "__main__":
    main()
