# app.py: Keyword to Page Mapping (front-end / orchestrator)
# ---------------------------------------------------------------------------
# Thin Streamlit UI. The heavy engine (pipeline.py) runs in a GitHub Action so a
# run survives the browser closing and has no practical timeout. This app:
#   1. uploads the keyword CSV to the data branch,
#   2. triggers the Action,
#   3. polls for the result,
#   4. renders the Topic > Pillar > Page map.
#
# It holds only a GitHub token (in Streamlit secrets). The OpenAI and DataForSEO
# credentials live in GitHub Actions secrets, so they never touch the front-end.

import io
import os
import json
import time
import base64
import datetime as dt

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Keyword & topical clustering tool", page_icon="🔵", layout="wide")

# --------------------------- Blue Array branding -------------------------- #
# 2026 palette lives in .streamlit/config.toml. Fonts (Source Serif 4 headings,
# Raleway body) need a Google Fonts @import, so they go in here. This block runs
# before the secrets gate below, so the brand shows even on the config screen.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600&family=Source+Serif+4:wght@600;700&display=swap');
    html, body, .stApp, [data-testid="stSidebar"], .stMarkdown, .stMetric,
    .stButton > button, .stDownloadButton > button, input, textarea, select {
        font-family: 'Raleway', sans-serif;
    }
    h1, h2, h3, h4, [data-testid="stHeading"] {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 600;
        color: #002140;
    }
    a, a:visited { color: #1291D2; }
    .ba-stepper { display:flex; align-items:center; margin:.25rem 0 1.75rem; }
    .ba-step { display:flex; align-items:center; white-space:nowrap; font-family:'Raleway',sans-serif; font-weight:500; font-size:.95rem; color:rgba(0,33,64,.45); }
    .ba-step .ba-dot { width:30px; height:30px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; margin-right:.5rem; background:#EAF1F9; color:#1291D2; font-weight:600; border:2px solid #cfe0f2; }
    .ba-step.active, .ba-step.done { color:#002140; }
    .ba-step.active .ba-dot, .ba-step.done .ba-dot { background:#1291D2; color:#fff; border-color:#1291D2; }
    .ba-conn { flex:1; height:2px; background:#cfe0f2; margin:0 .75rem; }
    .ba-conn.done { background:#1291D2; }
    </style>
    """,
    unsafe_allow_html=True,
)

_LOGO = os.path.join(os.path.dirname(__file__), "assets", "logo-colour-converts.png")
if os.path.exists(_LOGO):
    st.logo(_LOGO, size="large", link="https://www.bluearray.co.uk")

st.title("Keyword & topical clustering tool")

# --------------------------- config from secrets --------------------------- #
try:
    gh = st.secrets["github"]
    TOKEN = gh["token"]
    REPO = gh["repo"]                              # "owner/name"
    CODE_BRANCH = gh.get("code_branch", "main")    # where the workflow lives
    DATA_BRANCH = gh.get("data_branch", "runs")    # where inputs/results live
    WORKFLOW = gh.get("workflow", "run.yml")
except Exception:
    st.error(
        "Missing GitHub config. Add this to the app's Streamlit secrets:\n\n"
        '[github]\n'
        'token = "ghp_..."\n'
        'repo = "bluearray-nathan/topical-clustering-hdbscan"\n'
        'code_branch = "main"\n'
        'data_branch = "runs"\n'
        'workflow = "run.yml"'
    )
    st.stop()

API = "https://api.github.com"
HDRS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
POLL_SECONDS = 6


# ------------------------------- GitHub API -------------------------------- #
def gh_get_file(path, ref):
    """Return the file's bytes, or None if it does not exist yet."""
    r = requests.get(f"{API}/repos/{REPO}/contents/{path}",
                     headers=HDRS, params={"ref": ref}, timeout=30)
    if r.status_code == 200:
        return base64.b64decode(r.json()["content"])
    return None


def gh_ensure_branch(branch, base):
    """Make sure the data branch exists, creating it from base if needed."""
    r = requests.get(f"{API}/repos/{REPO}/git/ref/heads/{branch}", headers=HDRS, timeout=30)
    if r.status_code == 200:
        return True
    b = requests.get(f"{API}/repos/{REPO}/git/ref/heads/{base}", headers=HDRS, timeout=30)
    if b.status_code != 200:
        return False
    sha = b.json()["object"]["sha"]
    c = requests.post(f"{API}/repos/{REPO}/git/refs", headers=HDRS,
                      json={"ref": f"refs/heads/{branch}", "sha": sha}, timeout=30)
    return c.status_code in (200, 201)


def gh_put_file(path, content_bytes, branch, message):
    """Create or update a file on the given branch."""
    cur = requests.get(f"{API}/repos/{REPO}/contents/{path}",
                       headers=HDRS, params={"ref": branch}, timeout=30)
    sha = cur.json().get("sha") if cur.status_code == 200 else None
    body = {"message": message,
            "content": base64.b64encode(content_bytes).decode(),
            "branch": branch}
    if sha:
        body["sha"] = sha
    r = requests.put(f"{API}/repos/{REPO}/contents/{path}", headers=HDRS, json=body, timeout=30)
    return r.status_code in (200, 201), r


def gh_dispatch(run_id, location, pillar_threshold):
    body = {"ref": CODE_BRANCH,
            "inputs": {"run_id": run_id, "location": location,
                       "pillar_threshold": str(pillar_threshold), "data_branch": DATA_BRANCH}}
    r = requests.post(f"{API}/repos/{REPO}/actions/workflows/{WORKFLOW}/dispatches",
                      headers=HDRS, json=body, timeout=30)
    return r.status_code == 204, r


def latest_run_url():
    try:
        r = requests.get(f"{API}/repos/{REPO}/actions/workflows/{WORKFLOW}/runs",
                         headers=HDRS, params={"per_page": 1}, timeout=30)
        runs = r.json().get("workflow_runs", [])
        return runs[0]["html_url"] if runs else None
    except Exception:
        return None


def list_runs():
    """Return run ids present on the data branch, newest first."""
    r = requests.get(f"{API}/repos/{REPO}/contents/runs",
                     headers=HDRS, params={"ref": DATA_BRANCH}, timeout=30)
    if r.status_code != 200:
        return []
    ids = [x["name"] for x in r.json() if x.get("type") == "dir"]
    return sorted(ids, reverse=True)


# --------------------------------- helpers --------------------------------- #
def vol(x):
    try:
        v = float(x)
        return f"{int(v):,}" if pd.notna(v) else "-"
    except Exception:
        return "-"


def read_csv_bytes(b):
    if not b:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        return pd.DataFrame()


# --------------------------------- render ---------------------------------- #
def render_results(run_id):
    pj = gh_get_file(f"runs/{run_id}/result.json", DATA_BRANCH)
    ps_bytes = gh_get_file(f"runs/{run_id}/page_summary.csv", DATA_BRANCH)
    km_bytes = gh_get_file(f"runs/{run_id}/keyword_mapping.csv", DATA_BRANCH)
    meta = (json.loads(pj).get("meta", {}) if pj else {})
    page_summary = read_csv_bytes(ps_bytes)
    keyword_mapping = read_csv_bytes(km_bytes)

    st.subheader("Results")
    c = st.columns(5)
    c[0].metric("Topics", meta.get("n_topics", "-"))
    c[1].metric("Pillars", meta.get("n_pillars", "-"))
    c[2].metric("Pages", meta.get("n_pages", "-"))
    c[3].metric("Keywords", meta.get("n_keywords", "-"))
    c[4].metric("SERP cost", f"${meta.get('serp_cost_usd', 0)}")

    bits = []
    if meta.get("intent_from_serp") is not None:
        bits.append(f"{meta['intent_from_serp']} intents from the SERP, "
                    f"{meta.get('intent_from_text', 0)} from text")
    if meta.get("page_intent_distribution"):
        dist = ", ".join(f"{k}: {v}" for k, v in meta["page_intent_distribution"].items())
        bits.append(f"page intent split, {dist}")
    if meta.get("timed_out"):
        bits.append(f"{meta['timed_out']} SERP lookups timed out")
    if bits:
        st.caption(". ".join(bits) + ".")

    if page_summary.empty:
        st.warning("No page summary found for this run.")
        return

    has_role = "Role" in page_summary.columns
    real = page_summary[page_summary["Role"] != "Ungrouped"] if has_role else page_summary

    st.markdown("### Topic > Pillar > Page")
    for topic, tg in sorted(real.groupby("Topic"),
                            key=lambda kv: -kv[1]["Volume"].sum(min_count=1)
                            if kv[1]["Volume"].notna().any() else 0):
        head = (f"{topic}  ·  {tg['Pillar'].nunique()} pillars · "
                f"{len(tg)} pages · {vol(tg['Volume'].sum(min_count=1))} vol")
        with st.expander(head):
            for pillar, pg in sorted(tg.groupby("Pillar"),
                                     key=lambda kv: -kv[1]["Volume"].sum(min_count=1)
                                     if kv[1]["Volume"].notna().any() else 0):
                st.markdown(f"**{pillar}**  ·  {len(pg)} pages · {vol(pg['Volume'].sum(min_count=1))} vol")
                extra = (["Merged from"] if "Merged from" in pg.columns
                         and pg["Merged from"].fillna("").str.len().gt(0).any() else [])
                if has_role:                       # hub page first, flagged
                    pg = pg.assign(_o=(pg["Role"] != "Pillar page").astype(int))
                    show = pg.sort_values(["_o", "Volume"], ascending=[True, False], na_position="last")[
                        ["Role", "Page", "Intent", "Keywords", "Volume"] + extra]
                else:
                    show = pg.sort_values("Volume", ascending=False, na_position="last")[
                        ["Page", "Intent", "Keywords", "Volume"] + extra]
                st.dataframe(show, hide_index=True, use_container_width=True)

    if "Page role" in keyword_mapping.columns:
        ung = keyword_mapping[keyword_mapping["Page role"] == "Ungrouped"]
        if len(ung):
            with st.expander(f"Ungrouped / for review  ·  {len(ung)} keywords"):
                st.caption("Low-relevance or noisy keywords held out of the pages. Review and keep or discard.")
                cols = [c for c in ["Keyword", "Volume", "Keyword intent"] if c in ung.columns]
                st.dataframe(ung[cols], hide_index=True, use_container_width=True, height=300)

    with st.expander("Full keyword mapping"):
        st.dataframe(keyword_mapping, hide_index=True, use_container_width=True, height=400)

    st.markdown("### Download")
    d1, d2 = st.columns(2)
    if km_bytes:
        d1.download_button("Keyword mapping (CSV)", km_bytes,
                           f"keyword_mapping_{run_id}.csv", "text/csv")
    if ps_bytes:
        d2.download_button("Page summary (CSV)", ps_bytes,
                           f"page_summary_{run_id}.csv", "text/csv")


# ------------------------------- wizard setup ------------------------------ #
STEPS = ["Upload", "Settings", "Run", "Results"]
ss = st.session_state
ss.setdefault("step", 1)
ss.setdefault("csv_bytes", None)
ss.setdefault("n_rows", 0)
ss.setdefault("location", "United Kingdom")
ss.setdefault("pillar_threshold", 0.45)
ss.setdefault("running", False)


def goto(step):
    ss.step = step
    st.rerun()


def show_stepper(current):
    html = ['<div class="ba-stepper">']
    for i, label in enumerate(STEPS, start=1):
        cls = "active" if i == current else ("done" if i < current else "todo")
        dot = "&#10003;" if i < current else str(i)
        html.append(f'<div class="ba-step {cls}"><span class="ba-dot">{dot}</span>{label}</div>')
        if i < len(STEPS):
            html.append(f'<div class="ba-conn {"done" if i < current else ""}"></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


show_stepper(ss.step)

# --------------------------------- step 1 ---------------------------------- #
if ss.step == 1:
    st.subheader("1. Upload your keyword list")
    with st.expander("How to use this tool", expanded=ss.csv_bytes is None):
        st.markdown(
            "This tool turns a flat keyword list into a ready-to-build site structure: a "
            "three-level **Topic > Pillar > Page** map, with the search intent of every page.\n\n"
            "**What to upload.** A CSV with one keyword per row. Add a search volume column if you "
            "have one and it will order the results and name each page after its biggest term, but "
            "it is optional. Keywords can come from anywhere: Ahrefs, Semrush, Search Console, "
            "Reddit, or your own brainstorm.\n\n"
            "**How to read the result.**\n"
            "- **Page** is the level you build. One URL should target the keywords grouped under it.\n"
            "- **Pillar** gathers closely related pages under a shared theme.\n"
            "- **Topic** is the broad editorial section a pillar sits in.\n"
            "- **Intent** (Informational, Commercial, Transactional, Navigational, Local or Mixed) "
            "is read from the pages actually ranking for each keyword, so it reflects what Google "
            "rewards rather than a guess.\n\n"
            "**Getting the best from it.** Run one product area or section at a time rather than a "
            "whole site at once, so the groups stay clean. If the result feels over- or "
            "under-split, change **Pillar tightness** in Settings and run again. Re-running on the "
            "same keywords is quick and nearly free, so it is worth iterating.\n\n"
            "**What you get back.** A map you can browse on screen, plus two downloads: a "
            "keyword-level mapping and a page-level summary you can hand to a writer or developer."
        )
    up = st.file_uploader("CSV with a keyword column (volume optional).", type=["csv"])
    if up is not None:
        try:
            preview = pd.read_csv(up)
            preview.columns = [str(c).strip() for c in preview.columns]
            low = {c.lower(): c for c in preview.columns}
            if not any(c in low for c in ["keyword", "keywords", "query", "term"]):
                st.error(f"No keyword column found. Columns seen: {list(preview.columns)}")
                ss.csv_bytes = None
            else:
                up.seek(0)
                ss.csv_bytes = up.read()
                ss.n_rows = len(preview)
                st.success(f"{ss.n_rows} keywords ready.")
                st.dataframe(preview.head(10), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            ss.csv_bytes = None

    runs = list_runs()
    if runs:
        with st.expander("Or open a previous run"):
            pick = st.selectbox("Previous runs", ["-"] + runs, label_visibility="collapsed")
            if pick != "-" and st.button("Open this run"):
                ss.run_id = pick
                ss.running = False
                ss.loaded = True
                goto(4)

    st.divider()
    _, nxt = st.columns([3, 1])
    if nxt.button("Next", type="primary", use_container_width=True,
                  disabled=ss.csv_bytes is None):
        goto(2)

# --------------------------------- step 2 ---------------------------------- #
elif ss.step == 2:
    st.subheader("2. Settings")
    ss.location = st.text_input("SERP location", value=ss.location).strip() or "United Kingdom"
    st.caption("The country whose Google results are read to judge each keyword's intent.")
    with st.expander("Advanced"):
        ss.pillar_threshold = st.slider(
            "Pillar tightness (cosine distance)",
            min_value=0.35, max_value=0.55, value=ss.pillar_threshold, step=0.01,
            help="Lower = tighter, more pillars. 0.45 is the validated default.")
    st.divider()
    back, nxt = st.columns(2)
    if back.button("Back", use_container_width=True):
        goto(1)
    if nxt.button("Next", type="primary", use_container_width=True):
        goto(3)

# --------------------------------- step 3 ---------------------------------- #
elif ss.step == 3:
    st.subheader("3. Run the mapping")
    st.markdown(f"Ready to map **{ss.n_rows} keywords** for **{ss.location}** "
                f"(pillar tightness {ss.pillar_threshold:.2f}).")
    st.caption("The run happens on a server, so you can close this tab and come back. The first "
               "run on new keywords takes a few minutes while it reads the SERPs; later runs on "
               "the same keywords are quick.")

    if not ss.running:
        back, run = st.columns(2)
        if back.button("Back", use_container_width=True):
            goto(2)
        if run.button("Run mapping", type="primary", use_container_width=True):
            run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            with st.spinner("Submitting the run..."):
                if not gh_ensure_branch(DATA_BRANCH, CODE_BRANCH):
                    st.error(f"Could not find or create the `{DATA_BRANCH}` branch. "
                             "Check the token has repo access.")
                    st.stop()
                ok, r = gh_put_file(f"runs/{run_id}/input.csv", ss.csv_bytes, DATA_BRANCH,
                                    f"run {run_id}: input")
                if not ok:
                    st.error(f"Could not upload the keyword list (HTTP {r.status_code}). "
                             "Check the token scopes (repo).")
                    st.stop()
                ok, r = gh_dispatch(run_id, ss.location, ss.pillar_threshold)
                if not ok:
                    st.error(f"Could not trigger the workflow (HTTP {r.status_code}). "
                             "Check the workflow is on the default branch and the token "
                             "has the workflow scope.")
                    st.stop()
            ss.run_id = run_id
            ss.start_ts = time.time()
            ss.running = True
            ss.loaded = False
            st.rerun()
    else:
        rid = ss.run_id
        raw = gh_get_file(f"runs/{rid}/status.json", DATA_BRANCH)
        status = {}
        if raw:
            try:
                status = json.loads(raw)
            except Exception:
                status = {}
        s = status.get("status", "queued")
        elapsed = int(time.time() - ss.get("start_ts", time.time()))
        if s == "done":
            ss.running = False
            ss.loaded = True
            goto(4)
        elif s == "error":
            ss.running = False
            st.error(f"The run failed: {status.get('message', 'see the run log')}.")
            url = latest_run_url()
            if url:
                st.markdown(f"[View the run log]({url})")
            if st.button("Back to settings"):
                goto(2)
        else:
            msg = {"queued": "Queued, the run is starting", "running": "Running"}.get(s, s)
            st.info(f"{msg}... {elapsed}s elapsed. You can safely close this tab and come back.")
            url = latest_run_url()
            if url:
                st.caption(f"[Watch the live log]({url})")
            time.sleep(POLL_SECONDS)
            st.rerun()

# --------------------------------- step 4 ---------------------------------- #
elif ss.step == 4:
    if ss.get("run_id"):
        render_results(ss.run_id)
    else:
        st.info("No run to show yet. Go back to step 1 to upload a keyword list.")
    st.divider()
    if st.button("Start a new run"):
        for k in ("run_id", "start_ts"):
            ss.pop(k, None)
        ss.step = 1
        ss.csv_bytes = None
        ss.n_rows = 0
        ss.running = False
        ss.loaded = False
        st.rerun()
