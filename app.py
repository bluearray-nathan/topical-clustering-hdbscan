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

st.set_page_config(page_title="Keyword to page mapping", page_icon="🔵", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

_LOGO = os.path.join(os.path.dirname(__file__), "assets", "logo-colour-converts.png")
if os.path.exists(_LOGO):
    st.logo(_LOGO, size="large", link="https://www.bluearray.co.uk")

st.title("Keyword to page mapping")

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

    st.markdown("### Topic > Pillar > Page")
    for topic, tg in sorted(page_summary.groupby("Topic"),
                            key=lambda kv: -kv[1]["Volume"].sum(min_count=1)
                            if kv[1]["Volume"].notna().any() else 0):
        head = (f"{topic}  ·  {tg['Pillar'].nunique()} pillars · "
                f"{len(tg)} pages · {vol(tg['Volume'].sum(min_count=1))} vol")
        with st.expander(head):
            for pillar, pg in sorted(tg.groupby("Pillar"),
                                     key=lambda kv: -kv[1]["Volume"].sum(min_count=1)
                                     if kv[1]["Volume"].notna().any() else 0):
                st.markdown(f"**{pillar}**  ·  {len(pg)} pages · {vol(pg['Volume'].sum(min_count=1))} vol")
                show = pg[["Page", "Intent", "Keywords", "Volume"]].sort_values(
                    "Volume", ascending=False, na_position="last")
                st.dataframe(show, hide_index=True, use_container_width=True)

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


# --------------------------------- sidebar --------------------------------- #
with st.sidebar:
    st.header("Run settings")
    location = st.text_input("SERP location", value="United Kingdom").strip() or "United Kingdom"
    with st.expander("Advanced"):
        pillar_threshold = st.slider(
            "Pillar tightness (cosine distance)",
            min_value=0.35, max_value=0.55, value=0.45, step=0.01,
            help="Lower = tighter, more pillars. 0.45 is the validated default.")
    st.divider()
    st.caption(f"Engine: GitHub Actions on `{REPO}`")
    runs = list_runs()
    if runs:
        st.caption("Load a previous run")
        pick = st.selectbox("Previous runs", ["-"] + runs, label_visibility="collapsed")
        if pick != "-" and st.button("Load", use_container_width=True):
            st.session_state.run_id = pick
            st.session_state.running = False
            st.session_state.loaded = True
            st.rerun()

with st.expander("What this does"):
    st.markdown(
        "Maps a keyword list to a **Topic > Pillar > Page** structure. Pillars are semantic "
        "themes, pages blend meaning with SERP overlap, and topics are broad editorial sections. "
        "Each page carries an intent read from its ranking SERPs.\n\n"
        "The engine runs on GitHub Actions, so you can close this tab while it works. "
        "The first run on a fresh keyword set fetches SERPs (a few minutes); after that the "
        "SERPs are cached, so re-runs are quick and cost only a little OpenAI."
    )

# ------------------------------- upload + run ------------------------------ #
st.subheader("1. Upload your keyword list")
file = st.file_uploader("CSV with a keyword column (volume optional).", type=["csv"])

if file is not None:
    try:
        preview = pd.read_csv(file)
        preview.columns = [str(c).strip() for c in preview.columns]
        low = {c.lower(): c for c in preview.columns}
        has_kw = any(c in low for c in ["keyword", "keywords", "query", "term"])
        if not has_kw:
            st.error(f"No keyword column found. Columns seen: {list(preview.columns)}")
        else:
            st.success(f"{len(preview)} rows ready.")
            file.seek(0)
            csv_bytes = file.read()
            st.subheader("2. Run the mapping")
            if st.button("Run mapping", type="primary"):
                run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                with st.spinner("Submitting the run..."):
                    if not gh_ensure_branch(DATA_BRANCH, CODE_BRANCH):
                        st.error(f"Could not find or create the `{DATA_BRANCH}` branch. "
                                 "Check the token has repo access.")
                        st.stop()
                    ok, r = gh_put_file(f"runs/{run_id}/input.csv", csv_bytes, DATA_BRANCH,
                                        f"run {run_id}: input")
                    if not ok:
                        st.error(f"Could not upload the keyword list (HTTP {r.status_code}). "
                                 "Check the token scopes (repo).")
                        st.stop()
                    ok, r = gh_dispatch(run_id, location, pillar_threshold)
                    if not ok:
                        st.error(f"Could not trigger the workflow (HTTP {r.status_code}). "
                                 "Check the workflow is on the default branch and the token "
                                 "has the workflow scope.")
                        st.stop()
                st.session_state.run_id = run_id
                st.session_state.start_ts = time.time()
                st.session_state.running = True
                st.session_state.loaded = False
                st.rerun()
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

# ------------------------------- poll / show ------------------------------- #
if st.session_state.get("running"):
    rid = st.session_state.run_id
    raw = gh_get_file(f"runs/{rid}/status.json", DATA_BRANCH)
    status = {}
    if raw:
        try:
            status = json.loads(raw)
        except Exception:
            status = {}
    s = status.get("status", "queued")
    elapsed = int(time.time() - st.session_state.get("start_ts", time.time()))

    st.divider()
    if s == "done":
        st.session_state.running = False
        st.success(f"Done in about {elapsed}s.")
        render_results(rid)
    elif s == "error":
        st.session_state.running = False
        st.error(f"The run failed: {status.get('message', 'see the Actions log')}.")
        url = latest_run_url()
        if url:
            st.markdown(f"[View the Actions log]({url})")
    else:
        msg = {"queued": "Queued, the Action is starting",
               "running": "Running"}.get(s, s)
        st.info(f"{msg}... {elapsed}s elapsed. You can safely close this tab and come back.")
        url = latest_run_url()
        if url:
            st.caption(f"[Watch the live log on GitHub]({url})")
        time.sleep(POLL_SECONDS)
        st.rerun()

elif st.session_state.get("loaded") and st.session_state.get("run_id"):
    st.divider()
    st.caption(f"Showing run {st.session_state.run_id}")
    render_results(st.session_state.run_id)
