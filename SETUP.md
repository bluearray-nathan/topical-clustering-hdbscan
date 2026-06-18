# Setup: Keyword to Page Mapping (free hosting)

Two pieces, both free:

- **Front-end**: a Streamlit app (Community Cloud) where you upload a keyword CSV and view results.
- **Engine**: a GitHub Action that runs `pipeline.py` server-side, so a run survives the browser closing and has no practical timeout.

The front-end only holds a GitHub token. The OpenAI and DataForSEO credentials live in GitHub Actions secrets, so they never touch the front-end.

---

## 1. Rotate the keys first

The OpenAI key and DataForSEO password used during development were shared in chat, so treat them as compromised and rotate both before wiring them in:

- **OpenAI**: platform.openai.com, API keys, revoke the old key and create a new one.
- **DataForSEO**: dashboard, change the API password.

## 2. Add the engine's credentials as GitHub Actions secrets

Repo, Settings, Secrets and variables, Actions, New repository secret. Add three:

| Name | Value |
| --- | --- |
| `OPENAI_API_KEY` | your new OpenAI key |
| `DATAFORSEO_LOGIN` | nathan@bluearray.co.uk |
| `DATAFORSEO_PASSWORD` | your new DataForSEO password |

### Let the Action write its results back

The engine commits results to the `runs` branch using the built-in Actions token, so that token needs write access. Repo, Settings, Actions, General, Workflow permissions: select **Read and write permissions**, then Save. Without this, a run fails when it tries to push the results.

## 3. Put the code on the default branch (done)

`clustering-lab` has been merged into `main` and pushed, so the workflow is already on the default branch. The `runs` branch has also been created with the SERP cache seeded, so your first run on the 591-keyword UK set is free. Nothing to do here.

## 4. Create a GitHub token for the front-end

The app needs a token to upload the CSV and trigger the workflow.

- GitHub, Settings, Developer settings, Personal access tokens.
- **Classic token** with scopes: `repo` and `workflow`. Copy it (starts `ghp_`).
- Fine-grained alternative: this repo only, with Contents read/write, Actions read/write, Metadata read.

## 5. Deploy the front-end to Streamlit Community Cloud

- share.streamlit.io, New app, pick this repo, branch `main`, main file `app.py`.
- Advanced settings, Secrets, paste:

```
[github]
token = "ghp_your_token"
repo = "bluearray-nathan/topical-clustering-hdbscan"
code_branch = "main"
data_branch = "runs"
workflow = "run.yml"
```

- Deploy.

---

## How a run works

1. You upload a keyword CSV and click **Run mapping**.
2. The app commits the CSV to the `runs` branch and triggers the Action.
3. The Action runs the engine (embed, cluster into pillars, group pillars into topics, read SERPs, classify intent, split pages) and commits the results back to `runs`.
4. The app polls and shows the **Topic > Pillar > Page** result when it is ready. You can close the tab while it works.

**Cost and speed.** The first run on a fresh keyword set fetches SERPs, a few minutes and roughly $0.0006 per keyword, plus a little OpenAI. SERPs are cached on the `runs` branch, so later runs on the same keywords are quick and cost only the OpenAI labelling.

## Notes

- The `runs` branch holds the inputs, the results, and the shared SERP cache. It is kept separate from the app code so a run never triggers a redeploy of the front-end.
- Runs are serialised (one at a time per data branch) so the shared cache stays consistent.
- If a run fails, the app shows a link to the Actions log.
