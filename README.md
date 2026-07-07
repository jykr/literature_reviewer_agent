# Literature Reviewer Agent

[![Server status](https://img.shields.io/website?url=https%3A%2F%2Fliterature-reviewer-4055136070.us-central1.run.app%2Fhealth&up_message=online&down_message=offline&label=server)](https://literature-reviewer-4055136070.us-central1.run.app/reviewer)

Literature review is often a daunting task. This concierge agent runs as a hosted web app on **Google Cloud Run** — point it at your field of interest (a CV, a seed paper, or a few keywords) and it returns a single ranked, structured review of the last ~12 months of computational-biology papers. The public server generates reviews on **your own API key**, used per-request and never stored.

**Live server:** https://literature-reviewer-4055136070.us-central1.run.app/reviewer

## Example output

The server renders your review as a single self-contained page — an interactive queue you can tag, filter, and export notes from (state persists in `localStorage`). It's served same-origin at `/reviewer`; because it makes no external calls once loaded, you can also save the page ([`literature-reviewer.html`](literature-reviewer.html)) and reopen it offline.

**Overall layout** — ranked queue, sidebar, interests panel, and per-paper cards:

![Literature Reviewer — overall layout](docs/example-overview.png)

**Paper card (detail)** — Aim → Main approach → Evaluation → Results (with quantitative task/data/metric/result bullets) → Limitation:

![Literature Reviewer — paper card detail](docs/example-card.png)

See [`compbio-paper-review-SPEC.md`](compbio-paper-review-SPEC.md) for the complete specification of this output.

## Run or deploy your own server

**Using the hosted app needs no setup** — open the live URL and paste your own model API key (Gemini, Anthropic, or OpenAI) in the interests panel; it is used per-request and never stored. Reviews are always generated on the caller's key, so the server holds no model credentials of its own.

To run your **own copy**, the server needs **Google Cloud credentials** — it calls `google.auth.default()` and initializes Cloud Logging at startup ([app/fast_api_app.py](app/fast_api_app.py)). Copy [`.env.example`](.env.example) to `.env`, then:

1. `gcloud auth application-default login`
2. set `GOOGLE_CLOUD_PROJECT` (and `GOOGLE_CLOUD_LOCATION=global`) in `.env`

No server-side model key is required for the public review endpoints. *(Only if you enable the ADK agent API — `ENABLE_AGENT_API=true` — do you also need a model backend: Vertex AI by default, or `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` + `REVIEW_MODEL=anthropic/claude-sonnet-5`.)* Optional: `OPENALEX_MAILTO` for the OpenAlex polite pool.

### With Docker (recommended — reproducible)

The repo ships a [`Dockerfile`](Dockerfile) pinning Python 3.12 + `uv` on port 8080 — no local Python/uv install needed. Docker packages the code and dependencies, but not your Google credentials, so mount them in:

```bash
docker build -t literature-reviewer .
docker run --env-file .env -p 8080:8080 \
  -v ~/.config/gcloud/application_default_credentials.json:/adc.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/adc.json \
  literature-reviewer
# → open http://localhost:8080/reviewer
```

### Without Docker

```bash
uv sync                                                          # install deps (Python 3.11–3.13)
uv run uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8080
```

### Deploy to Cloud Run

On Cloud Run the credentials come from the service account automatically. `agents-cli deploy` (from the [Agents CLI](https://github.com/google/agents-cli): `uv tool install google-agents-cli`) builds the same container and ships it to **Google Cloud Run**. Infrastructure is Terraform under [`deployment/terraform/`](deployment/terraform/). See [`compbio-paper-review-SPEC.md`](compbio-paper-review-SPEC.md) for the full specification of the generated review app.

## Model-agnostic by design

Model choice is a **config knob** (`REVIEW_MODEL`), never chat input. A bare Gemini id runs ADK-native; a LiteLLM `provider/model` id (e.g. `anthropic/claude-sonnet-5`, `openai/gpt-4o`) routes through LiteLLM. The hosted BYO-key server supports Gemini, Anthropic, and OpenAI behind one `call_llm`, with loose/repairing JSON parsing so a truncated or fenced response still yields a valid review. Because search runs on keyless tools rather than provider-specific grounding, swapping models changes nothing downstream.

## Agent workflow

![1783389818754](image/README/1783389818754.png)

Generated with `agents-cli playground`. `root-agent` sequentially runs `scope_agent`, `research_fanout`, and `rank_agent`.
