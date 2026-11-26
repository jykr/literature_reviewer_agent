To run your **own copy**, the server needs **Google Cloud credentials** — it calls `google.auth.default()` and initializes Cloud Logging at startup ([app/fast_api_app.py](../app/fast_api_app.py)). Copy [`.env.example`](../.env.example) to `.env`, then:

1. `gcloud auth application-default login`
2. set `GOOGLE_CLOUD_PROJECT` (and `GOOGLE_CLOUD_LOCATION=global`) in `.env`

No server-side model key is required for the public review endpoints. *(Only if you enable the ADK agent API — `ENABLE_AGENT_API=true` — do you also need a model backend: Vertex AI by default, or `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` + `REVIEW_MODEL=anthropic/claude-sonnet-5`.)* Optional: `OPENALEX_MAILTO` for the OpenAlex polite pool.

### With Docker (recommended — reproducible)

The repo ships a [`Dockerfile`](../Dockerfile) pinning Python 3.12 + `uv` on port 8080 — no local Python/uv install needed. Docker packages the code and dependencies, but not your Google credentials, so mount them in:

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

## With LangChain
```bash
uv sync --extra langchain
uv run --extra langchain review-lc "recent single-cell foundation model papers" --open
uv run --extra langchain review-lc --cv https://example.edu/~me --topics "a; b" --model openai:gpt-5
```

- **LangGraph dev server:** `uv run --extra langchain langgraph dev` serves the graph (`langgraph.json` → `lc_app/studio.py`) at http://127.0.0.1:2024, with API docs at `/docs` and LangGraph Studio for step-by-step runs. Input is `{"request": "..."}` plus optional `manual_ids`, `from_date`, `refresh`, `skip_research`, `add_to_previous`; the page is written to `.lc_cache/studio/` and the path is in the final `message`.