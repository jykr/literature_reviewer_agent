# Paper Triage

[![Server status](https://img.shields.io/website?url=https%3A%2F%2Fliterature-reviewer-4055136070.us-central1.run.app%2Fhealth&up_message=online&down_message=offline&label=server)](https://literature-reviewer-4055136070.us-east1.run.app/reviewer)

Literature review is often a daunting task. Get some help from agents. 

— Point it at your field of interest (a CV, a seed paper, or a few keywords) and 
- Returns a single ranked, structured review of the last ~12 months of papers in that field. It works for any field — the pipeline adapts to whatever you point it at. The public server generates reviews on your own API key (never stored).

**Live server:** https://literature-reviewer-4055136070.us-east1.run.app/reviewer

## Example output

The server renders your review as a single self-contained page — an interactive queue you can tag, filter, and export notes from (state persists in `localStorage`). It's served same-origin at `/reviewer`; because it makes no external calls once loaded, you can also save the page ([`literature-reviewer.html`](literature-reviewer.html)) and reopen it offline.

**Overall layout** — ranked queue, sidebar, interests panel, and per-paper cards:

![Literature Reviewer — overall layout](docs/example-overview.png)

**Paper card (detail)** — Aim → Main approach → Evaluation → Results (with quantitative task/data/metric/result bullets) → Limitation:

<img src="docs/example-card.png" alt="Paper Triage — paper card detail" width="560">

See [`paper-review-SPEC.md`](paper-review-SPEC.md) for the complete specification of this output.

## Run or deploy your own server

The page (titled **Paper Triage**) has an editable **researcher profile** under the title, used verbatim for the next review, and a setup panel in two blocks:

- **Interests + Model:** CV/papers/keywords, then the model API (Anthropic by default; Gemini, OpenAI, or **OpenRouter sign-in** via OAuth PKCE, so no key needs pasting), and **Regenerate review** with live stage progress. Keys are used per request and never stored.
- **Paperpile (optional, beta):** with a Paperpile API key the review flags papers already in your library, reads the PDFs you saved when a paper has no open full text, and can include a label's papers (up to its 10 most recently added). Papers can be added from each card (with the agent's notes) and removed again if they were added from the page. **Check library** and **Re-read abstracts** update the current review without regenerating it.

**Pipeline** (the ADK agent in [`app/agent.py`](app/agent.py); the hosted `/review` endpoints run the same agent on the caller's key):

`intake` (paper links in the message) → `scope` → five parallel `researcher`s over pooled sources (OpenAlex, Europe PMC, PubMed, arXiv, bioRxiv/medRxiv) → `added_papers` → `rank` (batched; author-first credibility, coverage-first) → `deepen` (selected papers re-read from open full text: arXiv, PMC, Unpaywall) → `library` (Paperpile flags) → render.

- **Integrity in code:** a paper is kept only if a search tool returned it, and its URL/DOI/venue/citations come from the tool record; full-text result bullets quoting numbers absent from the text are dropped.
- **Contribution kinds**, not scores: *Application / resource*, *Adaptation* (existing components modified or combined) and *Algorithm*, highlighted when they apply (one or more). Main approach reads Key question → Data → Model → **Methodological novelty** (the concrete mechanism added).
- **Writing style:** impersonal and evidence-focused (claims stated as reported, no promotional language), with per-field length limits.
- **Focus lens:** set `LC_REVIEW_FOCUS` to steer scope, search and ranking (e.g. frontier ML for genomics); unset, the app stays field-general.
- The review logic (verification, credibility rules, grounding) is shared with the LangChain variant in [`lc_app/core.py`](lc_app/core.py).

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

### Agent workflow

![1783389818754](image/README/agents_cli_graph.png)

(Generated with `agents-cli playground`)

## LangChain variant (`lc_app/`)

The same pipeline and output, built on LangGraph instead of ADK, with paper search served by a local MCP server that pools five sources:

| Source | What it adds |
|---|---|
| OpenAlex | broadest coverage, citation counts + FWCI |
| Europe PMC | PubMed/PMC journal articles with abstracts |
| PubMed (E-utilities) | MEDLINE metadata, PubMed query syntax |
| arXiv | cs / q-bio / stat / physics preprints |
| bioRxiv / medRxiv | keyword search via Europe PMC's preprint index; DOI lookups (incl. "published as") via api.biorxiv.org |

`search_papers` queries them all in parallel and merges hits that share a DOI or title, so a preprint and its journal version count as one paper.

```bash
uv sync --extra langchain
uv run --extra langchain review-lc "recent single-cell foundation model papers" --open
uv run --extra langchain review-lc --cv https://example.edu/~me --topics "a; b" --model openai:gpt-5
```

- **Model:** `LC_REVIEW_MODEL` / `--model`, an `init_chat_model` id (default `anthropic:claude-sonnet-5-5`).
- **Focus:** `LC_REVIEW_FOCUS` / `--focus` sets a standing lens applied to scope, research and rank. The default is frontier ML for genomics/biology (new architectures, bio foundation models, scaling, rigorous benchmarks). `--no-focus` makes it field-general like the ADK app.
- **Pipeline:** scope → `Send` fan-out of one `create_agent` researcher per cluster over the MCP tools → structured-output ranker → the same `app/render.py` HTML.
- **Credibility weighting:** author credibility is the primary signal (senior-author h-index first, then first-author h-index, citations, FWCI); peer-reviewed venue is a secondary tie-breaker over arXiv/bioRxiv preprints, roughly half the weight. The ranker sees evidence, not names: venue type, arXiv `journal_ref`/"accepted at" notes, first/senior-author h-index (OpenAlex), citations and FWCI. Venue and preprint status are copied from the tool output, so the model can't upgrade a preprint.
- **Full text, for the final selection only:** after ranking (which stays abstract-based), the selected papers are re-analyzed from full text: arXiv HTML/PDF, PMC (NCBI and Europe PMC, incl. author manuscripts), and legal open copies found by Unpaywall (`UNPAYWALL_EMAIL`); bioRxiv optionally via its official requester-pays S3 buckets (`LC_BIORXIV_S3=1` + AWS credentials). Result bullets quoting numbers absent from the text are dropped; papers without open full text are labeled "Abstract only".
- **Paperpile (beta MCP, optional):** with `PAPERPILE_API_KEY` in `.env`, reviewed papers already in your library get an "In your Paperpile" badge. `--paperpile-profile [FOLDER]` adds up to 100 titles from your library (or one folder/label) to the request as an interest profile. `--save-to-paperpile [LABEL]` saves the reviewed papers under a label (default `review-lc`); existing copies only gain the label, and nothing is deleted or rewritten. `--include-paperpile "to read"` adds every paper in that folder/label to the review, analyzed like manually added papers and shown in purple with a "Paperpile: to read" badge (items without a DOI, arXiv id or paper URL, like tool web pages, are skipped). `--added-since 6m` (or a date) keeps only papers added to the library since then; Paperpile doesn't record when a label was applied. `--only-added` skips the literature search and reviews just the added papers. `--no-paperpile` skips Paperpile entirely. Check the connection with `python -m lc_app.paperpile`.
- **Integrity in code:** a researcher's or ranker's paper is kept only if an MCP tool actually returned it during the run, and its URL/DOI/citation count are copied from the tool output.
- **LangGraph dev server:** `uv run --extra langchain langgraph dev` serves the graph (`langgraph.json` → `lc_app/studio.py`) at http://127.0.0.1:2024, with API docs at `/docs` and LangGraph Studio for step-by-step runs. Input is `{"request": "..."}` plus optional `manual_ids`, `from_date`, `refresh`, `skip_research`, `add_to_previous`; the page is written to `.lc_cache/studio/` and the path is in the final `message`.
- **Use the MCP server elsewhere:** `claude mcp add paper-sources -- uv run --directory "$PWD" --extra langchain paper-sources-mcp`. Optional env: `OPENALEX_API_KEY` (recommended: keyless OpenAlex requests share a ~100-search/day per-IP budget), `NCBI_API_KEY`.

### Agent workflow

![langgraph](image/README/review-lc-graph.png)

Generated with `graph.get_graph().draw_mermaid_png()`. 
- `scope`: Determine the scope of the search
- `manual`: Include manually added papers by user
- `research`: Search for abstract from the databases
- `rank`: Rank the papers
- `deepen`: Fetch full text and analyze
- `render`: Render the report

## Model-agnostic
The hosted BYO-key server supports Gemini, Anthropic, and OpenAI behind one `call_llm`. Users can also sign up for OpenRouter to use their OAuth based model access.

## Contribution
Contributions are welcome through GitHub Issue / Discussion / Pull Request.