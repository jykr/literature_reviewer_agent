# Paper Triage

[![Server status](https://img.shields.io/website?url=https%3A%2F%2Fliterature-reviewer-4055136070.us-central1.run.app%2Fhealth&up_message=online&down_message=offline&label=server)](https://literature-reviewer-4055136070.us-east1.run.app/reviewer)

Literature review is often a daunting task. Get some help from agents. 

* Point it at your field of interest (a CV, a seed paper, or a few keywords) and 
* Returns a single ranked, structured review of the last ~12 months of papers in that field. It works for any field — the pipeline adapts to whatever you point it at. The public server generates reviews on your own API key (never stored).

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

- **Interests + Model:** 
  - CV/papers/keywords
  - Model API (Anthropic by default; Gemini, OpenAI, or **OpenRouter sign-in** via OAuth PKCE, so no key needs pasting), and 
  - **Regenerate review** with live stage progress. Keys are used per request and never stored.
- **Paperpile (optional, beta):** with a Paperpile API key 
  - the review flags papers already in your library, 
  - reads the PDFs you saved when a paper has no open full text, and 
  - can include a label's papers (up to its 10 most recently added). 
  - Papers can be added from each card (with the agent's notes) and removed again if they were added from the page. 
  - **Check library** and **Re-read abstracts** update the current review without regenerating it.

## Pipeline

`intake` (paper links in the message) → `scope` → five parallel `researcher`s over pooled sources (OpenAlex, Europe PMC, PubMed, arXiv, bioRxiv/medRxiv) → `added_papers` → `rank` (batched; author-first credibility, coverage-first) → `deepen` (selected papers re-read from open full text: arXiv, PMC, Unpaywall) → `library` (Paperpile flags) → render.

### **MCP server for paper source collection**
  The same pipeline and output, with paper search served by a local MCP server that pools five sources:
  | Source | What it adds |
  |---|---|
  | OpenAlex | broadest coverage, citation counts + FWCI |
  | Europe PMC | PubMed/PMC journal articles with abstracts |
  | PubMed (E-utilities) | MEDLINE metadata, PubMed query syntax |
  | arXiv | cs / q-bio / stat / physics preprints |
  | bioRxiv / medRxiv | keyword search via Europe PMC's preprint index; DOI lookups (incl. "published as") via api.biorxiv.org |
  - [Source](packages/paper-sources-mcp/)
  - **Use the MCP server elsewhere:** the pooled search runs as a standalone MCP server (`uv run paper-sources-mcp`; stdio by default, or `--transport streamable-http` to serve a URL), so any MCP client can use it. **[docs/paper-sources-mcp.md](docs/paper-sources-mcp.md)** has its tools, optional keys, and setup for Claude Code and Desktop, Cursor, Windsurf, VS Code, Gemini CLI, Codex CLI, and the MCP Python SDK, ADK and LangChain.

- **Focus lens:** set `LC_REVIEW_FOCUS` to steer scope, search and ranking (e.g. frontier ML for genomics); unset, the app stays field-general.
- [Run your own server](run_your_own_server.md)
<details>
<summary>Details</summary>

- **Model:** `LC_REVIEW_MODEL` / `--model`, an `init_chat_model` id (default `anthropic:claude-sonnet-5-5`).
- **Focus:** `LC_REVIEW_FOCUS` / `--focus` sets a standing lens applied to scope, research and rank. The default is frontier ML for genomics/biology (new architectures, bio foundation models, scaling, rigorous benchmarks). `--no-focus` makes it field-general like the ADK app.
- **Pipeline:** scope → `Send` fan-out of one `create_agent` researcher per cluster over the MCP tools → structured-output ranker → the same `app/render.py` HTML.
- **Credibility weighting:** author credibility is the primary signal (senior-author h-index first, then first-author h-index, citations, FWCI); peer-reviewed venue is a secondary tie-breaker over arXiv/bioRxiv preprints, roughly half the weight. The ranker sees evidence, not names: venue type, arXiv `journal_ref`/"accepted at" notes, first/senior-author h-index (OpenAlex), citations and FWCI. Venue and preprint status are copied from the tool output, so the model can't upgrade a preprint.
- **Full text, for the final selection only:** after ranking (which stays abstract-based), the selected papers are re-analyzed from full text: arXiv HTML/PDF, PMC (NCBI and Europe PMC, incl. author manuscripts), and legal open copies found by Unpaywall (`UNPAYWALL_EMAIL`); bioRxiv optionally via its official requester-pays S3 buckets (`LC_BIORXIV_S3=1` + AWS credentials). Result bullets quoting numbers absent from the text are dropped; papers without open full text are labeled "Abstract only".
- **Paperpile (beta MCP, optional):** with `PAPERPILE_API_KEY` in `.env`, reviewed papers already in your library get an "In your Paperpile" badge. `--paperpile-profile [FOLDER]` adds up to 100 titles from your library (or one folder/label) to the request as an interest profile. `--save-to-paperpile [LABEL]` saves the reviewed papers under a label (default `review-lc`); existing copies only gain the label, and nothing is deleted or rewritten. `--include-paperpile "to read"` adds every paper in that folder/label to the review, analyzed like manually added papers and shown in purple with a "Paperpile: to read" badge (items without a DOI, arXiv id or paper URL, like tool web pages, are skipped). `--added-since 6m` (or a date) keeps only papers added to the library since then; Paperpile doesn't record when a label was applied. `--only-added` skips the literature search and reviews just the added papers. `--no-paperpile` skips Paperpile entirely. Check the connection with `python -m lc_app.paperpile`.
- **LangGraph dev server:** `uv run --extra langchain langgraph dev` serves the graph (`langgraph.json` → `lc_app/studio.py`) at http://127.0.0.1:2024, with API docs at `/docs` and LangGraph Studio for step-by-step runs. Input is `{"request": "..."}` plus optional `manual_ids`, `from_date`, `refresh`, `skip_research`, `add_to_previous`; the page is written to `.lc_cache/studio/` and the path is in the final `message`.

</details>


## Agent workflow
### agents-cli
<img src="image/README/agents_cli_graph.png" alt="agents-cli graph" width="720">

(Generated with `agents-cli playground`)

### LangChain (`lc_app/`)
![langgraph](image/README/review-lc-graph.png)

(Generated with `graph.get_graph().draw_mermaid_png()`)

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