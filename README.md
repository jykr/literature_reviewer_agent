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


## Pipeline

| # | Step | What it does |
|---|---|---|
| 1 | `intake` | Pulls paper links, DOIs, arXiv ids and PMIDs out of the request. Those papers are reviewed as added papers. |
| 2 | `scope` | Reads the CV or lab page and the topics, then writes a researcher profile (or uses the one you typed) and up to five topic groups with search keywords. |
| 3 | `researcher` ×5, in parallel | One per topic group. Each searches OpenAlex, Europe PMC, PubMed, arXiv and bioRxiv/medRxiv through the pooled tools. A paper is kept only if a tool returned it, and its link, venue, citations and author h-index come from the tool record. |
| 4 | `added_papers` | Looks up the added papers: pasted links, and optionally the 10 most recently added papers under a Paperpile label. |
| 5 | `rank` | Orders papers by how many interests they cover, then by author credibility (weighted above venue). Added papers are ranked in batches of 10, and the ranker's output is re-checked against the tool records. |
| 6 | `deepen` | Re-reads the selected papers in full text (arXiv, PMC, Unpaywall, then PDFs saved in Paperpile) and rewrites their analysis. Result bullets that quote numbers not found in the text are dropped. |
| 7 | `library` | With a Paperpile key, flags papers already in your library. |
| 8 | `render` | Builds the self-contained Paper Triage page. |


### **MCP server for paper source collection**
  The same pipeline and output, with paper search served by a local MCP server that pools five sources:
  | Source | What it adds |
  |---|---|
  | OpenAlex | broadest coverage, citation counts + FWCI |
  | Europe PMC | PubMed/PMC journal articles with abstracts |
  | PubMed (E-utilities) | MEDLINE metadata, PubMed query syntax |
  | arXiv | cs / q-bio / stat / physics preprints |
  | bioRxiv / medRxiv | keyword search via Europe PMC's preprint index; DOI lookups (incl. "published as") via api.biorxiv.org |

  - **Use the MCP server elsewhere:** the pooled search runs as a standalone MCP server (`uv run paper-sources-mcp`; stdio by default, or `--transport streamable-http` to serve a URL), so any MCP client can use it. **[docs/paper-sources-mcp.md](docs/paper-sources-mcp.md)** has its tools, optional keys, and setup for Claude Code and Desktop, Cursor, Windsurf, VS Code, Gemini CLI, Codex CLI, and the MCP Python SDK, ADK and LangChain.

- **Focus lens:** set `LC_REVIEW_FOCUS` to steer scope, search and ranking (e.g. frontier ML for genomics); unset, the app stays field-general.




### Workflow visualization

<table>
  <tr>
    <th>agents-cli (ADK)</th>
    <th>LangChain (<code>lc_app/</code>)</th>
  </tr>
  <tr>
    <td align="center" valign="middle"><img src="image/README/agents_cli_graph.png" alt="agents-cli workflow graph" width="480"></td>
    <td align="center" valign="middle"><img src="image/README/review-lc-graph.png" alt="LangGraph workflow graph" width="230"></td>
  </tr>
  <tr>
    <td>Generated with <code>agents-cli playground</code></td>
    <td>Generated with <code>graph.get_graph().draw_mermaid_png()</code>.<br><code>manual</code>: papers added by the user</td>
  </tr>
</table>

### Deployment
The web app is deployed with `agents-cli deploy` on Google Cloud Platform. You can also [run your own server](run_your_own_server.md).


## Model-agnostic
The hosted BYO-key server supports Gemini, Anthropic, and OpenAI behind one `call_llm`. Users can also sign up for OpenRouter to use their OAuth based model access.

## Contribution
Contributions are welcome through GitHub Issue / Discussion / Pull Request.