# paper-sources-mcp

<!-- mcp-name: io.github.jykr/paper-sources -->

An [MCP](https://modelcontextprotocol.io) server that pools scholarly search across **OpenAlex, Europe PMC, PubMed, arXiv and bioRxiv/medRxiv**, and returns open full-text excerpts when they exist. It works without API keys and with any MCP client.

```bash
uvx paper-sources-mcp            # stdio, for clients that launch the server
uvx paper-sources-mcp --transport streamable-http --port 8765   # http://127.0.0.1:8765/mcp
```

## Tools

| Tool | What it does |
|---|---|
| `search_papers(query, from_date, max_results=12, sources=None)` | Searches all five sources at once and returns one deduplicated list (a preprint and its published version merge). Start here. |
| `search_openalex` / `search_pubmed` / `search_europepmc` / `search_biorxiv` / `search_arxiv` | One source each |
| `get_paper_details(identifier)` | Full metadata and abstract for a DOI, PMID, arXiv id or paper URL (bioRxiv DOIs also report `published_doi`) |
| `get_paper_fulltext(identifier)` | Open full-text excerpts (results, methods, discussion, tables, code/data links) from arXiv, PMC/Europe PMC or Unpaywall |

Paper records include title, authors, venue and venue type, date, DOI/URL, abstract, citation count, FWCI, first- and senior-author h-index (OpenAlex), preprint status, and the sources that returned each paper.

## Configure a client

Most clients take the same entry; this is Claude Desktop's `claude_desktop_config.json`, and Cursor, Windsurf and Gemini CLI use the same `mcpServers` block:

```json
{
  "mcpServers": {
    "paper-sources": {
      "command": "uvx",
      "args": ["paper-sources-mcp"],
      "env": { "NCBI_API_KEY": "optional", "UNPAYWALL_EMAIL": "you@example.org" }
    }
  }
}
```

- Claude Code: `claude mcp add paper-sources -- uvx paper-sources-mcp`
- VS Code (`.vscode/mcp.json`): `{"servers": {"paper-sources": {"type": "stdio", "command": "uvx", "args": ["paper-sources-mcp"]}}}`
- Codex CLI (`~/.codex/config.toml`): `[mcp_servers.paper-sources]` with `command = "uvx"` and `args = ["paper-sources-mcp"]`
- Without uv: `pip install paper-sources-mcp`, then use `paper-sources-mcp` as the command.

More clients, HTTP mode and agent-code examples (MCP SDK, Google ADK, LangChain): [full guide](https://github.com/jykr/paper_triage/blob/main/docs/paper-sources-mcp.md).

## Optional environment variables

| Variable | Effect |
|---|---|
| `OPENALEX_API_KEY` | Your own OpenAlex daily budget (keyless requests share a small per-IP budget) |
| `OPENALEX_MAILTO` | Contact email for OpenAlex's polite pool |
| `NCBI_API_KEY` | PubMed/PMC at 10 requests/s instead of 3 |
| `UNPAYWALL_EMAIL` | Enables legal open-access copies via Unpaywall in `get_paper_fulltext` |
| `LC_BIORXIV_S3=1` + AWS credentials | Experimental bioRxiv full text from its requester-pays S3 buckets; install with `paper-sources-mcp[s3]` |

Pass them through your client's `env` setting, or use `--env-file path/to/.env`. Variables that are already set take precedence over the file.

HTTP modes bind to `127.0.0.1` by default. The server has no authentication, so bind it publicly only behind your own access control.

Part of [Paper Triage](https://github.com/jykr/paper_triage), a literature-review agent.

MIT licensed. See [LICENSE](https://github.com/jykr/paper_triage/blob/main/packages/paper-sources-mcp/LICENSE).
