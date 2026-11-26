# paper-sources MCP server

A standalone [Model Context Protocol](https://modelcontextprotocol.io) server that pools scholarly search across five open sources. Any MCP client can use it: desktop chat apps, coding agents, IDEs, or your own agent code. It needs no API keys to work.

The Paper Triage apps use the same code: the ADK app calls it in-process, and the LangChain app launches this server over stdio.

## Tools

| Tool | What it does |
|---|---|
| `search_papers(query, from_date, max_results=12, sources=None)` | Searches all sources at once and returns one deduplicated list. A preprint and its published version merge into one record. Start here. |
| `search_openalex` | OpenAlex only: broadest coverage, plus citation counts and FWCI |
| `search_pubmed` | PubMed only (NCBI E-utilities); accepts PubMed query syntax |
| `search_europepmc` | Europe PMC journal articles with abstracts (no preprints) |
| `search_biorxiv` | bioRxiv and medRxiv preprints, by keyword |
| `search_arxiv` | arXiv preprints (cs, q-bio, stat, physics, math); rate-limited to 1 request every 3 s |
| `get_paper_details(identifier)` | Full metadata and abstract for a DOI, PMID, arXiv id or paper URL. For bioRxiv DOIs it also reports `published_doi`. |
| `get_paper_fulltext(identifier)` | Open full-text excerpts (results, methods, discussion, tables, code/data links) from arXiv, PMC/Europe PMC or Unpaywall. Paywalled papers return `status: "unavailable"`. |

Each paper record includes title, authors, venue and venue type, date, DOI/URL, abstract, `cited_by_count`, `fwci`, first- and senior-author h-index (OpenAlex), `is_preprint`, and the `sources` that returned it.

## Install

You need Python 3.11–3.13 and this repository. With [uv](https://docs.astral.sh/uv/), there's no separate install step: `uv run` creates the environment on first use.

```bash
git clone https://github.com/jykr/paper_triage.git
cd paper_triage
uv run paper-sources-mcp --help
```

Without uv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
paper-sources-mcp --help          # or: python -m paper_sources_mcp.server --help
```

In the client examples below, replace `/path/to/paper_triage` with your clone's absolute path. If you installed with pip, use the venv's executable instead: set `command` to `/path/to/paper_triage/.venv/bin/paper-sources-mcp` with no `args`.

## Optional configuration

All keys are optional. They raise rate limits or enable extra full-text sources.

| Variable | Effect |
|---|---|
| `OPENALEX_API_KEY` | Your own OpenAlex daily budget (keyless requests share a small per-IP budget) |
| `OPENALEX_MAILTO` | Contact email for OpenAlex's polite pool |
| `NCBI_API_KEY` | PubMed/PMC at 10 requests/s instead of 3 |
| `UNPAYWALL_EMAIL` | Enables legal open-access copies via Unpaywall in `get_paper_fulltext` (Unpaywall requires a contact email) |
| `LC_BIORXIV_S3=1` + AWS credentials | Experimental: bioRxiv full text from its requester-pays S3 buckets (billed to your AWS account) |

Pass them through the client's `env` setting, as in the examples below, or point the server at a file with `--env-file /path/to/.env`. Variables already set take precedence over the file. The server never sends these keys anywhere except the service each one belongs to.

## Run modes

| Mode | Command | Use it when |
|---|---|---|
| stdio (default) | `paper-sources-mcp` | The client launches the server itself. Most desktop apps, IDEs and CLIs work this way. |
| Streamable HTTP | `paper-sources-mcp --transport streamable-http --port 8765` | The client connects to a URL (`http://127.0.0.1:8765/mcp`), or several clients share one server. |
| SSE (legacy) | `paper-sources-mcp --transport sse --port 8765` | Older clients that only support SSE (`http://127.0.0.1:8765/sse`) |

HTTP modes bind to `127.0.0.1` by default. The server has **no authentication**, so bind it to a public address (`--host 0.0.0.0`) only behind your own authentication or a private network.

## Connect a client

Every stdio client needs the same three things: the **command** `uv`, the **args** `run --directory /path/to/paper_triage paper-sources-mcp`, and optionally the **env** keys. Only the config file format differs.

### Claude Code

```bash
claude mcp add paper-sources -- uv run --directory /path/to/paper_triage paper-sources-mcp
# with a key:
claude mcp add paper-sources -e NCBI_API_KEY=... -- uv run --directory /path/to/paper_triage paper-sources-mcp
```

### Claude Desktop

In `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart the app:

```json
{
  "mcpServers": {
    "paper-sources": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"],
      "env": { "NCBI_API_KEY": "optional", "UNPAYWALL_EMAIL": "you@example.org" }
    }
  }
}
```

### Cursor and Windsurf

The same `mcpServers` block as Claude Desktop goes in:
- **Cursor:** `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project).
- **Windsurf:** `~/.codeium/windsurf/mcp_config.json`.

### VS Code (GitHub Copilot agent mode)

`.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "paper-sources": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"]
    }
  }
}
```

### Gemini CLI

`~/.gemini/settings.json` (or `.gemini/settings.json` in a project):

```json
{
  "mcpServers": {
    "paper-sources": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"]
    }
  }
}
```

### OpenAI Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.paper-sources]
command = "uv"
args = ["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"]
env = { NCBI_API_KEY = "optional" }
```

### Clients that connect by URL

Start the server in HTTP mode (`paper-sources-mcp --transport streamable-http --port 8765`) and add `http://127.0.0.1:8765/mcp` as a remote or "streamable HTTP" MCP server in the client.

### Your own agent code

**MCP Python SDK** (stdio):

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(
    command="uv",
    args=["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"],
)

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_papers",
                {"query": "single-cell foundation model", "from_date": "2026-01-01"},
            )
            print(result.content[0].text)

asyncio.run(main())
```

**Google ADK:**

```python
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

papers = McpToolset(connection_params=StdioConnectionParams(server_params=StdioServerParameters(
    command="uv",
    args=["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"],
)))
agent = Agent(name="researcher", model="gemini-flash-latest", tools=[papers],
              instruction="Find recent papers with the paper-sources tools.")
```

**LangChain / LangGraph** ([langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)):

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

client = MultiServerMCPClient({"papers": {
    "transport": "stdio",
    "command": "uv",
    "args": ["run", "--directory", "/path/to/paper_triage", "paper-sources-mcp"],
}})
async with client.session("papers") as session:  # hold one session: one server process
    tools = await load_mcp_tools(session)
```

For HTTP mode, use `"transport": "streamable_http", "url": "http://127.0.0.1:8765/mcp"` (LangChain) or `StreamableHTTPConnectionParams(url=...)` (ADK).

## Try it without a client

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) opens a web UI where you can call each tool by hand (it needs Node.js):

```bash
npx @modelcontextprotocol/inspector uv run --directory /path/to/paper_triage paper-sources-mcp
```

## Notes

- **Agent instructions:** only cite papers a tool actually returned. The server's MCP instructions say so, and Paper Triage enforces it by checking every paper against the tool results.
- **Dates:** `from_date` is `YYYY-MM-DD`. Short keyword phrases work best as queries; avoid boolean syntax except in `search_pubmed`.
- **Logging:** the server logs warnings only, to stderr, so stdout stays clean for the MCP protocol.
