# literature-reviewer

Simple ReAct agent
Agent generated with `agents-cli` version `1.0.0`

## Project Structure

```
literature-reviewer/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Antigravity CLI](https://antigravity.google/) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        |
| `agents-cli deploy`  | Deploy agent to Agent Runtime                                                                |
| `agents-cli publish gemini-enterprise` | Register deployed agent to Gemini Enterprise                    || [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## Security

The agent fetches **user-supplied URLs** (`load_web_page` on the CV) and ingests
untrusted web/API text, so guardrails live in [`app/security.py`](app/security.py)
as ADK callbacks on the scope agent:

- **SSRF defense** (`url_ssrf_guard`, a `before_tool_callback`) — before any fetch,
  enforces an http/https scheme allowlist and resolves the host, rejecting every
  non-public address: loopback, RFC-1918 private ranges, and link-local — including
  the cloud **metadata** endpoint (`169.254.169.254` / `metadata.google.internal`).
  All resolved IPs are checked, so a public name pointing at an internal address
  (DNS rebinding) is blocked too.
- **Prompt-injection / input-abuse guard** (`injection_guard`, a
  `before_model_callback`) — caps input size and refuses known instruction-override
  payloads before they reach the model.
- **Model-layer safety settings** ([`app/models.py`](app/models.py) `safety_config()`)
  — Gemini harm categories blocked at `MEDIUM_AND_ABOVE` on the agents that read
  untrusted text.
- **Stored-XSS defense** ([`app/render.py`](app/render.py) `_json_for_script()`) —
  paper data is `\uXXXX`-escaped when embedded in the generated file's `<script>`
  block, so a hostile paper title can't break out of the tag.

Tests: [`tests/unit/test_security.py`](tests/unit/test_security.py)
(`uv run pytest tests/unit`).

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
