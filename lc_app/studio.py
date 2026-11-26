"""Graph factory for the LangGraph dev server (``langgraph dev``).

    uv run --extra langchain langgraph dev      # from the repo root

``langgraph.json`` points here. Each run opens its own session to the
paper-sources MCP server (and to Paperpile when PAPERPILE_API_KEY is set),
held for the whole run, as ``run_review`` does. Requests that only inspect
the graph (Studio's graph view, schemas) get the same topology without
spawning anything.

Run input is small: ``{"request": "..."}`` plus optional ``manual_ids``,
``from_date``, ``refresh``, ``skip_research`` and ``add_to_previous``. A
``prepare`` node expands it into the full review state (dates, research
cache). The review page is written to ``<LC_CACHE_DIR>/studio/`` and its
path is in the final ``message``.
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.types import Overwrite
from typing_extensions import TypedDict  # pydantic schema on Python < 3.12

from lc_app import cache
from lc_app.graph import (
    ReviewState,
    build_graph,
    initial_state,
    paper_server_connection,
)


class ReviewInput(TypedDict, total=False):
    request: str  # topics, a CV/website URL, or both
    manual_ids: list[str]  # DOIs / PMIDs / arXiv ids / URLs to add
    from_date: str  # ISO date; default one year back
    refresh: bool  # ignore cached research and analyses
    skip_research: bool  # review only manual_ids
    add_to_previous: bool  # reuse the last review's request + research
    profile: str  # researcher profile, used verbatim


def prepare_node(state: ReviewState) -> dict:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = cache.cache_dir() / "studio" / f"review-{stamp}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = initial_state(
        state.get("request", ""),
        str(out),
        state.get("from_date"),
        state.get("manual_ids"),
        bool(state.get("refresh")),
        bool(state.get("add_to_previous")),
        skip_research=bool(state.get("skip_research")),
        profile=state.get("profile") or "",
    )
    # A thread keeps its state across runs: start each run clean, bypassing
    # the add reducers so earlier runs' papers don't accumulate.
    return {
        "scope": {},
        "cached": False,
        "queue": {},
        "library_msg": "",
        "message": "",
        **new,
        "candidates": Overwrite(new.get("candidates", [])),
        "manual": Overwrite([]),
        "manual_failed": Overwrite([]),
    }


def _graph(tools, library=None):
    return build_graph(tools, library, prepare=prepare_node, input_schema=ReviewInput)


# Names are all build_graph needs to lay out the nodes. A file loaded by path
# (langgraph.json) can't use relative imports, hence lc_app.* above.
_STUB_TOOLS = [
    SimpleNamespace(name=n) for n in ("get_paper_details", "get_paper_fulltext")
]


@asynccontextmanager
async def make_graph(config: RunnableConfig):
    paperpile = bool(os.getenv("PAPERPILE_API_KEY"))
    if not (config.get("configurable") or {}).get("__is_for_execution__", True):
        yield _graph(_STUB_TOOLS, object() if paperpile else None)
        return
    async with AsyncExitStack() as stack:
        client = MultiServerMCPClient({"papers": paper_server_connection()})
        session = await stack.enter_async_context(client.session("papers"))
        tools = await load_mcp_tools(session)
        library = None
        if paperpile:
            from lc_app.paperpile import PaperpileLibrary, paperpile_connection

            pp = MultiServerMCPClient({"paperpile": paperpile_connection()})
            library = PaperpileLibrary(
                await stack.enter_async_context(pp.session("paperpile"))
            )
        yield _graph(tools, library)
