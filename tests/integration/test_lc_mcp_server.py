"""The paper-sources MCP server starts over stdio and exposes its tools (no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_mcp_adapters")

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from lc_app.graph import paper_server_connection


@pytest.mark.asyncio
async def test_mcp_server_lists_pooled_tools():
    client = MultiServerMCPClient({"papers": paper_server_connection()})
    async with client.session("papers") as session:
        tools = {t.name: t for t in await load_mcp_tools(session)}
    assert set(tools) == {
        "search_papers",
        "search_openalex",
        "search_pubmed",
        "search_europepmc",
        "search_biorxiv",
        "search_arxiv",
        "get_paper_details",
        "get_paper_fulltext",
    }
    schema = tools["search_papers"].args_schema
    props = (
        schema["properties"]
        if isinstance(schema, dict)
        else schema.model_json_schema()["properties"]
    )
    assert {"query", "from_date", "max_results", "sources"} <= set(props)
