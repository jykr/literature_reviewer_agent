# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pooled scholarly search tools for the ADK researchers.

The same five-source adapters the LangChain variant's MCP server exposes
(``lc_app.paper_sources``: OpenAlex, Europe PMC, PubMed, arXiv, bioRxiv/medRxiv),
called in-process as ADK function tools. ``record_tool_results`` (an
``after_tool_callback``) keeps every returned paper record in state, so a
researcher's papers can be checked against what the tools actually returned.
"""

from __future__ import annotations

from typing import Any

from lc_app import paper_sources as ps


async def search_papers(query: str, from_date: str, max_results: int = 12) -> dict:
    """Search ALL sources at once and return one deduplicated list (preferred).

    Queries OpenAlex, Europe PMC, PubMed, arXiv and bioRxiv/medRxiv in parallel
    and merges hits that share a DOI or title (a preprint and its published
    version merge). Each result lists the `sources` that returned it.

    Args:
        query: Short keyword phrase, e.g. "single-cell foundation model perturbation".
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: Max merged results (default 12).
    """
    return await ps.search_papers(query, from_date, max_results)


async def search_openalex(query: str, from_date: str, max_results: int = 8) -> dict:
    """OpenAlex only: broadest coverage plus cited_by_count and FWCI.

    Args:
        query: Keyword query.
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_openalex(query, from_date, max_results)


async def search_pubmed(query: str, from_date: str, max_results: int = 8) -> dict:
    """PubMed only: peer-reviewed biomedical literature (PubMed query syntax OK).

    Args:
        query: PubMed query.
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_pubmed(query, from_date, max_results)


async def search_europepmc(query: str, from_date: str, max_results: int = 8) -> dict:
    """Europe PMC only: journal articles with abstracts (no preprints).

    Args:
        query: Keyword query.
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_europepmc(query, from_date, max_results)


async def search_biorxiv(query: str, from_date: str, max_results: int = 8) -> dict:
    """bioRxiv + medRxiv preprints only, by keyword.

    Args:
        query: Keyword query.
        from_date: Earliest posting date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_biorxiv(query, from_date, max_results)


async def search_arxiv(query: str, from_date: str, max_results: int = 8) -> dict:
    """arXiv preprints only (cs, q-bio, stat, physics). Rate-limited: 1 req / 3 s.

    Args:
        query: Keyword query.
        from_date: Earliest submission date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_arxiv(query, from_date, max_results)


async def get_paper_details(identifier: str) -> dict:
    """Full metadata and untruncated abstract for one paper.

    Args:
        identifier: DOI, bioRxiv DOI, PubMed ID, arXiv id, or a URL containing one.
    """
    return await ps.get_paper_details(identifier)


RESEARCH_TOOLS = [
    search_papers,
    search_openalex,
    search_pubmed,
    search_europepmc,
    search_biorxiv,
    search_arxiv,
    get_paper_details,
]


def records_in(response: Any) -> list[dict]:
    """Paper records in one tool response ({results: [...]} or {paper: {...}})."""
    if not isinstance(response, dict):
        return []
    recs = [r for r in response.get("results") or [] if isinstance(r, dict)]
    if isinstance(response.get("paper"), dict):
        recs.append(response["paper"])
    return recs


def record_tool_results(tool, args, tool_context, tool_response) -> None:
    """after_tool_callback: append returned records to state['tool_records_<agent>'].

    Per-agent keys, so parallel researchers never write the same key.
    """
    recs = records_in(tool_response)
    if recs:
        key = f"tool_records_{tool_context.agent_name}"
        tool_context.state[key] = list(tool_context.state.get(key) or []) + recs
    return None
