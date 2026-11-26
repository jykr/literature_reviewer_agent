"""MCP server that pools scholarly search across OpenAlex, Europe PMC, PubMed,
arXiv and bioRxiv/medRxiv (stdio transport, keyless).

The LangGraph pipeline launches it as a subprocess via ``langchain-mcp-adapters``,
but it is a standalone server any MCP client can use:

    uv run --extra langchain paper-sources-mcp            # stdio
    claude mcp add paper-sources -- uv run --directory "$PWD" --extra langchain paper-sources-mcp

Optional env: ``OPENALEX_API_KEY`` (own daily budget; keyless requests share a
small per-IP one), ``OPENALEX_MAILTO``, ``NCBI_API_KEY``
(PubMed 10 req/s instead of 3).
"""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from paper_sources_mcp import sources as ps

Source = Literal["openalex", "europepmc", "pubmed", "arxiv", "biorxiv"]

mcp = FastMCP(
    "paper-sources",
    log_level="WARNING",  # stderr is the client's terminal; don't log every request
    instructions=(
        "Scholarly search pooled across OpenAlex, Europe PMC, PubMed, arXiv and "
        "bioRxiv/medRxiv. Start with search_papers; use get_paper_details for a "
        "full abstract. Only cite papers these tools return."
    ),
)


@mcp.tool()
async def search_papers(
    query: str,
    from_date: str,
    max_results: int = 12,
    sources: list[Source] | None = None,
) -> dict:
    """Search ALL sources at once and return one deduplicated list (preferred).

    Queries OpenAlex, Europe PMC (PubMed/PMC journals), PubMed, arXiv and
    bioRxiv/medRxiv in parallel, merges hits that share a DOI or title (a
    preprint and its published version merge), and orders papers found by more
    sources first. Each result lists the `sources` that returned it.

    Args:
        query: Keyword query, e.g. "single-cell foundation model perturbation".
            Short keyword phrases work best; avoid boolean syntax.
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: Max merged results to return (default 12).
        sources: Optional subset of sources to query; default is all five.
            Drop "arxiv" for purely biomedical topics (it is rate-limited to
            one request per 3 s); drop "pubmed"/"biorxiv" for CS/physics topics.
    """
    return await ps.search_papers(query, from_date, max_results, sources)


@mcp.tool()
async def search_openalex(query: str, from_date: str, max_results: int = 8) -> dict:
    """OpenAlex only: broadest coverage plus cited_by_count and FWCI.

    Args:
        query: Keyword query.
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_openalex(query, from_date, max_results)


@mcp.tool()
async def search_pubmed(query: str, from_date: str, max_results: int = 8) -> dict:
    """PubMed only (NCBI E-utilities): peer-reviewed biomedical literature.

    Accepts PubMed query syntax, e.g. 'CRISPR screen[tiab] AND T cell[mh]'.

    Args:
        query: PubMed query.
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_pubmed(query, from_date, max_results)


@mcp.tool()
async def search_europepmc(query: str, from_date: str, max_results: int = 8) -> dict:
    """Europe PMC only: PubMed + PMC journal articles with abstracts (no preprints).

    Args:
        query: Keyword query (Europe PMC syntax also accepted).
        from_date: Earliest publication date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_europepmc(query, from_date, max_results)


@mcp.tool()
async def search_biorxiv(query: str, from_date: str, max_results: int = 8) -> dict:
    """bioRxiv + medRxiv preprints only, by keyword (via Europe PMC's preprint index).

    Args:
        query: Keyword query.
        from_date: Earliest posting date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_biorxiv(query, from_date, max_results)


@mcp.tool()
async def search_arxiv(query: str, from_date: str, max_results: int = 8) -> dict:
    """arXiv preprints only (cs, q-bio, stat, physics, math). Rate-limited: 1 req / 3 s.

    Args:
        query: Keyword query (<= 3 words: all must appear; longer: relevance-ranked).
        from_date: Earliest submission date, 'YYYY-MM-DD'.
        max_results: How many results (default 8).
    """
    return await ps.search_arxiv(query, from_date, max_results)


@mcp.tool()
async def get_paper_details(identifier: str) -> dict:
    """Full metadata and untruncated abstract for one paper.

    For bioRxiv/medRxiv DOIs it also reports `published_doi` when the preprint
    has since appeared in a journal.

    Args:
        identifier: A DOI ('10.1038/...'), bioRxiv/medRxiv DOI ('10.1101/...'),
            PubMed ID ('41443191'), arXiv id ('2407.06405'), or a URL that
            contains one (doi.org, arxiv.org, pubmed, biorxiv.org, publishers).
    """
    return await ps.get_paper_details(identifier)


@mcp.tool()
async def get_paper_fulltext(identifier: str) -> dict:
    """Open full-text excerpts of one paper: results, methods and discussion
    sections (each budgeted, sampled across subsections), table text, and
    code/data links. Intro, related work and references are skipped.

    Sources: arXiv HTML; open-access journal articles via Europe PMC/PMC;
    bioRxiv/medRxiv via Europe PMC or the published version. Returns
    status "unavailable" for paywalled papers or blocked preprint servers.

    Args:
        identifier: DOI, PMID, arXiv id, or a URL containing one.
    """
    return await ps.get_paper_fulltext(identifier)


def main(argv: list[str] | None = None) -> None:
    """Run the server: stdio (default) or streamable HTTP for remote clients.

    paper-sources-mcp                                   # stdio, for local clients
    paper-sources-mcp --transport streamable-http --port 8765   # http://127.0.0.1:8765/mcp
    paper-sources-mcp --env-file .env                   # also read optional keys from a file
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="paper-sources-mcp",
        description="Pooled scholarly search MCP server (OpenAlex, Europe PMC, "
        "PubMed, arXiv, bioRxiv/medRxiv).",
    )
    ap.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="stdio (default) for clients that launch the server; streamable-http "
        "(or legacy sse) to serve it over HTTP.",
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (default 127.0.0.1). The server has no "
        "authentication: only bind publicly behind your own auth/proxy.",
    )
    ap.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765).")
    ap.add_argument(
        "--env-file",
        default=None,
        help="Optional .env file with OPENALEX_API_KEY, NCBI_API_KEY, UNPAYWALL_EMAIL, "
        "... (environment variables already set take precedence).",
    )
    args = ap.parse_args(argv)
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
