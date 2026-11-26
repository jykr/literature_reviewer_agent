"""LangGraph pipeline: scope -> parallel research (MCP tools) -> rank -> render.

    scope      fetch any CV/site URLs, structured-output Scope (<= 5 clusters)
      |  Send() one branch per cluster (dynamic fan-out, runs concurrently)
    research   create_agent ReAct loop over the paper-sources MCP tools,
               structured CandidateList; candidates are VERIFIED against what
               the tools actually returned (SPEC §2.5) and their url/doi are
               overwritten with the tool's values
      |  candidates merged via an operator.add reducer
    rank       structured-output ReviewQueue (same editorial prompt as the ADK
               ranker), re-verified against the candidate pool
      |
    render     app.render.build_html -> self-contained literature-reviewer.html

Reuses the ADK app's Pydantic schemas, prompts and renderer so both variants
emit byte-compatible review apps.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import json
import logging
import operator
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, Any, TypedDict

import httpx
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.analysis import RANK_INSTRUCTION
from app.byo_review import SCOPE_SYS
from app.render import build_html, categories_from_scope
from app.schemas import Approach, Paper, ReviewQueue, Scope
from app.security import _url_is_safe
from lc_app import cache
from lc_app.focus import focus_block, review_focus
from lc_app.models import build_model, model_id
from lc_app.paper_sources import (
    _ARXIV_ID,
    normalize_doi,
    normalize_identifier,
    title_key,
)

logger = logging.getLogger(__name__)

MAX_CLUSTERS = 5  # RANK_INSTRUCTION has five {papers_k} slots
MAX_INPUT_CHARS = 8000
CV_CHARS = 6000
_URL_RE = re.compile(r"https?://[^\s<>\"'),\]]+")  # commas separate pasted lists


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ReviewState(TypedDict, total=False):
    request: str
    from_date: str
    output_path: str
    scope: dict
    candidates: Annotated[list[dict], operator.add]  # [{"cluster": k, "papers": [...]}]
    manual_ids: list[str]  # user-supplied DOIs / PMIDs / arXiv ids / URLs
    manual_origin: dict[str, str]  # id -> origin label, e.g. "Paperpile: to read"
    skip_research: bool  # review only the added papers (no literature search)
    manual: Annotated[list[dict], operator.add]  # analyzed manual papers
    manual_failed: Annotated[list[str], operator.add]
    refresh: bool  # ignore cached manual-paper analyses
    cache_key: str
    cached: bool  # scope + candidates came from the research cache
    queue: dict
    library_msg: str  # Paperpile flag/save summary for the final message
    message: str


class ManualTask(TypedDict):
    identifier: str
    profile: str
    refresh: bool
    origin: str  # "" = added by hand; else e.g. "Paperpile: to read"


class LabeledPaper(Paper):
    """Render-time Paper with the 'manually added' label (not ranker output,
    so the ranker's structured-output schema stays unchanged)."""

    manual: bool = False
    evidence: str = ""  # "fulltext" | "abstract" (set by the deepen stage)
    in_library: bool = False  # already in the user's Paperpile library
    origin: str = ""  # where an added paper came from, e.g. "Paperpile: to read"


class LabeledQueue(ReviewQueue):
    papers: list[LabeledPaper]


class ResearchTask(TypedDict):
    k: int
    cluster: dict
    profile: str
    from_date: str


class Candidate(BaseModel):
    """One paper found by a researcher, described from its abstract.

    Kept minimal on purpose: ids, metrics, venue status and h-index are NOT
    model output. ``verify_candidates`` copies them from the matching tool
    record (``_TOOL_TRUTH_FIELDS``), and fewer optional fields keep the
    schema under providers' structured-output complexity limits.
    """

    title: str = Field(description="Exactly as returned by the search tool.")
    authors: str
    institution: str = ""
    venue: str = ""
    date: str = ""
    url: str = Field(description="Exactly as returned by the search tool.")
    doi: str = ""
    algo: str = Field(description="Core method/algorithm name.")
    data: str = Field(description="Data used (train + eval, scale).")
    model: str = Field(description="How the method works / how it was adapted.")
    evaluation: str = Field(description="Metrics, eval data, design novelty, limits.")
    question: str = Field(description="The key question it answers.")
    app_tags: list[str] = Field(default_factory=list)
    method_tags: list[str] = Field(default_factory=list)


class CandidateList(BaseModel):
    papers: list[Candidate] = Field(description="4-6 verified papers for this cluster.")


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def fetch_page_text(url: str, limit: int = CV_CHARS) -> str:
    """Fetch a CV/site as plain text, SSRF-checking every redirect hop."""
    try:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            for _ in range(4):
                ok, why = _url_is_safe(url)
                if not ok:
                    return f"(refused to fetch {url}: {why})"
                resp = client.get(
                    url, headers={"User-Agent": "literature-reviewer/0.1"}
                )
                if resp.is_redirect and "location" in resp.headers:
                    url = str(resp.url.join(resp.headers["location"]))
                    continue
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text[:400_000], "lxml")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                return " ".join(soup.get_text(" ").split())[:limit]
        return f"(too many redirects fetching {url})"
    except Exception as e:
        return f"(could not fetch {url}: {e})"


async def scope_node(state: ReviewState) -> dict:
    if state.get("cached"):
        return {}  # scope came from the research cache
    request = (state.get("request") or "").strip()[:MAX_INPUT_CHARS]
    parts = [f"User request:\n{request}"] if request else []
    for url in list(dict.fromkeys(_URL_RE.findall(request)))[:2]:
        text = fetch_page_text(url)
        logger.info("scope: fetched %s (%d chars)", url, len(text))
        parts.append(f"Content of {url}:\n{text}")
    if not parts:
        return {"scope": {"profile": "", "clusters": []}}

    llm = build_model().with_structured_output(Scope)
    scope: Scope = await llm.ainvoke(
        [
            SystemMessage(SCOPE_SYS + focus_block("scope")),
            HumanMessage("Researcher interests:\n\n" + "\n\n".join(parts)),
        ]
    )
    scope.clusters = scope.clusters[:MAX_CLUSTERS]
    logger.info(
        "scope: %d clusters: %s", len(scope.clusters), [c.name for c in scope.clusters]
    )
    return {"scope": scope.model_dump()}


def fan_out(state: ReviewState):
    scope = state.get("scope") or {}
    clusters = scope.get("clusters") or []
    if not clusters:
        return "no_scope"
    profile = scope.get("profile", "")
    sends = [
        Send(
            "research",
            ResearchTask(k=k, cluster=c, profile=profile, from_date=state["from_date"]),
        )
        for k, c in enumerate(clusters)
        if not state.get("cached") and not state.get("skip_research")
    ]
    sends += [
        Send(
            "manual",
            ManualTask(
                identifier=i,
                profile=profile,
                refresh=bool(state.get("refresh")),
                origin=(state.get("manual_origin") or {}).get(i, ""),
            ),
        )
        for i in dict.fromkeys(state.get("manual_ids") or [])
    ]
    return sends or "rank"  # cached research, nothing to add: straight to rank


def no_scope_node(state: ReviewState) -> dict:
    return {
        "message": "No research scope found. Provide a CV/website URL and/or a few "
        "research topics or keywords."
    }


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """\
You research recent papers for ONE subfield cluster of a literature review
tailored to one researcher.

Researcher profile: {profile}
Cluster: {name}
Keywords: {keywords}

Find 4-6 REAL papers matching this cluster, published on or after {from_date}.

Tools (all keyless, pooled from OpenAlex, Europe PMC, PubMed, arXiv, bioRxiv/medRxiv):
- `search_papers` — START HERE. Searches every source at once and dedupes.
  Issue 2-4 focused queries, each a short keyword phrase combining different
  cluster keywords. Always pass from_date="{from_date}".
- `search_pubmed`, `search_biorxiv`, `search_arxiv`, `search_openalex`,
  `search_europepmc` — dig into one source when the pooled results are thin.
- `get_paper_details` — full abstract for a DOI/PMID/arXiv id, when the
  search abstract is too short to describe the method and evaluation.

Prefer papers with substantive methods or results over reviews/editorials.

Credibility: weigh author credibility most: among comparable papers prefer
higher senior_author_h_index, then first_author_h_index, cited_by_count and
fwci. Venue is a secondary tie-breaker: when a preprint has a published
version (a merged record with a journal/conference venue, or `published_doi`
from `get_paper_details`), use the published one, and between otherwise
comparable papers prefer the peer-reviewed one. Preprints from established
groups are fine to include.

Integrity rule: include ONLY papers the tools returned. Copy `title`, `url`
and `doi` exactly from the tool output (venue status, citations and h-index
are attached automatically from the tool record). Describe
algo/data/model/evaluation from the abstract only; write "not stated in
abstract" rather than guessing.
{focus}"""


CREDIBILITY_RULES = """

Credibility weighting (applies within every coverage level). Author
credibility is the PRIMARY signal; venue is SECONDARY (weigh authors roughly
twice as much as venue):
- Authors (primary): judge from evidence: `senior_author_h_index` first, then
  `first_author_h_index`, `institution`, `cited_by_count` and `fwci`. A paper
  from a clearly more established group (e.g. a much higher senior-author
  h-index) ranks above one from a less established group even when the latter
  has the better venue. Do not infer credibility from author or institution
  names alone; when the fields are missing, treat credibility as unknown
  rather than low.
- Venue (secondary): at comparable relevance AND comparable author
  credibility, a peer-reviewed journal or conference paper (`is_preprint`
  false, `venue_type` journal/conference) ranks above a preprint (arXiv,
  bioRxiv, medRxiv). Venue alone never outweighs a clear gap in author
  credibility. An "accepted at" `venue_note` is self-reported: between a
  preprint and a published paper. Workshops rank below main conference and
  journal papers.
- For every preprint, start `comments` with "Preprint (not peer-reviewed)."
"""


def _content_records(content: Any) -> list[dict]:
    """Paper records in one MCP tool result (str or list of content blocks)."""
    records: list[dict] = []
    for chunk in content if isinstance(content, list) else [content]:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            records += payload.get("results") or []
            if isinstance(payload.get("paper"), dict):
                records.append(payload["paper"])
    return records


def _tool_records(messages: list) -> list[dict]:
    """Every paper record that any MCP tool returned during an agent run."""
    return [
        r
        for m in messages
        if isinstance(m, ToolMessage)
        for r in _content_records(m.content)
    ]


# Copied from the tool record over whatever the model wrote: ids, metrics and
# venue status, so a model can't mangle a link or upgrade a preprint to a journal.
_TOOL_TRUTH_FIELDS = (
    "title",
    "url",
    "doi",
    "cited_by_count",
    "fwci",
    "venue",
    "venue_type",
    "venue_note",
    "is_preprint",
    "first_author_h_index",
    "senior_author_h_index",
)


def verify_candidates(
    papers: list[dict], records: list[dict]
) -> tuple[list[dict], int]:
    """Keep only papers matching a tool record (by DOI or title); copy its ids.

    Returns (kept, n_dropped). The tool record is the source of truth for
    url/doi/citation fields, so a model-mangled link can't reach the review.
    """
    by_doi = {normalize_doi(r.get("doi")): r for r in records if r.get("doi")}
    by_title = {title_key(r.get("title", "")): r for r in records if r.get("title")}
    by_url = {r["url"].rstrip("/"): r for r in records if r.get("url")}
    kept = []
    for p in papers:
        rec = (
            (by_doi.get(normalize_doi(p.get("doi"))) if p.get("doi") else None)
            or by_title.get(title_key(p.get("title", "")))
            or by_url.get((p.get("url") or "").rstrip("/"))
        )
        if rec is None:
            continue
        kept.append(_apply_tool_truth(p, rec))
    return kept, len(papers) - len(kept)


def _apply_tool_truth(paper: dict, rec: dict, extra: tuple[str, ...] = ()) -> dict:
    p = dict(paper)
    for f in _TOOL_TRUTH_FIELDS + extra:
        if rec.get(f) not in (None, ""):
            p[f] = rec[f]
    p["sources"] = rec.get("sources", [])
    return p


# ---------------------------------------------------------------------------
# Manually added papers
# ---------------------------------------------------------------------------

MANUAL_PROMPT = """\
You analyze ONE paper that the researcher added to their literature review by
hand. It will be ranked alongside papers found by search, so describe it the
same way.

Researcher profile: {profile}

You get the paper's metadata and abstract as JSON. Copy `title`, `url` and
`doi` exactly. Describe algo/data/model/evaluation/question and the
application/method tags from the abstract only; write "not stated in
abstract" rather than guessing.
"""


_MANUAL_CONCURRENCY = 6  # a Paperpile label can add ~100 papers at once


def make_manual_node(tools: list[BaseTool]):
    details = next(t for t in tools if t.name == "get_paper_details")
    sem = asyncio.Semaphore(_MANUAL_CONCURRENCY)

    async def manual_node(task: ManualTask) -> dict:
        async with sem:
            out = await _analyze_manual(task)
        for paper in out.get("manual", []):
            paper["origin"] = task.get("origin", "")
        return out

    async def _analyze_manual(task: ManualTask) -> dict:
        ident = task["identifier"]
        if not task.get("refresh") and (hit := cache.load_manual(ident)):
            logger.info("manual: %s (cached analysis)", ident)
            return {"manual": [hit]}
        try:
            records = _content_records(await details.ainvoke({"identifier": ident}))
        except Exception as e:
            logger.warning("manual: lookup failed for %s: %s", ident, e)
            records = []
        if not records:
            logger.warning("manual: no record found for %s; skipped", ident)
            return {"manual_failed": [ident]}
        rec = records[0]
        llm = build_model().with_structured_output(Candidate)
        analysis: Candidate = await llm.ainvoke(
            [
                SystemMessage(MANUAL_PROMPT.format(profile=task["profile"])),
                HumanMessage(json.dumps(rec, ensure_ascii=False)),
            ]
        )
        # Metadata is the tool's, not the model's (this paper skipped search).
        paper = _apply_tool_truth(
            analysis.model_dump(), rec, extra=("authors", "institution", "date")
        )
        paper.update(manual=True, identifier=ident)
        cache.save_manual(ident, paper)
        logger.info("manual: analyzed %s: %s", ident, paper["title"][:70])
        return {"manual": [paper]}

    return manual_node


def make_research_node(tools: list[BaseTool]):
    async def research_node(task: ResearchTask) -> dict:
        k, cluster = task["k"], task["cluster"]
        agent = create_agent(
            build_model(),
            tools,
            system_prompt=RESEARCH_PROMPT.format(
                profile=task["profile"],
                name=cluster.get("name", ""),
                keywords=", ".join(cluster.get("keywords", [])),
                from_date=task["from_date"],
                focus=focus_block("research"),
            ),
            response_format=CandidateList,
            name=f"researcher_{k}",
        )
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(f"Research cluster {k}: {cluster.get('name', '')}")
                    ]
                },
                {"recursion_limit": 30},
            )
        except Exception as e:
            logger.warning("research[%d] failed: %s: %s", k, type(e).__name__, e)
            return {"candidates": [{"cluster": k, "papers": []}]}

        found: CandidateList | None = result.get("structured_response")
        raw = [p.model_dump() for p in (found.papers if found else [])]
        records = _tool_records(result["messages"])
        papers, dropped = verify_candidates(raw, records)
        n_calls = sum(isinstance(m, ToolMessage) for m in result["messages"])
        logger.info(
            "research[%d] %s: %d tool calls, %d papers kept, %d unverified dropped",
            k,
            cluster.get("name", ""),
            n_calls,
            len(papers),
            dropped,
        )
        return {"candidates": [{"cluster": k, "papers": papers}]}

    return research_node


# ---------------------------------------------------------------------------
# Rank + render
# ---------------------------------------------------------------------------


async def rank_node(state: ReviewState) -> dict:
    scope = state["scope"]
    if (
        state.get("cache_key")
        and not state.get("cached")
        and not state.get("skip_research")
    ):
        cache.save_research(
            state["cache_key"],
            state.get("request", ""),
            scope,
            copy.deepcopy(state.get("candidates", [])),
            from_date=state.get("from_date"),
            focus=review_focus(),
            model=model_id(),
        )
    if state.get("cache_key") and not state.get("skip_research"):
        cache.save_last(state["cache_key"])
    manual = list({_paper_key(m): m for m in state.get("manual", [])}.values())
    manual_keys = {_paper_key(m) for m in manual}
    manual_origin = {_paper_key(m): m.get("origin", "") for m in manual}
    by_cluster: dict[int, list[dict]] = {}
    found_in: dict[
        str, set[int]
    ] = {}  # paper key -> clusters whose researcher found it
    for c in state.get("candidates", []):
        by_cluster.setdefault(c["cluster"], []).extend(c["papers"])
        for p in c["papers"]:
            found_in.setdefault(_paper_key(p), set()).add(c["cluster"])
    for papers in by_cluster.values():
        for p in papers:
            p["found_in_clusters"] = sorted(found_in[_paper_key(p)])
    # Manual last, so a manual record wins verification when it duplicates a
    # searched candidate.
    pool = [p for ps in by_cluster.values() for p in ps] + manual
    if not pool:
        return {
            "queue": {"papers": []},
            "message": "The search tools returned no usable papers.",
        }

    # The ranker writes every paper's full review, so ~100 added papers (a
    # Paperpile label) can't fit one response: rank added papers in batches;
    # the first batch also gets the searched candidates.
    batches = [manual[i : i + _RANK_BATCH] for i in range(0, len(manual), _RANK_BATCH)]
    batches = batches or [[]]
    sem = asyncio.Semaphore(3)

    async def one(i: int, batch: list[dict]):
        async with sem:
            return await _rank_batch(scope, by_cluster if i == 0 else {}, batch, pool)

    results = await asyncio.gather(*(one(i, b) for i, b in enumerate(batches)))
    kept = [p for ps, _ in results for p in ps]
    dropped = sum(d for _, d in results)
    seen: set[str] = set()
    kept = [p for p in kept if not (_paper_key(p) in seen or seen.add(_paper_key(p)))]
    if len(batches) > 1:
        logger.info(
            "rank: %d added papers ranked in %d batches", len(manual), len(batches)
        )
    # Coverage (SPEC §3d): the clusters that actually surfaced a paper are a
    # floor under the ranker's judgment, then re-rank coverage-first with the
    # ranker's order as the tie-break.
    n_clusters = len(scope.get("clusters", []))
    for p in kept:
        cats = {c for c in p.get("cats", []) if 0 <= c < n_clusters}
        p["cats"] = sorted(cats | found_in.get(_paper_key(p), set()))
        p.pop("sources", None)
        p["manual"] = _paper_key(p) in manual_keys
        p["origin"] = manual_origin.get(_paper_key(p), "")
        if p.get("is_preprint") and not p.get("comments", "").startswith("Preprint"):
            p["comments"] = "Preprint (not peer-reviewed). " + p.get("comments", "")
    if len(batches) == 1:
        kept.sort(key=lambda p: (-len(p["cats"]), p["rank"]))
    else:  # batch-local ranks aren't comparable: scores break coverage ties
        kept.sort(key=lambda p: (-len(p["cats"]), -(p["rel"] + p["impact"]), p["rank"]))
    for i, p in enumerate(kept, 1):
        p["rank"] = i
    logger.info(
        "rank: %d papers ranked, %d dropped as not in candidate pool",
        len(kept),
        dropped,
    )
    return {"queue": {"papers": kept}}


_RANK_BATCH = 10  # added papers per ranker call


async def _rank_batch(
    scope: dict, by_cluster: dict[int, list[dict]], manual: list[dict], pool: list[dict]
) -> tuple[list[dict], int]:
    """One ranker call over ``by_cluster`` candidates + ``manual`` papers.
    Returns (verified papers, n dropped)."""
    prompt = (
        RANK_INSTRUCTION.format(
            scope=json.dumps(scope, ensure_ascii=False),
            **{
                f"papers_{k}": json.dumps(by_cluster.get(k, []), ensure_ascii=False)
                for k in range(MAX_CLUSTERS)
            },
        )
        + CREDIBILITY_RULES
        + focus_block("rank")
        + (
            MANUAL_RANK_RULES.format(papers=json.dumps(manual, ensure_ascii=False))
            if manual
            else ""
        )
    )
    llm = build_model(max_tokens=16000).with_structured_output(
        ReviewQueue, include_raw=True
    )
    messages = [SystemMessage(prompt), HumanMessage("Produce the ranked ReviewQueue.")]
    for attempt in (1, 2):
        out = await llm.ainvoke(messages)
        if out["parsed"] is None:
            err = str(out["parsing_error"])[:2000]
            logger.warning(
                "rank: ReviewQueue failed validation (attempt %d): %s", attempt, err
            )
            fix = f"Your previous ReviewQueue failed schema validation:\n{err}\n"
        else:
            # Re-verify: the ranker may only use papers from the verified pool.
            kept, dropped = verify_candidates(
                [p.model_dump() for p in out["parsed"].papers], pool
            )
            got = {_paper_key(p) for p in kept}
            missing = [m["title"] for m in manual if _paper_key(m) not in got]
            if not missing or attempt == 2:
                if missing:
                    logger.warning(
                        "rank: ranker still omitted manual papers: %s", missing
                    )
                return kept, dropped
            logger.warning(
                "rank: ranker omitted %d manual paper(s); retrying", len(missing)
            )
            fix = (
                "Your ReviewQueue omitted these manually added papers, which must all "
                "be included: " + "; ".join(missing) + "\n"
            )
        messages = [
            *messages[:2],
            HumanMessage(fix + "Return the complete, corrected ReviewQueue."),
        ]
    raise RuntimeError("Ranker did not produce a valid ReviewQueue after a retry.")


def _paper_key(p: dict) -> str:
    return normalize_doi(p.get("doi")) or title_key(p.get("title", ""))


MANUAL_RANK_RULES = """

Manually added papers (the researcher added these by hand; JSON):
{papers}
Every manually added paper MUST appear in `papers`, ranked by the same rules as
the searched candidates. They do NOT count toward the 12-paper limit. If one
duplicates a searched candidate, list it once.
"""


# ---------------------------------------------------------------------------
# Deepen: full-text analysis of the SELECTED papers only (after ranking)
# ---------------------------------------------------------------------------


class DeepAnalysis(BaseModel):
    """Fields re-derived from full text. Rank, scores, coverage, tags,
    relevance and comments stay as the ranker set them."""

    insights: str = Field(
        description='"New metric: <...> New eval data: <...> Design & novelty: <...> '
        'Eval limits: <...>" (four labels verbatim, in order, one string).'
    )
    results: list[str] = Field(
        description="3-6 main results, one sentence each. Quantitative ones name "
        "task, dataset/benchmark, metric, value and the baseline compared against."
    )
    limitation: str
    resources: str = Field(
        default="",
        description="Space-separated code/data URLs from the provided links only.",
    )
    approach: Approach


DEEPEN_PROMPT = """\
You are deepening the analysis of ONE paper already selected for a literature
review, now using excerpts of its full text (results, methods, discussion,
tables, links). Researcher profile: {profile}

Rewrite these fields from the full text:
- insights: the four labels verbatim, in order: "New metric: ... New eval data:
  ... Design & novelty: ... Eval limits: ..."
- results: 3-6 main results. For a quantitative result name the task, the
  dataset/benchmark, the metric, the value and what it is compared against.
  Copy every number exactly as it appears in the excerpts. Never compute,
  round or estimate a number (no derived differences).
- limitation: the main limitation, stated by the authors or evident from the
  evaluation.
- resources: code/data URLs, ONLY from the provided `links` list.
- approach: algo, nov (0 = pure engineering ... 100 = fundamentally new
  algorithm), aim ("<goal>. Limitation addressed: <gap>."), data, model,
  question.

Base every statement on the excerpts. Where they don't cover a field, keep
the current (abstract-based) value given below.
"""

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")


def grounded_results(results: list[str], source: str) -> tuple[list[str], int]:
    """Drop result bullets quoting a decimal/percentage absent from the source.

    Integers are not checked (years, counts and section numbers are too
    common to tell apart); decimals and percentages are what matter.
    """
    kept = []
    for r in results:
        nums = [
            n.lstrip("+-").rstrip("%")
            for n in _NUMBER.findall(r)
            if "." in n or n.endswith("%")
        ]
        if all(n in source for n in nums):
            kept.append(r)
    return kept, len(results) - len(kept)


def _json_payload(content: Any) -> dict:
    for chunk in content if isinstance(content, list) else [content]:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _fulltext_ids(p: dict) -> list[str]:
    url = p.get("url") or ""
    ids = [
        p.get("identifier"),
        url if "arxiv.org" in url else None,
        p.get("doi"),
        p.get("preprint_doi"),
        url,
    ]
    return list(dict.fromkeys(i for i in ids if i))


def make_deepen_node(tools: list[BaseTool]):
    fulltext = next((t for t in tools if t.name == "get_paper_fulltext"), None)

    async def deepen_node(state: ReviewState) -> dict:
        papers = (state.get("queue") or {}).get("papers") or []
        if not papers or fulltext is None:
            return {}
        profile = state["scope"].get("profile", "")
        llm = build_model(max_tokens=4096).with_structured_output(DeepAnalysis)
        sem = asyncio.Semaphore(4)
        stats = {"dropped_results": 0}

        async def one(p: dict) -> dict:
            key = _paper_key(p)
            if not state.get("refresh") and (hit := cache.load_deep(key)):
                return {**p, **hit}
            ft = None
            for ident in _fulltext_ids(p):
                payload = _json_payload(await fulltext.ainvoke({"identifier": ident}))
                if payload.get("status") == "success":
                    ft = payload
                    break
            if ft is None:
                return {**p, "evidence": "abstract"}
            current = {
                k: p.get(k)
                for k in ("insights", "results", "limitation", "resources", "approach")
            }
            excerpt = {k: ft[k] for k in ("title", "sections", "tables", "links")}
            async with sem:
                analysis: DeepAnalysis = await llm.ainvoke(
                    [
                        SystemMessage(DEEPEN_PROMPT.format(profile=profile)),
                        HumanMessage(
                            "Current analysis:\n"
                            + json.dumps(current, ensure_ascii=False)
                            + "\n\nFull-text excerpts:\n"
                            + json.dumps(excerpt, ensure_ascii=False)
                        ),
                    ]
                )
            d = analysis.model_dump()
            source = "\n".join(ft["sections"] + ft["tables"])
            d["results"], dropped = grounded_results(d["results"], source)
            stats["dropped_results"] += dropped
            if not d["results"]:
                d["results"] = p.get("results") or []
            allowed = set(ft["links"]) | set((p.get("resources") or "").split())
            urls = [u for u in d["resources"].split() if u in allowed]
            d["resources"] = " ".join(
                dict.fromkeys(urls + (p.get("resources") or "").split())
            )
            hit = {**d, "evidence": "fulltext", "fulltext_source": ft["source"]}
            cache.save_deep(key, hit)
            return {**p, **hit}

        outs = await asyncio.gather(*(one(p) for p in papers), return_exceptions=True)
        deepened = []
        for p, out in zip(papers, outs, strict=True):
            if isinstance(out, BaseException):
                logger.warning("deepen: %s failed: %s", p.get("title", "")[:50], out)
                out = {**p, "evidence": "abstract"}
            deepened.append(out)
        n_full = sum(p.get("evidence") == "fulltext" for p in deepened)
        logger.info(
            "deepen: %d/%d selected papers analyzed from full text, %d abstract only; "
            "%d ungrounded result bullet(s) dropped",
            n_full,
            len(deepened),
            len(deepened) - n_full,
            stats["dropped_results"],
        )
        return {"queue": {"papers": deepened}}

    return deepen_node


def render_node(state: ReviewState) -> dict:
    queue = LabeledQueue.model_validate(state.get("queue") or {"papers": []})
    if not queue.papers:
        return {"message": state.get("message") or "No papers to review."}
    scope = state["scope"]
    html = build_html(
        queue,
        profile=scope.get("profile", ""),
        generated=dt.date.today().isoformat(),
        categories=categories_from_scope(scope),
    )
    path = Path(state.get("output_path") or "literature-reviewer.html")
    path.write_text(html, encoding="utf-8")
    n = len(queue.papers)
    added: dict[str, int] = {}
    for p in queue.papers:
        if p.manual:
            label = p.origin or "manually added"
            added[label] = added.get(label, 0) + 1
    msg = f"Reviewed and ranked {n} paper{'s' * (n != 1)}"
    if added:
        msg += (
            " ("
            + ", ".join(
                f"{k} from {lab}" if lab != "manually added" else f"{k} {lab}"
                for lab, k in added.items()
            )
            + ")"
        )
    if failed := state.get("manual_failed"):
        msg += f". Could not find: {', '.join(failed)}"
    if lib := state.get("library_msg"):
        msg += f". {lib}"
    return {"message": f"{msg}. Wrote {path.resolve()}"}


# ---------------------------------------------------------------------------
# Paperpile (optional): flag papers already in the library, save picks
# ---------------------------------------------------------------------------


def reference_identifier(ref: dict) -> str:
    """A Paperpile reference -> DOI, arXiv id or URL that get_paper_details takes."""
    if doi := normalize_doi(ref.get("doi") or ""):
        return doi
    if arxiv := (ref.get("arxivid") or "").strip():
        return arxiv
    url = ref.get("url") or ""
    url = (url[0] if isinstance(url, list) and url else url) or ""
    # Only URLs that resolve to a paper id (arXiv/PubMed/DOI-bearing links).
    ident = normalize_identifier(url) if url else ""
    return "" if ident == url.strip().rstrip("/") else ident


def paper_dois(papers: list[dict], state: ReviewState) -> dict[int, str]:
    """rank -> DOI for queue papers, from the tool records they came from
    (arXiv-only papers get arXiv's DataCite DOI)."""
    from .paperpile import arxiv_doi

    pool = [p for c in state.get("candidates") or [] for p in c.get("papers", [])]
    pool += state.get("manual") or []
    by_title = {title_key(r.get("title", "")): r for r in pool if r.get("title")}
    by_url = {r["url"].rstrip("/"): r for r in pool if r.get("url")}
    out = {}
    for p in papers:
        rec = by_url.get((p.get("url") or "").rstrip("/")) or by_title.get(
            title_key(p.get("title", ""))
        )
        doi = normalize_doi((rec or {}).get("doi") or "")
        if not doi:
            ident = normalize_identifier(p.get("url") or "")
            if _ARXIV_ID.match(ident):
                doi = arxiv_doi(ident)
            elif ident.startswith("10."):
                doi = normalize_doi(ident)
        if doi:
            out[p["rank"]] = doi
    return out


def make_library_node(library, save_label: str | None = None):
    """Flag queue papers already in Paperpile; optionally save them all under
    ``save_label``. Failures are logged and never break the review."""

    async def library_node(state: ReviewState) -> dict:
        queue = dict(state.get("queue") or {"papers": []})
        papers = [dict(p) for p in queue.get("papers", [])]
        dois = paper_dois(papers, state)
        parts = []
        try:
            have = {
                normalize_doi(r.get("doi") or "")
                for r in await library.lookup(list(dois.values()))
            }
            # Papers pulled from Paperpile are in the library by definition
            # (their DOI may differ from the tool's, e.g. preprint vs journal).
            searched = [
                p for p in papers if not p.get("origin", "").startswith("Paperpile")
            ]
            for p in papers:
                p["in_library"] = p not in searched or dois.get(p["rank"]) in have
            if searched:
                n_have = sum(p["in_library"] for p in searched)
                parts.append(f"{n_have} already in your Paperpile library")
                logger.info(
                    "paperpile: %d/%d searched papers already in library",
                    n_have,
                    len(searched),
                )
            if save_label:
                res = await library.save(list(dois.values()), save_label)
                parts.append(
                    f"saved to Paperpile label '{save_label}': {res['added']} added, "
                    f"{res['labeled_existing']} already there"
                    + (f", {res['failed']} not resolvable" if res["failed"] else "")
                )
                logger.info("paperpile: saved under %r: %s", save_label, res)
                missing = len(papers) - len(dois)
                if missing:
                    parts[-1] += f", {missing} without a DOI skipped"
        except Exception as e:
            logger.warning("paperpile: skipped (%s: %s)", type(e).__name__, e)
            parts.append("Paperpile step failed; see log")
        queue["papers"] = papers
        msg = "; ".join(parts)
        return {"queue": queue, "library_msg": msg[:1].upper() + msg[1:]}

    return library_node


def build_graph(tools: list[BaseTool], library=None, save_label: str | None = None):
    g = StateGraph(ReviewState)
    g.add_node("scope", scope_node)
    g.add_node("no_scope", no_scope_node)
    g.add_node("research", make_research_node(tools))
    g.add_node("manual", make_manual_node(tools))
    g.add_node("rank", rank_node)
    g.add_node("deepen", make_deepen_node(tools))
    g.add_node("render", render_node)
    g.add_edge(START, "scope")
    g.add_conditional_edges(
        "scope", fan_out, ["research", "manual", "rank", "no_scope"]
    )
    g.add_edge("no_scope", END)
    g.add_edge("research", "rank")
    g.add_edge("manual", "rank")
    g.add_edge("rank", "deepen")
    if library is not None:
        g.add_node("library", make_library_node(library, save_label))
        g.add_edge("deepen", "library")
        g.add_edge("library", "render")
    else:
        g.add_edge("deepen", "render")
    g.add_edge("render", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# Only these reach the tool server: source keys and (opt-in) AWS credentials
# for bioRxiv's S3 buckets. Model API keys never do.
_SERVER_ENV = (
    "OPENALEX_API_KEY",
    "OPENALEX_MAILTO",
    "NCBI_API_KEY",
    "UNPAYWALL_EMAIL",
    "LC_BIORXIV_S3",
    "LC_CACHE_DIR",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


def paper_server_connection() -> dict[str, Any]:
    """stdio connection that launches lc_app.paper_mcp_server in a subprocess."""
    env = {k: os.environ[k] for k in _SERVER_ENV if os.getenv(k)}
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "lc_app.paper_mcp_server"],
        "env": env,
        "cwd": str(Path(__file__).resolve().parent.parent),
    }


async def run_review(
    request: str,
    output_path: str = "literature-reviewer.html",
    from_date: str | None = None,
    manual_ids: list[str] | None = None,
    refresh: bool = False,
    add_to_previous: bool = False,
    paperpile: bool = False,
    paperpile_profile: str | None = None,
    save_label: str | None = None,
    include_collection: str | None = None,
    skip_research: bool = False,
    added_since: str | None = None,
) -> ReviewState:
    """Run the full pipeline. One MCP session is held for the whole run (the
    adapter's default ``get_tools()`` would spawn a server per tool call).

    Research (scope + verified candidates) is cached per request + focus; a
    cache hit skips straight to analyzing ``manual_ids`` and ranking.
    ``refresh`` ignores both the research and manual-paper caches.
    ``add_to_previous`` reuses the most recent review's request and research
    (``request`` is ignored), so papers can be added without restating topics.

    Paperpile (``paperpile=True``, needs PAPERPILE_API_KEY or a sign-in):
    reviewed papers already in the library are flagged; ``paperpile_profile``
    ("" = whole library, else a folder/label name) adds library titles to the
    request as an interest profile; ``save_label`` saves the reviewed papers
    to the library under that label; ``include_collection`` adds every paper
    in that folder/label (e.g. "to read") to the review, labeled by origin;
    ``added_since`` (ISO date) keeps only those added to the library since
    then (Paperpile doesn't record when a label was applied).

    ``skip_research`` reviews only the added papers: no literature search and
    nothing written to the research cache.
    """
    manual_ids = list(manual_ids or [])
    origin: dict[str, str] = {}
    async with AsyncExitStack() as stack:
        library = None
        if (
            paperpile
            or paperpile_profile is not None
            or save_label
            or include_collection
        ):
            from .paperpile import PaperpileLibrary, paperpile_connection

            pp = MultiServerMCPClient({"paperpile": paperpile_connection()})
            library = PaperpileLibrary(
                await stack.enter_async_context(pp.session("paperpile"))
            )
            if paperpile_profile is not None and not add_to_previous:
                if profile := await library.profile(paperpile_profile or None):
                    request = f"{request}\n\n{profile}".strip()
                    logger.info("paperpile: added library interest profile to request")
            if include_collection:
                name, refs = await library.collection_refs(include_collection)
                if added_since:
                    cut = dt.datetime.fromisoformat(added_since).timestamp()
                    n_all = len(refs)
                    refs = [r for r in refs if (r.get("created_at") or 0) >= cut]
                    logger.info(
                        "paperpile: %d of %d in %r added to the library since %s",
                        len(refs),
                        n_all,
                        name,
                        added_since,
                    )
                label = f"Paperpile: {name}"
                no_id = []
                for ref in refs:
                    if ident := reference_identifier(ref):
                        manual_ids.append(ident)
                        origin[ident] = label
                    else:
                        no_id.append(ref.get("title") or "(untitled)")
                logger.info(
                    "paperpile: %d paper(s) from %r; %d without DOI/arXiv id/URL skipped",
                    len(refs) - len(no_id),
                    name,
                    len(no_id),
                )
                if skip_research and not request.strip():
                    from .paperpile import profile_text

                    # Scope from exactly the papers under review (after filters).
                    request = profile_text(refs[:100], f"label/folder '{name}'")
        if skip_research and not request.strip():
            raise ValueError("Nothing to review: no request and no added papers.")
        return await _run_review(
            request,
            output_path,
            from_date,
            manual_ids,
            refresh,
            add_to_previous,
            library,
            save_label,
            origin,
            skip_research,
        )


async def _run_review(
    request: str,
    output_path: str,
    from_date: str | None,
    manual_ids: list[str] | None,
    refresh: bool,
    add_to_previous: bool,
    library,
    save_label: str | None,
    origin: dict[str, str],
    skip_research: bool,
) -> ReviewState:
    from_date = from_date or (dt.date.today() - dt.timedelta(days=365)).isoformat()
    if add_to_previous:
        last = cache.load_last()
        if last is None:
            raise ValueError(
                "No previous review to add papers to. Include research topics or a "
                "CV URL in the request."
            )
        key, entry = last
        request = entry["request"]
        logger.info("intake: adding to previous review: %r", request[:100])
    elif skip_research:
        key = ""  # added papers only: don't cache an empty search as research
    else:
        key = cache.research_key(request, review_focus())
    state: ReviewState = {
        "request": request,
        "from_date": from_date,
        "output_path": output_path,
        "manual_ids": list(manual_ids or []),
        "refresh": refresh,
        "cache_key": key,
        "manual_origin": origin,
        "skip_research": skip_research,
    }
    hit = (
        None
        if not key or (refresh and not add_to_previous)
        else cache.load_research(key)
    )
    if hit:
        n = sum(len(c["papers"]) for c in hit["candidates"])
        logger.info(
            "research: using cache from %s (%d candidates, model %s, since %s); "
            "--refresh to search again",
            hit.get("saved"),
            n,
            hit.get("model"),
            hit.get("from_date"),
        )
        state.update(
            scope=hit["scope"],
            candidates=hit["candidates"],
            from_date=hit.get("from_date") or from_date,
            cached=True,
        )
    client = MultiServerMCPClient({"papers": paper_server_connection()})
    async with client.session("papers") as session:
        tools = await load_mcp_tools(session)
        logger.info("MCP tools: %s", [t.name for t in tools])
        graph = build_graph(tools, library, save_label)
        return await graph.ainvoke(state)
