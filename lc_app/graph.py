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
from app.schemas import (  # noqa: F401 (re-exported)
    LabeledPaper,
    LabeledQueue,
    ReviewQueue,
    Scope,
)
from app.security import _url_is_safe
from lc_app import cache
from lc_app.core import (  # shared with the ADK app
    CREDIBILITY_RULES,
    DEEPEN_PROMPT,
    MANUAL_RANK_RULES,
    RESEARCH_PROMPT,
    WRITING_STYLE,
    DeepAnalysis,
    _apply_tool_truth,
    _fulltext_ids,
    _json_payload,
    apply_deep,
    finalize_ranked,
    reference_identifier,
    verify_candidates,
)
from lc_app.core import paper_dois as _core_paper_dois
from lc_app.core import paper_key as _paper_key
from lc_app.focus import focus_block, review_focus
from lc_app.models import build_model, model_id
from lc_app.paper_sources import (
    normalize_doi,
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
    add_to_previous: bool  # server input: reuse the last review (read by prepare)
    profile: str  # researcher profile to use verbatim instead of the model's
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
    scope = scope.model_dump()
    if state.get("profile"):  # the user's own wording, verbatim
        scope["profile"] = state["profile"]
    return {"scope": scope}


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
        if not task.get("refresh") and (
            hit := await asyncio.to_thread(cache.load_manual, ident)
        ):
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
        await asyncio.to_thread(cache.save_manual, ident, paper)
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
        # File I/O off the event loop (langgraph dev rejects blocking calls).
        await asyncio.to_thread(
            cache.save_research,
            state["cache_key"],
            state.get("request", ""),
            scope,
            copy.deepcopy(state.get("candidates", [])),
            from_date=state.get("from_date"),
            focus=review_focus(),
            model=model_id(),
        )
    if state.get("cache_key") and not state.get("skip_research"):
        await asyncio.to_thread(cache.save_last, state["cache_key"])
    manual = list({_paper_key(m): m for m in state.get("manual", [])}.values())
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
    finalize_ranked(
        kept,
        len(scope.get("clusters", [])),
        found_in,
        manual,
        batched=len(batches) > 1,
    )
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
        + WRITING_STYLE
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


# ---------------------------------------------------------------------------
# Deepen: full-text analysis of the SELECTED papers only (after ranking)
# ---------------------------------------------------------------------------


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
            if not state.get("refresh") and (
                hit := await asyncio.to_thread(cache.load_deep, key)
            ):
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
                for k in (
                    "insights",
                    "results",
                    "limitation",
                    "comments",
                    "resources",
                    "approach",
                )
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
            hit, dropped = apply_deep(p, analysis.model_dump(), ft)
            stats["dropped_results"] += dropped
            await asyncio.to_thread(cache.save_deep, key, hit)
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


def paper_dois(papers: list[dict], state: ReviewState) -> dict[int, str]:
    """rank -> DOI for queue papers, from this run's tool records."""
    pool = [p for c in state.get("candidates") or [] for p in c.get("papers", [])]
    return _core_paper_dois(papers, pool + (state.get("manual") or []))


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


def build_graph(
    tools: list[BaseTool],
    library=None,
    save_label: str | None = None,
    prepare=None,
    input_schema=None,
):
    """``prepare`` (optional) runs first to turn a bare request into the full
    input state, for servers whose callers send only a few fields."""
    g = StateGraph(ReviewState, input_schema=input_schema or ReviewState)
    g.add_node("scope", scope_node)
    g.add_node("no_scope", no_scope_node)
    g.add_node("research", make_research_node(tools))
    g.add_node("manual", make_manual_node(tools))
    g.add_node("rank", rank_node)
    g.add_node("deepen", make_deepen_node(tools))
    g.add_node("render", render_node)
    if prepare is not None:
        g.add_node("prepare", prepare)
        g.add_edge(START, "prepare")
        g.add_edge("prepare", "scope")
    else:
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
    """stdio connection that launches the paper-sources MCP server in a subprocess."""
    env = {k: os.environ[k] for k in _SERVER_ENV if os.getenv(k)}
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "paper_sources_mcp.server"],
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
    profile: str = "",
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
            profile,
        )


def initial_state(
    request: str,
    output_path: str,
    from_date: str | None = None,
    manual_ids: list[str] | None = None,
    refresh: bool = False,
    add_to_previous: bool = False,
    origin: dict[str, str] | None = None,
    skip_research: bool = False,
    profile: str = "",
) -> ReviewState:
    """The graph's input: request + defaults, pre-filled from the research
    cache on a hit. Shared by run_review and the ``langgraph dev`` server."""
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
        # Profile joins the key only when set, so existing caches stay valid.
        key = cache.research_key(
            f"{request}\x1f{profile}" if profile else request, review_focus()
        )
    state: ReviewState = {
        "request": request,
        "from_date": from_date,
        "output_path": output_path,
        "manual_ids": list(manual_ids or []),
        "refresh": refresh,
        "cache_key": key,
        "manual_origin": dict(origin or {}),
        "skip_research": skip_research,
        "profile": profile,
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
    return state


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
    profile: str = "",
) -> ReviewState:
    state = initial_state(
        request,
        output_path,
        from_date,
        manual_ids,
        refresh,
        add_to_previous,
        origin,
        skip_research,
        profile,
    )
    client = MultiServerMCPClient({"papers": paper_server_connection()})
    async with client.session("papers") as session:
        tools = await load_mcp_tools(session)
        logger.info("MCP tools: %s", [t.name for t in tools])
        graph = build_graph(tools, library, save_label)
        return await graph.ainvoke(state)
