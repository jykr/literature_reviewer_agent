"""Review logic shared by the LangGraph app (lc_app.graph) and the ADK app.

LangChain-free on purpose: the ADK agent imports this without the
``langchain`` extra. Covers tool-truth verification, credibility and
added-paper ranking rules, coverage-first finalization, full-text grounding,
and paper-identifier parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from app.schemas import Approach
from lc_app.paper_sources import (
    _ARXIV_ID,
    normalize_doi,
    normalize_identifier,
    title_key,
)

# ---------------------------------------------------------------------------
# Ranking rules and tool-truth verification
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

WRITING_STYLE = """

Writing style for every text field (relevance, comments, insights, results,
limitation, aim, approach):
- Impersonal and objective: third person, no "you"/"your" and no address to
  the reader. Refer to the profile's interests by their topic names (e.g.
  "variant effect prediction"); never mention clusters, categories, indices
  or their numbers.
- Scientific and skeptical: state the paper's findings as reported claims
  ("the authors report ...", "reported to ..."), not as established fact.
  Do not adopt the paper's promotional language ("state-of-the-art",
  "breakthrough", "unprecedented", "powerful", "revolutionary", "paves the
  way") and do not add enthusiasm of your own.
- Separate evidence from interpretation: name what supports a claim
  (datasets, baselines, ablations, independent validation, sample size) and
  what is not shown or not compared.
- `comments`: a neutral appraisal of what the work contributes and how strong
  its evidence is, including its main caveat. No praise or hype.
- `relevance`: one plain sentence on which of the profile's interests the
  paper bears on and how, without value judgments.

Be concise: the review is scanned, not read. Hard limits (words):
- insights: each of the four labels gets one short phrase (<= 12 words);
  write "none" when a label does not apply.
- results: the 2-3 most important results, one sentence each (<= 25
  words); quantitative ones keep task, dataset, metric, value, baseline.
- limitation: one sentence (<= 25 words).
- comments: an exception to brevity: three or four sentences (about 70-100
  words) of appraisal, after any preprint prefix.
- aim: one sentence (<= 25 words). question: <= 20 words. data: <= 20 words.
  model: <= 12 words. method: at most two sentences (<= 45 words) naming the
  concrete mechanism.
No lists of every experiment, no restating the abstract, no filler.
"""


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


def paper_key(p: dict) -> str:
    return normalize_doi(p.get("doi")) or title_key(p.get("title", ""))


MANUAL_RANK_RULES = """

Manually added papers (the researcher added these by hand; JSON):
{papers}
Every manually added paper MUST appear in `papers`, ranked by the same rules as
the searched candidates. They do NOT count toward the 12-paper limit. If one
duplicates a searched candidate, list it once.
"""


def finalize_ranked(
    papers: list[dict],
    n_clusters: int,
    found_in: dict[str, set[int]],
    added: list[dict],
    batched: bool = False,
) -> list[dict]:
    """Coverage-first order (SPEC §3d) with added-paper labels, in place.

    The clusters whose researcher actually surfaced a paper are a floor under
    the ranker's ``cats``. Sorted by coverage, then the ranker's order (or,
    when added papers were ranked in batches whose ranks aren't comparable,
    by rel + impact). Preprints get the "Preprint (not peer-reviewed)." prefix.
    """
    added_keys = {paper_key(m): m.get("origin", "") for m in added}
    for p in papers:
        k = paper_key(p)
        cats = {c for c in p.get("cats", []) if 0 <= c < n_clusters}
        p["cats"] = sorted(cats | found_in.get(k, set()))
        p.pop("sources", None)
        p["manual"] = k in added_keys
        p["origin"] = added_keys.get(k, "")
        if p.get("is_preprint") and not p.get("comments", "").startswith("Preprint"):
            p["comments"] = "Preprint (not peer-reviewed). " + p.get("comments", "")
    if batched:
        papers.sort(
            key=lambda p: (-len(p["cats"]), -(p["rel"] + p["impact"]), p["rank"])
        )
    else:
        papers.sort(key=lambda p: (-len(p["cats"]), p["rank"]))
    for i, p in enumerate(papers, 1):
        p["rank"] = i
    return papers


# ---------------------------------------------------------------------------
# Full-text deepening (selected papers only)
# ---------------------------------------------------------------------------


class DeepAnalysis(BaseModel):
    """Fields re-derived from full text. Rank, scores, coverage, tags and
    relevance stay as the ranker set them."""

    insights: str = Field(
        description='"New metric: <...> New eval data: <...> Design & novelty: <...> '
        'Eval limits: <...>" (four labels verbatim, in order, one string).'
    )
    results: list[str] = Field(
        description="The 2-3 most important results, one sentence each (<= 25 "
        "words). Quantitative ones name task, dataset/benchmark, metric, value and "
        "the baseline compared against."
    )
    limitation: str
    comments: str = Field(
        description="Objective appraisal of the contribution and the strength of "
        "its evidence as seen in the full text, with the main caveat."
    )
    resources: str = Field(
        default="",
        description="Space-separated code/data URLs from the provided links only.",
    )
    approach: Approach


DEEPEN_PROMPT = (
    """\
You are deepening the analysis of ONE paper already selected for a literature
review, now using excerpts of its full text (results, methods, discussion,
tables, links). Researcher profile: {profile}

Rewrite these fields from the full text:
- insights: the four labels verbatim, in order: "New metric: ... New eval data:
  ... Design & novelty: ... Eval limits: ..."
- results: the 2-3 most important results. For a quantitative result name the
  task, the dataset/benchmark, the metric, the value and what it is compared
  against.
  Copy every number exactly as it appears in the excerpts. Never compute,
  round or estimate a number (no derived differences).
- limitation: the main limitation, stated by the authors or evident from the
  evaluation.
- comments: an objective appraisal of the contribution and how strong its
  evidence is in the full text (baselines, ablations, validation), with the
  main caveat. Do not comment on the abstract or on missing text.
- resources: code/data URLs, ONLY from the provided `links` list.
- approach: algo; nov, the list of every kind of contribution that applies
  (one or more of "Application": existing methods applied, or a
  dataset/benchmark/tool resource; "Adaptation": existing components
  modified or combined, e.g. a pretrained encoder projected into an LLM's
  token space; "Algorithm": a fundamentally new learning algorithm or
  objective), judged from what was done, not from the paper's claims; aim
  ("<goal>. Limitation addressed: <gap>."); question; data; model (the base
  model or architecture built on); method (the methodological novelty: the
  specific mechanism added or changed relative to prior work, stated
  concretely from the methods section: what is combined, projected, trained
  or optimized, and how).

Base every statement on the excerpts. Where they don't cover a field, keep
the current (abstract-based) value given below.
"""
    + WRITING_STYLE
)

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


def apply_deep(p: dict, analysis: dict, ft: dict) -> tuple[dict, int]:
    """Merge a DeepAnalysis into paper ``p``; return (cache entry, n dropped).

    Result bullets quoting numbers absent from the excerpts are dropped (the
    abstract-based ones stay if none survive), and resources are limited to
    links found in the paper or already listed.
    """
    d = dict(analysis)
    old = p.get("comments") or ""
    prefix = "Preprint (not peer-reviewed). "
    if (
        d.get("comments")
        and old.startswith(prefix.strip())
        and not d["comments"].startswith(prefix.strip())
    ):
        d["comments"] = prefix + d["comments"]
    if not d.get("comments"):
        d.pop("comments", None)  # keep the ranker's
    source = "\n".join(ft["sections"] + ft["tables"])
    d["results"], dropped = grounded_results(d["results"], source)
    if not d["results"]:
        d["results"] = p.get("results") or []
    allowed = set(ft["links"]) | set((p.get("resources") or "").split())
    urls = [u for u in d["resources"].split() if u in allowed]
    d["resources"] = " ".join(dict.fromkeys(urls + (p.get("resources") or "").split()))
    return {**d, "evidence": "fulltext", "fulltext_source": ft["source"]}, dropped


# ---------------------------------------------------------------------------
# Paper identifiers
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


def paper_dois(papers: list[dict], pool: list[dict]) -> dict[int, str]:
    """rank -> DOI for queue papers, from the tool records in ``pool`` they
    came from (arXiv-only papers get arXiv's DataCite DOI)."""
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


def doi_from_link(doi: str = "", url: str = "") -> str:
    """A DOI from a DOI field or a paper link (arXiv links -> arXiv's DOI)."""
    if d := normalize_doi(doi):
        return d
    ident = normalize_identifier(url) if url else ""
    if ident.startswith("10."):
        return normalize_doi(ident)
    return arxiv_doi(ident) if ident and _ARXIV_ID.match(ident) else ""


def arxiv_doi(arxiv_id: str) -> str:
    """arXiv's DataCite DOI, version dropped: 2608.24688v2 -> 10.48550/arxiv.2608.24688."""
    base = re.sub(r"v\d+$", "", arxiv_id.split(":")[-1])
    return f"10.48550/arxiv.{base.lower()}"


_URL = re.compile(r"https?://[^\s<>\"'),\]]+")
_BARE_DOI = re.compile(r"(?<![\w/.])10\.\d{4,9}/[^\s,;<>\"')\]]+")
_BARE_ARXIV = re.compile(
    r"(?<![\w/.])(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?(?![\w.])", re.I
)
_PMID = re.compile(r"\bPMID:?\s*(\d{5,9})\b", re.I)


def _is_paper_url(url: str) -> bool:
    return normalize_identifier(url) != url.strip().rstrip("/")


def _dedupe_key(ident: str) -> str:
    return re.sub(r"v\d+$", "", ident.lower())  # arXiv versions are one paper


def extract_papers(text: str) -> tuple[str, list[str]]:
    """Pull paper identifiers out of ``text``; return (remaining text, ids)."""
    found: list[str] = []

    def take(m: re.Match, value: str | None = None) -> str:
        found.append(normalize_identifier(value or m.group(0)))
        return " "

    def url_sub(m: re.Match) -> str:
        url = m.group(0).rstrip(".")
        return take(m, url) if _is_paper_url(url) else m.group(0)

    text = _URL.sub(url_sub, text)
    text = _PMID.sub(lambda m: take(m, m.group(1)), text)
    text = _BARE_DOI.sub(lambda m: take(m, m.group(0).rstrip(".")), text)
    text = _BARE_ARXIV.sub(lambda m: take(m, m.group(0).split(":")[-1]), text)

    ids, seen = [], set()
    for i in found:
        if (k := _dedupe_key(i)) not in seen:
            seen.add(k)
            ids.append(i)
    remainder = re.sub(r"[\s,;]+", " ", text).strip(" ,;:")
    return remainder, ids
