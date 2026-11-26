"""Natural-language intake: split a free-form request into review inputs.

    "recent DNA language model papers, also include arxiv.org/abs/2611.27924"
        -> new review on the topics + 1 manually added paper
    "add these to my review: https://arxiv.org/abs/...,https://doi.org/..."
    "https://arxiv.org/abs/2611.27924v1?ref=newsletter"   (papers only)
        -> add the papers to the PREVIOUS review (cached research reused)

Paper identifiers (arXiv / DOI / PubMed / bioRxiv URLs, bare DOIs, arXiv ids,
PMIDs) are extracted deterministically. Remaining URLs stay in the request and
are read as CV / lab pages. A model call is made only when the message mixes
papers with other text, to decide "new review" vs "add to previous".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from lc_app.models import build_model
from lc_app.paper_sources import normalize_identifier

_URL = re.compile(r"https?://[^\s<>\"'),\]]+")
_BARE_DOI = re.compile(r"(?<![\w/.])10\.\d{4,9}/[^\s,;<>\"')\]]+")
_BARE_ARXIV = re.compile(
    r"(?<![\w/.])(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?(?![\w.])", re.I
)
_PMID = re.compile(r"\bPMID:?\s*(\d{5,9})\b", re.I)


@dataclass
class ParsedRequest:
    request: str  # topics / CV text for the scope stage ("" when add_to_previous)
    manual_ids: list[str] = field(default_factory=list)
    add_to_previous: bool = False


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


class _Intent(BaseModel):
    add_to_previous: bool = Field(
        description="True if the user wants the papers added to their EXISTING/previous "
        "review rather than starting a new one on new topics."
    )
    research_request: str = Field(
        description="For a new review: the user's research interests, topics and any "
        "CV/site URLs, copied in the user's own words, minus instructions about the "
        "extracted papers. Empty when add_to_previous is true."
    )


_INTENT_SYS = """\
You route requests for a literature-review tool. Paper links were already
extracted from the user's message ({n} paper(s)) and replaced by spaces.
Decide whether the user (a) wants those papers added to their previous review,
or (b) is describing a new review (topics, research interests, a CV/site URL),
possibly with those papers included. Copy topic wording verbatim; do not
paraphrase or add topics."""


async def parse_request(text: str) -> ParsedRequest:
    remainder, ids = extract_papers(text)
    if not ids:
        return ParsedRequest(request=text.strip())
    # No words beyond the links: papers for the previous review.
    if len(re.findall(r"[a-zA-Z]{3,}", remainder)) == 0:
        return ParsedRequest(request="", manual_ids=ids, add_to_previous=True)
    intent: _Intent = (
        await build_model(max_tokens=1024)
        .with_structured_output(_Intent)
        .ainvoke(
            [SystemMessage(_INTENT_SYS.format(n=len(ids))), HumanMessage(remainder)]
        )
    )
    if intent.add_to_previous or not intent.research_request.strip():
        return ParsedRequest(request="", manual_ids=ids, add_to_previous=True)
    return ParsedRequest(request=intent.research_request.strip(), manual_ids=ids)
