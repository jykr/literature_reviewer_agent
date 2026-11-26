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

from lc_app.core import extract_papers
from lc_app.models import build_model


@dataclass
class ParsedRequest:
    request: str  # topics / CV text for the scope stage ("" when add_to_previous)
    manual_ids: list[str] = field(default_factory=list)
    add_to_previous: bool = False


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
