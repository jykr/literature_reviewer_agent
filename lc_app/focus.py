"""Review focus: a standing lens applied to scope, research and rank.

A CONFIG knob (env ``LC_REVIEW_FOCUS`` or ``--focus``), LangChain variant only:
the shared ADK prompts in ``app/`` stay field-general. The focus sharpens what
the user asked for; it never replaces their CV/topics.

    unset              -> DEFAULT_FOCUS (frontier ML for genomics / biology)
    "<text>"           -> that lens
    "" or "none"       -> no lens (same behavior as the ADK app)
"""

from __future__ import annotations

import os

DEFAULT_FOCUS = (
    "Frontier machine learning for genomics and biology: new model architectures, "
    "biological foundation models (DNA, RNA, protein, single-cell, multimodal), "
    "scaling and training methods, and rigorous benchmarks or evaluations of such "
    "models. Deprioritize papers that only apply existing off-the-shelf tools to a "
    "new dataset."
)

_STAGE_GUIDANCE = {
    "scope": (
        "Shape the clusters and their keywords through this lens: name each cluster "
        "by method/model class plus application where possible, and pick keywords "
        "that surface work fitting the focus. Derive clusters from the user's input; "
        "the focus sharpens it and never replaces it."
    ),
    "research": (
        "Prefer papers that fit this focus and skip ones outside it. Phrase queries "
        "with the method terms the focus implies. ML-heavy work often appears on "
        "arXiv/bioRxiv before journals and conferences: search those too, but "
        "follow the credibility rule and prefer peer-reviewed versions."
    ),
    "rank": (
        "Coverage stays the primary rank driver. Within a coverage level, rank "
        "papers that better fit this focus higher, and reflect the fit in `rel` "
        "and `comments`."
    ),
}


def review_focus() -> str:
    """The active focus text, or "" when disabled."""
    focus = os.environ.get("LC_REVIEW_FOCUS", DEFAULT_FOCUS).strip()
    return "" if focus.lower() == "none" else focus


def focus_block(stage: str) -> str:
    """Prompt text for one stage ('scope' | 'research' | 'rank'), "" if no focus."""
    focus = review_focus()
    if not focus:
        return ""
    return f"\n\nReview focus: {focus}\n{_STAGE_GUIDANCE[stage]}\n"
