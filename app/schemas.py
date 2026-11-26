# ruff: noqa
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

"""Pydantic data models shared across the review pipeline.

These mirror the data model in paper-review-SPEC.md (§3). The ranker
agent emits a ``ReviewQueue`` via ``output_schema``; the HTML assembler consumes
it to render the self-contained ``literature-reviewer.html`` app.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class Cluster(BaseModel):
    """A subfield the researcher works in, used to fan out research (§2.2)."""

    name: str = Field(description="Short subfield label, e.g. 'Variant effect prediction'.")
    keywords: list[str] = Field(
        default_factory=list,
        description="3-8 search keywords/terms characterising this subfield.",
    )


class Scope(BaseModel):
    """Output of the scope agent: the researcher profile + clusters to research."""

    profile: str = Field(
        description="2-4 sentence summary of the researcher: subfields, methods, seniority. "
        "Drives the per-paper relevance score."
    )
    clusters: list[Cluster] = Field(
        description="Up to 5 subfield clusters to research in parallel."
    )


NOVELTY = ("Application", "Adaptation", "Algorithm")
# Older reviews: a 0-100 score, an Application/Adaptation/"New method" label,
# or three Low/Medium/High ratings (the ML rating maps onto this scale).
_LEGACY = {
    "application": "Application",
    "resource": "Application",
    "adaptation": "Adaptation",
    "algorithm": "Algorithm",
    "new method": "Algorithm",
    "low": "Application",
    "medium": "Adaptation",
    "high": "Algorithm",
}


def novelty(value) -> str:
    """One kind: 'Application' | 'Adaptation' | 'Algorithm' (or "" when unknown)."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _LEGACY:
            return _LEGACY[v]
        if v.startswith("application"):  # "Application/resource"
            return "Application"
        try:
            value = float(v)
        except ValueError:
            return ""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return NOVELTY[0 if value < 34 else 1 if value < 67 else 2]
    return ""


def novelty_kinds(value) -> list[str]:
    """The kinds of contribution that apply, in scale order (one or more)."""
    items = value if isinstance(value, list | tuple) else [value]
    found = {novelty(v) for v in items} - {""}
    return [k for k in NOVELTY if k in found]


class Approach(BaseModel):
    """Detailed 'Main approach' narrative (SPEC §3c)."""

    algo: str = Field(description="Short label of the core method/algorithm.")
    nov: list[str] = Field(
        description="Every kind of contribution that applies, one or more of "
        "'Application' (existing methods applied to new data or a new problem, or a "
        "dataset/benchmark/tool resource), 'Adaptation' (existing components "
        "modified or combined, e.g. a pretrained encoder connected to an LLM "
        "through a projection layer, however large the effort) and 'Algorithm' (a "
        "fundamentally new learning algorithm, objective or architectural principle)."
    )
    aim: str = Field(description="Goal + the limitation it addresses.")
    question: str = Field(description="The key question the paper addresses.")
    data: str = Field(description="What data the model consumes (train + eval, scale).")
    model: str = Field(description="The base model or architecture the work builds on.")
    method: str = Field(
        description="Methodological novelty: the specific technical mechanism the "
        "paper introduces or changes relative to prior work, stated concretely (what "
        "is combined, projected, trained or optimized, and how)."
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("nov") and data.get("nov_ml"):
                data["nov"] = data["nov_ml"]
            data.setdefault("nov", [])
            data.setdefault("method", "")
        return data

    @field_validator("nov", mode="before")
    @classmethod
    def _novelty(cls, v):
        return novelty_kinds(v)  # also accepts one value from older reviews


class Tags(BaseModel):
    """Application (biological problem) + Method (computational class) tags (§3b)."""

    app: list[str] = Field(description="Application tags, e.g. ['Variant effect'].")
    method: list[str] = Field(description="Method tags, e.g. ['DNA language model'].")

    @model_validator(mode="before")
    @classmethod
    def _strip_keys(cls, data):
        # Models occasionally emit " method" (leading space) for this key.
        if isinstance(data, dict):
            return {k.strip() if isinstance(k, str) else k: v for k, v in data.items()}
        return data


class Paper(BaseModel):
    """One reviewed paper. Combines DATA, TAGS and APPROACH from SPEC §3."""

    rank: int = Field(description="1-based rank; primary key and default sort.")
    title: str
    authors: str = Field(description="'First A., ..., Senior Z.'")
    institution: str
    venue: str
    date: str
    url: str
    insights: str = Field(
        description="Evaluation text with the four inline labels verbatim: "
        "'New metric:' ... 'New eval data:' ... 'Design & novelty:' ... 'Eval limits:' ..."
    )
    results: list[str] = Field(
        default_factory=list,
        description="Main results of the paper as concise bullet points (SPEC §2.3). "
        "For a QUANTITATIVE result, state the task, the data/benchmark, the metric, "
        "and the value, e.g. 'Variant effect (ClinVar): 0.91 auROC, +0.04 over Enformer'. "
        "Prefer characterizing results you cannot confirm over quoting exact numbers.",
    )
    limitation: str
    resources: str = Field(default="", description="Bare URLs to code/data/blog; auto-linkified.")
    comments: str = Field(
        description="Objective appraisal of the contribution and the strength of its "
        "evidence, with its main caveat. Impersonal; no hype."
    )
    relevance: str = Field(
        description="Which of the profile's interests the paper bears on, and how. "
        "Impersonal; name the interests, never clusters or indices."
    )
    impact: int = Field(ge=1, le=10, description="Field impact score.")
    rel: int = Field(ge=1, le=10, description="Relevance-to-CV score.")
    cats: list[int] = Field(
        default_factory=list,
        description="0-based indices of the scope clusters (categories) this paper "
        "matches; drives coverage scoring (SPEC §3d). More categories ranks higher.",
    )
    tags: Tags
    approach: Approach


class ReviewQueue(BaseModel):
    """Final ranked queue emitted by the ranker agent."""

    papers: list[Paper] = Field(description="Papers, ranked (rank=1 is best).")


class LabeledPaper(Paper):
    """Render-time Paper with the 'manually added' label (not ranker output,
    so the ranker's structured-output schema stays unchanged). Set by the
    LangChain variant and the ADK pipeline's post-rank stages."""

    manual: bool = False
    evidence: str = ""  # "fulltext" | "abstract" (set by the deepen stage)
    in_library: bool = False  # already in the user's Paperpile library
    origin: str = ""  # where an added paper came from, e.g. "Paperpile: to read"
    doi: str = ""  # from the tool record (used to save the paper to Paperpile)


class LabeledQueue(ReviewQueue):
    papers: list[LabeledPaper]
