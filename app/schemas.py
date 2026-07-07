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

These mirror the data model in compbio-paper-review-SPEC.md (§3). The ranker
agent emits a ``ReviewQueue`` via ``output_schema``; the HTML assembler consumes
it to render the self-contained ``literature-reviewer.html`` app.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class Approach(BaseModel):
    """Detailed 'Main approach' narrative (SPEC §3c)."""

    algo: str = Field(description="Short label of the core method/algorithm.")
    nov: int = Field(ge=0, le=100, description="0=pure engineering ... 100=novel algorithm.")
    aim: str = Field(description="Goal + the limitation it addresses.")
    data: str = Field(description="What data the model consumes (train + eval, scale).")
    model: str = Field(description="The model/algorithm: if novel, how; else what was done.")
    bio: str = Field(description="The biological question the paper answers.")


class Tags(BaseModel):
    """Application (biological problem) + Method (computational class) tags (§3b)."""

    app: list[str] = Field(description="Application tags, e.g. ['Variant effect'].")
    method: list[str] = Field(description="Method tags, e.g. ['DNA language model'].")


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
    comments: str = Field(description="Reviewer significance note.")
    relevance: str = Field(description="Why it matters to THIS researcher.")
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
