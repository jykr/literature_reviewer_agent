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

"""Parallel research fan-out (SPEC §2.2): one searcher per subfield cluster.

Fixed-slot design: ``N_SLOTS`` researcher agents run concurrently in a
``ParallelAgent``. A ``before_agent_callback`` splits the scope's clusters into
per-slot state keys so slot k researches ``cluster_k`` (or no-ops if there are
fewer clusters than slots). Each searcher writes to its own ``output_key`` to
avoid parallel write races.

Searchers use the pooled scholarly tools in ``app.sources`` (OpenAlex, Europe
PMC, PubMed, arXiv, bioRxiv/medRxiv), shared with the LangChain variant, so
the pipeline is model-agnostic. Their instructions come from the same
``lc_app.core.RESEARCH_PROMPT`` (credibility weighting, review focus).

Integrity (SPEC §2.5) is enforced in code, not just asked for: every record a
tool returns is kept in state (``record_tool_results``), and after the
researcher finishes, ``verify_researcher`` keeps only papers matching one of
those records and copies ids, venue status, citations and h-index from it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

from google.adk.agents import Agent, ParallelAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext

from app.byo_review import _json_loads_loose
from app.models import build_model, safety_config
from app.sources import RESEARCH_TOOLS, record_tool_results
from lc_app.core import RESEARCH_PROMPT, verify_candidates
from lc_app.focus import focus_block

logger = logging.getLogger(__name__)

N_SLOTS = 5


def _scope(state) -> dict:
    scope = state.get("scope")
    if isinstance(scope, str):
        try:
            scope = json.loads(scope)
        except json.JSONDecodeError:
            scope = {}
    return scope if isinstance(scope, dict) else {}


def default_from_date() -> str:
    return (dt.date.today() - dt.timedelta(days=365)).isoformat()


def split_clusters(callback_context: CallbackContext) -> None:
    """Populate state['cluster_0'..'cluster_{N-1}'] from state['scope'].

    With ``skip_research`` (review only added papers) every slot is left empty.
    """
    state = callback_context.state
    clusters = [] if state.get("skip_research") else _scope(state).get("clusters", [])
    for k in range(N_SLOTS):
        state[f"cluster_{k}"] = (
            json.dumps(clusters[k], ensure_ascii=False) if k < len(clusters) else ""
        )
        state[f"tool_records_researcher_{k}"] = []


OUTPUT_FORMAT = """
Output ONLY a JSON array (no prose, no code fences). Each element:
{"title": "...", "authors": "First A., ..., Senior Z.", "institution": "...",
 "venue": "...", "date": "...", "url": "...", "doi": "...",
 "algo": "core method", "data": "data used (train + eval, scale)",
 "model": "how the method works", "evaluation": "metrics, eval data, limits",
 "question": "key question", "app_tags": ["..."], "method_tags": ["..."]}
"""


def research_instruction(k: int):
    """InstructionProvider for slot k (formats the shared prompt from state)."""

    def provide(ctx: ReadonlyContext) -> str:
        raw = ctx.state.get(f"cluster_{k}") or ""
        if not raw:
            return "There is no cluster for this slot. Output exactly `[]` and stop."
        cluster = json.loads(raw)
        return (
            RESEARCH_PROMPT.format(
                profile=_scope(ctx.state).get("profile", ""),
                name=cluster.get("name", ""),
                keywords=", ".join(cluster.get("keywords", [])),
                from_date=ctx.state.get("from_date") or default_from_date(),
                focus=focus_block("research", use_default=False),
            )
            + OUTPUT_FORMAT
        )

    return provide


def verify_researcher(callback_context: CallbackContext) -> None:
    """after_agent_callback: keep only papers some tool returned (tool truth)."""
    state = callback_context.state
    name = callback_context.agent_name
    k = int(name.rsplit("_", 1)[-1])
    raw = state.get(f"papers_{k}")
    try:
        found = _json_loads_loose(raw) if isinstance(raw, str) and raw.strip() else raw
    except (json.JSONDecodeError, ValueError):
        found = []
    found = [p for p in (found or []) if isinstance(p, dict)]
    papers, dropped = verify_candidates(found, state.get(f"tool_records_{name}") or [])
    state[f"papers_{k}"] = json.dumps(papers, ensure_ascii=False)
    if found:
        logger.info(
            "research[%d]: %d papers kept, %d unverified dropped", k, len(papers), dropped
        )


def create_researcher(k: int, model=None) -> Agent:
    """Factory for researcher slot k (call it, don't pass the reference)."""
    return Agent(
        name=f"researcher_{k}",
        model=model or build_model(),
        description=f"Searches for recent papers in subfield cluster {k}.",
        instruction=research_instruction(k),
        tools=RESEARCH_TOOLS,
        output_key=f"papers_{k}",
        generate_content_config=safety_config(),
        after_tool_callback=record_tool_results,
        after_agent_callback=verify_researcher,
    )


def create_research_fanout(model=None) -> ParallelAgent:
    return ParallelAgent(
        name="research_fanout",
        description="Runs one searcher per subfield cluster, concurrently.",
        sub_agents=[create_researcher(k, model) for k in range(N_SLOTS)],
        before_agent_callback=split_clusters,
    )


research_fanout = create_research_fanout()
