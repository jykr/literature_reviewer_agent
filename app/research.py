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

Searchers use the scholarly FunctionTools in ``search_tools`` (OpenAlex, Europe
PMC, arXiv) rather than Gemini's ``google_search`` grounding — so the pipeline is
model-agnostic (works on Claude/GPT too) and returns structured, verifiable
metadata plus citation counts for ranking.

The dynamic-fan-out upgrade (a custom BaseAgent that spawns exactly one searcher
per discovered cluster) is described at the bottom of this file.
"""

from __future__ import annotations

import json

from google.adk.agents import Agent, ParallelAgent
from google.adk.agents.callback_context import CallbackContext

from app.models import build_model, safety_config
from app.search_tools import SCHOLARLY_TOOLS

N_SLOTS = 5


def split_clusters(callback_context: CallbackContext) -> None:
    """Populate state['cluster_0'..'cluster_{N-1}'] from state['scope']."""
    scope = callback_context.state.get("scope")
    if isinstance(scope, str):
        try:
            scope = json.loads(scope)
        except json.JSONDecodeError:
            scope = {}
    clusters = (scope or {}).get("clusters", []) if isinstance(scope, dict) else []
    for k in range(N_SLOTS):
        callback_context.state[f"cluster_{k}"] = (
            json.dumps(clusters[k], ensure_ascii=False) if k < len(clusters) else ""
        )


RESEARCH_INSTRUCTION = """\
You research recent computational-biology papers for ONE subfield cluster.

The cluster to research (JSON, may be empty) is:
{cluster_%d}

If the cluster is empty, output exactly `[]` and stop.

Otherwise, find 4-6 REAL, verifiable papers matching this cluster's keywords,
published from mid-2025 to now (max ~1 year old). Use the search tools:
- `search_openalex(query, from_date, max_results)` — your PRIMARY source; broad
  coverage plus citation metrics. Pass from_date="2025-06-01".
- `search_europepmc(query, max_results)` — biomedical papers and preprints.
- `search_arxiv(query, max_results)` — q-bio / CS preprints.
Issue a few focused queries combining the cluster keywords; call more than one
tool when useful. Prefer papers with a resolvable DOI or preprint URL, and carry
`cited_by_count`/`fwci` from OpenAlex when present (used later for ranking).

Integrity rule (SPEC §2.5): only include papers actually returned by the tools.
Do NOT invent titles, authors, venues, or URLs. If unsure about a figure, omit it
rather than guess.

Output ONLY a JSON array (no prose, no code fences). Each element:
{
  "title": "...",
  "authors": "First A., ..., Senior Z.",
  "institution": "lead institution(s)",
  "venue": "journal / conference / 'bioRxiv preprint'",
  "date": "e.g. 'Jan 2026 (bioRxiv Jun 2025)'",
  "url": "canonical DOI/preprint link",
  "cited_by_count": 0,
  "fwci": null,
  "algo": "core method/algorithm name",
  "data": "what data it uses (train + eval, scale)",
  "model": "how the method works / how it was adapted",
  "evaluation": "how it was evaluated: metrics, eval data, design novelty, limits",
  "bio_question": "the biological question it answers",
  "app_tags": ["Application tag", "..."],
  "method_tags": ["Method/computational-class tag", "..."]
}
"""


def create_researcher(k: int) -> Agent:
    """Factory for researcher slot k (call it, don't pass the reference)."""
    return Agent(
        name=f"researcher_{k}",
        model=build_model(),
        description=f"Searches for recent papers in subfield cluster {k}.",
        instruction=RESEARCH_INSTRUCTION % k,
        tools=SCHOLARLY_TOOLS,
        output_key=f"papers_{k}",
        generate_content_config=safety_config(),
    )


research_fanout = ParallelAgent(
    name="research_fanout",
    description="Runs one searcher per subfield cluster, concurrently.",
    sub_agents=[create_researcher(k) for k in range(N_SLOTS)],
    before_agent_callback=split_clusters,
)
