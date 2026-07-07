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

"""Rank + structure agent (SPEC §2.3-2.4): merge, deepen, rank, emit ReviewQueue.

Uses ``output_schema=ReviewQueue`` to force clean, validated JSON. That DISABLES
tools/delegation (ADK rule) — fine, because all web research already happened in
the fan-out. Its ``after_agent_callback`` renders the HTML app.
"""

from google.adk.agents import Agent

from app.models import build_model
from app.render import assemble_html_callback
from app.schemas import ReviewQueue

RANK_INSTRUCTION = """\
You are the editor of a computational-biology paper review tailored to one
researcher.

Researcher profile and scope (JSON):
{scope}

Candidate papers gathered by the search agents (each a JSON array; some may be
empty or `[]`):
- {papers_0}
- {papers_1}
- {papers_2}
- {papers_3}
- {papers_4}

Do the following:
1. MERGE all candidate papers into one list and DEDUPLICATE (same title or URL =
   same paper). Drop anything that looks unverifiable or fabricated.
2. Deepen each paper into the three axes (SPEC §2.3):
   - Main approach: name the core algorithm; judge whether it is off-the-shelf or
     novel and set `nov` (0 = pure engineering/integration ... 100 = fundamentally
     new algorithm; middle = modified existing architecture).
   - Evaluation: new metric? new eval data? design novelty? eval limitations?
   - Results: the paper's MAIN results as a list of short bullet strings in
     `results`. For any QUANTITATIVE result, name the task, the data/benchmark,
     the metric, and the value (e.g. "Contact prediction (CASP15): 0.72 long-range
     precision, +0.05 over the baseline"). Keep each bullet to one sentence.
3. For each paper, set `cats` to the 0-based indices (into the scope's
   `clusters` array, in order) of EVERY cluster/category it matches. A paper may
   match several. This is the primary rank driver (SPEC §3d).
4. RANK primarily by COVERAGE — papers matching MORE categories (longer `cats`)
   rank higher — then, as tie-breakers within a coverage level, by a blended
   judgment of field impact, relevance to THIS researcher's profile,
   author/institution credibility, and venue. rank=1 is the best.
5. Score each paper: `impact` (1-10 field impact) and `rel` (1-10 relevance to
   the profile). These are editorial judgments.
6. Keep the queue focused: at most 12 papers.

For the `insights` field you MUST use these four inline labels VERBATIM and in
this order, each followed by a short phrase, all in one string:
"New metric: <...> New eval data: <...> Design & novelty: <...> Eval limits: <...>"

Fill `tags.app` with the biological-problem tags and `tags.method` with the
computational-class tags. Fill `approach` with {{algo, nov, aim, data, model,
bio}} where `aim` is "<goal>. Limitation addressed: <gap>." Put code/data URLs
(space-separated, bare) in `resources`. Write `relevance` as why it matters to
THIS researcher, and `comments` as your significance note.

Integrity rule (SPEC §2.5): prefer characterizing HOW a paper evaluated over
quoting exact numbers; hedge anything you cannot confirm.
"""

rank_agent = Agent(
    name="rank_agent",
    model=build_model(),
    description="Merges, deepens, and ranks candidate papers into the final ReviewQueue.",
    instruction=RANK_INSTRUCTION,
    output_schema=ReviewQueue,
    output_key="review_queue",
    after_agent_callback=assemble_html_callback,
)
