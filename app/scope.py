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

"""Scope agent (SPEC §2.0-2.1): read the CV, derive subfield clusters.

Design note: this agent needs the ``load_web_page`` *function tool* to fetch the
CV, and ``output_schema`` would DISABLE tool-calling. So we cannot force a
Pydantic schema here. Instead we instruct the model to emit JSON matching
``Scope`` and stash it verbatim in ``state['scope']`` via ``output_key``.
Downstream agents read it with the ``{scope}`` placeholder; the ranker (which
has no tools) is where we enforce a real ``output_schema``.
"""

from google.adk.agents import Agent
from google.adk.tools.load_web_page import load_web_page

from app.models import build_model, safety_config
from app.security import injection_guard, url_ssrf_guard

SCOPE_INSTRUCTION = """\
You define the research scope for a computational-biology paper review tailored to
one researcher.

The user's message may contain ANY combination of:
- CV(s) or personal-site URL(s)
- Link(s) or title(s) of paper(s)
- free-text research topics / keywords / subfield names to scope the review,
- any combination of the above, or none.

Rules:
1. If a URL is present, call `load_web_page` on it and extract the researcher's
   subfields, recurring methods, and seniority. If the fetch fails or is thin,
   work from the rest and say so briefly in `profile`.
2. If the user gave free-text topics/keywords, treat EACH as a research category:
   turn them into clusters directly. User-provided topics take priority and are
   merged with anything the CV implies.
3. If the message has NEITHER a URL NOR any usable topics (e.g. it is empty, a
   greeting, or unrelated chit-chat), DO NOT invent a scope. Instead, reply in
   plain language asking the user to provide a CV/website URL or a few research
   topics/keywords, and stop. (Do not emit the JSON in this case.)

When you DO have scope (rule 1 and/or 2), produce UP TO 5 distinct subfield
clusters (minimise overlap), each with 3-8 concrete search keywords, and output
ONLY this JSON object — no prose, no code fences:
{
  "profile": "<2-4 sentences: subfields, methods, seniority; note if derived only from topics>",
  "clusters": [
    {"name": "<short subfield label>", "keywords": ["kw1", "kw2", ...]}
  ]
}
"""

scope_agent = Agent(
    name="scope_agent",
    model=build_model(),
    description="Derives up to 5 subfield clusters from a CV URL and/or free-text research topics; asks for input if given neither.",
    instruction=SCOPE_INSTRUCTION,
    tools=[load_web_page],
    output_key="scope",
    generate_content_config=safety_config(),
    # Security guardrails (app/security.py): block SSRF-prone URLs before the
    # load_web_page fetch, and refuse oversized / injection-style user input.
    before_tool_callback=url_ssrf_guard,
    before_model_callback=injection_guard,
)
