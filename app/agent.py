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

"""Root agent for the recent-literature Review Queue.

Pipeline (SPEC §2), as a linear SequentialAgent:

    intake_agent       paper links in the message -> added papers
    scope_agent        read CV / topics -> subfield clusters        (state['scope'])
    research_fanout    one pooled-search researcher per cluster     (state['papers_*'])
    added_papers       look up added papers (+ a Paperpile label)   (state['added'])
    rank_agent         batched ranker, tool-verified, coverage-first (state['review_queue'])
    deepen_agent       selected papers re-analyzed from open full text
    library_agent      flag papers already in Paperpile; render literature-reviewer.html

Input: a chat message with a CV/site URL, research topics, and/or paper links.
Output: the self-contained review app, saved as an artifact.

``build_root_agent(model)`` builds the tree on a given model (the hosted page
passes one bound to the caller's own API key); ``root_agent`` uses the
configured ``REVIEW_MODEL``.
"""

from google.adk.agents import SequentialAgent
from google.adk.apps import App

from app.models import build_model
from app.render import assemble_html_callback
from app.research import create_research_fanout
from app.scope import create_scope_agent
from app.stages import (
    AddedPapersAgent,
    DeepenAgent,
    IntakeAgent,
    LibraryAgent,
    RankAgent,
)


def build_root_agent(model=None) -> SequentialAgent:
    model = model or build_model()
    return SequentialAgent(
        name="root_agent",
        description=(
            "Given a researcher's CV URL, topics or papers, researches recent papers, "
            "ranks them, reads the selected ones in full, and renders a review app."
        ),
        sub_agents=[
            IntakeAgent(name="intake_agent", description="Collects added paper links."),
            create_scope_agent(model),
            create_research_fanout(model),
            AddedPapersAgent(name="added_papers", description="Looks up added papers."),
            RankAgent(name="rank_agent", description="Ranks papers.", llm=model),
            DeepenAgent(name="deepen_agent", description="Reads full text.", llm=model),
            LibraryAgent(
                name="library_agent",
                description="Flags papers in Paperpile and renders the review.",
                after_agent_callback=assemble_html_callback,
            ),
        ],
    )


root_agent = build_root_agent()

# App name MUST match the agent directory ("app") or eval/sessions break.
app = App(
    root_agent=root_agent,
    name="app",
)
