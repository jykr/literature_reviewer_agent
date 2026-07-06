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

"""Root agent for the Recent Computational Biology Review Queue.

Pipeline (SPEC §2), as a linear SequentialAgent:

    scope_agent        read CV -> subfield clusters          (state['scope'])
        |
    research_fanout    one google_search agent per cluster   (state['papers_*'])
        |
    rank_agent         merge + deepen + rank -> ReviewQueue  (state['review_queue'])
                       then after_agent_callback renders literature-reviewer.html

Input: a chat message with the researcher's CV/site URL (+ optional scope hints).
Output: the self-contained review app, saved as an artifact.
"""

from google.adk.agents import SequentialAgent
from google.adk.apps import App

from app.analysis import rank_agent
from app.research import research_fanout
from app.scope import scope_agent

root_agent = SequentialAgent(
    name="root_agent",
    description=(
        "Given a researcher's CV URL, researches recent computational-biology "
        "papers, ranks them by relevance, and renders a self-contained review app."
    ),
    sub_agents=[scope_agent, research_fanout, rank_agent],
)

# App name MUST match the agent directory ("app") or eval/sessions break.
app = App(
    root_agent=root_agent,
    name="app",
)
