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

"""Shared model factory: pick the model/provider, set retry/backoff.

Model choice is a CONFIG knob (env ``REVIEW_MODEL``), never chat input:

- A bare Gemini id (default ``gemini-flash-latest``) -> ADK-native ``Gemini`` with
  retries, and the researchers can use native ``google_search`` grounding.
- A LiteLLM id with a provider prefix (e.g. ``anthropic/claude-sonnet-5``,
  ``openai/gpt-4o``) -> ``LiteLlm``. Non-Gemini models can't use Gemini's internal
  ``google_search``, so the researchers fall back to a model-agnostic web-search
  function tool (see ``search_tools.web_search``). Set the provider's key in .env
  (e.g. ``ANTHROPIC_API_KEY``).

``uses_native_grounding()`` is what ``research.py`` branches on to choose the tool.
"""

from __future__ import annotations

import os

from google.adk.models import Gemini
from google.genai import types

DEFAULT_MODEL = "gemini-flash-latest"

# Safety config (defense-in-depth alongside app/security.py): block medium+ harmful
# content at the model layer. Applied via each Agent's generate_content_config on
# the agents that ingest untrusted external text (CV pages, search results).
_SAFETY_SETTINGS = [
    types.SafetySetting(category=cat, threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
    for cat in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


def safety_config() -> types.GenerateContentConfig:
    """GenerateContentConfig applying the shared safety settings.

    A fresh instance per call (ADK/pydantic may mutate per-agent config)."""
    return types.GenerateContentConfig(safety_settings=list(_SAFETY_SETTINGS))


def model_id() -> str:
    return os.getenv("REVIEW_MODEL", DEFAULT_MODEL)


def uses_native_grounding() -> bool:
    """True when the configured model is a Gemini model (supports google_search).

    LiteLLM ids are 'provider/model'; a bare 'gemini-*' id is ADK-native Gemini.
    """
    mid = model_id().lower()
    provider = mid.split("/", 1)[0] if "/" in mid else "gemini"
    return provider == "gemini" or mid.startswith("gemini")


def build_model():
    """Return the configured model (Gemini native, or LiteLlm for others)."""
    mid = model_id()
    if uses_native_grounding():
        return Gemini(model=mid, retry_options=types.HttpRetryOptions(attempts=5))
    # Non-Gemini via LiteLLM (needs the provider's API key in the environment).
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=mid)
