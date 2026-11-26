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

"""Run the ADK review pipeline for one hosted /review request, on the caller's keys.

Each request builds the agent tree on a model bound to the caller's own API key
(``LiteLlm(api_key=...)``), so the project's credentials are never used and
concurrent requests never share a key. The optional Paperpile key is held in a
context variable for the run only: never written to session state, logs or disk.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from collections.abc import Callable

from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agent import build_root_agent
from app.render import build_payload, categories_from_scope
from app.research import _scope
from app.schemas import LabeledQueue
from app.stages import paperpile_key

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    # OpenRouter: the caller signs in (OAuth PKCE) and gets a key on their own
    # OpenRouter credits; model ids are OpenRouter slugs.
    "openrouter": "anthropic/claude-sonnet-5.5",
}
OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/auth/keys"
SUPPORTED_PROVIDERS = tuple(DEFAULT_MODELS)

# Progress label shown when a stage's agent first emits an event.
STAGE_LABELS = {
    "intake_agent": "Reading your request…",
    "scope_agent": "Analyzing your interests…",
    "researcher": "Searching OpenAlex, PubMed, Europe PMC, arXiv and bioRxiv…",
    "added_papers": "Looking up added papers…",
    "rank_agent": "Ranking papers…",
    "deepen_agent": "Reading the selected papers in full text…",
    "library_agent": "Checking your Paperpile and rendering…",
}
_INPUT_LABELS = {"cv": "CV / profile", "paper": "Paper", "keyword": "Research topic"}


def caller_model(provider: str, model: str, api_key: str) -> LiteLlm:
    """A model bound to the caller's key (LiteLLM ids: 'gemini/...', 'anthropic/...')."""
    name = model.strip() or DEFAULT_MODELS[provider]
    return LiteLlm(model=f"{provider}/{name}", api_key=api_key)


async def exchange_openrouter_code(code: str, code_verifier: str) -> str:
    """Finish OpenRouter's OAuth PKCE sign-in: authorization code -> the user's key.

    The key is returned to the caller's browser and never stored or logged here.
    Raises ValueError with a user-facing message when OpenRouter refuses.
    """
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            OPENROUTER_KEYS_URL,
            json={
                "code": code,
                "code_verifier": code_verifier,
                "code_challenge_method": "S256",
            },
        )
    if resp.status_code != 200:
        if resp.status_code == 403:
            raise ValueError(
                "OpenRouter sign-in expired or was already used; sign in again."
            )
        raise ValueError(f"OpenRouter sign-in failed (HTTP {resp.status_code}).")
    key = (resp.json() or {}).get("key") or ""
    if not key:
        raise ValueError("OpenRouter returned no key; sign in again.")
    return key


# Fields the full-text step rewrites; ranking, scores and tags stay as shown.
DEEP_FIELDS = (
    "insights",
    "results",
    "limitation",
    "comments",
    "resources",
    "approach",
    "evidence",
    "fulltext_source",
)


async def reanalyze_with_paperpile(
    papers: list[dict],
    provider: str,
    model: str,
    api_key: str,
    paperpile_api_key: str,
    profile: str = "",
) -> dict:
    """Re-read abstract-only papers from the PDFs saved in the caller's Paperpile.

    No search or ranking: each paper keeps its rank, scores and tags, and only
    the full-text fields are rewritten. Papers without a saved PDF (or with an
    open copy that still fails) come back unchanged. Returns
    {"papers": [{rank, ...updated fields}], "read": n, "missing": [titles]}.
    """
    from contextlib import AsyncExitStack

    from app.stages import DeepenAgent, open_paperpile

    llm = caller_model(provider, model, api_key)
    deepen = DeepenAgent(name="deepen_agent", llm=llm)
    state = {"scope": json.dumps({"profile": profile}), "candidate_pool": papers}
    async with AsyncExitStack() as stack:
        library = await stack.enter_async_context(open_paperpile(paperpile_api_key))
        delta, _ = await deepen._deepen(state, papers, library)
    out = delta.get("review_queue", {}).get("papers") or []
    updated = [
        {"rank": p["rank"], **{k: p[k] for k in DEEP_FIELDS if k in p}}
        for p in out
        if p.get("evidence") == "fulltext"
    ]
    done = {p["rank"] for p in updated}
    return {
        "papers": updated,
        "read": len(updated),
        "missing": [p.get("title", "") for p in papers if p.get("rank") not in done],
    }


def request_message(inputs: list[dict]) -> str:
    lines = [
        f"- {_INPUT_LABELS.get(i.get('kind', ''), 'Interest')}: {i['value'].strip()}"
        for i in inputs
        if (i.get("value") or "").strip()
    ]
    return "Researcher interests:\n" + "\n".join(lines) if lines else ""


async def run_review(
    inputs: list[dict],
    provider: str,
    model: str,
    api_key: str,
    *,
    paperpile_api_key: str = "",
    paperpile_label: str = "",
    paperpile_added_since: str = "",
    profile: str = "",
    on_stage: Callable[[str, str], None] | None = None,
) -> dict:
    """Run the pipeline; return the SPEC §3 payload {data, tags, approach, categories,
    profile, generated, inputs}. Raises ValueError when there is nothing to review."""
    message = request_message(inputs)
    if not message and not paperpile_label:
        raise ValueError("Provide at least one interest or a Paperpile label.")
    agent = build_root_agent(caller_model(provider, model, api_key))
    runner = InMemoryRunner(agent=agent, app_name="app")
    state = {
        "hosted": True,  # render: no disk write (the served page is shared)
        "paperpile_include": paperpile_label.strip(),
        "paperpile_added_since": paperpile_added_since.strip(),
        "profile_override": profile.strip(),
        "skip_research": not message,  # a Paperpile label alone: review just it
    }
    session = await runner.session_service.create_session(
        app_name="app", user_id="web", session_id=uuid.uuid4().hex, state=state
    )
    token = paperpile_key.set(paperpile_api_key.strip())
    seen: set[str] = set()
    try:
        content = types.Content(
            role="user",
            parts=[types.Part(text=message or "Review my Paperpile papers.")],
        )
        async for ev in runner.run_async(
            user_id="web", session_id=session.id, new_message=content
        ):
            stage = "researcher" if ev.author.startswith("researcher_") else ev.author
            if stage in STAGE_LABELS and stage not in seen:
                seen.add(stage)
                if on_stage:
                    on_stage(stage, STAGE_LABELS[stage])
    finally:
        paperpile_key.reset(token)
    final = await runner.session_service.get_session(
        app_name="app", user_id="web", session_id=session.id
    )
    st = final.state if final else {}
    queue = LabeledQueue.model_validate(st.get("review_queue") or {"papers": []})
    if not queue.papers:
        raise ValueError(
            "No papers to review. Add a CV URL, research topics, papers, "
            "or a Paperpile label."
        )
    scope = _scope(st)
    payload = build_payload(
        queue,
        profile=scope.get("profile", ""),
        generated=dt.date.today().isoformat(),
        categories=categories_from_scope(scope),
    )
    payload["inputs"] = inputs
    payload["added_failed"] = st.get("added_failed") or []
    logger.info(
        "hosted review: %d papers (%s)",
        len(queue.papers),
        json.dumps({"provider": provider, "paperpile": bool(paperpile_api_key)}),
    )
    return payload
