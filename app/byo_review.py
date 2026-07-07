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

"""Bring-your-own-key review pipeline (no ADK).

A provider-agnostic re-implementation of the scope -> research -> rank pipeline
that runs entirely on a caller-supplied API key, so each request bills the
caller (not the project). Wired into ``/review`` and ``/review/stream`` so the
deployed app can be exposed publicly without spending the project's quota.

Providers: gemini (AI Studio), anthropic, openai -- ``call_llm`` branches on the
provider. Research uses the keyless OpenAlex / Europe PMC / arXiv tools in
``app.search_tools``. Rendering reuses the same ``app.render.build_payload`` the
ADK path uses, so the JSON payload shape is byte-for-byte identical.

Ported from ``deployment/noadk/byo_server.py`` (which stays as a standalone
zero-dependency server for local use).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from types import SimpleNamespace as NS

from app import search_tools
from app.render import build_payload, categories_from_scope

logger = logging.getLogger(__name__)

# Called with (stage, human_label) as each pipeline stage starts; used by the
# SSE endpoint to stream progress. Never receives the API key.
ProgressFn = Callable[[str, str], None]

DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
}
SUPPORTED_PROVIDERS = tuple(DEFAULT_MODELS)

# OpenAlex polite-pool contact; overridable via env, defaulted so research works
# out of the box.
os.environ.setdefault("OPENALEX_MAILTO", "review-server@example.com")


# =========================================================================
# LLM providers -- one call_llm() over three REST APIs, key supplied per call
# =========================================================================
def _post_json(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call_llm(provider: str, model: str, api_key: str, system: str, user: str) -> str:
    """Return the model's text output. Raises on HTTP/provider error."""
    provider = (provider or "gemini").lower()
    model = model or DEFAULT_MODELS.get(provider, "")
    if provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
                "maxOutputTokens": 32768,
            },
        }
        data = _post_json(url, {"x-goog-api-key": api_key}, body)
        cand = (data.get("candidates") or [{}])[0]
        # Gemini may split output across several parts -> join ALL of them.
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
        if cand.get("finishReason") == "MAX_TOKENS":
            raise ValueError("Gemini response hit MAX_TOKENS (truncated). Try fewer interests.")
        if not text:
            raise ValueError(f"Gemini returned no text (finishReason={cand.get('finishReason')}).")
        return text
    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        body = {
            "model": model,
            "max_tokens": 16000,
            "system": system + "\nOutput ONLY valid JSON, no prose, no code fences.",
            "messages": [{"role": "user", "content": user}],
        }
        data = _post_json(url, {"x-api-key": api_key, "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []))
        if data.get("stop_reason") == "max_tokens":
            raise ValueError("Claude response hit max_tokens (truncated). Try fewer interests.")
        return text
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system + " Respond with a single JSON object."},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 16000,
        }
        data = _post_json(url, {"Authorization": f"Bearer {api_key}"}, body)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ValueError("OpenAI response hit the token limit (truncated). Try fewer interests.")
        return choice.get("message", {}).get("content", "")
    raise ValueError(f"unknown provider {provider!r}")


# =========================================================================
# Loose JSON parsing (models occasionally fence / truncate their output)
# =========================================================================
def _repair_truncated(t: str):
    """Best-effort parse of a truncated JSON object/array: close an open string
    and any unclosed brackets, then drop the trailing partial element."""
    # close an unterminated string (odd count of unescaped quotes)
    quotes = len(re.findall(r'(?<!\\)"', t))
    if quotes % 2:
        t += '"'
    # balance brackets using a simple stack (ignoring those inside strings)
    stack, in_str, esc = [], False, False
    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    t += "".join("}" if b == "{" else "]" for b in reversed(stack))
    for candidate in (t, re.sub(r",\s*([}\]])", r"\1", t)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_loads_loose(text: str):
    """Parse JSON from a model response, tolerating code fences / stray prose /
    truncation."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # grab the outermost { } or [ ]
        for op, cl in (("{", "}"), ("[", "]")):
            i, j = t.find(op), t.rfind(cl)
            if 0 <= i < j:
                try:
                    return json.loads(t[i : j + 1])
                except json.JSONDecodeError:
                    pass
        # last resort: repair a truncated structure
        repaired = _repair_truncated(t)
        if repaired is not None:
            return repaired
        raise


# =========================================================================
# Pipeline stages
# =========================================================================
_INPUT_LABELS = {"cv": "CV / profile", "paper": "Paper", "keyword": "Research topic/keyword"}

SCOPE_SYS = """You define the research scope for a computational-biology paper review tailored to one researcher.
Given the user's interests (CV/site text, papers, and/or free-text topics), produce UP TO 5 distinct subfield
clusters (minimise overlap), each with 3-8 concrete search keywords.
Output ONLY this JSON object: {"profile":"2-4 sentences: subfields, methods, seniority","clusters":[{"name":"short label","keywords":["kw1","kw2"]}]}
If there is no usable input at all, output {"profile":"","clusters":[]}."""

RANK_SYS = """You are the editor of a computational-biology paper review tailored to one researcher.
You receive the researcher scope (JSON) and candidate papers (JSON arrays, one per cluster; some may be empty).
Do this:
1. MERGE all candidates and DEDUPLICATE (same title or URL). Drop anything off-topic or unverifiable.
2. For each kept paper set `cats` = 0-based indices (into scope.clusters, in order) of EVERY cluster it matches.
3. RANK primarily by COVERAGE (more matched categories = higher), then by blended impact/relevance/venue. rank=1 is best.
4. Keep at most 12 papers.
For `insights` use these four inline labels VERBATIM in order, each followed by a short phrase, all in one string:
"New metric: <...> New eval data: <...> Design & novelty: <...> Eval limits: <...>"
For `results` list the paper's MAIN results as short bullet strings; for a QUANTITATIVE result name the
task, data/benchmark, metric, and value (e.g. "Contact prediction (CASP15): 0.72 long-range precision, +0.05 over baseline").
Output ONLY JSON: {"papers":[{
 "rank":int,"title":str,"authors":str,"institution":str,"venue":str,"date":str,"url":str,
 "insights":str,"results":[str],"limitation":str,"resources":str,"comments":str,"relevance":str,
 "impact":int(1-10),"rel":int(1-10),"cats":[int],
 "tags":{"app":[str],"method":[str]},
 "approach":{"algo":str,"nov":int(0-100),"aim":str,"data":str,"model":str,"bio":str}}]}
Only use papers actually present in the candidates; do NOT invent titles/authors/URLs."""


def _fetch_url_text(url: str, limit: int = 6000) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "review-server/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            html = r.read(400_000).decode("utf-8", "ignore")
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]
    except Exception as e:  # noqa: BLE001
        return f"(could not fetch {url}: {e})"


def run_pipeline(
    inputs: list[dict],
    provider: str,
    model: str,
    api_key: str,
    on_stage: ProgressFn | None = None,
) -> dict:
    """Run scope -> research -> rank on the caller's key and return the SPEC 3
    payload ({data, tags, approach, categories, inputs}).

    ``on_stage(stage, label)`` is invoked as each stage starts (for SSE progress)
    and is never passed the API key. Blocking (urllib) -- call from a threadpool
    inside async endpoints.
    """
    tag = f"[{provider}/{model or DEFAULT_MODELS.get(provider, '?')}]"

    def stage(name: str, label: str) -> None:
        logger.info("%s %s", tag, label)
        if on_stage is not None:
            on_stage(name, label)

    # ---- assemble the user message (fetch CV URLs, like load_web_page) ----
    lines = []
    for it in inputs:
        kind, val = it.get("kind", "keyword"), (it.get("value") or "").strip()
        if not val:
            continue
        if kind == "cv" and val.startswith(("http://", "https://")):
            lines.append(f"- CV / profile ({val}):\n{_fetch_url_text(val)}")
        else:
            lines.append(f"- {_INPUT_LABELS.get(kind, kind)}: {val}")
    if not lines:
        raise ValueError("Provide at least one interest.")
    user_msg = "Researcher interests:\n" + "\n".join(lines)

    # ---- stage 1: scope ----
    stage("scope", "Analyzing your interests…")
    scope = _json_loads_loose(call_llm(provider, model, api_key, SCOPE_SYS, user_msg))
    clusters = scope.get("clusters") or []
    if not clusters:
        raise ValueError("No usable scope from the given inputs; add a CV URL or research topics.")

    # ---- stage 2: research fan-out (real keyless tools, with fallback) ----
    clusters = clusters[:5]
    candidates = {}
    for k, c in enumerate(clusters):
        stage(
            "research",
            f"Researching recent papers… ({k + 1} of {len(clusters)} clusters)",
        )
        q = " ".join([c.get("name", "")] + list(c.get("keywords", []))).strip()
        r = search_tools.search_openalex(q, "2025-06-01", 8)
        if r.get("status") != "success" or not r.get("results"):
            r = search_tools.search_europepmc(q, 8)
        if r.get("status") != "success" or not r.get("results"):
            r = search_tools.search_arxiv(q, 8)
        candidates[k] = [p for p in r.get("results", []) if p.get("title")]

    # ---- stage 3: rank ----
    stage("rank", "Ranking and writing summaries…")
    rank_user = (
        "Scope:\n" + json.dumps(scope, ensure_ascii=False)
        + "\n\nCandidates by cluster index:\n" + json.dumps(candidates, ensure_ascii=False)
    )
    ranked = _json_loads_loose(call_llm(provider, model, api_key, RANK_SYS, rank_user))
    papers = ranked.get("papers") or []
    logger.info("%s done: %d ranked papers", tag, len(papers))

    # ---- render via the project's real build_payload ----
    def obj(p):
        d = dict(p)
        d.setdefault("tags", {"app": [], "method": []})
        d.setdefault("approach", {"algo": "", "nov": 0, "aim": "", "data": "", "model": "", "bio": ""})
        d["tags"] = NS(**{"app": [], "method": [], **d["tags"]})
        d["approach"] = NS(
            **{"algo": "", "nov": 0, "aim": "", "data": "", "model": "", "bio": "", **d["approach"]}
        )
        d.setdefault("cats", [])
        d.setdefault("results", [])
        return NS(**d)

    queue = NS(papers=[obj(p) for p in papers])
    payload = build_payload(
        queue, profile=scope.get("profile", ""), categories=categories_from_scope(scope)
    )
    payload["inputs"] = inputs
    return payload
