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

"""Code-driven pipeline stages (ADK custom agents), ported from the LangChain variant.

    intake_agent   paper links in the message -> state['manual_ids'], date window
    added_papers   added papers (links, a Paperpile label) -> tool records
    rank_agent     ranker in batches -> verified, coverage-first ReviewQueue
    deepen_agent   selected papers re-analyzed from open full text
    library_agent  flags papers already in the user's Paperpile; renders the page

Model calls here go straight to the ADK model (``BaseLlm.generate_content_async``)
with a response schema, so batches and per-paper analyses run concurrently.
The review rules (verification, credibility, grounding) live in
``lc_app.core``, shared with the LangChain variant.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel, ConfigDict

from app.analysis import RANK_INSTRUCTION
from app.byo_review import _json_loads_loose
from app.research import N_SLOTS, _scope, default_from_date
from app.schemas import ReviewQueue
from lc_app import cache
from lc_app import paper_sources as ps
from lc_app.core import (
    CREDIBILITY_RULES,
    DEEPEN_PROMPT,
    MANUAL_RANK_RULES,
    WRITING_STYLE,
    DeepAnalysis,
    _fulltext_ids,
    apply_deep,
    extract_papers,
    finalize_ranked,
    paper_dois,
    paper_key,
    reference_identifier,
    verify_candidates,
)
from lc_app.focus import focus_block
from lc_app.paper_sources import normalize_doi, parse_pdf_text

logger = logging.getLogger(__name__)

# A Paperpile key supplied per request by the hosted page. Held only in this
# context (never in session state, which may be persisted). None = not a
# hosted request: fall back to PAPERPILE_API_KEY from the environment.
paperpile_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "paperpile_key", default=None
)


def current_paperpile_key() -> str:
    key = paperpile_key.get()
    return (key if key is not None else os.getenv("PAPERPILE_API_KEY", "")).strip()


@asynccontextmanager
async def open_paperpile(key: str):
    """A PaperpileLibrary over Paperpile's MCP server, authenticated with ``key``."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from lc_app.paperpile import SERVER_URL, PaperpileLibrary

    headers = {"Authorization": f"Bearer {key}"}
    async with streamablehttp_client(SERVER_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield PaperpileLibrary(session)


async def structured_call(llm, schema: type[BaseModel], system: str, user: str) -> dict:
    """One model call whose reply must validate against ``schema``.

    Gemini gets a native response schema. Through LiteLLM (Claude, GPT) the
    JSON Schema goes in the prompt instead: LiteLLM enforces response formats
    with a forced tool call, which some Claude models reject. An invalid reply
    is retried once with the validation error.
    """
    native = not isinstance(llm, LiteLlm)
    if not native:
        system += (
            "\n\nReturn ONLY a JSON object (no prose, no code fences) that "
            "validates against this JSON Schema:\n"
            + json.dumps(schema.model_json_schema())
        )
    ask = user
    for attempt in (1, 2):
        req = LlmRequest(
            model=llm.model,
            contents=[types.Content(role="user", parts=[types.Part(text=ask)])],
            config=types.GenerateContentConfig(
                system_instruction=system,
                **(
                    {
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    }
                    if native
                    else {}
                ),
            ),
        )
        text = ""
        async for resp in llm.generate_content_async(req, stream=False):
            if resp.content and resp.content.parts:
                text += "".join(p.text or "" for p in resp.content.parts)
        try:
            return schema.model_validate(_json_loads_loose(text)).model_dump()
        except (ValueError, TypeError) as e:  # JSON or schema error
            if attempt == 2:
                raise
            ask = f"{user}\n\nYour previous reply was invalid: {str(e)[:800]}\nReply again."
    raise AssertionError("unreachable")


class _Stage(BaseAgent):
    """A code-driven step: subclasses return (state delta, message)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Any = None

    async def run(self, state: dict, ctx: InvocationContext) -> tuple[dict, str]:
        raise NotImplementedError

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        delta, message = await self.run(dict(ctx.session.state), ctx)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
            actions=EventActions(state_delta=delta),
            content=types.Content(role="model", parts=[types.Part(text=message)])
            if message
            else None,
        )


# ---------------------------------------------------------------------------
# Intake and added papers
# ---------------------------------------------------------------------------


class IntakeAgent(_Stage):
    """Paper links/DOIs/arXiv ids in the user's message become added papers."""

    async def run(self, state, ctx):
        text = "".join(
            p.text or "" for p in (ctx.user_content.parts if ctx.user_content else [])
        )
        _, ids = extract_papers(text)
        manual = list(dict.fromkeys(list(state.get("manual_ids") or []) + ids))
        delta = {
            "manual_ids": manual,
            "from_date": state.get("from_date") or default_from_date(),
        }
        return delta, (f"Added {len(ids)} paper(s) from your message." if ids else "")


LABEL_LIMIT = 10  # papers taken from a Paperpile label: the most recently added


class AddedPapersAgent(_Stage):
    """Look up every added paper (and a Paperpile label's papers, if asked)."""

    async def run(self, state, ctx):
        ids = list(state.get("manual_ids") or [])
        origin = dict(state.get("manual_origin") or {})
        notes = []
        label = (state.get("paperpile_include") or "").strip()
        if label:
            key = current_paperpile_key()
            if not key:
                notes.append(f"Paperpile label '{label}' skipped: no Paperpile key.")
            else:
                try:
                    async with open_paperpile(key) as lib:
                        name, refs = await lib.collection_refs(label)
                    since = state.get("paperpile_added_since") or ""
                    if since:
                        import datetime as dt

                        cut = dt.datetime.fromisoformat(since).timestamp()
                        refs = [r for r in refs if (r.get("created_at") or 0) >= cut]
                    # The most recently added papers that have a usable id.
                    refs = sorted(
                        (r for r in refs if reference_identifier(r)),
                        key=lambda r: r.get("created_at") or 0,
                        reverse=True,
                    )
                    n_all = len(refs)
                    for r in refs[:LABEL_LIMIT]:
                        ident = reference_identifier(r)
                        ids.append(ident)
                        origin[ident] = f"Paperpile: {name}"
                    notes.append(
                        f"{min(n_all, LABEL_LIMIT)} paper(s) from Paperpile '{name}'"
                        + (
                            f" (the {LABEL_LIMIT} most recent of {n_all})."
                            if n_all > LABEL_LIMIT
                            else "."
                        )
                    )
                except Exception as e:  # a Paperpile failure never stops the review
                    logger.warning("paperpile include failed: %s", e)
                    notes.append(f"Paperpile label '{label}' could not be read.")
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {"added": [], "added_failed": []}, " ".join(notes)
        sem = asyncio.Semaphore(6)

        async def one(ident: str):
            async with sem:
                res = await ps.get_paper_details(ident)
            rec = res.get("paper") if isinstance(res, dict) else None
            if not rec:
                return ident, None
            return ident, {
                **rec,
                "identifier": ident,
                "manual": True,
                "origin": origin.get(ident, ""),
            }

        added, failed = [], []
        for ident, rec in await asyncio.gather(*(one(i) for i in ids)):
            (added.append(rec) if rec else failed.append(ident))
        msg = f"Looked up {len(added)} added paper(s)" + (
            f"; not found: {', '.join(failed)}" if failed else ""
        )
        return {"added": added, "added_failed": failed}, " ".join([*notes, msg + "."])


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------

RANK_BATCH = 10  # added papers per ranker call


class RankAgent(_Stage):
    """The ranker, in batches of added papers, with tool-truth re-verification."""

    async def run(self, state, ctx):
        scope = _scope(state)
        by_cluster: dict[int, list[dict]] = {}
        found_in: dict[str, set[int]] = {}
        for k in range(N_SLOTS):
            raw = state.get(f"papers_{k}") or "[]"
            try:
                papers = json.loads(raw) if isinstance(raw, str) else list(raw)
            except json.JSONDecodeError:
                papers = []
            by_cluster[k] = papers
            for p in papers:
                found_in.setdefault(paper_key(p), set()).add(k)
        added = list({paper_key(m): m for m in state.get("added") or []}.values())
        pool = [p for ps_ in by_cluster.values() for p in ps_] + added
        if not pool:
            return {
                "review_queue": {"papers": []}
            }, "The search returned no usable papers."
        batches = [added[i : i + RANK_BATCH] for i in range(0, len(added), RANK_BATCH)]
        batches = batches or [[]]
        sem = asyncio.Semaphore(3)

        async def one(i: int, batch: list[dict]):
            async with sem:
                return await self._rank(
                    scope, by_cluster if i == 0 else {}, batch, pool
                )

        results = await asyncio.gather(*(one(i, b) for i, b in enumerate(batches)))
        kept, seen = [], set()
        for papers, _ in results:
            for p in papers:
                if (k := paper_key(p)) not in seen:
                    seen.add(k)
                    kept.append(p)
        finalize_ranked(
            kept, len(scope.get("clusters", [])), found_in, added, len(batches) > 1
        )
        delta = {"review_queue": {"papers": kept}, "candidate_pool": pool}
        return delta, f"Ranked {len(kept)} papers."

    async def _rank(self, scope, by_cluster, batch, pool):
        system = (
            RANK_INSTRUCTION.format(
                scope=json.dumps(scope, ensure_ascii=False),
                **{
                    f"papers_{k}": json.dumps(by_cluster.get(k, []), ensure_ascii=False)
                    for k in range(N_SLOTS)
                },
            )
            + CREDIBILITY_RULES
            + WRITING_STYLE
            + focus_block("rank", use_default=False)
            + (
                MANUAL_RANK_RULES.format(papers=json.dumps(batch, ensure_ascii=False))
                if batch
                else ""
            )
        )
        ask = "Produce the ranked ReviewQueue."
        for attempt in (1, 2):
            try:
                out = await structured_call(self.llm, ReviewQueue, system, ask)
            except Exception as e:  # invalid JSON: retry once
                if attempt == 2:
                    raise
                ask = f"Your previous ReviewQueue was invalid ({str(e)[:500]}). Return it again."
                continue
            kept, dropped = verify_candidates(out["papers"], pool)
            got = {paper_key(p) for p in kept}
            missing = [m["title"] for m in batch if paper_key(m) not in got]
            if not missing or attempt == 2:
                return kept, dropped
            ask = (
                "Your ReviewQueue omitted these added papers, which must all be "
                "included: " + "; ".join(missing) + ". Return the complete ReviewQueue."
            )
        return [], 0


# ---------------------------------------------------------------------------
# Deepen: full text for the selected papers
# ---------------------------------------------------------------------------


class DeepenAgent(_Stage):
    """Re-analyze each selected paper from full text: open copies (arXiv, PMC,
    Unpaywall) first, then a PDF the user saved in Paperpile (needs their key)."""

    async def run(self, state, ctx):
        papers = (state.get("review_queue") or {}).get("papers") or []
        if not papers:
            return {}, ""
        async with AsyncExitStack() as stack:
            library = None
            if key := current_paperpile_key():
                try:
                    library = await stack.enter_async_context(open_paperpile(key))
                except Exception as e:  # without Paperpile, open copies only
                    logger.warning("paperpile unavailable for full text: %s", e)
            return await self._deepen(state, papers, library)

    async def _saved_pdfs(self, library, papers, pool) -> dict[int, dict]:
        """rank -> library reference with a PDF, for the selected papers."""
        dois = paper_dois(papers, pool)
        refs = await library.lookup(list(dois.values()))
        by_doi = {normalize_doi(r.get("doi") or ""): r for r in refs if r.get("files")}
        return {rank: by_doi[d] for rank, d in dois.items() if d in by_doi}

    async def _deepen(self, state, papers, library):
        profile = _scope(state).get("profile", "")
        sem = asyncio.Semaphore(4)
        stats = {"dropped": 0, "paperpile": 0}
        saved: dict[int, dict] = {}
        if library is not None:
            try:
                saved = await self._saved_pdfs(
                    library, papers, state.get("candidate_pool") or []
                )
            except Exception as e:
                logger.warning("paperpile lookup for full text failed: %s", e)

        async def one(p: dict) -> dict:
            key = paper_key(p)
            if hit := await asyncio.to_thread(cache.load_deep, key):
                return {**p, **hit}
            ft = None
            for ident in _fulltext_ids(p):
                res = await ps.get_paper_fulltext(ident)
                if res.get("status") == "success":
                    ft = res
                    break
            if ft is None and (ref := saved.get(p.get("rank"))):
                text = await library.pdf_text(ref["id"])
                if len(text) > 2000:
                    ft = {
                        **parse_pdf_text(text),
                        "status": "success",
                        "source": "paperpile",
                        "title": ref.get("title", ""),
                    }
                    stats["paperpile"] += 1
            if ft is None:
                return {**p, "evidence": "abstract"}
            current = {
                k: p.get(k)
                for k in (
                    "insights",
                    "results",
                    "limitation",
                    "comments",
                    "resources",
                    "approach",
                )
            }
            excerpt = {k: ft[k] for k in ("title", "sections", "tables", "links")}
            async with sem:
                d = await structured_call(
                    self.llm,
                    DeepAnalysis,
                    DEEPEN_PROMPT.format(profile=profile),
                    "Current analysis:\n"
                    + json.dumps(current, ensure_ascii=False)
                    + "\n\nFull-text excerpts:\n"
                    + json.dumps(excerpt, ensure_ascii=False),
                )
            hit, dropped = apply_deep(p, d, ft)
            stats["dropped"] += dropped
            await asyncio.to_thread(cache.save_deep, key, hit)
            return {**p, **hit}

        outs = await asyncio.gather(*(one(p) for p in papers), return_exceptions=True)
        deepened = []
        for p, out in zip(papers, outs, strict=True):
            if isinstance(out, BaseException):
                logger.warning("deepen: %s failed: %s", p.get("title", "")[:50], out)
                out = {**p, "evidence": "abstract"}
            deepened.append(out)
        n_full = sum(p.get("evidence") == "fulltext" for p in deepened)
        msg = f"Read {n_full} of {len(deepened)} papers in full text"
        msg += (
            f" ({stats['paperpile']} from your Paperpile PDFs)."
            if stats["paperpile"]
            else "."
        )
        if stats["dropped"]:
            msg += f" Dropped {stats['dropped']} result(s) quoting numbers not in the text."
        return {"review_queue": {"papers": deepened}}, msg


# ---------------------------------------------------------------------------
# Paperpile flags (then render)
# ---------------------------------------------------------------------------


class LibraryAgent(_Stage):
    """Flag reviewed papers already in the user's Paperpile library."""

    async def run(self, state, ctx):
        papers = [
            dict(p) for p in (state.get("review_queue") or {}).get("papers") or []
        ]
        key = current_paperpile_key()
        if not papers or not key:
            return {}, ""
        dois = paper_dois(papers, state.get("candidate_pool") or [])
        try:
            async with open_paperpile(key) as lib:
                refs = await lib.lookup(list(dois.values()))
        except Exception as e:  # never break the review over Paperpile
            logger.warning("paperpile lookup failed: %s", e)
            return {}, "Paperpile check failed."
        have = {normalize_doi(r.get("doi") or "") for r in refs}
        searched = [
            p for p in papers if not p.get("origin", "").startswith("Paperpile")
        ]
        for p in papers:
            p["in_library"] = p not in searched or dois.get(p["rank"]) in have
        n = sum(p["in_library"] for p in searched)
        return {"review_queue": {"papers": papers}}, (
            f"{n} reviewed paper(s) already in your Paperpile library."
            if searched
            else ""
        )
