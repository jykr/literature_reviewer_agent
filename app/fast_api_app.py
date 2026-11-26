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

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from pydantic import BaseModel, Field

# Load .env BEFORE importing app modules: some resolve config (e.g. REVIEW_MODEL)
# at import time, so the environment must be populated first.
load_dotenv()

from app import hosted_review  # noqa: E402

from app.app_utils import services  # noqa: E402
from app.app_utils.a2a import attach_a2a_routes  # noqa: E402
from app.app_utils.reasoning_engine_adapter import (  # noqa: E402
    attach_reasoning_engine_routes,
)
from app.app_utils.telemetry import setup_telemetry  # noqa: E402
from app.app_utils.typing import Feedback  # noqa: E402

# Sets the OTel providers/resource; must run before get_fast_api_app.
setup_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The full ADK agent API (get_fast_api_app's /run, /run_sse and dev UI), the A2A
# routes, and the reasoning_engine proxy all execute the agent on the PROJECT's
# credentials. The public web app only needs the BYO-key /review endpoints, so
# these agent-protocol surfaces are OFF by default and must be opted into (e.g.
# for the Vertex AI playground or A2A clients sitting behind their own auth).
ENABLE_AGENT_API = os.getenv("ENABLE_AGENT_API", "").lower() in ("1", "true", "yes")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Runner for the A2A path, sharing the same session/artifact services as the
    # adk_api and reasoning_engine paths (see services.py). Imported here so the
    # agent is built after env/telemetry setup.
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    # Shared by the A2A path and the reasoning_engine adapter routes.
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


if ENABLE_AGENT_API:
    app: FastAPI = get_fast_api_app(
        agents_dir=AGENT_DIR,
        web=True,
        artifact_service_uri=services.ARTIFACT_SERVICE_URI,
        allow_origins=allow_origins,
        session_service_uri=services.SESSION_SERVICE_URI,
        otel_to_cloud=False,
        lifespan=lifespan,
    )
    # Proxy routes so the Vertex AI Console Playground (reasoning_engine SDK) can
    # talk to this agent alongside the native adk_api routes.
    attach_reasoning_engine_routes(app)
else:
    # Public deployment: only the BYO-key web app. No agent-protocol routes, no
    # ADK session/artifact services, and no agent ever built on project creds.
    app = FastAPI()
    if allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

app.title = "literature-reviewer"
app.description = "API for interacting with the Agent literature-reviewer"


class InputItem(BaseModel):
    """One interest input from the web app (SPEC §2.0/§3d)."""

    kind: str  # 'cv' | 'paper' | 'keyword'
    value: str


class ReviewRequest(BaseModel):
    """A regenerate request from the interests panel.

    ``api_key`` (aliased ``apiKey`` on the wire) is supplied by the user and used
    for that request only -- never stored or logged. The project's own
    credentials are never used to generate reviews, so the service is safe to
    expose publicly: each request bills the caller's key.
    """

    inputs: list[InputItem] = []
    provider: str = "anthropic"
    model: str = ""
    api_key: str = Field("", alias="apiKey")
    # Optional, also per request and never stored: the caller's Paperpile key
    # (flags papers already in their library; reads `paperpile_label`).
    paperpile_key: str = Field("", alias="paperpileKey")
    paperpile_label: str = Field("", alias="paperpileLabel")
    paperpile_added_since: str = Field("", alias="paperpileAddedSince")
    profile: str = ""  # the researcher profile, verbatim (else the model writes one)

    model_config = {"populate_by_name": True}


def _require_key(req: ReviewRequest) -> tuple[str, str]:
    """Validate the provider + user key, returning ``(provider, key)``.

    Raises ``HTTPException`` (400) when the key is missing or the provider is
    unsupported -- generation never falls back to project credentials.
    """
    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Enter your API key to regenerate the review.",
        )
    provider = (req.provider or "anthropic").lower()
    if provider not in hosted_review.SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported provider {provider!r}. Choose one of: "
                f"{', '.join(hosted_review.SUPPORTED_PROVIDERS)}."
            ),
        )
    return provider, key


@app.get("/")
def root() -> RedirectResponse:
    """Send the bare URL to the review app."""
    return RedirectResponse(url="/reviewer")


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict[str, str]:
    """Liveness probe for Cloud Run (also HEAD, for uptime/status pings)."""
    return {"status": "ok"}


@app.get("/reviewer")
def reviewer_page() -> FileResponse:
    """Serve the single-file review app same-origin (so /review needs no CORS)."""
    return FileResponse(os.path.join(AGENT_DIR, "literature-reviewer.html"))


def _run_kwargs(req: ReviewRequest) -> dict:
    return {
        "paperpile_api_key": req.paperpile_key,
        "paperpile_label": req.paperpile_label,
        "paperpile_added_since": req.paperpile_added_since,
        "profile": req.profile,
    }


def _error_detail(exc: Exception, provider: str) -> str:
    """A safe, user-facing message (never echoes keys or request bodies)."""
    name = type(exc).__name__
    if "Authentication" in name or "PermissionDenied" in name:
        return f"{provider} rejected the API key."
    if "RateLimit" in name:
        return f"{provider} rate limit reached; try again shortly."
    return "Internal error generating review."


@app.post("/review")
async def review(req: ReviewRequest) -> dict:
    """Regenerate the review on the caller's own API key and return the SPEC §3
    JSON.

    Runs the ADK pipeline (:mod:`app.hosted_review`) on a model bound to the
    caller's key -- never the project's credentials. The interests panel uses
    ``/review/stream`` instead for per-stage progress.
    """
    provider, key = _require_key(req)
    inputs = [it.model_dump() for it in req.inputs]
    try:
        return await hosted_review.run_review(
            inputs, provider, req.model, key, **_run_kwargs(req)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=_error_detail(exc, provider)
        ) from exc


@app.post("/review/stream")
async def review_stream(req: ReviewRequest) -> StreamingResponse:
    """Same pipeline as /review, streamed as Server-Sent Events.

    Emits ``{type:'progress', stage, label}`` as each stage starts, then a final
    ``{type:'done', payload}`` (or ``{type:'error', detail}``). The payload is
    identical to what /review returns.
    """
    provider, key = _require_key(req)
    inputs = [it.model_dump() for it in req.inputs]

    def sse(obj: dict) -> str:
        # json.dumps emits no literal newlines, so one event is always one line.
        return f"data: {json.dumps(obj)}\n\n"

    async def generate():
        events: asyncio.Queue = asyncio.Queue()

        def on_stage(stage: str, label: str) -> None:
            events.put_nowait({"type": "progress", "stage": stage, "label": label})

        async def worker():
            try:
                payload = await hosted_review.run_review(
                    inputs,
                    provider,
                    req.model,
                    key,
                    on_stage=on_stage,
                    **_run_kwargs(req),
                )
                events.put_nowait({"type": "done", "payload": payload})
            except ValueError as exc:
                events.put_nowait({"type": "error", "detail": str(exc)})
            except Exception as exc:  # noqa: BLE001 - stream must not raise mid-flight
                logger.log_struct(
                    {"event": "review_stream_error", "error": type(exc).__name__},
                    severity="ERROR",
                )
                events.put_nowait(
                    {"type": "error", "detail": _error_detail(exc, provider)}
                )
            finally:
                events.put_nowait(None)  # sentinel: worker finished

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await events.get()
                if item is None:
                    break
                yield sse(item)
        finally:
            await task

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SaveItem(BaseModel):
    title: str = ""
    url: str = ""
    doi: str = ""
    note: str = Field("", max_length=20000)  # Markdown note written by the agent


class PaperpileSaveRequest(BaseModel):
    """Save reviewed papers to the caller's Paperpile library (key never stored)."""

    paperpile_key: str = Field("", alias="paperpileKey")
    papers: list[SaveItem] = Field(default_factory=list, max_length=50)
    label: str = Field("", max_length=100)

    model_config = {"populate_by_name": True}


def _paper_doi(item: SaveItem) -> str:
    from lc_app.core import doi_from_link

    return doi_from_link(item.doi, item.url)


@app.post("/paperpile/save")
async def paperpile_save(req: PaperpileSaveRequest) -> dict:
    """Add papers to the caller's Paperpile, each with the agent's note if sent.

    Papers already in the library are left as they are: their notes are never
    overwritten. Returns one {title, doi, status} per paper.
    """
    from app.stages import open_paperpile

    key = req.paperpile_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Enter your Paperpile API key.")
    items, results = [], []
    for p in req.papers:
        if doi := _paper_doi(p):
            items.append({"doi": doi, "note": p.note.strip(), "title": p.title})
        else:
            results.append({"title": p.title, "doi": "", "status": "no DOI"})
    if items:
        try:
            async with open_paperpile(key) as lib:
                saved = await lib.save_items(items, req.label.strip() or None)
        except Exception as exc:
            logger.log_struct(
                {"event": "paperpile_save_error", "error": type(exc).__name__},
                severity="WARNING",
            )
            raise HTTPException(
                status_code=502,
                detail="Paperpile rejected the request. Check that your key is "
                "valid and your account is enrolled in Paperpile's API beta.",
            ) from exc
        titles = {i["doi"]: i["title"] for i in items}
        results += [{**r, "title": titles.get(r["doi"], "")} for r in saved]
    return {"results": results}


class RemoveItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    doi: str = Field(..., min_length=1, max_length=300)
    title: str = ""


class PaperpileRemoveRequest(BaseModel):
    """Undo papers this page added to the caller's Paperpile (moved to trash)."""

    paperpile_key: str = Field("", alias="paperpileKey")
    refs: list[RemoveItem] = Field(default_factory=list, max_length=50)

    model_config = {"populate_by_name": True}


@app.post("/paperpile/remove")
async def paperpile_remove(req: PaperpileRemoveRequest) -> dict:
    """Move papers the review page added back out of the library (to the trash).

    Only references the page itself created (their ids came back from
    /paperpile/save) are sent here, and each id must match its DOI.
    """
    from app.stages import open_paperpile

    key = req.paperpile_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Enter your Paperpile API key.")
    try:
        async with open_paperpile(key) as lib:
            results = await lib.trash([r.model_dump() for r in req.refs])
    except Exception as exc:
        logger.log_struct(
            {"event": "paperpile_remove_error", "error": type(exc).__name__},
            severity="WARNING",
        )
        raise HTTPException(
            status_code=502,
            detail="Paperpile rejected the request. Check that your key is "
            "valid and your account is enrolled in Paperpile's API beta.",
        ) from exc
    titles = {r.doi.lower(): r.title for r in req.refs}
    return {"results": [{**r, "title": titles.get(r["doi"], "")} for r in results]}


class PaperpileCheckRequest(BaseModel):
    """Which of the shown papers are already in the caller's Paperpile (read-only)."""

    paperpile_key: str = Field("", alias="paperpileKey")
    papers: list[SaveItem] = Field(default_factory=list, max_length=100)

    model_config = {"populate_by_name": True}


@app.post("/paperpile/check")
async def paperpile_check(req: PaperpileCheckRequest) -> dict:
    """Flag papers already in the caller's library, without regenerating the review.

    Read-only: looks papers up by DOI and changes nothing in Paperpile.
    Returns one {title, doi, in_library} per paper.
    """
    from app.stages import open_paperpile
    from lc_app.paper_sources import normalize_doi

    key = req.paperpile_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Enter your Paperpile API key.")
    dois = [_paper_doi(p) for p in req.papers]
    try:
        async with open_paperpile(key) as lib:
            refs = await lib.lookup([d for d in dois if d])
    except Exception as exc:
        logger.log_struct(
            {"event": "paperpile_check_error", "error": type(exc).__name__},
            severity="WARNING",
        )
        raise HTTPException(
            status_code=502,
            detail="Paperpile rejected the request. Check that your key is "
            "valid and your account is enrolled in Paperpile's API beta.",
        ) from exc
    have = {normalize_doi(r.get("doi") or "") for r in refs}
    return {
        "results": [
            {"title": p.title, "doi": d, "in_library": bool(d) and d in have}
            for p, d in zip(req.papers, dois, strict=True)
        ]
    }


class ReanalyzeRequest(BaseModel):
    """Re-read abstract-only papers from the caller's Paperpile PDFs (no new search)."""

    provider: str = "anthropic"
    model: str = ""
    api_key: str = Field("", alias="apiKey")
    paperpile_key: str = Field("", alias="paperpileKey")
    profile: str = Field("", max_length=4000)
    papers: list[dict] = Field(default_factory=list, max_length=20)

    model_config = {"populate_by_name": True}


@app.post("/review/reanalyze")
async def review_reanalyze(req: ReanalyzeRequest) -> dict:
    """Redo the full-text analysis for the given papers from their Paperpile PDFs.

    Uses the caller's model key and Paperpile key for this request only;
    ranking, scores and tags are left as they are on the page.
    """
    provider, key = _require_key(ReviewRequest(provider=req.provider, apiKey=req.api_key))
    if not req.paperpile_key.strip():
        raise HTTPException(status_code=400, detail="Enter your Paperpile API key.")
    papers = [
        {k: p.get(k) for k in ("rank", "title", "url", "doi", "insights", "results",
                               "limitation", "resources", "comments", "approach")}
        for p in req.papers
        if isinstance(p.get("rank"), int)
    ]
    try:
        return await hosted_review.reanalyze_with_paperpile(
            papers, provider, req.model, key, req.paperpile_key.strip(), req.profile
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=_error_detail(exc, provider)
        ) from exc


class OpenRouterExchange(BaseModel):
    code: str = Field(..., min_length=1, max_length=512)
    code_verifier: str = Field(..., min_length=43, max_length=128)


@app.post("/openrouter/exchange")
async def openrouter_exchange(req: OpenRouterExchange) -> dict:
    """Swap an OpenRouter sign-in code for the user's key (OAuth PKCE).

    The key goes straight back to the caller's browser; it is not stored or
    logged, and every review request carries it like a pasted API key.
    """
    try:
        return {"key": await hosted_review.exchange_openrouter_code(
            req.code, req.code_verifier
        )}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


OPENROUTER_CALLBACK_HTML = """<!doctype html><meta charset="utf-8">
<title>OpenRouter sign-in</title>
<p style="font-family:sans-serif">Signing in with OpenRouter&hellip; you can close this window.</p>
<script>
  const q = new URLSearchParams(location.search);
  if (window.opener) {
    window.opener.postMessage(
      {type: "openrouter-code", code: q.get("code"), state: q.get("state")},
      location.origin);
    window.close();
  }
</script>"""


@app.get("/openrouter/callback")
def openrouter_callback() -> HTMLResponse:
    """Popup landing page: hands the sign-in code back to the review page."""
    return HTMLResponse(OPENROUTER_CALLBACK_HTML)


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
