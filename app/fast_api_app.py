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
import urllib.error
from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from pydantic import BaseModel, Field

from app import byo_review

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

load_dotenv()
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

    inputs: list[InputItem]
    provider: str = "gemini"
    model: str = ""
    api_key: str = Field("", alias="apiKey")

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
    provider = (req.provider or "gemini").lower()
    if provider not in byo_review.SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported provider {provider!r}. Choose one of: "
                f"{', '.join(byo_review.SUPPORTED_PROVIDERS)}."
            ),
        )
    return provider, key


@app.get("/")
def root() -> RedirectResponse:
    """Send the bare URL to the review app."""
    return RedirectResponse(url="/reviewer")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Cloud Run."""
    return {"status": "ok"}


@app.get("/reviewer")
def reviewer_page() -> FileResponse:
    """Serve the single-file review app same-origin (so /review needs no CORS)."""
    return FileResponse(os.path.join(AGENT_DIR, "literature-reviewer.html"))


@app.post("/review")
async def review(req: ReviewRequest) -> dict:
    """Regenerate the review on the caller's own API key and return the SPEC §3
    JSON.

    Runs the provider-agnostic pipeline (:mod:`app.byo_review`) -- no ADK, no
    project credentials. Synchronous; the interests panel may use
    ``/review/stream`` instead for per-stage progress.
    """
    provider, key = _require_key(req)
    inputs = [it.model_dump() for it in req.inputs]
    try:
        return await run_in_threadpool(
            byo_review.run_pipeline, inputs, provider, req.model, key
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:400]
        raise HTTPException(
            status_code=502, detail=f"{provider} API {exc.code}: {detail}"
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
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()

        def on_stage(stage: str, label: str) -> None:
            # Invoked from the worker thread -> hop back onto the event loop.
            loop.call_soon_threadsafe(
                events.put_nowait, {"type": "progress", "stage": stage, "label": label}
            )

        async def worker():
            try:
                payload = await run_in_threadpool(
                    byo_review.run_pipeline, inputs, provider, req.model, key, on_stage
                )
                events.put_nowait({"type": "done", "payload": payload})
            except ValueError as exc:
                events.put_nowait({"type": "error", "detail": str(exc)})
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:400]
                events.put_nowait(
                    {"type": "error", "detail": f"{provider} API {exc.code}: {detail}"}
                )
            except Exception as exc:  # noqa: BLE001 - stream must not raise mid-flight
                logger.log_struct(
                    {"event": "review_stream_error", "error": str(exc)}, severity="ERROR"
                )
                events.put_nowait(
                    {"type": "error", "detail": "Internal error generating review."}
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
