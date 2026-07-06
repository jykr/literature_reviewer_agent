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

import contextlib
import json
import os
import uuid
from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from google.genai import types as genai_types
from pydantic import BaseModel

from app.render import build_payload, categories_from_scope
from app.schemas import ReviewQueue

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from app.app_utils.telemetry import (
    setup_agent_engine_telemetry,
    setup_telemetry,
)
from app.app_utils.typing import Feedback

load_dotenv()
setup_telemetry()
# Must run before get_fast_api_app to set the tracer provider resource.
setup_agent_engine_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    lifespan=lifespan,
)
app.title = "literature-reviewer"
app.description = "API for interacting with the Agent literature-reviewer"


# Proxy routes so the Vertex AI Console Playground (reasoning_engine SDK) can
# talk to this agent alongside the native adk_api routes.
attach_reasoning_engine_routes(app)


class InputItem(BaseModel):
    """One interest input from the web app (SPEC §2.0/§3d)."""

    kind: str  # 'cv' | 'paper' | 'keyword'
    value: str


class ReviewRequest(BaseModel):
    inputs: list[InputItem]


_INPUT_LABELS = {
    "cv": "CV / profile",
    "paper": "Paper",
    "keyword": "Research topic/keyword",
}


@app.get("/reviewer")
def reviewer_page() -> FileResponse:
    """Serve the single-file review app same-origin (so /review needs no CORS)."""
    return FileResponse(os.path.join(AGENT_DIR, "literature-reviewer.html"))


@app.post("/review")
async def review(req: ReviewRequest, request: Request) -> dict:
    """Run the scope→research→rank pipeline for the given inputs, return JSON.

    Mirrors the offline file's data model (SPEC §3) but returns it as an API
    response the interests panel re-renders in place, instead of a baked file.
    """
    runner: Runner = request.app.state.runner
    app_name: str = request.app.state.agent_app_name

    lines = [
        f"- {_INPUT_LABELS.get(it.kind, it.kind)}: {it.value.strip()}"
        for it in req.inputs
        if it.value and it.value.strip()
    ]
    if not lines:
        raise HTTPException(status_code=400, detail="Provide at least one input.")
    text = (
        "Build my recent computational-biology review from these interest inputs:\n"
        + "\n".join(lines)
    )

    user_id = "web"
    session_id = uuid.uuid4().hex  # auto-created by the Runner (auto_create_session)
    message = genai_types.Content(
        role="user", parts=[genai_types.Part(text=text)]
    )
    async for _ in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        pass  # drain the event stream; final data lands in session state

    session = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    state = session.state if session else {}

    raw = state.get("review_queue")
    if raw is None:
        # scope_agent asked for more input, or nothing verifiable was found.
        raise HTTPException(
            status_code=422,
            detail="No review produced. Provide a CV/website URL or a few "
            "research topics/keywords.",
        )
    if isinstance(raw, str):
        raw = json.loads(raw)
    queue = ReviewQueue.model_validate(raw)

    scope = state.get("scope")
    profile = ""
    if isinstance(scope, str):
        with contextlib.suppress(json.JSONDecodeError):
            profile = json.loads(scope).get("profile", "")
    elif isinstance(scope, dict):
        profile = scope.get("profile", "")

    payload = build_payload(
        queue, profile=profile, categories=categories_from_scope(scope)
    )
    payload["inputs"] = [it.model_dump() for it in req.inputs]
    return payload


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
