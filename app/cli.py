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

"""Interactive CLI for the review queue.

Unlike `agents-cli run "<msg>"` (single-shot) or `agents-cli playground` (chat UI),
this *asks* the user for a CV URL and/or research topics, then drives the ADK
pipeline programmatically with a Runner and streams progress to the terminal.

Run:
    uv run python -m app.cli                      # interactive prompts
    uv run python -m app.cli "recent scRNA papers" # free-form natural language
    uv run python -m app.cli --topics "a; b; c"   # non-interactive, structured
    uv run python -m app.cli --cv URL --open      # open the result in a browser
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from app.agent import root_agent  # noqa: E402

HTML_PATH = Path.cwd() / "literature-reviewer.html"


def _prompt_inputs() -> str:
    """Ask the user for a CV URL and/or topics; loop until something is given."""
    print("\nRecent Computational Biology Review Queue")
    print("=========================================")
    print("Provide a CV/website URL, some research topics, or both.\n")
    while True:
        cv = input("CV / website URL (Enter to skip): ").strip()
        topics = input("Research topics / keywords, ';'-separated (Enter to skip): ").strip()
        msg = _build_message(cv, topics)
        if msg:
            return msg
        print("\nPlease enter at least a URL or one topic (or Ctrl-C to quit).\n")


def _build_message(cv: str, topics: str) -> str:
    parts = []
    if cv:
        parts.append(f"CV: {cv}")
    if topics:
        parts.append(f"Research topics: {topics}")
    return "\n".join(parts)


async def _run(message: str) -> str | None:
    """Drive the pipeline, printing progress; return the final message."""
    runner = InMemoryRunner(agent=root_agent, app_name="app")
    session = await runner.session_service.create_session(app_name="app", user_id="cli")
    final, last_author = None, None
    async for ev in runner.run_async(
        user_id="cli",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if ev.author and ev.author != last_author:
            print(f"▶ {ev.author}")
            last_author = ev.author
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if p.function_call:
                    print(f"    ↳ {p.function_call.name}(…)")
        if ev.is_final_response() and ev.content and ev.content.parts:
            txt = ev.content.parts[0].text
            if txt:
                final = txt
    return final


def _run_remote(message: str, url: str) -> None:
    """Drive a DEPLOYED agent (Agent Runtime / Cloud Run) via `agents-cli run`.

    Reuses agents-cli's auth + session handling. The deployed agent saves the
    review as a GCS artifact (LOGS_BUCKET_NAME); fetch it via the artifact API or
    the bucket rather than expecting a local file.
    """
    cmd = ["agents-cli", "run", "--url", url, "--mode", "a2a", message]
    print(f"\nSending to deployed agent at {url} …\n")
    subprocess.run(cmd, check=False)
    print("\n(The deployed agent saves literature-reviewer.html as a GCS artifact.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a computational-biology paper review.")
    ap.add_argument(
        "prompt",
        nargs="*",
        help="Free-form natural-language request, sent verbatim to the agent "
        "(e.g. \"recent single-cell foundation-model papers for a variant-effect PI\"). "
        "Takes precedence over --cv/--topics.",
    )
    ap.add_argument("--cv", default=None, help="CV / website URL")
    ap.add_argument("--topics", default=None, help="Research topics/keywords, ';'-separated")
    ap.add_argument("--model", default=None, help="Override REVIEW_MODEL for this run")
    ap.add_argument("--url", default=None, help="Call a DEPLOYED agent at this URL instead of running locally")
    ap.add_argument("--open", action="store_true", help="Open the result in a browser")
    args = ap.parse_args()

    if args.model:
        os.environ["REVIEW_MODEL"] = args.model

    if args.prompt:
        message = " ".join(args.prompt)  # free-form NL, sent verbatim
    elif args.cv or args.topics:
        message = _build_message(args.cv or "", args.topics or "")
    else:
        message = _prompt_inputs()

    if args.url:
        _run_remote(message, args.url)
        return

    print("\nResearching… (scope → parallel search → rank → render)\n")
    final = asyncio.run(_run(message))
    print("\n" + (final or "(no output)"))

    if HTML_PATH.exists():
        print(f"\nReview app: file://{HTML_PATH}")
        if args.open:
            webbrowser.open(f"file://{HTML_PATH}")


if __name__ == "__main__":
    main()
