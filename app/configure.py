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

"""Interactive setup: pick a model + enter its API key, written to .env.

Run:  uv run python -m app.configure

Sets ``REVIEW_MODEL`` (the knob ``app.models.build_model`` reads) and the
matching provider key. Uses getpass so the key is never echoed, and updates
.env in place without touching your other settings. .env is gitignored.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# (label, REVIEW_MODEL value, provider key env var). Key var None = Vertex/ADC.
PRESETS = [
    ("Gemini Flash — AI Studio key (default)", "gemini-flash-latest", "GEMINI_API_KEY"),
    ("Gemini Pro — AI Studio key", "gemini-pro-latest", "GEMINI_API_KEY"),
    ("Gemini Flash — Vertex AI (uses gcloud ADC, no key)", "gemini-flash-latest", None),
    ("Claude Sonnet — Anthropic key", "anthropic/claude-sonnet-5", "ANTHROPIC_API_KEY"),
    ("OpenAI GPT — OpenAI key", "openai/gpt-4o", "OPENAI_API_KEY"),
    ("Custom LiteLLM model id", None, None),
]


def _upsert_env(updates: dict[str, str]) -> None:
    """Set each KEY=value in .env, replacing existing lines or appending."""
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    remaining = dict(updates)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for key, val in remaining.items():
        out.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(out) + "\n")


def main() -> None:
    print("\nConfigure the review model\n" + "=" * 26)
    for i, (label, _, _) in enumerate(PRESETS, 1):
        print(f"  {i}. {label}")
    choice = input(f"\nSelect a model [1-{len(PRESETS)}] (default 1): ").strip() or "1"
    try:
        label, model, key_var = PRESETS[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection; aborting.")
        return

    if model is None:  # custom
        model = input("Enter the model id (e.g. anthropic/claude-opus-4-8): ").strip()
        if not model:
            print("No model id given; aborting.")
            return
        key_var = input("API key env var for it (blank if none, e.g. ANTHROPIC_API_KEY): ").strip() or None

    updates = {"REVIEW_MODEL": model}

    if key_var is None and model.startswith("gemini") and "Vertex" in label:
        # Vertex path: no API key; ensure Vertex backend + project are set.
        updates["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        project = input("GOOGLE_CLOUD_PROJECT (Vertex): ").strip()
        if project:
            updates["GOOGLE_CLOUD_PROJECT"] = project
        updates.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
        print("\nUsing Vertex AI with gcloud ADC (run `gcloud auth application-default login`).")
    elif key_var:
        key = getpass.getpass(f"Enter {key_var} (input hidden): ").strip()
        if key:
            updates[key_var] = key
        else:
            print(f"\nNo key entered — set {key_var} in .env yourself before running.")

    _upsert_env(updates)
    shown = ", ".join(k for k in updates if "KEY" not in k) or "(keys only)"
    print(f"\n✓ Wrote to {ENV_PATH}\n  REVIEW_MODEL={model}\n  updated: {shown}")
    print("\nTest it:  agents-cli run \"CV: <your-cv-url>\"\n")


if __name__ == "__main__":
    main()
