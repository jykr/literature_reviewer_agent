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

"""Security guardrails (ADK callbacks).

The scope agent fetches a USER-SUPPLIED URL with ``load_web_page`` and then feeds
that (untrusted) page text into the model. Two exposures follow, guarded here:

- **SSRF** (``url_ssrf_guard``, a ``before_tool_callback``): a crafted URL could
  point the fetch at internal services or the cloud *metadata* endpoint
  (169.254.169.254 / metadata.google.internal), leaking credentials. We allow only
  http/https to PUBLIC hosts; everything else is blocked before the tool runs.

- **Prompt injection / input abuse** (``injection_guard``, a
  ``before_model_callback``): the user message (and, transitively, fetched page
  text) may try to override instructions or be abusively large. We cap size and
  refuse on known override patterns.

ADK short-circuit contract:
- a ``before_tool_callback`` that returns a dict SKIPS the tool and returns that
  dict to the model as the tool result;
- a ``before_model_callback`` that returns an ``LlmResponse`` SKIPS the model call
  and returns that response.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# ---------------------------------------------------------------------------
# Layer A — SSRF guard for load_web_page
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES = {"http", "https"}
# Hostnames that must never be resolved/fetched, regardless of IP.
_BLOCKED_HOSTNAMES = {"metadata.google.internal", "metadata"}


def _ip_is_public(ip_str: str) -> bool:
    """True only for globally-routable unicast addresses."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # is_private covers RFC-1918 + loopback + link-local (incl. 169.254.169.254).
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _url_is_safe(url: str) -> tuple[bool, str]:
    """Validate scheme + resolve host and reject any non-public address."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, f"scheme '{parsed.scheme}' not allowed (only http/https)"
    host = parsed.hostname
    if not host:
        return False, "URL has no host"
    if host.lower() in _BLOCKED_HOSTNAMES:
        return False, f"host '{host}' is blocked (cloud metadata endpoint)"
    try:
        # Resolve ALL addresses; every one must be public (defends against a
        # public name that resolves to a private/metadata IP — DNS rebinding).
        infos = socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return False, f"could not resolve host '{host}': {e}"
    addrs = {info[4][0] for info in infos}
    for addr in addrs:
        if not _ip_is_public(addr):
            return False, f"host '{host}' resolves to non-public address {addr}"
    return True, "ok"


def url_ssrf_guard(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]:
    """before_tool_callback: block SSRF-prone URLs on load_web_page.

    Returns a dict (skipping the fetch) when the URL is unsafe; returns None to
    let a safe fetch proceed.
    """
    if tool.name != "load_web_page":
        return None
    url = args.get("url", "")
    if not isinstance(url, str) or not url:
        return {"status": "blocked", "reason": "no URL provided"}
    safe, reason = _url_is_safe(url)
    if not safe:
        return {
            "status": "blocked",
            "reason": f"Refused to fetch '{url}': {reason}.",
            "hint": "Provide a public CV/website URL, or paste research topics instead.",
        }
    return None


# ---------------------------------------------------------------------------
# Layer B — prompt-injection / input-abuse guard
# ---------------------------------------------------------------------------

# Cap on the user's message size (characters). Guards cost/DoS; a CV URL plus a
# few topics is well under this.
_MAX_INPUT_CHARS = 8000

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all|any|the)?\s*(previous|prior|above)\s+(instructions|prompts?)",
        r"disregard (the|all|any)?\s*(previous|prior|above|system)",
        r"you are now (an?|the)\b",
        r"system prompt",
        r"reveal (your|the) (system )?(prompt|instructions)",
        r"</?(system|instructions?)>",
    )
]


def _latest_user_text(llm_request: LlmRequest) -> str:
    """Concatenate text parts of the last user turn."""
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            return " ".join(p.text or "" for p in (content.parts or []) if p.text)
    return ""


def _refuse(message: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=message)])
    )


def injection_guard(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """before_model_callback: cap input size and refuse known injection payloads.

    Returns an LlmResponse (skipping the model) when input is abusive; returns
    None to let a clean request proceed.
    """
    text = _latest_user_text(llm_request)
    if len(text) > _MAX_INPUT_CHARS:
        return _refuse(
            "Your message is too long to process safely. Please send a CV/website "
            "URL and a short list of research topics."
        )
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return _refuse(
                "That request looks like an attempt to override my instructions, so "
                "I can't act on it. Send a researcher CV/website URL or a few "
                "research topics and I'll build the review."
            )
    return None
