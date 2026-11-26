"""Chat-model factory for the LangChain pipeline.

Model choice is a CONFIG knob (env ``LC_REVIEW_MODEL`` or ``--model``), never
chat input. Values are ``init_chat_model`` ids, ``provider:model``:

    anthropic:claude-sonnet-5-5     (default; needs ANTHROPIC_API_KEY)
    openai:gpt-5                     (pip install langchain-openai; OPENAI_API_KEY)
    google_genai:gemini-flash-latest (pip install langchain-google-genai; GOOGLE_API_KEY)
    ollama:qwen3:14b                 (local open-weight model; `ollama pull qwen3:14b`)

Ollama-specific env: ``LC_OLLAMA_NUM_CTX`` (context window, default 32768;
Ollama's own default is far below this pipeline's ~20k-token prompts) and
``LC_OLLAMA_THINK`` ("1" keeps a reasoning model's thinking on; default off,
which is several times faster on a laptop).

A separate variable from the ADK app's ``REVIEW_MODEL``, whose LiteLLM-style ids
(``anthropic/claude-sonnet-5``) don't parse here.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"


def model_id() -> str:
    return os.getenv("LC_REVIEW_MODEL", DEFAULT_MODEL)


def build_model(max_tokens: int = 8192) -> BaseChatModel:
    """Return the configured chat model with retries.

    ``max_tokens`` matters for the ranker: a 12-paper ReviewQueue is long, and
    provider defaults (often 1-4k) would truncate it mid-JSON.
    """
    mid = model_id()
    if mid.startswith("ollama:"):
        # ChatOllama has no max_tokens/max_retries; it truncates silently at
        # num_ctx, so size the window for tool-result-heavy prompts.
        return init_chat_model(
            mid,
            num_predict=max_tokens,
            num_ctx=int(os.getenv("LC_OLLAMA_NUM_CTX", "32768")),
            reasoning=os.getenv("LC_OLLAMA_THINK", "0") == "1",
        )
    return init_chat_model(mid, max_tokens=max_tokens, max_retries=4)
