"""On-disk cache for research results and manually added paper analyses.

Re-running the same request (same text + review focus) reuses the cached scope
and verified candidates, so adding papers or re-ranking skips the slow search
stage. Manually added papers are cached per identifier so each is analyzed once.

Location: ``LC_CACHE_DIR`` (default ``.lc_cache/`` in the working directory).
Plain JSON: safe to inspect, edit or delete.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.getenv("LC_CACHE_DIR", ".lc_cache"))


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def research_key(request: str, focus: str) -> str:
    """Whitespace/case-insensitive key over what shapes the research."""
    norm = " ".join(request.lower().split())
    return _digest(norm, focus)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_research(key: str) -> dict | None:
    """{'scope', 'candidates', 'from_date', 'model', 'saved'} or None."""
    return _read(cache_dir() / f"research-{key}.json")


def save_research(
    key: str, request: str, scope: dict, candidates: list[dict], **meta
) -> None:
    _write(
        cache_dir() / f"research-{key}.json",
        {
            "request": request,
            "scope": scope,
            "candidates": candidates,
            "saved": dt.datetime.now().isoformat(timespec="seconds"),
            **meta,
        },
    )


def save_last(key: str) -> None:
    """Remember the most recent review, for "add these papers to my review"."""
    _write(cache_dir() / "last.json", {"key": key})


def load_last() -> tuple[str, dict] | None:
    """(key, research entry) of the most recent review, or None."""
    last = _read(cache_dir() / "last.json") or {}
    entry = load_research(last.get("key", "")) if last.get("key") else None
    return (last["key"], entry) if entry else None


def manual_key(identifier: str) -> str:
    return _digest(re.sub(r"\s+", "", identifier.lower()))


def load_manual(identifier: str) -> dict | None:
    return _read(cache_dir() / "manual" / f"{manual_key(identifier)}.json")


def save_manual(identifier: str, paper: dict) -> None:
    _write(cache_dir() / "manual" / f"{manual_key(identifier)}.json", paper)


# Bump when the full-text instructions or DeepAnalysis fields change.
# 3 comments + method novelty; 4 length limits; 5 novelty kinds; 6 comment length
DEEP_VERSION = 6


def load_deep(paper_key: str) -> dict | None:
    """Cached full-text ("deepen") analysis for one paper, by DOI/title key.

    Entries written under older instructions (an older ``DEEP_VERSION``) count
    as missing, so the paper is re-read once with the current ones.
    """
    hit = _read(cache_dir() / "deep" / f"{_digest(paper_key)}.json")
    return hit if hit and hit.get("v") == DEEP_VERSION else None


def save_deep(paper_key: str, analysis: dict) -> None:
    _write(
        cache_dir() / "deep" / f"{_digest(paper_key)}.json",
        {**analysis, "v": DEEP_VERSION},
    )
