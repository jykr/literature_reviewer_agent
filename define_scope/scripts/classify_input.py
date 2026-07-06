#!/usr/bin/env python3
"""Classify raw interest inputs into {cv | paper | keyword}.

Deterministic front-half of the scope step (SPEC §2.0). Takes the raw items the
user supplied and tags each with its kind so the model knows how to normalize it.
Keeping this out of the prompt makes classification testable and consistent.

Usage:
    echo '["https://jdoe.github.io/cv", "single-cell atlases", "10.1101/2025.06.01.123456"]' \
        | python classify_input.py
    # or newline-separated:
    printf 'https://arxiv.org/abs/2506.01234\nvariant effect prediction\n' | python classify_input.py

Output: JSON array of {"kind": "cv"|"paper"|"keyword", "value": "<raw>"}.
Heuristics are documented in references/classification_rules.md.
"""
from __future__ import annotations

import json
import re
import sys

# A DOI: 10.<registrant>/<suffix>  (RFC-ish, good enough for triage).
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Hosts that almost always mean "a paper", not "a personal site".
PAPER_HOSTS = (
    "doi.org",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "ncbi.nlm.nih.gov",
    "pubmed",
    "nature.com",
    "science.org",
    "cell.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "oup.com",
    "academic.oup.com",
    "plos.org",
    "elifesciences.org",
    "openreview.net",
    "proceedings.",
)


def classify_one(raw: str) -> dict[str, str]:
    value = raw.strip()
    if not value:
        return {"kind": "keyword", "value": value}

    # A bare DOI (no scheme) is a paper.
    if DOI_RE.fullmatch(value) or (DOI_RE.search(value) and not URL_RE.search(value)):
        return {"kind": "paper", "value": value}

    m = URL_RE.search(value)
    if m:
        host = m.group(0).split("//", 1)[-1].split("/", 1)[0].lower()
        if any(h in host or h in m.group(0).lower() for h in PAPER_HOSTS):
            return {"kind": "paper", "value": value}
        # Any other URL is treated as a CV / personal site to fetch.
        return {"kind": "cv", "value": value}

    # No URL, no DOI: free-text. Long title-like strings *might* be a paper, but
    # without a resolvable id we treat them as keywords the model can refine.
    return {"kind": "keyword", "value": value}


def read_items(argv: list[str]) -> list[str]:
    if len(argv) > 1:
        return argv[1:]
    data = sys.stdin.read().strip()
    if not data:
        return []
    # Accept a JSON array, or newline-separated lines.
    if data.startswith("["):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return [ln for ln in (line.strip() for line in data.splitlines()) if ln]


def main() -> int:
    items = read_items(sys.argv)
    result = [classify_one(x) for x in items]
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
