#!/usr/bin/env python3
"""Deduplicate, merge, and cap a raw cluster list into the final Scope clusters.

Deterministic back-half of the scope step (SPEC §2.1: "Deduplicate/merge
near-identical clusters ... keep each cluster's source list ... cap"). Merging by
hand in a prompt is error-prone; doing it here is repeatable and testable.

Input (stdin): JSON, either a bare array of clusters or {"clusters": [...]}.
Each cluster: {"name"|"label": str, "keywords"|"terms": [str], "source"?: [str]}.

Output (stdout): {"clusters": [{"name", "keywords", "source"}]}, at most
MAX_CLUSTERS, each with 3–8 keywords, sources unioned across merged clusters.

Merge rule: two clusters merge when their names match case-insensitively OR their
keyword sets overlap by Jaccard >= MERGE_THRESHOLD.
"""
from __future__ import annotations

import json
import sys

MAX_CLUSTERS = 5
MIN_KEYWORDS = 3
MAX_KEYWORDS = 8
MERGE_THRESHOLD = 0.6


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _kw(cluster: dict) -> list[str]:
    return cluster.get("keywords") or cluster.get("terms") or []


def _name(cluster: dict) -> str:
    return cluster.get("name") or cluster.get("label") or ""


def _source(cluster: dict) -> list[str]:
    src = cluster.get("source")
    if src is None:
        return []
    return src if isinstance(src, list) else [src]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _should_merge(c1: dict, c2: dict) -> bool:
    if _norm(_name(c1)) and _norm(_name(c1)) == _norm(_name(c2)):
        return True
    k1 = {_norm(k) for k in _kw(c1)}
    k2 = {_norm(k) for k in _kw(c2)}
    return _jaccard(k1, k2) >= MERGE_THRESHOLD


def _merge_into(base: dict, other: dict) -> dict:
    # Union keywords preserving first-seen order; longer/more-specific name wins.
    seen: dict[str, str] = {}
    for k in _kw(base) + _kw(other):
        key = _norm(k)
        if key and key not in seen:
            seen[key] = k.strip()
    name = _name(base)
    if len(_name(other)) > len(name):
        name = _name(other)
    sources: list[str] = []
    for s in _source(base) + _source(other):
        if s and s not in sources:
            sources.append(s)
    return {"name": name, "keywords": list(seen.values()), "source": sources}


def consolidate(raw: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for cluster in raw:
        if not _name(cluster) and not _kw(cluster):
            continue
        for i, existing in enumerate(merged):
            if _should_merge(existing, cluster):
                merged[i] = _merge_into(existing, cluster)
                break
        else:
            merged.append(_merge_into(cluster, cluster))  # normalize shape

    # Cap keyword counts (trim overflow; leave thin clusters for the model to fill).
    for c in merged:
        c["keywords"] = c["keywords"][:MAX_KEYWORDS]

    # Cap cluster count: keep the richest (most keywords, then most sources).
    merged.sort(key=lambda c: (len(c["keywords"]), len(c["source"])), reverse=True)
    capped = merged[:MAX_CLUSTERS]

    # Drop empty source arrays for cleaner output when nothing tracked them.
    for c in capped:
        if not c["source"]:
            c.pop("source", None)
    return capped


def main() -> int:
    data = sys.stdin.read().strip()
    if not data:
        json.dump({"clusters": []}, sys.stdout)
        sys.stdout.write("\n")
        return 0
    parsed = json.loads(data)
    raw = parsed["clusters"] if isinstance(parsed, dict) else parsed
    result = consolidate(raw)
    thin = [c["name"] for c in result if len(c["keywords"]) < MIN_KEYWORDS]
    if thin:
        # Advisory only (stderr): the model should top these up to >=3 keywords.
        sys.stderr.write(f"warning: clusters below {MIN_KEYWORDS} keywords: {thin}\n")
    json.dump({"clusters": result}, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
