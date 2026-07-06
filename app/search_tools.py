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

"""Scholarly search FunctionTools (OpenAlex, Europe PMC, arXiv).

These are ordinary FunctionTools (plain REST calls, keyless), so ANY model —
Gemini, Claude, GPT — can call them; unlike Gemini's internal ``google_search``
grounding, there's no provider lock-in. They return STRUCTURED, verifiable paper
metadata, which serves the SPEC integrity rule (§2.5), and OpenAlex adds
citations + FWCI for bibliometric ranking (§2.4).

Implementation uses the stdlib (urllib + json/xml) to avoid extra dependencies.
Tools run in ADK's threadpool, so blocking HTTP is fine.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_TIMEOUT = 20
_UA = "literature-reviewer/0.1 (compbio paper review agent)"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def search_openalex(query: str, from_date: str, max_results: int) -> dict:
    """Search OpenAlex for recent papers, with citation metrics.

    Best default source: broad coverage (journals + bioRxiv/medRxiv preprints)
    plus bibliometrics (cited_by_count, FWCI) useful for ranking.

    Args:
        query: Free-text search, e.g. topic plus keywords.
        from_date: Earliest publication date as 'YYYY-MM-DD' (e.g. '2025-06-01').
        max_results: How many results to return (e.g. 8).

    Returns:
        dict with 'status' and 'results': a list of paper objects with title,
        authors, institution, venue, date, url/doi, cited_by_count, fwci, abstract.
    """
    mailto = os.getenv("OPENALEX_MAILTO", "")
    params = {
        "search": query,
        "filter": f"from_publication_date:{from_date}",
        "sort": "relevance_score:desc",
        "per_page": max_results,
    }
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    try:
        data = json.loads(_get(url))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "results": []}

    results = []
    for w in data.get("results", []):
        auths = w.get("authorships", [])
        names = [a.get("author", {}).get("display_name", "") for a in auths]
        inst = ""
        for a in auths:
            insts = a.get("institutions", [])
            if insts:
                inst = insts[0].get("display_name", "")
                break
        doi = w.get("doi") or ""
        results.append(
            {
                "title": w.get("display_name", ""),
                "authors": _fmt_authors(names),
                "institution": inst,
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get(
                    "display_name", ""
                )
                or w.get("type", ""),
                "date": w.get("publication_date", ""),
                "url": doi or w.get("id", ""),
                "doi": doi,
                "cited_by_count": w.get("cited_by_count", 0),
                "fwci": w.get("fwci"),
                "oa_url": (w.get("open_access") or {}).get("oa_url", ""),
            }
        )
    return {"status": "success", "results": results}


def search_europepmc(query: str, max_results: int) -> dict:
    """Search Europe PMC (biomedical literature incl. preprints).

    Args:
        query: Free-text query, e.g. topic plus keywords.
        max_results: How many results to return (e.g. 8).

    Returns:
        dict with 'status' and 'results': papers with title, authors, venue,
        date, url/doi, source.
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": max_results,
        "sort": "P_PDATE_D desc",
        "resultType": "core",
    }
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        params
    )
    try:
        data = json.loads(_get(url))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "results": []}

    results = []
    for r in data.get("resultList", {}).get("result", []):
        doi = r.get("doi", "")
        src, ext_id = r.get("source", ""), r.get("id", "")
        url_ = (
            f"https://doi.org/{doi}"
            if doi
            else (f"https://europepmc.org/article/{src}/{ext_id}" if src and ext_id else "")
        )
        results.append(
            {
                "title": r.get("title", ""),
                "authors": r.get("authorString", ""),
                "venue": r.get("journalTitle", "") or r.get("bookOrReportDetails", ""),
                "date": r.get("firstPublicationDate", ""),
                "url": url_,
                "doi": doi,
                "source": src,
                "is_preprint": r.get("pubType", "").lower().find("preprint") >= 0,
            }
        )
    return {"status": "success", "results": results}


def search_arxiv(query: str, max_results: int) -> dict:
    """Search arXiv (e.g. q-bio, cs preprints), newest first.

    Args:
        query: Free-text query, e.g. topic plus keywords.
        max_results: How many results to return (e.g. 8).

    Returns:
        dict with 'status' and 'results': papers with title, authors, date, url,
        abstract.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    try:
        root = ET.fromstring(_get(url))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "results": []}

    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        names = [
            (a.findtext("a:name", default="", namespaces=ns) or "")
            for a in entry.findall("a:author", ns)
        ]
        results.append(
            {
                "title": " ".join((entry.findtext("a:title", "", ns) or "").split()),
                "authors": _fmt_authors(names),
                "venue": "arXiv preprint",
                "date": (entry.findtext("a:published", "", ns) or "")[:10],
                "url": entry.findtext("a:id", "", ns) or "",
                "abstract": " ".join((entry.findtext("a:summary", "", ns) or "").split()),
            }
        )
    return {"status": "success", "results": results}


def _fmt_authors(names: list[str]) -> str:
    """'First A., ..., Senior Z.' — first + last when the list is long."""
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) <= 2:
        return ", ".join(names)
    return f"{names[0]}, ..., {names[-1]}"


SCHOLARLY_TOOLS = [search_openalex, search_europepmc, search_arxiv]
