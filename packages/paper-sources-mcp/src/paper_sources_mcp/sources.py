"""Async adapters for keyless scholarly APIs, plus a pooled, deduplicated search.

Every adapter returns the same normalized record shape (see ``_record``) so the
pooled ``search_papers`` can merge hits across sources:

- OpenAlex       broad coverage (journals + preprints) and citation metrics
- Europe PMC     PubMed/PMC + preprints, real keyword search, abstracts
- PubMed         NCBI E-utilities (esearch + efetch), MEDLINE metadata
- arXiv          preprints (cs, q-bio, stat, physics, math)
- bioRxiv/medRxiv  keyword search via Europe PMC's preprint index (the official
                 bioRxiv API has no keyword search); DOI lookups via
                 api.biorxiv.org

No ADK / LangChain imports: this module runs inside the MCP server subprocess.
Rate limits are enforced per process (NCBI: 3 req/s, 10 with NCBI_API_KEY;
arXiv: 1 req / 3 s), and identical queries are cached for the process lifetime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 20.0
_UA = "literature-reviewer/0.1 (paper review agent; mailto:{mailto})"
_ABSTRACT_CHARS = 800

ALL_SOURCES = ("openalex", "europepmc", "pubmed", "arxiv", "biorxiv")

_cache: dict[tuple, Any] = {}
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        mailto = os.getenv("OPENALEX_MAILTO", "") or "unknown"
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA.format(mailto=mailto)},
            follow_redirects=True,
        )
    return _client


class _Spacer:
    """Minimum interval between requests to one host (process-wide)."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


_NCBI = _Spacer(0.11 if os.getenv("NCBI_API_KEY") else 0.35)
_ARXIV = _Spacer(3.0)


def _auth_headers(url: str) -> dict:
    # OpenAlex: keyless requests share a small per-IP daily budget; a free
    # personal key (https://help.openalex.org/api/authentication/) has its own.
    if "api.openalex.org" in url and os.getenv("OPENALEX_API_KEY"):
        return {"Authorization": f"Bearer {os.environ['OPENALEX_API_KEY']}"}
    return {}


async def _fetch(
    url: str, params: dict, spacer: _Spacer | None = None
) -> httpx.Response:
    """GET with a polite spacer and a short retry on transient 429/5xx."""
    client = _get_client()
    for attempt in range(3):
        if spacer is not None:
            await spacer.wait()
        resp = await client.get(url, params=params, headers=_auth_headers(url))
        retry_after = resp.headers.get("retry-after", "0")
        if resp.status_code == 429 and retry_after.isdigit() and int(retry_after) > 60:
            # A quota, not a burst limit: retrying won't help; surface the reason.
            raise RuntimeError(
                f"429 quota exhausted for {resp.url.host}: {resp.text[:300]}"
            )
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")


async def _cached(key: tuple, fn) -> dict:
    if key in _cache:
        return _cache[key]
    try:
        records = await fn()
        for r in records:  # keep tool output compact; get_paper_details is untruncated
            r["abstract"] = _truncate(r["abstract"], _ABSTRACT_CHARS)
        result = {"status": "success", "results": records}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "results": []}
    _cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _record(**kw) -> dict:
    rec = {
        "title": "",
        "authors": "",
        "institution": "",
        "venue": "",
        "date": "",
        "url": "",
        "doi": "",
        "abstract": "",
        "cited_by_count": None,
        "fwci": None,
        "is_preprint": False,
        "venue_type": "",  # "journal" | "conference" | "workshop" | "preprint" | ""
        "first_author_h_index": None,
        "senior_author_h_index": None,
        "sources": [],
    }
    rec.update({k: v for k, v in kw.items() if v is not None})
    if rec["is_preprint"]:
        rec["venue_type"] = "preprint"
    rec["title"] = _clean(rec["title"])
    rec["doi"] = normalize_doi(rec["doi"])
    if not rec["url"] and rec["doi"]:
        rec["url"] = f"https://doi.org/{rec['doi']}"
    rec["abstract"] = _clean(rec["abstract"])
    return rec


def _clean(s: str | None) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return " ".join(s.split())


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def normalize_doi(doi: str | None) -> str:
    d = (doi or "").strip().lower()
    return re.sub(r"^(https?://)?(dx\.)?doi\.org/|^doi:", "", d)


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def fmt_authors(names: list[str]) -> str:
    """'First A., ..., Senior Z.': first + last when the list is long."""
    names = [n for n in names if n]
    if len(names) <= 2:
        return ", ".join(names)
    return f"{names[0]}, ..., {names[-1]}"


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    return " ".join(w for _, w in sorted(pos))


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------


async def search_openalex(query: str, from_date: str, max_results: int = 8) -> dict:
    async def run():
        params = {
            "search": query,
            "filter": f"from_publication_date:{from_date}",
            "sort": "relevance_score:desc",
            "per_page": max_results,
        }
        if os.getenv("OPENALEX_MAILTO"):
            params["mailto"] = os.environ["OPENALEX_MAILTO"]
        works = (await _fetch("https://api.openalex.org/works", params)).json()
        works = works.get("results", [])
        records = [_openalex_record(w) for w in works]
        await _attach_h_index(works, records)
        return records

    return await _cached(("openalex", query, from_date, max_results), run)


def _openalex_record(w: dict) -> dict:
    auths = w.get("authorships", [])
    inst = next(
        (
            a["institutions"][0].get("display_name", "")
            for a in auths
            if a.get("institutions")
        ),
        "",
    )
    loc = w.get("primary_location") or {}
    source = loc.get("source") or {}
    return _record(
        title=w.get("display_name", ""),
        authors=fmt_authors(
            [(a.get("author") or {}).get("display_name", "") for a in auths]
        ),
        institution=inst,
        venue=source.get("display_name", "") or w.get("type", ""),
        date=w.get("publication_date", ""),
        doi=w.get("doi") or "",
        url=w.get("doi") or w.get("id", ""),
        abstract=_openalex_abstract(w.get("abstract_inverted_index")),
        cited_by_count=w.get("cited_by_count"),
        fwci=w.get("fwci"),
        is_preprint=source.get("type") == "repository" or w.get("type") == "preprint",
        venue_type=_OPENALEX_VENUE_TYPES.get(source.get("type", ""), ""),
        sources=["openalex"],
    )


_OPENALEX_VENUE_TYPES = {"journal": "journal", "conference": "conference"}


def _author_id(authorship: dict) -> str:
    return ((authorship.get("author") or {}).get("id") or "").rsplit("/", 1)[-1]


async def _attach_h_index(works: list[dict], records: list[dict]) -> None:
    """Add first/senior (last) author h-index via ONE batched /authors call.

    Credibility evidence for the ranker. Best-effort: a failure leaves the
    fields as None rather than failing the search.
    """
    pairs = []
    for w in works:
        auths = w.get("authorships") or []
        pairs.append(
            (_author_id(auths[0]), _author_id(auths[-1])) if auths else ("", "")
        )
    ids = sorted({i for pair in pairs for i in pair if i})
    if not ids:
        return
    try:
        resp = await _fetch(
            "https://api.openalex.org/authors",
            {
                "filter": "ids.openalex:" + "|".join(ids[:50]),
                "select": "id,summary_stats",
                "per_page": 50,
            },
        )
        h = {
            a["id"].rsplit("/", 1)[-1]: (a.get("summary_stats") or {}).get("h_index")
            for a in resp.json().get("results", [])
        }
    except Exception:
        return
    for rec, (first, last) in zip(records, pairs, strict=True):
        rec["first_author_h_index"] = h.get(first)
        rec["senior_author_h_index"] = h.get(last)


async def search_europepmc(
    query: str,
    from_date: str,
    max_results: int = 8,
    preprint_servers: tuple[str, ...] = (),
) -> dict:
    """Europe PMC keyword search; ``preprint_servers`` restricts to e.g. bioRxiv."""

    async def run():
        q = f"({query}) AND FIRST_PDATE:[{from_date} TO 3000-12-31]"
        if preprint_servers:
            pubs = " OR ".join(f'PUBLISHER:"{p}"' for p in preprint_servers)
            q += f" AND SRC:PPR AND ({pubs})"
        else:
            # Preprints are the biorxiv adapter's job; excluding them here keeps
            # "found by N sources" counting independent indexes only.
            q += " NOT SRC:PPR"
        params = {
            "query": q,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core",
        }
        data = (
            await _fetch(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search", params
            )
        ).json()
        tag = "biorxiv" if preprint_servers else "europepmc"
        return [
            _europepmc_record(r, tag)
            for r in data.get("resultList", {}).get("result", [])
        ]

    key = ("europepmc", query, from_date, max_results, preprint_servers)
    return await _cached(key, run)


def _europepmc_record(r: dict, tag: str) -> dict:
    src, ext_id = r.get("source", ""), r.get("id", "")
    affil = ""
    for a in (r.get("authorList") or {}).get("author", []):
        details = (a.get("authorAffiliationDetailsList") or {}).get(
            "authorAffiliation", []
        )
        if details:
            affil = details[0].get("affiliation", "")
            break
    venue = (
        (r.get("journalInfo") or {}).get("journal", {}).get("title", "")
        or (r.get("bookOrReportDetails") or {}).get("publisher", "")
        or r.get("journalTitle", "")
    )
    is_preprint = src == "PPR"
    return _record(
        title=r.get("title", ""),
        authors=fmt_authors(
            [s.strip() for s in (r.get("authorString") or "").rstrip(".").split(",")]
        ),
        institution=affil,
        venue=venue
        + (" preprint" if is_preprint and "preprint" not in venue.lower() else ""),
        date=r.get("firstPublicationDate", ""),
        doi=r.get("doi", ""),
        url=""
        if r.get("doi")
        else (f"https://europepmc.org/article/{src}/{ext_id}" if src else ""),
        abstract=r.get("abstractText", ""),
        cited_by_count=r.get("citedByCount"),
        is_preprint=is_preprint,
        venue_type="" if is_preprint else "journal",
        pmid=r.get("pmid") or None,
        sources=[tag],
    )


async def search_pubmed(query: str, from_date: str, max_results: int = 8) -> dict:
    async def run():
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        common = {"db": "pubmed"}
        if os.getenv("NCBI_API_KEY"):
            common["api_key"] = os.environ["NCBI_API_KEY"]
        search = await _fetch(
            f"{base}/esearch.fcgi",
            {
                **common,
                "term": query,
                "mindate": from_date.replace("-", "/"),
                "maxdate": "3000",
                "datetype": "pdat",
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
            },
            _NCBI,
        )
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetched = await _fetch(
            f"{base}/efetch.fcgi",
            {**common, "id": ",".join(ids), "retmode": "xml"},
            _NCBI,
        )
        return parse_pubmed_xml(fetched.text)

    return await _cached(("pubmed", query, from_date, max_results), run)


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        a = cit.find("Article") if cit is not None else None
        if a is None:
            continue
        pmid = cit.findtext("PMID", "")
        names, inst = [], ""
        for au in a.findall("AuthorList/Author"):
            last, ini = au.findtext("LastName", ""), au.findtext("Initials", "")
            names.append(f"{last} {ini}".strip() or au.findtext("CollectiveName", ""))
            inst = inst or au.findtext("AffiliationInfo/Affiliation", "")
        doi = next(
            (
                e.text or ""
                for e in art.findall("PubmedData/ArticleIdList/ArticleId")
                if e.get("IdType") == "doi"
            ),
            "",
        )
        date = ""
        pd = a.find("Journal/JournalIssue/PubDate")
        if pd is not None:
            ym = [pd.findtext("Year", ""), pd.findtext("Month", "")]
            date = " ".join(x for x in reversed(ym) if x) or pd.findtext(
                "MedlineDate", ""
            )
        title_el = a.find("ArticleTitle")
        abstract = " ".join(
            "".join(t.itertext()) for t in a.findall("Abstract/AbstractText")
        )
        out.append(
            _record(
                title="".join(title_el.itertext()) if title_el is not None else "",
                authors=fmt_authors(names),
                institution=inst,
                venue=a.findtext("Journal/Title", ""),
                date=date,
                doi=doi,
                url="" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                abstract=abstract,
                pmid=pmid,
                venue_type="journal",
                sources=["pubmed"],
            )
        )
    return out


async def search_arxiv(query: str, from_date: str, max_results: int = 8) -> dict:
    async def run():
        start = from_date.replace("-", "") + "0000"
        words = [w for w in re.split(r"\W+", query) if w]
        # AND for short queries; long agent phrases AND down to ~0 hits, so OR
        # them and let relevance sort rank multi-term matches first.
        terms = (
            " AND ".join(f"all:{w}" for w in words)
            if len(words) <= 3
            else f"all:({' '.join(words)})"
        )
        params = {
            "search_query": f"({terms}) AND submittedDate:[{start} TO 300001010000]",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        resp = await _fetch("https://export.arxiv.org/api/query", params, _ARXIV)
        return parse_arxiv_atom(resp.text)

    return await _cached(("arxiv", query, from_date, max_results), run)


def parse_arxiv_atom(xml_text: str) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out = []
    for e in ET.fromstring(xml_text).findall("a:entry", ns):
        out.append(
            _record(
                title=e.findtext("a:title", "", ns),
                authors=fmt_authors(
                    [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
                ),
                institution=e.findtext("a:author/arxiv:affiliation", "", ns),
                date=(e.findtext("a:published", "", ns) or "")[:10],
                url=(e.findtext("a:id", "", ns) or "").replace("http://", "https://"),
                doi=e.findtext("arxiv:doi", "", ns),
                abstract=e.findtext("a:summary", "", ns),
                **_arxiv_venue(
                    e.findtext("arxiv:journal_ref", "", ns),
                    e.findtext("arxiv:comment", "", ns),
                ),
                sources=["arxiv"],
            )
        )
    return out


_CONFERENCE = re.compile(
    r"\b(NeurIPS|NIPS|ICML|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|KDD|AISTATS|"
    r"UAI|COLT|RECOMB|ISMB|PSB|MLCB|MICCAI|Proceedings|Conference|Symposium)\b",
    re.I,
)
_ACCEPTED = re.compile(r"\b(accepted|to appear|published)\b[^.;]*", re.I)


def _arxiv_venue(journal_ref: str, comment: str) -> dict:
    """Venue fields for an arXiv entry.

    ``journal_ref`` (author-supplied publication reference) makes it a published
    paper. An "Accepted at X" comment is weaker, self-reported evidence, so it is
    surfaced as ``venue_note`` without dropping the preprint flag.
    """
    journal_ref = " ".join(journal_ref.split())
    if journal_ref:
        kind = "conference" if _CONFERENCE.search(journal_ref) else "journal"
        if re.search(r"workshop", journal_ref, re.I):
            kind = "workshop"
        return {"venue": journal_ref, "is_preprint": False, "venue_type": kind}
    note = _ACCEPTED.search(comment or "")
    return {
        "venue": "arXiv preprint",
        "is_preprint": True,
        "venue_note": " ".join(note.group(0).split()) if note else None,
    }


async def search_biorxiv(query: str, from_date: str, max_results: int = 8) -> dict:
    return await search_europepmc(
        query, from_date, max_results, preprint_servers=("bioRxiv", "medRxiv")
    )


_ADAPTERS = {
    "openalex": search_openalex,
    "europepmc": search_europepmc,
    "pubmed": search_pubmed,
    "arxiv": search_arxiv,
    "biorxiv": search_biorxiv,
}


# ---------------------------------------------------------------------------
# Pooled search + detail lookup
# ---------------------------------------------------------------------------


_MAX_FIELDS = ("cited_by_count", "first_author_h_index", "senior_author_h_index")


def merge_records(per_source: list[list[dict]], limit: int) -> list[dict]:
    """Merge per-source ranked lists into one deduplicated list.

    Two records are the same paper if they share a DOI or a normalized title
    (which also folds a preprint into its published version). The published
    version's venue/DOI/URL win over a preprint's; otherwise merged records keep
    the first non-empty value per field, the max citation count and h-index,
    and the union of ``sources``. Order: papers found by more sources first, then by
    best rank position in any source.
    """
    merged: list[dict] = []
    by_doi: dict[str, int] = {}
    by_title: dict[str, int] = {}
    best_pos: list[int] = []
    for results in per_source:
        for pos, rec in enumerate(results):
            if not rec.get("title"):
                continue
            doi, tkey = rec.get("doi", ""), title_key(rec["title"])
            idx = by_doi.get(doi) if doi else None
            if idx is None:
                idx = by_title.get(tkey)
            if idx is None:
                merged.append(dict(rec, sources=list(rec.get("sources", []))))
                best_pos.append(pos)
                idx = len(merged) - 1
            else:
                cur = merged[idx]
                if cur.get("is_preprint") and not rec.get("is_preprint"):
                    # Published version wins venue/ids; keep the preprint's DOI.
                    if cur.get("doi") and cur["doi"] != rec.get("doi"):
                        cur.setdefault("preprint_doi", cur["doi"])
                    for f in ("venue", "venue_type", "doi", "url", "date"):
                        if rec.get(f):
                            cur[f] = rec[f]
                    cur["is_preprint"] = False
                for k, v in rec.items():
                    if k == "sources":
                        cur["sources"] += [s for s in v if s not in cur["sources"]]
                    elif k == "is_preprint":
                        continue  # handled above: published if any source says so
                    elif k in _MAX_FIELDS and v is not None:
                        cur[k] = max(cur[k] or 0, v)
                    elif not cur.get(k) and v:
                        cur[k] = v
                best_pos[idx] = min(best_pos[idx], pos)
            if doi:
                by_doi.setdefault(doi, idx)
            if merged[idx].get("doi"):
                by_doi.setdefault(merged[idx]["doi"], idx)
            by_title.setdefault(tkey, idx)
    order = sorted(
        range(len(merged)), key=lambda i: (-len(merged[i]["sources"]), best_pos[i])
    )
    return [merged[i] for i in order[:limit]]


async def search_papers(
    query: str,
    from_date: str,
    max_results: int = 12,
    sources: list[str] | None = None,
) -> dict:
    chosen = [s for s in (sources or ALL_SOURCES) if s in _ADAPTERS] or list(
        ALL_SOURCES
    )
    per = max(3, min(10, max_results))
    outcomes = await asyncio.gather(
        *(_ADAPTERS[s](query, from_date, per) for s in chosen)
    )
    errors = {
        s: o["error"]
        for s, o in zip(chosen, outcomes, strict=True)
        if o["status"] != "success"
    }
    results = merge_records([o["results"] for o in outcomes], max_results)
    return {
        "status": "success" if results or not errors else "error",
        "sources_queried": chosen,
        "source_errors": errors,
        "results": results,
    }


_ARXIV_ID = re.compile(
    r"^(arxiv:)?(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+(\.[A-Z]{2})?/\d{7})$", re.I
)


_DOI_IN_TEXT = re.compile(r"10\.\d{4,9}/[^\s?#]+")


def normalize_identifier(identifier: str) -> str:
    """Reduce a pasted id or URL to a DOI, PMID or arXiv id.

    Handles arxiv.org/abs|pdf links, pubmed.ncbi.nlm.nih.gov/<pmid>, doi.org
    links, and publisher/preprint URLs that embed a DOI (e.g. bioRxiv's
    /content/10.1101/...v2, whose version suffix is dropped).
    """
    ident = identifier.strip().rstrip("/")
    # Tracking params (?ref=..., utm_*) and fragments aren't part of the id.
    bare = re.sub(r"[?#].*$", "", ident).rstrip("/") if "://" in ident else ident
    if m := re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?$", bare):
        return m.group(1)
    if m := re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", bare):
        return m.group(1)
    if ident.lower().startswith("pmid:"):
        return ident[5:].strip()
    # nature.com/articles/<suffix> omits the DOI prefix (10.1038/<suffix>).
    if m := re.search(r"nature\.com/articles/([a-z][\w.-]+)$", bare):
        return f"10.1038/{m.group(1)}"
    if ident.startswith(("http://", "https://")) and (m := _DOI_IN_TEXT.search(ident)):
        doi = m.group(0)
        if doi.startswith(("10.1101/", "10.64898/")):
            doi = re.sub(r"v\d+(\.full(\.pdf)?)?$", "", doi)
        return doi
    return ident


async def get_paper_details(identifier: str) -> dict:
    """Full metadata + abstract for a DOI, PMID, arXiv id, or bioRxiv/medRxiv DOI."""
    ident = normalize_identifier(identifier)
    doi = normalize_doi(ident)
    try:
        if ident.isdigit():
            resp = await _fetch(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                {"db": "pubmed", "id": ident, "retmode": "xml"},
                _NCBI,
            )
            recs = parse_pubmed_xml(resp.text)
        elif _ARXIV_ID.match(ident):
            resp = await _fetch(
                "https://export.arxiv.org/api/query",
                {"id_list": ident.split(":", 1)[-1]},
                _ARXIV,
            )
            recs = parse_arxiv_atom(resp.text)
        elif doi.startswith(("10.1101/", "10.64898/")):
            recs = await _biorxiv_details(doi)
        elif doi.startswith("10."):
            resp = await _fetch(f"https://api.openalex.org/works/doi:{doi}", {})
            recs = [_openalex_record(resp.json())]
        else:
            return {
                "status": "error",
                "error": f"Unrecognized identifier: {identifier!r}",
            }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    if not recs:
        return {"status": "error", "error": f"No record found for {identifier!r}"}
    rec = recs[0]
    return {"status": "success", "paper": rec}


async def _biorxiv_details(doi: str) -> list[dict]:
    for server in ("biorxiv", "medrxiv"):
        resp = await _fetch(f"https://api.biorxiv.org/details/{server}/{doi}", {})
        coll = resp.json().get("collection") or []
        if coll:
            c = coll[-1]  # latest version
            published = c.get("published") or ""
            return [
                _record(
                    title=c.get("title", ""),
                    authors=fmt_authors(
                        [a.strip() for a in c.get("authors", "").split(";")]
                    ),
                    institution=c.get("author_corresponding_institution", ""),
                    venue=f"{c.get('server', server)} preprint",
                    date=c.get("date", ""),
                    doi=c.get("doi", doi),
                    abstract=c.get("abstract", ""),
                    is_preprint=True,
                    published_doi="" if published in ("", "NA") else published,
                    category=c.get("category", ""),
                    sources=["biorxiv"],
                )
            ]
    return []


# ---------------------------------------------------------------------------
# Full text (for the post-ranking "deepen" stage; never used for ranking)
# ---------------------------------------------------------------------------
#
# arXiv: LaTeXML HTML at arxiv.org/html/<id> (most papers since late 2023).
# Journals: open-access PMC JATS via Europe PMC fullTextXML.
# bioRxiv/medRxiv: Europe PMC preprint full text, else the published version
# in PMC, else bioRxiv's own JATS (often behind a bot challenge).
# Output is section-budgeted: results first, then methods, then discussion,
# each long section sampled across its subsections rather than head-truncated.

_ARXIV_WEB = _Spacer(1.0)
_FULLTEXT_TIMEOUT = 60.0
_ft_cache: dict[str, dict] = {}

_SKIP_SECTION = re.compile(
    r"introduc|related|background|preliminar|acknowledg|reference|bibliograph|"
    r"author|funding|competing|contribution|availability|supplement|ethics|declaration",
    re.I,
)
_SECTION_BUDGETS = (  # (title pattern, char budget), checked in order
    (
        re.compile(
            r"result|experiment|evaluat|benchmark|ablation|performance|finding", re.I
        ),
        10_000,
    ),
    (
        re.compile(
            r"method|approach|model|architecture|framework|training|dataset|design|algorithm",
            re.I,
        ),
        8_000,
    ),
    (re.compile(r"discussion|conclusion|limitation|future", re.I), 4_000),
)
_TABLE_BUDGET = 3_000
_MIN_SELECTED = 14_000  # below this, topical (non-standard) sections are added
_CODE_HOST = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co|zenodo\.org|"
    r"figshare\.com|codeocean\.com|bitbucket\.org)/[^\s\"'<>)\]]+",
    re.I,
)
_BOILERPLATE_LINK = re.compile(r"github\.com/(arXiv|brucemiller)/", re.I)


def _squash(text: str) -> str:
    return " ".join(text.split())


def _budget_section(title: str, parts: list[tuple[str, str]], budget: int) -> str:
    """Fit a section into ``budget`` chars, giving each subsection a fair share."""
    parts = [(t, _squash(x)) for t, x in parts if _squash(x)]
    total = sum(len(x) for _, x in parts)
    if total <= budget:
        body = "\n".join(f"[{t}] {x}" if t else x for t, x in parts)
    else:
        share = max(400, budget // max(1, len(parts)))
        body = "\n".join(
            (f"[{t}] " if t else "") + _truncate(x, share) for t, x in parts
        )
    return f"## {title}\n{_truncate(body, budget)}"


def _select_sections(sections: list[tuple[str, list[tuple[str, str]]]]) -> list[str]:
    """Pick and budget sections by priority; returns rendered text blocks in
    priority order (results, methods, discussion)."""
    buckets: list[list[str]] = [[] for _ in _SECTION_BUDGETS]
    other: list[tuple[str, list[tuple[str, str]]]] = []
    for title, parts in sections:
        if not title or _SKIP_SECTION.search(title):
            continue
        for i, (pat, budget) in enumerate(_SECTION_BUDGETS):
            if pat.search(title):
                buckets[i].append(_budget_section(title, parts, budget))
                break
        else:
            other.append((title, parts))  # e.g. topical headings instead of "Results"
    out = []
    for (_, budget), blocks in zip(_SECTION_BUDGETS, buckets, strict=True):
        # Several matching sections share their bucket's budget.
        per = budget // max(1, len(blocks))
        out += [_truncate(b, per) for b in blocks]
    # Papers with topical section titles: fill up with those, in order.
    room = _MIN_SELECTED - sum(map(len, out))
    if other and room > 1000:
        per = room // len(other)
        out += [_budget_section(t, parts, max(per, 1000)) for t, parts in other]
    return out


def _links(urls: list[str]) -> list[str]:
    seen = []
    for u in urls:
        u = u.rstrip(".,;")
        if _CODE_HOST.match(u) and not _BOILERPLATE_LINK.search(u) and u not in seen:
            seen.append(u)
    return seen[:10]


def parse_arxiv_html(html: str) -> dict:
    from bs4 import BeautifulSoup  # lazy: only the full-text path needs it

    soup = BeautifulSoup(html, "lxml")
    for m in soup.find_all("math"):  # LaTeXML math -> its TeX alttext
        m.replace_with(f" {m.get('alttext', '')} ")
    for junk in soup.select("cite, .ltx_bibliography, .ltx_note_outer, nav, footer"):
        junk.decompose()

    def title_of(el) -> str:
        t = el.find(class_="ltx_title")
        return re.sub(r"^\s*[\dA-Z.]+\s+", "", t.get_text(" ", strip=True)) if t else ""

    def body_of(el) -> str:  # text without the element's own heading
        t = el.find(class_="ltx_title")
        return (
            el.get_text(" ").replace(t.get_text(" "), "", 1) if t else el.get_text(" ")
        )

    sections = []
    for sec in soup.select("section.ltx_section"):
        subs = sec.select("section.ltx_subsection")
        parts = (
            [(title_of(s), body_of(s)) for s in subs] if subs else [("", body_of(sec))]
        )
        sections.append((title_of(sec), parts))
    tables = [_squash(t.get_text(" | ")) for t in soup.select("figure.ltx_table")]
    title = soup.select_one("h1.ltx_title")
    return {
        "title": _squash(title.get_text(" ")) if title else "",
        "sections": _select_sections(sections),
        "tables": tables,
        "links": _links([a["href"] for a in soup.select("a[href]")]),
    }


def parse_jats(xml: str) -> dict:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml, "lxml-xml")
    body = soup.find("body")
    if body is None:
        return {"title": "", "sections": [], "tables": [], "links": []}
    for junk in body.find_all(["xref", "fig", "disp-formula"]):
        junk.decompose()
    tables = []
    for tw in soup.find_all("table-wrap"):
        tables.append(_squash(tw.get_text(" | ")))
        tw.decompose()

    def title_of(sec) -> str:
        t = sec.find("title", recursive=False)
        return _squash(t.get_text(" ")) if t else ""

    def body_of(sec) -> str:  # text without the section's own heading
        t = sec.find("title", recursive=False)
        return (
            sec.get_text(" ").replace(t.get_text(" "), "", 1)
            if t
            else sec.get_text(" ")
        )

    sections = []
    for sec in body.find_all("sec", recursive=False):
        subs = sec.find_all("sec", recursive=False)
        parts = (
            [(title_of(s), body_of(s)) for s in subs] if subs else [("", body_of(sec))]
        )
        sections.append((title_of(sec), parts))
    urls = [e.get("xlink:href", "") for e in soup.find_all("ext-link")]
    urls += _CODE_HOST.findall(soup.get_text(" "))
    t = soup.find("article-title")
    return {
        "title": _squash(t.get_text(" ")) if t else "",
        "sections": _select_sections(sections),
        "tables": tables,
        "links": _links(urls),
    }


async def _safe(coro, what: str):
    """Run one full-text source; a timeout or error just moves to the next."""
    try:
        return await coro
    except Exception as e:
        logger.debug("fulltext: %s failed: %s: %s", what, type(e).__name__, e)
        return None


async def _pmc_efetch(pmcid: str) -> dict | None:
    """PMC JATS via NCBI E-utilities: also covers author manuscripts that
    Europe PMC's fullTextXML (open-access subset only) does not serve."""
    params = {"db": "pmc", "id": pmcid.removeprefix("PMC")}
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    await _NCBI.wait()
    resp = await _get_client().get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params=params,
        timeout=_FULLTEXT_TIMEOUT,
    )
    if resp.status_code == 200 and "<body" in resp.text:
        return dict(parse_jats(resp.text), source=f"pmc:{pmcid}")
    return None


async def _pmc_fulltext(src_id: str) -> dict | None:
    """PMCID -> NCBI first, Europe PMC second; preprint (PPR) ids -> Europe PMC."""
    if src_id.startswith("PMC") and (
        found := await _safe(_pmc_efetch(src_id), "pmc efetch")
    ):
        return found
    return await _epmc_xml(src_id)


async def _epmc_xml(src_id: str) -> dict | None:
    """Europe PMC full-text XML for a PMCID or preprint (PPR) id, no search."""
    resp = await _get_client().get(
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/{src_id}/fullTextXML",
        timeout=_FULLTEXT_TIMEOUT,  # full JATS XML is often several hundred KB
    )
    if resp.status_code == 200 and "<body" in resp.text:
        return dict(parse_jats(resp.text), source=f"europepmc:{src_id}")
    return None


async def _epmc_fulltext(query: str) -> dict | None:
    """Open-access full text from Europe PMC (PMC articles and some preprints)."""
    resp = await _get_client().get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "format": "json", "resultType": "lite", "pageSize": 3},
        timeout=30.0,
    )
    resp.raise_for_status()
    for r in resp.json().get("resultList", {}).get("result", []):
        src_id = r.get("pmcid") or (r.get("id") if r.get("source") == "PPR" else None)
        if src_id and (found := await _pmc_fulltext(src_id)):
            return found
    return None


_PDF_HEADING = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?\s+|[IVX]{1,4}\.\s+)?"
    r"(abstract|introduction|background|related work|results?(?: and discussion)?|"
    r"experiments?|evaluation|benchmarks?|methods?|materials and methods|"
    r"online methods|approach|model|discussion|conclusions?|limitations|"
    r"references|acknowledg(?:e)?ments?|data availability|code availability)\s*$",
    re.I,
)
_MAX_PDF_BYTES = 30_000_000


def parse_pdf_text(text: str) -> dict:
    """Split extracted PDF text on standard section headings.

    PDFs have no structure, so this is heuristic: a short line that is just a
    known section name (optionally numbered) starts a section. With no
    recognizable headings the whole text becomes one budgeted section.
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    title, buf = "", []
    for line in text.splitlines():
        if len(line) < 60 and (m := _PDF_HEADING.match(line)):
            if buf:
                sections.append((title, [("", " ".join(buf))]))
            title, buf = m.group(1).strip().title(), []
        else:
            buf.append(line)
    if buf:
        sections.append((title, [("", " ".join(buf))]))
    selected = _select_sections(sections)
    if not selected:
        selected = [_budget_section("Full text", [("", text)], 16_000)]
    return {
        "title": "",
        "sections": selected,
        "tables": [],
        "links": _links(_CODE_HOST.findall(text)),
    }


async def _pdf_fulltext(url: str) -> dict | None:
    resp = await _get_client().get(url, timeout=_FULLTEXT_TIMEOUT)
    body = resp.content
    if (
        resp.status_code != 200
        or not body.startswith(b"%PDF")
        or len(body) > _MAX_PDF_BYTES
    ):
        return None
    import io

    from pypdf import PdfReader  # lazy: only the PDF fallback needs it

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-XObject noise

    reader = PdfReader(io.BytesIO(body))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:60])
    return parse_pdf_text(text) if len(text) > 2000 else None


_PMCID = re.compile(r"(PMC\d{5,9})")


async def _unpaywall_fulltext(doi: str) -> dict | None:
    """Legal open copy via Unpaywall: PMC (JATS) first, then repository and
    publisher PDFs. Needs UNPAYWALL_EMAIL (Unpaywall's required contact)."""
    email = os.getenv("UNPAYWALL_EMAIL")
    if not email:
        return None
    resp = await _get_client().get(
        f"https://api.unpaywall.org/v2/{doi}", params={"email": email}
    )
    if resp.status_code != 200:
        return None
    locations = resp.json().get("oa_locations") or []
    urls = [
        u
        for loc in locations
        for u in (
            loc.get("url_for_landing_page"),
            loc.get("url_for_pdf"),
            loc.get("url"),
        )
        if u
    ]
    # 1) PMC copy: structured JATS, fetched by id (PMC's PDF links serve HTML).
    for pmcid in dict.fromkeys(m.group(1) for u in urls if (m := _PMCID.search(u))):
        if found := await _safe(_pmc_fulltext(pmcid), f"unpaywall pmc {pmcid}"):
            return dict(found, source=f"unpaywall→{found['source']}")
    # 2) arXiv copy: its HTML beats parsing a PDF.
    for u in urls:
        if "arxiv.org/" in u and (aid := normalize_identifier(u)) != u.rstrip("/"):
            if found := await _safe(_fulltext_for(aid), f"unpaywall arxiv {aid}"):
                return dict(found, source=f"unpaywall→{found['source']}")
    # 3) PDFs, best version first.
    rank = {"publishedVersion": 0, "acceptedVersion": 1, "submittedVersion": 2}
    for loc in sorted(locations, key=lambda x: rank.get(x.get("version"), 3)):
        pdf = loc.get("url_for_pdf")
        if not pdf or _PMCID.search(pdf):
            continue
        if found := await _safe(_pdf_fulltext(pdf), f"unpaywall pdf {pdf}"):
            host = httpx.URL(pdf).host
            return dict(
                found, source=f"unpaywall→pdf:{host} ({loc.get('version', '?')})"
            )
    return None


async def _fulltext_for(ident: str) -> dict | None:
    doi = normalize_doi(ident)
    if _ARXIV_ID.match(ident) or doi.startswith("10.48550/arxiv."):
        arxiv_id = (
            ident.split(":", 1)[-1]
            if _ARXIV_ID.match(ident)
            else doi.split("arxiv.", 1)[1]
        )
        await _ARXIV_WEB.wait()
        resp = await _get_client().get(
            f"https://arxiv.org/html/{arxiv_id}", timeout=_FULLTEXT_TIMEOUT
        )
        if resp.status_code == 200 and "ltx_section" in resp.text:
            return dict(parse_arxiv_html(resp.text), source="arxiv_html")
        # No HTML rendering (older or LaTeXML-incompatible source): the PDF.
        await _ARXIV_WEB.wait()
        if found := await _pdf_fulltext(f"https://arxiv.org/pdf/{arxiv_id}"):
            return dict(found, source="arxiv_pdf")
        return None
    if ident.isdigit():
        return await _safe(_epmc_fulltext(f"EXT_ID:{ident} AND SRC:MED"), "epmc pmid")
    if not doi.startswith("10."):
        return None
    if found := await _safe(_epmc_fulltext(f'DOI:"{doi}"'), "epmc doi"):
        return found
    if doi.startswith(("10.1101/", "10.64898/")):
        recs = await _safe(_biorxiv_details(doi), "biorxiv details") or []
        published = recs[0].get("published_doi") if recs else ""
        for src, coro in (
            ("epmc published", lambda: _epmc_fulltext(f'DOI:"{published}"')),
            ("unpaywall published", lambda: _unpaywall_fulltext(published)),
        ):
            if published and (found := await _safe(coro(), src)):
                return dict(found, source=found["source"] + " (published version)")
        if found := await _safe(_biorxiv_s3_fulltext(doi), "biorxiv s3"):
            return found
    return await _safe(_unpaywall_fulltext(doi), "unpaywall")


async def _biorxiv_s3_fulltext(doi: str) -> dict | None:
    """bioRxiv/medRxiv JATS from their official requester-pays S3 buckets.

    Opt-in (LC_BIORXIV_S3=1): requests are billed to your AWS account
    (cents per month of content indexed). See paper_sources_mcp/biorxiv_s3.py.
    """
    if os.getenv("LC_BIORXIV_S3") != "1":
        return None
    from paper_sources_mcp import biorxiv_s3

    xml = await asyncio.to_thread(biorxiv_s3.fetch_jats, doi)
    return dict(parse_jats(xml), source="biorxiv_s3") if xml else None


async def get_paper_fulltext(identifier: str) -> dict:
    ident = normalize_identifier(identifier)
    if ident in _ft_cache:
        return _ft_cache[ident]
    try:
        found = await _fulltext_for(ident)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    if not found or not found["sections"]:
        result = {
            "status": "unavailable",
            "error": f"No open full text found for {identifier!r} "
            "(paywalled, or the preprint server blocks automated access).",
        }
    else:
        tables, used = [], 0
        for t in found["tables"]:
            if used + len(t) > _TABLE_BUDGET:
                break
            tables.append(t)
            used += len(t)
        result = {
            "status": "success",
            "identifier": ident,
            "source": found["source"],
            "title": found["title"],
            "sections": found["sections"],
            "tables": tables,
            "links": found["links"],
        }
    _ft_cache[ident] = result
    return result
