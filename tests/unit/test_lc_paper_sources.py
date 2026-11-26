"""Offline tests for the pooled paper-sources layer (lc_app/paper_sources.py)."""

from __future__ import annotations

import pytest

from lc_app import paper_sources as ps

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle>
  <MedlineCitation><PMID>123</PMID><Article>
    <Journal><Title>Nature methods</Title>
      <JournalIssue><PubDate><Year>2025</Year><Month>Dec</Month></PubDate></JournalIssue>
    </Journal>
    <ArticleTitle>Nicheformer: a <i>foundation</i> model.</ArticleTitle>
    <Abstract><AbstractText Label="BACKGROUND">Part one.</AbstractText>
      <AbstractText>Part two.</AbstractText></Abstract>
    <AuthorList>
      <Author><LastName>Schaar</LastName><Initials>AC</Initials>
        <AffiliationInfo><Affiliation>Helmholtz Munich</Affiliation></AffiliationInfo></Author>
      <Author><LastName>Mid</LastName><Initials>B</Initials></Author>
      <Author><LastName>Theis</LastName><Initials>FJ</Initials></Author>
    </AuthorList>
  </Article></MedlineCitation>
  <PubmedData><ArticleIdList>
    <ArticleId IdType="pubmed">123</ArticleId>
    <ArticleId IdType="doi">10.1038/S41592-025-02814-Z</ArticleId>
  </ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>"""

ARXIV_XML = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v2</id>
    <published>2025-01-02T00:00:00Z</published>
    <title>A   Model
      for Cells</title>
    <summary>We propose.</summary>
    <author><name>Ada L</name></author>
    <arxiv:doi>10.48550/arXiv.2501.00001</arxiv:doi>
  </entry>
</feed>"""


def test_parse_pubmed_xml():
    [rec] = ps.parse_pubmed_xml(PUBMED_XML)
    assert rec["title"] == "Nicheformer: a foundation model."
    assert rec["doi"] == "10.1038/s41592-025-02814-z"  # normalized lower-case
    assert rec["url"] == "https://doi.org/10.1038/s41592-025-02814-z"
    assert rec["authors"] == "Schaar AC, ..., Theis FJ"
    assert rec["institution"] == "Helmholtz Munich"
    assert rec["date"] == "Dec 2025"
    assert rec["abstract"] == "Part one. Part two."
    assert rec["sources"] == ["pubmed"]


def test_parse_arxiv_atom():
    [rec] = ps.parse_arxiv_atom(ARXIV_XML)
    assert rec["title"] == "A Model for Cells"
    assert rec["url"] == "https://arxiv.org/abs/2501.00001v2"
    assert rec["is_preprint"] and rec["venue"] == "arXiv preprint"


def test_normalize_doi():
    assert ps.normalize_doi("https://doi.org/10.1101/ABC") == "10.1101/abc"
    assert ps.normalize_doi("doi:10.1/x") == "10.1/x"
    assert ps.normalize_doi(None) == ""


def _r(title, doi="", src="openalex", cites=None, **kw):
    return ps._record(title=title, doi=doi, sources=[src], cited_by_count=cites, **kw)


def test_merge_dedupes_by_doi_and_title_and_ranks_multi_source_first():
    openalex = [_r("Solo paper"), _r("Shared Paper", doi="10.1/s", cites=5)]
    pubmed = [
        _r("Shared paper.", doi="10.1/S", src="pubmed", cites=9, institution="MIT"),
        _r("Other", src="pubmed"),
    ]
    biorxiv = [_r("SHARED PAPER", src="biorxiv")]  # preprint: no DOI match, title match

    merged = ps.merge_records([openalex, pubmed, biorxiv], limit=10)

    assert [m["title"] for m in merged] == ["Shared Paper", "Solo paper", "Other"]
    shared = merged[0]
    assert shared["sources"] == ["openalex", "pubmed", "biorxiv"]
    assert shared["cited_by_count"] == 9  # max across sources
    assert shared["institution"] == "MIT"  # first non-empty backfills


def test_merge_respects_limit_and_skips_untitled():
    recs = [_r(f"P{i}") for i in range(5)] + [_r("")]
    assert len(ps.merge_records([recs], limit=3)) == 3
    assert all(m["title"] for m in ps.merge_records([recs], limit=10))


@pytest.mark.asyncio
async def test_search_papers_pools_and_reports_errors(monkeypatch):
    async def ok(src):
        async def f(query, from_date, max_results):
            return {
                "status": "success",
                "results": [_r(f"{src} hit", src=src), _r("Common", src=src)],
            }

        return f

    async def boom(query, from_date, max_results):
        return {"status": "error", "error": "HTTPStatusError: 503", "results": []}

    adapters = {s: await ok(s) for s in ps.ALL_SOURCES}
    adapters["arxiv"] = boom
    monkeypatch.setattr(ps, "_ADAPTERS", adapters)

    out = await ps.search_papers("q", "2025-01-01", max_results=3)
    assert out["status"] == "success"
    assert out["source_errors"] == {"arxiv": "HTTPStatusError: 503"}
    assert out["results"][0]["title"] == "Common"
    assert len(out["results"][0]["sources"]) == 4

    sub = await ps.search_papers("q", "2025-01-01", sources=["pubmed", "bogus"])
    assert sub["sources_queried"] == ["pubmed"]


def test_merge_prefers_published_version_over_preprint():
    preprint = ps._record(
        title="Great Model",
        doi="10.1101/2025.01.01.1",
        venue="bioRxiv preprint",
        is_preprint=True,
        sources=["biorxiv"],
        senior_author_h_index=40,
    )
    journal = ps._record(
        title="Great model.",
        doi="10.1038/s41592-026-1",
        venue="Nature Methods",
        venue_type="journal",
        sources=["pubmed"],
        senior_author_h_index=42,
    )
    [m] = ps.merge_records([[preprint], [journal]], limit=5)
    assert m["venue"] == "Nature Methods" and m["venue_type"] == "journal"
    assert m["is_preprint"] is False
    assert m["doi"] == "10.1038/s41592-026-1"
    assert m["url"] == "https://doi.org/10.1038/s41592-026-1"
    assert m["preprint_doi"] == "10.1101/2025.01.01.1"
    assert m["senior_author_h_index"] == 42

    # Published first, preprint second: still published.
    [m2] = ps.merge_records([[journal], [preprint]], limit=5)
    assert m2["is_preprint"] is False and m2["venue"] == "Nature Methods"


def test_arxiv_venue_from_journal_ref_and_comment():
    assert ps._arxiv_venue("Proc. ICML 2026, PMLR 267", "") == {
        "venue": "Proc. ICML 2026, PMLR 267",
        "is_preprint": False,
        "venue_type": "conference",
    }
    assert ps._arxiv_venue("ICLR 2026 Workshop on GEM", "")["venue_type"] == "workshop"
    assert ps._arxiv_venue("Bioinformatics 41(3), 2025", "")["venue_type"] == "journal"
    v = ps._arxiv_venue("", "12 pages. Accepted at NeurIPS 2025; code at github")
    assert v["is_preprint"] is True and v["venue_note"] == "Accepted at NeurIPS 2025"
    assert ps._arxiv_venue("", "34 pages, 7 figures")["venue_note"] is None


def test_record_marks_preprints():
    assert ps._record(title="x", is_preprint=True)["venue_type"] == "preprint"
    assert ps._record(title="x", venue_type="journal")["venue_type"] == "journal"


@pytest.mark.asyncio
async def test_attach_h_index_batches_one_call_and_tolerates_failure(monkeypatch):
    calls = []

    class _Resp:
        def json(self):
            return {
                "results": [
                    {"id": "https://openalex.org/A1", "summary_stats": {"h_index": 12}},
                    {"id": "https://openalex.org/A3", "summary_stats": {"h_index": 80}},
                ]
            }

    async def fake_fetch(url, params, spacer=None):
        calls.append((url, params["filter"]))
        return _Resp()

    monkeypatch.setattr(ps, "_fetch", fake_fetch)
    works = [
        {
            "authorships": [
                {"author": {"id": "https://openalex.org/A1"}},
                {"author": {"id": None}},
                {"author": {"id": "https://openalex.org/A3"}},
            ]
        },
        {"authorships": []},
    ]
    records = [ps._record(title="a"), ps._record(title="b")]
    await ps._attach_h_index(works, records)
    assert calls == [("https://api.openalex.org/authors", "ids.openalex:A1|A3")]
    assert (
        records[0]["first_author_h_index"],
        records[0]["senior_author_h_index"],
    ) == (12, 80)
    assert records[1]["senior_author_h_index"] is None

    async def broken(url, params, spacer=None):
        raise RuntimeError("429")

    monkeypatch.setattr(ps, "_fetch", broken)
    fresh = [ps._record(title="a"), ps._record(title="b")]
    await ps._attach_h_index(works, fresh)  # must not raise
    assert fresh[0]["senior_author_h_index"] is None


def test_openalex_key_header_only_for_openalex(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "abc")
    assert ps._auth_headers("https://api.openalex.org/works") == {
        "Authorization": "Bearer abc"
    }
    assert ps._auth_headers("https://eutils.ncbi.nlm.nih.gov/x") == {}


ARXIV_HTML = """<html><body><h1 class="ltx_title">Paper</h1>
<section class="ltx_section"><h2 class="ltx_title">1 Introduction</h2><p>Skip me.</p></section>
<section class="ltx_section"><h2 class="ltx_title">2 Results</h2>
  <section class="ltx_subsection"><h3 class="ltx_title">2.1 Benchmark</h3><p>We reach <math alttext="0.91"></math> auROC.</p></section>
  <section class="ltx_subsection"><h3 class="ltx_title">2.2 Ablation</h3><p>Ablation text.</p></section>
</section>
<section class="ltx_section"><h2 class="ltx_title">3 Methods</h2><p>We train. Code: <a href="https://github.com/lab/model">repo</a>
 <a href="https://github.com/arXiv/html_feedback/issues">feedback</a></p></section>
<figure class="ltx_table"><figcaption>Table 1: scores</figcaption><table><tr><td>A</td><td>0.91</td></tr></table></figure>
</body></html>"""


def test_parse_arxiv_html_prioritizes_results_and_filters_links():
    out = ps.parse_arxiv_html(ARXIV_HTML)
    assert out["sections"][0].startswith("## Results")
    assert "[Benchmark] We reach 0.91 auROC." in out["sections"][0]
    assert not any("Skip me" in s for s in out["sections"])
    assert out["links"] == ["https://github.com/lab/model"]
    assert "0.91" in out["tables"][0]


def test_budget_section_samples_every_subsection():
    parts = [(f"S{i}", "x" * 5000) for i in range(4)]
    text = ps._budget_section("Results", parts, budget=4000)
    assert all(f"[S{i}]" in text for i in range(4)) and len(text) <= 4000 + 20


def test_parse_pdf_text_splits_on_headings():
    text = "\n".join(
        [
            "A Great Paper",
            "Abstract",
            "We do things.",
            "1 Introduction",
            "Skip this intro.",
            "2 Results",
            "We reach 0.91 auROC. Code at https://github.com/lab/tool.",
            "3. Methods",
            "Train a model.",
            "References",
            "[1] Someone.",
        ]
        + ["filler line"] * 400
    )
    out = ps.parse_pdf_text(text)
    heads = [s.split("\n")[0] for s in out["sections"]]
    assert heads[:2] == ["## Results", "## Methods"]
    assert not any("Skip this intro" in s for s in out["sections"])
    assert out["links"] == ["https://github.com/lab/tool."[:-1]]


def test_select_sections_fills_with_topical_headings():
    secs = [
        ("Introduction", [("", "intro " * 500)]),
        ("Benchmark saturation", [("", "finding " * 400)]),
        ("Knowledge transfer across species", [("", "transfer " * 400)]),
    ]
    out = ps._select_sections(secs)
    assert [s.split("\n")[0] for s in out] == [
        "## Benchmark saturation",
        "## Knowledge transfer across species",
    ]


@pytest.mark.asyncio
async def test_unpaywall_prefers_pmc_then_arxiv_then_pdf(monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "me@lab.edu")
    order = []

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "oa_locations": [
                    {
                        "version": "submittedVersion",
                        "url_for_pdf": "https://www.biorxiv.org/x.pdf",
                    },
                    {
                        "version": "publishedVersion",
                        "url_for_pdf": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
                    },
                    {
                        "version": "submittedVersion",
                        "url_for_pdf": "https://arxiv.org/pdf/2602.16696",
                    },
                ]
            }

    class _Client:
        async def get(self, url, params=None, timeout=None):
            assert params == {"email": "me@lab.edu"}
            return _Resp()

    async def pmc(src_id):
        order.append(("pmc", src_id))
        raise TimeoutError  # a failing source must not stop the chain

    async def full(ident):
        order.append(("arxiv", ident))
        return {
            "source": "arxiv_html",
            "sections": ["## Results\nx"],
            "tables": [],
            "links": [],
            "title": "",
        }

    async def pdf(url):
        order.append(("pdf", url))
        return None

    monkeypatch.setattr(ps, "_get_client", lambda: _Client())
    monkeypatch.setattr(ps, "_pmc_fulltext", pmc)
    monkeypatch.setattr(ps, "_fulltext_for", full)
    monkeypatch.setattr(ps, "_pdf_fulltext", pdf)
    found = await ps._unpaywall_fulltext("10.64898/2026.02.11.705358")
    assert found["source"] == "unpaywall→arxiv_html"
    assert order == [("pmc", "PMC1234567"), ("arxiv", "2602.16696")]


@pytest.mark.asyncio
async def test_unpaywall_needs_email(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    assert await ps._unpaywall_fulltext("10.1/x") is None


def test_remote_zip_reads_jats_head_and_full_entry(tmp_path):
    import io
    import os
    import zipfile

    from lc_app import biorxiv_s3 as b3

    jats = (
        '<article><front><article-meta><article-id pub-id-type="doi">10.1101/2026.11.03.1'
        "</article-id></article-meta></front><body>"
        + "<p>text</p>" * 5000
        + "</body></article>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.xml", "<manifest/>")
        z.writestr("content/12345.pdf", b"%PDF" + os.urandom(300_000))
        z.writestr("content/12345.xml", jats)
    data = buf.getvalue()
    reads = []

    def read(start, end):
        reads.append(end - start + 1)
        return data[start : end + 1]

    entries = b3.zip_entries(read, len(data))
    name, entry = b3.jats_entry(entries)
    assert name == "content/12345.xml"
    head = b3.read_entry(read, entry, limit=2000)
    assert b3._DOI_IN_JATS.search(head).group(1) == b"10.1101/2026.11.03.1"
    assert b3.read_entry(read, entry).decode() == jats
    assert sum(reads[:3]) < len(data)  # indexing never downloads the whole package
