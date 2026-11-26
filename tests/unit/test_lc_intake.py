"""Natural-language intake (lc_app/intake.py)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain")

from lc_app import intake

USER_INPUT = (
    "https://arxiv.org/abs/2611.27924v1?ref=biomldigest.com,"
    "https://arxiv.org/abs/2611.24763v1?ref=biomldigest.com,"
    "https://arxiv.org/abs/2611.24763v1?ref=biomldigest.com"
)


def test_comma_separated_tracked_arxiv_links_dedupe():
    remainder, ids = intake.extract_papers(USER_INPUT)
    assert ids == ["2611.27924v1", "2611.24763v1"]
    assert remainder == ""


def test_nature_article_and_arxiv_pdf_links():
    _, ids = intake.extract_papers(
        "https://www.nature.com/articles/s41598-026-71466-6\n"
        "https://arxiv.org/pdf/2608.24688"
    )
    assert ids == ["10.1038/s41598-026-71466-6", "2608.24688"]


def test_mixed_text_keeps_cv_urls_and_topics():
    text = (
        "Recent DNA language model papers for https://lab.example.edu/~me, "
        "also include 10.1038/s41592-025-02814-z, PMID: 41443191, arXiv:2407.06405 and "
        "https://www.biorxiv.org/content/10.1101/2023.04.30.538439v2.full."
    )
    remainder, ids = intake.extract_papers(text)
    assert ids == [
        "10.1101/2023.04.30.538439",
        "41443191",
        "10.1038/s41592-025-02814-z",
        "2407.06405",
    ]
    assert "https://lab.example.edu/~me" in remainder
    assert "DNA language model" in remainder and "biorxiv" not in remainder


def test_no_papers_is_a_plain_request_without_a_model_call(monkeypatch):
    monkeypatch.setattr(
        intake, "build_model", lambda **kw: pytest.fail("no LLM call expected")
    )
    parsed = pytest.importorskip("asyncio").run(
        intake.parse_request("single-cell foundation models")
    )
    assert parsed == intake.ParsedRequest(request="single-cell foundation models")


@pytest.mark.asyncio
async def test_papers_only_add_to_previous_without_a_model_call(monkeypatch):
    monkeypatch.setattr(
        intake, "build_model", lambda **kw: pytest.fail("no LLM call expected")
    )
    parsed = await intake.parse_request(USER_INPUT)
    assert parsed.add_to_previous and parsed.request == ""
    assert parsed.manual_ids == ["2611.27924v1", "2611.24763v1"]


class _IntentStub:
    def __init__(self, reply):
        self.reply, self.seen = reply, None

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.seen = messages[1].content
        return self.reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (intake._Intent(add_to_previous=True, research_request=""), ("", True)),
        (
            intake._Intent(
                add_to_previous=False, research_request="DNA language models"
            ),
            ("DNA language models", False),
        ),
    ],
)
async def test_mixed_message_routes_by_intent(monkeypatch, reply, expected):
    stub = _IntentStub(reply)
    monkeypatch.setattr(intake, "build_model", lambda **kw: stub)
    parsed = await intake.parse_request("add these to my review: " + USER_INPUT)
    assert (parsed.request, parsed.add_to_previous) == expected
    assert stub.seen == "add these to my review"  # links never reach the model
    assert len(parsed.manual_ids) == 2


@pytest.mark.asyncio
async def test_add_to_previous_without_a_previous_review_errors(tmp_path, monkeypatch):
    from lc_app import graph

    monkeypatch.setenv("LC_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="No previous review"):
        await graph.run_review("", manual_ids=["2611.27924v1"], add_to_previous=True)


def test_last_review_pointer_roundtrip(tmp_path, monkeypatch):
    from lc_app import cache

    monkeypatch.setenv("LC_CACHE_DIR", str(tmp_path))
    assert cache.load_last() is None
    cache.save_research("k1", "topics a", {"clusters": []}, [])
    cache.save_last("k1")
    key, entry = cache.load_last()
    assert key == "k1" and entry["request"] == "topics a"
