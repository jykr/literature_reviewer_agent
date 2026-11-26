"""ADK pipeline stages ported from the LangChain variant (app/stages.py etc.)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import hosted_review, stages
from app.research import N_SLOTS, split_clusters, verify_researcher
from app.sources import record_tool_results, records_in


def ctx(state: dict, name: str = "researcher_0"):
    return SimpleNamespace(state=state, agent_name=name)


def test_tool_records_are_kept_per_researcher():
    state: dict = {}
    t = ctx(state, "researcher_2")
    record_tool_results(None, {}, t, {"results": [{"title": "A"}]})
    record_tool_results(None, {}, t, {"paper": {"title": "B"}})
    record_tool_results(None, {}, t, {"status": "error"})
    assert [r["title"] for r in state["tool_records_researcher_2"]] == ["A", "B"]
    assert records_in("not a dict") == []


def test_verify_researcher_drops_papers_no_tool_returned():
    state = {
        "papers_0": json.dumps(
            [
                {"title": "Real Paper", "url": "model-mangled"},
                {"title": "Invented Paper", "url": "x"},
            ]
        ),
        "tool_records_researcher_0": [
            {
                "title": "Real Paper",
                "url": "https://doi.org/10.1/real",
                "doi": "10.1/real",
            }
        ],
    }
    verify_researcher(ctx(state))
    kept = json.loads(state["papers_0"])
    assert [p["title"] for p in kept] == ["Real Paper"]
    assert kept[0]["url"] == "https://doi.org/10.1/real"  # tool truth wins


def test_split_clusters_skips_research_for_added_only_reviews():
    scope = json.dumps({"clusters": [{"name": "a"}, {"name": "b"}]})
    state = {"scope": scope}
    split_clusters(ctx(state))
    assert state["cluster_0"] and state["cluster_1"] and not state["cluster_2"]
    state = {"scope": scope, "skip_research": True}
    split_clusters(ctx(state))
    assert not any(state[f"cluster_{k}"] for k in range(N_SLOTS))


@pytest.mark.asyncio
async def test_intake_collects_paper_links_from_the_message():
    msg = SimpleNamespace(
        user_content=SimpleNamespace(
            parts=[
                SimpleNamespace(
                    text="DNA models, also https://arxiv.org/abs/2608.24688v1"
                )
            ]
        )
    )
    delta, note = await stages.IntakeAgent(name="intake_agent").run({}, msg)
    assert delta["manual_ids"] == ["2608.24688v1"] and delta["from_date"]
    assert "1 paper" in note


@pytest.mark.asyncio
async def test_rank_batches_added_papers_and_verifies(monkeypatch):
    calls = []
    added = [
        {
            "title": f"P{i}",
            "url": f"u{i}",
            "doi": f"10.1/{i}",
            "origin": "Paperpile: to read",
        }
        for i in range(12)
    ]

    async def fake_call(llm, schema, system, user):
        batch = [m for m in added if f'"title": "{m["title"]}"' in system]
        calls.append(len(batch))
        papers = [
            {
                "rank": i + 1,
                "title": m["title"],
                "url": m["url"],
                "cats": [],
                "rel": 5,
                "impact": 5,
            }
            for i, m in enumerate(batch)
        ]
        if '"Searched"' in system:  # batch 0 also gets the searched papers
            papers += [
                {
                    "rank": 1,
                    "title": "Searched",
                    "url": "s",
                    "cats": [0],
                    "rel": 9,
                    "impact": 9,
                },
                {
                    "rank": 2,
                    "title": "Invented",
                    "url": "i",
                    "cats": [0],
                    "rel": 9,
                    "impact": 9,
                },
            ]
        return {"papers": papers}

    monkeypatch.setattr(stages, "structured_call", fake_call)
    state = {
        "scope": json.dumps({"clusters": [{"name": "c"}]}),
        "papers_0": json.dumps([{"title": "Searched", "url": "s", "doi": "10.9/s"}]),
        "added": added,
    }
    delta, _ = await stages.RankAgent(name="rank_agent").run(state, None)
    papers = delta["review_queue"]["papers"]
    assert sorted(calls) == [2, 10]
    titles = {p["title"] for p in papers}
    assert "Invented" not in titles and len(papers) == 13  # 12 added + 1 searched
    assert all(
        p["origin"] == "Paperpile: to read"
        for p in papers
        if p["title"].startswith("P")
    )
    assert [p["rank"] for p in papers] == list(range(1, len(papers) + 1))


def test_paperpile_key_never_falls_back_to_env_for_hosted_requests(monkeypatch):
    monkeypatch.setenv("PAPERPILE_API_KEY", "owner-key")
    assert stages.current_paperpile_key() == "owner-key"  # local / agents-cli runs
    token = stages.paperpile_key.set("")  # a hosted request with no key typed in
    try:
        assert stages.current_paperpile_key() == ""
    finally:
        stages.paperpile_key.reset(token)


def test_hosted_model_is_bound_to_the_callers_key():
    m = hosted_review.caller_model("anthropic", "", "caller-key")
    assert m.model == "anthropic/" + hosted_review.DEFAULT_MODELS["anthropic"]
    msg = hosted_review.request_message(
        [
            {"kind": "cv", "value": "https://lab.example"},
            {"kind": "keyword", "value": " "},
        ]
    )
    assert msg == "Researcher interests:\n- CV / profile: https://lab.example"


def test_adk_applies_a_focus_only_when_configured(monkeypatch):
    from lc_app.focus import focus_block

    monkeypatch.delenv("LC_REVIEW_FOCUS", raising=False)
    assert focus_block("rank", use_default=False) == ""  # field-general by default
    assert "Frontier machine learning" in focus_block("rank")  # LangChain default
    monkeypatch.setenv("LC_REVIEW_FOCUS", "Plant genomics")
    assert "Plant genomics" in focus_block("rank", use_default=False)


@pytest.mark.asyncio
async def test_save_items_adds_new_papers_with_their_note_and_never_touches_existing():
    from lc_app.paperpile import PaperpileLibrary
    from tests.unit.test_lc_paperpile import FakeSession

    s = FakeSession(refs=[{"id": "R1", "doi": "10.1/old", "labels": []}])
    orig = s.call_tool

    async def call_tool(name, args):
        if name == "save_to_library":
            args = {**args, "labels": args.get("labels", [])}
        return await orig(name, args)

    s.call_tool = call_tool
    out = await PaperpileLibrary(s).save_items(
        [{"doi": "10.1/OLD", "note": "n1"}, {"doi": "10.2/new", "note": "agent note"}]
    )
    assert out == [
        {"doi": "10.1/old", "status": "already in library"},
        {"doi": "10.2/new", "status": "added", "id": "R10.2/new"},
    ]
    saves = [a for n, a in s.calls if n == "save_to_library"]
    assert len(saves) == 1 and saves[0]["dois"] == ["10.2/new"]
    assert saves[0]["note"] == "agent note"
    assert not any(n == "update_references" for n, _ in s.calls)  # notes untouched


def test_save_resolves_dois_from_links():
    from lc_app.core import doi_from_link

    assert doi_from_link(url="https://doi.org/10.1038/s41592-025-0281") == (
        "10.1038/s41592-025-0281"
    )
    assert doi_from_link(url="https://arxiv.org/abs/2608.24688v2") == (
        "10.48550/arxiv.2608.24688"
    )
    assert doi_from_link(doi="10.1/X") == "10.1/x"
    assert doi_from_link(url="https://example.org/blog") == ""


@pytest.mark.asyncio
async def test_deepen_reads_a_saved_paperpile_pdf_when_no_open_copy(monkeypatch):
    async def no_open_copy(ident):
        return {"status": "unavailable"}

    async def fake_call(llm, schema, system, user):
        assert "Saved PDF body" in user
        return {
            "insights": "New metric: a New eval data: b Design & novelty: c Eval limits: d",
            "results": ["Accuracy 0.93 on the held-out set."],
            "limitation": "l",
            "resources": "",
            "approach": {
                "algo": "x",
                "nov": ["Adaptation"],
                "aim": "",
                "question": "",
                "data": "",
                "model": "",
                "method": "",
            },
        }

    class FakeLib:
        async def lookup(self, dois):
            return [
                {"id": "R9", "doi": "10.5/paper", "title": "T", "files": [{"id": "f"}]}
            ]

        async def pdf_text(self, ref_id):
            assert ref_id == "R9"
            return "Saved PDF body. Results: accuracy 0.93 on the held-out set. " * 60

    monkeypatch.setattr(stages.ps, "get_paper_fulltext", no_open_copy)
    monkeypatch.setattr(stages, "structured_call", fake_call)
    monkeypatch.setattr(stages.cache, "load_deep", lambda k: None)
    monkeypatch.setattr(stages.cache, "save_deep", lambda k, v: None)
    paper = {
        "rank": 1,
        "title": "T",
        "url": "https://doi.org/10.5/paper",
        "doi": "10.5/paper",
    }
    out, note = await stages.DeepenAgent(name="deepen_agent")._deepen(
        {"scope": "{}", "candidate_pool": [paper]}, [paper], FakeLib()
    )
    p = out["review_queue"]["papers"][0]
    assert p["evidence"] == "fulltext" and p["fulltext_source"] == "paperpile"
    assert "1 from your Paperpile PDFs" in note


def test_full_text_comment_keeps_the_preprint_prefix():
    from lc_app.core import apply_deep

    p = {
        "comments": "Preprint (not peer-reviewed). Abstract-based note.",
        "results": [],
    }
    ft = {"sections": ["x"], "tables": [], "links": [], "source": "arxiv"}
    d = {
        "insights": "",
        "results": [],
        "limitation": "",
        "resources": "",
        "comments": "The authors report gains on one benchmark only.",
        "approach": {},
    }
    hit, _ = apply_deep(p, d, ft)
    assert hit["comments"] == (
        "Preprint (not peer-reviewed). The authors report gains on one benchmark only."
    )
    hit, _ = apply_deep(p, {**d, "comments": ""}, ft)
    assert "comments" not in hit  # keep the ranker's


def test_writing_style_is_in_every_review_prompt():
    from lc_app.core import DEEPEN_PROMPT, WRITING_STYLE

    assert WRITING_STYLE in DEEPEN_PROMPT
    assert "never mention clusters" in WRITING_STYLE


@pytest.mark.asyncio
async def test_a_paperpile_label_adds_only_its_10_most_recent_papers(monkeypatch):
    from contextlib import asynccontextmanager

    refs = [
        {"title": f"P{i}", "doi": f"10.1/{i}", "created_at": 1000 + i}
        for i in range(15)
    ] + [{"title": "No id", "created_at": 9999}]

    class Lib:
        async def collection_refs(self, name):
            return "to read", refs

    @asynccontextmanager
    async def fake_open(key):
        yield Lib()

    async def details(ident):
        return {"paper": {"title": ident, "doi": ident}}

    monkeypatch.setattr(stages, "open_paperpile", fake_open)
    monkeypatch.setattr(stages.ps, "get_paper_details", details)
    token = stages.paperpile_key.set("k")
    try:
        delta, note = await stages.AddedPapersAgent(name="added_papers").run(
            {"paperpile_include": "to read"}, None
        )
    finally:
        stages.paperpile_key.reset(token)
    dois = sorted(p["doi"] for p in delta["added"])
    assert dois == sorted(f"10.1/{i}" for i in range(5, 15))  # newest 10
    assert all(p["origin"] == "Paperpile: to read" for p in delta["added"])
    assert "the 10 most recent of 15" in note


def _mock_openrouter(monkeypatch, handler):
    import httpx

    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *a, **k):
            super().__init__(*a, transport=httpx.MockTransport(handler), **k)

    monkeypatch.setattr(httpx, "AsyncClient", Client)


@pytest.mark.asyncio
async def test_openrouter_sign_in_exchanges_the_code_for_a_key(monkeypatch):
    import httpx

    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"key": "sk-or-user"})

    _mock_openrouter(monkeypatch, handler)
    key = await hosted_review.exchange_openrouter_code("c0de", "v" * 43)
    assert key == "sk-or-user"
    assert seen["url"] == "https://openrouter.ai/api/v1/auth/keys"
    assert seen["body"] == {
        "code": "c0de",
        "code_verifier": "v" * 43,
        "code_challenge_method": "S256",
    }


@pytest.mark.asyncio
async def test_openrouter_expired_code_is_a_clear_error(monkeypatch):
    import httpx

    _mock_openrouter(monkeypatch, lambda r: httpx.Response(403, json={"error": "x"}))
    with pytest.raises(ValueError, match="sign in again"):
        await hosted_review.exchange_openrouter_code("old", "v" * 43)


def test_openrouter_model_routes_through_litellm_with_the_users_key():
    m = hosted_review.caller_model("openrouter", "", "sk-or-user")
    assert m.model == "openrouter/anthropic/claude-sonnet-5.5"
    assert "openrouter" in hosted_review.SUPPORTED_PROVIDERS


@pytest.mark.asyncio
async def test_remove_trashes_only_a_reference_whose_id_matches_its_doi():
    from lc_app.paperpile import PaperpileLibrary
    from tests.unit.test_lc_paperpile import FakeSession

    s = FakeSession(
        refs=[
            {"id": "R1", "doi": "10.1/mine"},
            {"id": "R2", "doi": "10.2/theirs"},
        ]
    )
    lib = PaperpileLibrary(s)
    out = await lib.trash(
        [{"id": "R1", "doi": "10.1/MINE"}, {"id": "R1", "doi": "10.2/theirs"}]
    )
    assert out == [
        {"doi": "10.1/mine", "status": "removed"},
        {"doi": "10.2/theirs", "status": "not found"},  # id/DOI mismatch: untouched
    ]
    assert s.refs["10.1/mine"].get("trashed") and not s.refs["10.2/theirs"].get(
        "trashed"
    )
    trashes = [u for n, a in s.calls if n == "update_references" for u in a["updates"]]
    assert trashes == [{"id": "R1", "trashed": True}]  # trash only, nothing else edited


@pytest.mark.asyncio
async def test_save_returns_the_new_reference_id_for_undo():
    from lc_app.paperpile import PaperpileLibrary
    from tests.unit.test_lc_paperpile import FakeSession

    s = FakeSession(refs=[{"id": "R1", "doi": "10.1/old"}])
    orig = s.call_tool

    async def call_tool(name, args):
        if name == "save_to_library":
            args = {**args, "labels": args.get("labels", [])}
        return await orig(name, args)

    s.call_tool = call_tool
    out = await PaperpileLibrary(s).save_items(
        [{"doi": "10.1/old"}, {"doi": "10.2/new"}]
    )
    assert out[0] == {
        "doi": "10.1/old",
        "status": "already in library",
    }  # no id: not ours
    assert out[1]["status"] == "added" and out[1]["id"] == "R10.2/new"


@pytest.mark.asyncio
async def test_reanalyze_rewrites_only_full_text_fields_of_papers_it_could_read(
    monkeypatch,
):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_open(key):
        assert key == "pp-key"
        yield object()

    async def fake_deepen(self, state, papers, library):
        assert json.loads(state["scope"])["profile"] == "ML for genomics"
        read, unread = papers
        return {
            "review_queue": {
                "papers": [
                    {
                        **read,
                        "results": ["new"],
                        "evidence": "fulltext",
                        "fulltext_source": "paperpile",
                        "rel": 1,
                    },
                    {**unread, "evidence": "abstract"},
                ]
            }
        }, ""

    monkeypatch.setattr(stages, "open_paperpile", fake_open)
    monkeypatch.setattr(stages.DeepenAgent, "_deepen", fake_deepen)
    out = await hosted_review.reanalyze_with_paperpile(
        [{"rank": 3, "title": "A"}, {"rank": 5, "title": "B"}],
        "anthropic",
        "",
        "model-key",
        "pp-key",
        "ML for genomics",
    )
    assert out["read"] == 1 and out["missing"] == ["B"]
    upd = out["papers"][0]
    assert upd["rank"] == 3 and upd["results"] == ["new"]
    assert upd["fulltext_source"] == "paperpile"
    assert "rel" not in upd  # scores and ranking are never touched
