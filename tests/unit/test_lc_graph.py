"""Offline tests for the LangGraph pipeline's deterministic pieces (lc_app/graph.py)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, ToolMessage

from lc_app import graph


def _tool_msg(payload, as_blocks=False):
    text = json.dumps(payload)
    content = [{"type": "text", "text": text}] if as_blocks else text
    return ToolMessage(content=content, tool_call_id="t1")


def test_tool_records_reads_results_and_details_in_both_content_shapes():
    msgs = [
        AIMessage("thinking"),
        _tool_msg({"results": [{"title": "A"}]}),
        _tool_msg({"status": "success", "paper": {"title": "B"}}, as_blocks=True),
        ToolMessage(content="not json", tool_call_id="t2"),
    ]
    assert [r["title"] for r in graph._tool_records(msgs)] == ["A", "B"]


def test_verify_candidates_drops_fabrications_and_trusts_tool_ids():
    records = [
        {
            "title": "Real Paper",
            "url": "https://doi.org/10.1/real",
            "doi": "10.1/real",
            "cited_by_count": 7,
            "sources": ["openalex", "pubmed"],
        },
        {"title": "By URL only", "url": "https://arxiv.org/abs/2501.1", "doi": ""},
    ]
    papers = [
        {
            "title": "Real paper",
            "url": "https://made-up.example/x",
            "doi": "",
            "cited_by_count": 999,
        },
        {"title": "Re-worded title", "url": "https://arxiv.org/abs/2501.1/"},
        {
            "title": "Invented Paper",
            "url": "https://doi.org/10.9/fake",
            "doi": "10.9/fake",
        },
    ]
    kept, dropped = graph.verify_candidates(papers, records)
    assert dropped == 1
    assert kept[0]["url"] == "https://doi.org/10.1/real"  # model's link replaced
    assert kept[0]["cited_by_count"] == 7
    assert kept[0]["sources"] == ["openalex", "pubmed"]
    assert kept[1]["title"] == "By URL only"


def test_fan_out_sends_one_task_per_cluster():
    state = {
        "scope": {
            "profile": "p",
            "clusters": [{"name": "a", "keywords": []}, {"name": "b", "keywords": []}],
        },
        "from_date": "2025-01-01",
    }
    sends = graph.fan_out(state)
    assert [s.node for s in sends] == ["research", "research"]
    assert [s.arg["k"] for s in sends] == [0, 1]
    assert graph.fan_out({"scope": {"clusters": []}, "from_date": "x"}) == "no_scope"


def test_fetch_page_text_refuses_private_addresses():
    assert graph.fetch_page_text("http://169.254.169.254/latest/meta-data").startswith(
        "(refused"
    )
    assert graph.fetch_page_text("file:///etc/passwd").startswith("(refused")


def test_paper_server_connection_passes_only_allowed_env(monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "k")
    monkeypatch.setenv("OPENALEX_MAILTO", "me@example.org")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    conn = graph.paper_server_connection()
    assert conn["transport"] == "stdio"
    assert conn["args"] == ["-m", "lc_app.paper_mcp_server"]
    assert conn["env"] == {
        "NCBI_API_KEY": "k",
        "OPENALEX_MAILTO": "me@example.org",
    }  # model keys never reach the tool server


class _StubLLM:
    """Stands in for a chat model; with_structured_output(...).ainvoke -> fixed reply."""

    def __init__(self, reply):
        self.reply = reply

    def with_structured_output(self, schema, include_raw=False):
        self.include_raw = include_raw
        return self

    async def ainvoke(self, messages):
        self.prompt = messages[0].content
        reply = self.replies.pop(0) if hasattr(self, "replies") else self.reply
        if not self.include_raw:
            return reply
        if isinstance(reply, Exception):
            return {"raw": None, "parsed": None, "parsing_error": reply}
        return {"raw": None, "parsed": reply, "parsing_error": None}


def _paper(rank, title, url, cats=()):
    from app.schemas import Approach, Paper, Tags

    return Paper(
        rank=rank,
        title=title,
        authors="A",
        institution="I",
        venue="V",
        date="2026",
        url=url,
        insights="New metric: x New eval data: x Design & novelty: x Eval limits: x",
        limitation="l",
        comments="c",
        relevance="r",
        impact=5,
        rel=5,
        cats=list(cats),
        tags=Tags(app=[], method=[]),
        approach=Approach(
            algo="a", nov="Application", aim="a", data="d", model="m", question="q"
        ),
    )


@pytest.mark.asyncio
async def test_rank_node_floors_cats_by_discovery_and_reranks_coverage_first(
    monkeypatch,
):
    from app.schemas import ReviewQueue

    single = {"title": "Single", "url": "https://doi.org/10.1/s", "doi": "10.1/s"}
    both = {"title": "Both", "url": "https://doi.org/10.1/b", "doi": "10.1/b"}
    reply = ReviewQueue(
        papers=[
            _paper(1, "Single", "https://doi.org/10.1/s"),  # ranker omitted cats
            _paper(2, "Both", "https://doi.org/10.1/b"),
            _paper(3, "Hallucinated", "https://doi.org/10.9/x", cats=[0]),
        ]
    )
    stub = _StubLLM(reply)
    monkeypatch.setattr(graph, "build_model", lambda **kw: stub)

    out = await graph.rank_node(
        {
            "scope": {
                "profile": "p",
                "clusters": [
                    {"name": "c0", "keywords": []},
                    {"name": "c1", "keywords": []},
                ],
            },
            "candidates": [
                {"cluster": 0, "papers": [dict(single), dict(both)]},
                {"cluster": 1, "papers": [dict(both)]},
            ],
        }
    )

    papers = out["queue"]["papers"]
    assert [(p["rank"], p["title"], p["cats"]) for p in papers] == [
        (1, "Both", [0, 1]),  # found by two clusters -> promoted
        (2, "Single", [0]),
    ]
    assert '"found_in_clusters": [0, 1]' in stub.prompt


@pytest.mark.asyncio
async def test_rank_node_retries_once_on_invalid_structured_output(monkeypatch):
    from app.schemas import ReviewQueue

    cand = {"title": "Only", "url": "https://doi.org/10.1/o", "doi": "10.1/o"}
    stub = _StubLLM(None)
    stub.replies = [
        ValueError("papers.0.impact: Input should be less than or equal to 10"),
        ReviewQueue(papers=[_paper(1, "Only", "https://doi.org/10.1/o")]),
    ]
    monkeypatch.setattr(graph, "build_model", lambda **kw: stub)

    out = await graph.rank_node(
        {
            "scope": {"profile": "p", "clusters": [{"name": "c0", "keywords": []}]},
            "candidates": [{"cluster": 0, "papers": [cand]}],
        }
    )
    assert [p["title"] for p in out["queue"]["papers"]] == ["Only"]
    assert stub.replies == []  # second attempt consumed


def test_focus_defaults_overrides_and_disables(monkeypatch):
    from lc_app import focus

    monkeypatch.delenv("LC_REVIEW_FOCUS", raising=False)
    assert focus.review_focus() == focus.DEFAULT_FOCUS
    assert "Coverage stays the primary rank driver" in focus.focus_block("rank")

    monkeypatch.setenv("LC_REVIEW_FOCUS", "clinical genomics")
    assert focus.focus_block("scope").startswith(
        "\n\nReview focus: clinical genomics\n"
    )

    for off in ("", "none", " None "):
        monkeypatch.setenv("LC_REVIEW_FOCUS", off)
        assert focus.focus_block("research") == ""


def test_research_prompt_formats_with_and_without_focus(monkeypatch):
    kw = {"profile": "p", "name": "n", "keywords": "k", "from_date": "2025-01-01"}
    monkeypatch.setenv("LC_REVIEW_FOCUS", "none")
    assert graph.RESEARCH_PROMPT.format(**kw, focus="").rstrip().endswith("guessing.")
    monkeypatch.delenv("LC_REVIEW_FOCUS")
    from lc_app.focus import focus_block

    assert "Review focus: Frontier machine learning" in graph.RESEARCH_PROMPT.format(
        **kw, focus=focus_block("research")
    )


def test_verify_candidates_venue_status_comes_from_tool():
    records = [
        {
            "title": "P",
            "url": "https://arxiv.org/abs/1",
            "venue": "arXiv preprint",
            "venue_type": "preprint",
            "is_preprint": True,
            "senior_author_h_index": 30,
        }
    ]
    papers = [
        {
            "title": "P",
            "url": "https://arxiv.org/abs/1",
            "venue": "Nature",
            "venue_type": "journal",
            "is_preprint": False,
        }
    ]
    [kept], _ = graph.verify_candidates(papers, records)
    assert (kept["venue"], kept["venue_type"], kept["is_preprint"]) == (
        "arXiv preprint",
        "preprint",
        True,
    )
    assert kept["senior_author_h_index"] == 30


@pytest.mark.asyncio
async def test_rank_prompt_has_credibility_rules_and_preprints_are_labeled(monkeypatch):
    from app.schemas import ReviewQueue

    cand = {
        "title": "Pre",
        "url": "https://arxiv.org/abs/1",
        "is_preprint": True,
        "venue": "arXiv preprint",
    }
    stub = _StubLLM(ReviewQueue(papers=[_paper(1, "Pre", "https://arxiv.org/abs/1")]))
    monkeypatch.setattr(graph, "build_model", lambda **kw: stub)
    monkeypatch.setenv("LC_REVIEW_FOCUS", "none")

    out = await graph.rank_node(
        {
            "scope": {"profile": "p", "clusters": [{"name": "c0", "keywords": []}]},
            "candidates": [{"cluster": 0, "papers": [cand]}],
        }
    )
    assert "credibility is the PRIMARY signal" in stub.prompt
    [p] = out["queue"]["papers"]
    assert p["comments"].startswith("Preprint (not peer-reviewed). ")
    assert p["venue"] == "arXiv preprint"


def test_candidate_schema_stays_small():
    # Anthropic rejects structured-output schemas past a complexity limit
    # ("Schema is too complex"); tool-truth fields must not be model output.
    props = graph.Candidate.model_json_schema()["properties"]
    assert not set(props) & {
        "is_preprint",
        "venue_type",
        "cited_by_count",
        "fwci",
        "first_author_h_index",
        "senior_author_h_index",
    }
    nullable = [k for k, v in props.items() if "anyOf" in v]
    assert nullable == []


def test_build_model_ollama_maps_limits(monkeypatch):
    pytest.importorskip("langchain_ollama")
    from lc_app import models

    monkeypatch.setenv("LC_REVIEW_MODEL", "ollama:qwen3:14b")
    monkeypatch.delenv("LC_OLLAMA_NUM_CTX", raising=False)
    m = models.build_model(max_tokens=16000)
    assert (m.model, m.num_predict, m.num_ctx, m.reasoning) == (
        "qwen3:14b",
        16000,
        32768,
        False,
    )


class _SchemaStub:
    """build_model() stand-in that answers per structured-output schema."""

    def __init__(self, replies):
        self.replies, self.prompts = replies, []

    def with_structured_output(self, schema, include_raw=False):
        outer = self

        class _Bound:
            async def ainvoke(self, messages):
                outer.prompts.append((schema.__name__, messages[0].content))
                reply = outer.replies[schema.__name__].pop(0)
                if include_raw:
                    return {"raw": None, "parsed": reply, "parsing_error": None}
                return reply

        return _Bound()


@pytest.mark.asyncio
async def test_cached_research_plus_manual_paper_end_to_end(tmp_path, monkeypatch):
    from langchain_core.tools import tool

    from app.schemas import ReviewQueue
    from lc_app import cache

    monkeypatch.setenv("LC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LC_REVIEW_FOCUS", "none")
    searched = {"title": "Searched", "url": "https://doi.org/10.1/s", "doi": "10.1/s"}
    manual_rec = {
        "title": "Mine",
        "url": "https://doi.org/10.2/m",
        "doi": "10.2/m",
        "authors": "Real A., ..., Real Z.",
        "venue": "Cell",
        "venue_type": "journal",
        "is_preprint": False,
        "senior_author_h_index": 55,
    }
    lookups = []

    @tool
    def get_paper_details(identifier: str) -> str:
        """Lookup."""
        lookups.append(identifier)
        return json.dumps({"status": "success", "paper": manual_rec})

    scope = {"profile": "p", "clusters": [{"name": "c0", "keywords": []}]}
    request = "Research topics: c0"
    cache.save_research(
        cache.research_key(request, ""),
        request,
        scope,
        [{"cluster": 0, "papers": [searched]}],
        from_date="2025-01-01",
    )

    analysis = graph.Candidate(
        title="Mine",
        authors="Made Up",
        url="https://wrong.example",
        algo="a",
        data="d",
        model="m",
        evaluation="e",
        question="q",
    )

    def rank_reply():
        return ReviewQueue(
            papers=[
                _paper(1, "Searched", "https://doi.org/10.1/s"),
                _paper(2, "Mine", "https://doi.org/10.2/m"),
            ]
        )

    stub = _SchemaStub(
        {
            "Candidate": [analysis],
            "ReviewQueue": [
                ReviewQueue(papers=[_paper(1, "Searched", "https://doi.org/10.1/s")]),
                rank_reply(),
                rank_reply(),
            ],
        }
    )
    monkeypatch.setattr(graph, "build_model", lambda **kw: stub)

    async def run():
        g = graph.build_graph([get_paper_details])
        return await g.ainvoke(
            {
                "request": request,
                "from_date": "x",
                "output_path": str(tmp_path / "out.html"),
                "manual_ids": ["https://doi.org/10.2/m"],
                "cache_key": cache.research_key(request, ""),
                "cached": True,
                "scope": scope,
                "candidates": cache.load_research(cache.research_key(request, ""))[
                    "candidates"
                ],
            }
        )

    final = await run()
    papers = final["queue"]["papers"]
    # First ranker reply omitted the manual paper -> exactly one retry (see the
    # Candidate, ReviewQueue, ReviewQueue call sequence below).
    assert [(p["title"], p["manual"]) for p in papers] == [
        ("Searched", False),
        ("Mine", True),
    ]
    mine = papers[1]
    assert mine["url"] == "https://doi.org/10.2/m"  # tool truth, not the model's
    assert "Manually added" in (tmp_path / "out.html").read_text()
    assert "(1 manually added)" in final["message"]
    assert [n for n, _ in stub.prompts] == ["Candidate", "ReviewQueue", "ReviewQueue"]
    assert lookups == ["https://doi.org/10.2/m"]

    # Second run: the manual analysis comes from cache (no lookup, no LLM call).
    await run()
    assert lookups == ["https://doi.org/10.2/m"]
    assert [n for n, _ in stub.prompts].count("Candidate") == 1


def test_fan_out_with_cache_and_manual():
    scope = {"profile": "p", "clusters": [{"name": "a", "keywords": []}]}
    base = {"scope": scope, "from_date": "x"}
    assert graph.fan_out({**base, "cached": True}) == "rank"
    sends = graph.fan_out({**base, "cached": True, "manual_ids": ["a", "a", "b"]})
    assert [(s.node, s.arg["identifier"]) for s in sends] == [
        ("manual", "a"),
        ("manual", "b"),
    ]
    assert [s.node for s in graph.fan_out({**base, "manual_ids": ["a"]})] == [
        "research",
        "manual",
    ]


def test_grounded_results_drops_numbers_not_in_source():
    source = "On ClinVar the model reaches 0.91 auROC versus 0.87 for Enformer; recall 85% at 2026."
    from lc_app.core import grounded_results

    kept, dropped = grounded_results(
        [
            "Variant effect (ClinVar): 0.91 auROC vs 0.87 for Enformer.",
            "Recall reaches 85% on held-out variants.",
            "Improves auROC by +0.04 over Enformer.",  # derived number, not in source
            "Trained on 12 species.",  # integers are not checked
        ],
        source,
    )
    assert dropped == 1
    assert all("+0.04" not in k for k in kept) and len(kept) == 3


@pytest.mark.asyncio
async def test_deepen_rewrites_selected_papers_and_marks_abstract_only(
    tmp_path, monkeypatch
):
    from langchain_core.tools import tool

    from app.schemas import Approach

    monkeypatch.setenv("LC_CACHE_DIR", str(tmp_path))
    calls = []

    @tool
    def get_paper_fulltext(identifier: str) -> str:
        """Full text."""
        calls.append(identifier)
        if "2611.27924" in identifier:
            return json.dumps(
                {
                    "status": "success",
                    "source": "arxiv_html",
                    "title": "BioEmu2",
                    "sections": [
                        "## Results\nFree energies within 0.8 kcal/mol of experiment."
                    ],
                    "tables": [],
                    "links": ["https://github.com/example/bioemu2"],
                }
            )
        return json.dumps({"status": "unavailable", "error": "paywalled"})

    deep = graph.DeepAnalysis(
        insights="New metric: a New eval data: b Design & novelty: c Eval limits: d",
        results=[
            "Free energy: within 0.8 kcal/mol of experiment.",
            "Speedup of 37.5% (made up).",
        ],
        limitation="L",
        comments="The authors report agreement with experiment on two systems.",
        resources="https://github.com/example/bioemu2 https://evil.example/x",
        approach=Approach(
            algo="a", nov="Adaptation", aim="x", data="d", model="m", question="q"
        ),
    )
    stub = _SchemaStub({"DeepAnalysis": [deep]})
    monkeypatch.setattr(graph, "build_model", lambda **kw: stub)
    node = graph.make_deepen_node([get_paper_fulltext])
    state = {
        "scope": {"profile": "p"},
        "queue": {
            "papers": [
                {
                    "rank": 1,
                    "title": "BioEmu2",
                    "url": "https://arxiv.org/abs/2611.27924v1",
                    "doi": "",
                    "results": ["old"],
                    "resources": "",
                    "rel": 7,
                },
                {
                    "rank": 2,
                    "title": "Paywalled",
                    "url": "https://doi.org/10.1016/x",
                    "doi": "10.1016/x",
                    "results": ["kept"],
                    "rel": 5,
                },
            ]
        },
    }
    out = (await node(state))["queue"]["papers"]

    bio, pay = out
    assert (
        bio["evidence"] == "fulltext" and bio["rank"] == 1 and bio["rel"] == 7
    )  # ranking untouched
    assert bio["results"] == [
        "Free energy: within 0.8 kcal/mol of experiment."
    ]  # 37.5% dropped
    assert (
        bio["resources"] == "https://github.com/example/bioemu2"
    )  # link not in paper dropped
    assert bio["approach"]["nov"] == ["Adaptation"]
    assert pay["evidence"] == "abstract" and pay["results"] == ["kept"]

    # Second pass: BioEmu2's analysis comes from cache (no fetch, no model call).
    calls.clear()
    out2 = (await node(state))["queue"]["papers"]
    assert out2[0]["results"] == bio["results"]
    assert not any("2611.27924" in c for c in calls)


def test_server_env_allowlist_passes_fulltext_keys_only(monkeypatch):
    for k in ("UNPAYWALL_EMAIL", "LC_BIORXIV_S3", "AWS_PROFILE"):
        monkeypatch.setenv(k, "v")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    env = graph.paper_server_connection()["env"]
    assert {"UNPAYWALL_EMAIL", "LC_BIORXIV_S3", "AWS_PROFILE"} <= set(env)
    assert "ANTHROPIC_API_KEY" not in env


def test_novelty_is_a_set_of_contribution_kinds():
    from app.schemas import Approach, novelty_kinds

    assert novelty_kinds(["algorithm", "Application/resource"]) == [
        "Application",
        "Algorithm",
    ]
    assert novelty_kinds("resource") == ["Application"]
    assert novelty_kinds(["x", ""]) == []
    a = Approach(
        algo="x",
        nov=["adaptation", "Application"],
        aim="",
        question="",
        data="",
        model="",
        method="DNA embeddings projected into the LLM token space",
    )
    assert a.nov == ["Application", "Adaptation"]
    # Older cached reviews: one score, one label, or the three-rating format.
    base = {"algo": "x", "aim": "", "data": "", "model": "", "question": ""}
    assert Approach.model_validate({**base, "nov": 80}).nov == ["Algorithm"]
    assert Approach.model_validate({**base, "nov": "New method"}).nov == ["Algorithm"]
    assert Approach.model_validate({**base, "nov_ml": "Medium"}).nov == ["Adaptation"]
    assert Approach.model_validate(base).nov == []
