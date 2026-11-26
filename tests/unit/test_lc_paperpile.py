"""Paperpile integration (lc_app/paperpile.py + the graph's library node)."""

from __future__ import annotations

import json
import stat
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain")

from lc_app import paperpile
from lc_app.graph import make_library_node, paper_dois


class FakeSession:
    """Minimal in-memory Paperpile: references keyed by DOI, collections by id."""

    def __init__(self, refs=(), collections=()):
        self.refs = {r["doi"].lower(): dict(r) for r in refs}
        self.collections = list(collections)
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "get_collections_from_library":
            data = {"data": self.collections}
        elif name == "get_references_from_library" and "dois" in args:
            data = {"data": [self.refs[d] for d in args["dois"] if d in self.refs]}
        elif name == "get_references_from_library":
            data = {"data": list(self.refs.values())[: args.get("limit", 100)]}
        elif name == "update_collections":
            for u in args["updates"]:
                self.collections.append(
                    {
                        "id": f"L{len(self.collections)}",
                        "name": u["name"],
                        "collection_type": u["type"],
                    }
                )
            data = {"ok": True}
        elif name == "save_to_library":
            for d in args["dois"]:
                self.refs[d] = {"id": f"R{d}", "doi": d, "labels": list(args["labels"])}
            data = {"ok": True}
        elif name == "update_references":
            by_id = {r["id"]: r for r in self.refs.values()}
            for u in args["updates"]:
                by_id[u["id"]].setdefault("labels", []).extend(u["add_labels"])
            data = {"ok": True}
        else:
            raise AssertionError(name)
        return SimpleNamespace(
            isError=False, content=[SimpleNamespace(text=json.dumps(data))]
        )


def test_arxiv_doi_drops_version():
    assert paperpile.arxiv_doi("2608.24688v2") == "10.48550/arxiv.2608.24688"
    assert paperpile.arxiv_doi("arXiv:2608.24688") == "10.48550/arxiv.2608.24688"


def test_paper_dois_from_records_urls_and_arxiv():
    state = {
        "candidates": [
            {"papers": [{"title": "Alpha Model", "url": "u1", "doi": "10.1/A"}]}
        ],
        "manual": [],
    }
    papers = [
        {"rank": 1, "title": "alpha model", "url": "other"},
        {"rank": 2, "title": "B", "url": "https://arxiv.org/abs/2608.24688v1"},
        {"rank": 3, "title": "C", "url": "https://doi.org/10.1038/s41598-026-71466-6"},
        {"rank": 4, "title": "D", "url": "https://example.org/x"},
    ]
    assert paper_dois(papers, state) == {
        1: "10.1/a",
        2: "10.48550/arxiv.2608.24688",
        3: "10.1038/s41598-026-71466-6",
    }


@pytest.mark.asyncio
async def test_save_creates_label_adds_new_and_labels_existing_only():
    s = FakeSession(refs=[{"id": "R1", "doi": "10.1/a", "labels": []}])
    lib = paperpile.PaperpileLibrary(s)
    res = await lib.save(["10.1/A", "10.2/b"], "review-lc")
    assert res == {"added": 1, "labeled_existing": 1, "failed": 0}
    label = next(c["id"] for c in s.collections if c["name"] == "review-lc")
    assert all(label in r["labels"] for r in s.refs.values())
    tools = [n for n, _ in s.calls]
    # Never deletes, trashes or rewrites notes/metadata.
    for _name, args in s.calls:
        for u in args.get("updates", []):
            assert u.get("action") != "delete"
            assert not {"trashed", "note", "metadata"} & set(u)
    saved = [a for n, a in s.calls if n == "save_to_library"]
    assert saved[0]["dois"] == ["10.2/b"] and "update_references" in tools


@pytest.mark.asyncio
async def test_library_node_flags_papers_in_library():
    s = FakeSession(refs=[{"id": "R1", "doi": "10.1/a"}])
    node = make_library_node(paperpile.PaperpileLibrary(s))
    state = {
        "candidates": [
            {
                "papers": [
                    {"title": "A", "url": "ua", "doi": "10.1/a"},
                    {"title": "B", "url": "ub", "doi": "10.2/b"},
                ]
            }
        ],
        "queue": {
            "papers": [
                {"rank": 1, "title": "A", "url": "ua"},
                {"rank": 2, "title": "B", "url": "ub"},
            ]
        },
    }
    out = await node(state)
    assert [p["in_library"] for p in out["queue"]["papers"]] == [True, False]
    assert "1 already in your Paperpile library" in out["library_msg"]
    assert not any(n == "save_to_library" for n, _ in s.calls)


@pytest.mark.asyncio
async def test_profile_unknown_folder_lists_available():
    s = FakeSession(
        collections=[{"id": "F1", "name": "Genomics", "collection_type": "folder"}]
    )
    with pytest.raises(ValueError, match="Available: Genomics"):
        await paperpile.PaperpileLibrary(s).profile("nope")


def test_api_key_uses_bearer_header(monkeypatch):
    monkeypatch.setenv("PAPERPILE_API_KEY", "k123")
    conn = paperpile.paperpile_connection()
    assert conn["headers"] == {"Authorization": "Bearer k123"} and "auth" not in conn


@pytest.mark.asyncio
async def test_token_storage_is_owner_only(tmp_path):
    from mcp.shared.auth import OAuthToken

    store = paperpile.FileTokenStorage(tmp_path / "pp")
    await store.set_tokens(OAuthToken(access_token="t", token_type="Bearer"))
    assert (await store.get_tokens()).access_token == "t"
    assert stat.S_IMODE((tmp_path / "pp" / "tokens.json").stat().st_mode) == 0o600


def test_reference_identifier_prefers_doi_then_arxiv_then_paper_urls():
    from lc_app.graph import reference_identifier

    assert reference_identifier({"doi": "10.1/X", "arxivid": "2601.1"}) == "10.1/x"
    assert reference_identifier({"arxivid": "2608.24688"}) == "2608.24688"
    assert (
        reference_identifier({"url": ["https://arxiv.org/abs/2608.24688v1"]})
        == "2608.24688v1"
    )
    # A plain web page (tool docs, blog) has no paper id: skipped, not guessed.
    assert reference_identifier({"url": "https://github.com/org/tool"}) == ""


def test_fan_out_skips_research_and_passes_origin():
    from lc_app.graph import fan_out

    state = {
        "scope": {"clusters": [{"name": "c"}], "profile": "p"},
        "from_date": "2026-01-01",
        "skip_research": True,
        "manual_ids": ["10.1/a", "10.2/b"],
        "manual_origin": {"10.1/a": "Paperpile: to read"},
    }
    sends = fan_out(state)
    assert [s.node for s in sends] == ["manual", "manual"]
    assert [s.arg["origin"] for s in sends] == ["Paperpile: to read", ""]


@pytest.mark.asyncio
async def test_rank_batches_added_papers_and_keeps_origin(monkeypatch):
    from lc_app import graph

    manual = [
        {
            "title": f"P{i}",
            "url": f"u{i}",
            "doi": f"10.1/{i}",
            "origin": "Paperpile: to read",
        }
        for i in range(23)
    ]
    calls = []

    async def fake_batch(scope, by_cluster, batch, pool):
        calls.append((bool(by_cluster), len(batch)))
        return [
            {**m, "rank": j + 1, "cats": [], "rel": 5, "impact": 5 + j % 3}
            for j, m in enumerate(batch)
        ], 0

    monkeypatch.setattr(graph, "_rank_batch", fake_batch)
    out = await graph.rank_node(
        {
            "scope": {"clusters": [{"name": "c"}]},
            "manual": manual,
            "skip_research": True,
            "candidates": [
                {"cluster": 0, "papers": [{"title": "S", "url": "s", "doi": "10.9/s"}]}
            ],
        }
    )
    papers = out["queue"]["papers"]
    assert sorted(n for _, n in calls) == [3, 10, 10]
    assert sum(has for has, _ in calls) == 1  # searched candidates go to one batch only
    assert len(papers) == 23 and [p["rank"] for p in papers] == list(range(1, 24))
    assert all(p["origin"] == "Paperpile: to read" and p["manual"] for p in papers)
    scores = [p["rel"] + p["impact"] for p in papers]
    assert scores == sorted(scores, reverse=True)


def test_since_date_spans_and_dates():
    import datetime as dt

    from lc_app.cli import since_date

    today = dt.date(2026, 11, 25)
    assert since_date("6m", today) == "2026-05-26"
    assert since_date("30d", today) == "2026-10-26"
    assert since_date("2026-05-01", today) == "2026-05-01"
    with pytest.raises(SystemExit):
        since_date("half a year", today)
