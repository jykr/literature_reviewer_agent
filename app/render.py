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

"""Deterministic renderer: ReviewQueue -> self-contained literature-reviewer.html.

This is plain Python string templating on purpose (SPEC §2.5, §10): the LLM
produces *data*, never HTML. We split each Paper back into the SPEC's three JS
structures (DATA / TAGS / APPROACH, §3) and inject them into a fixed template
that implements the render + interaction contract (§4-§9).
"""

from __future__ import annotations

import json

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from app.schemas import LabeledQueue, ReviewQueue


# Category chip palette (SPEC §3d/§8): cycled per cluster index.
_CAT_COLORS = [
    "#2b6cb0", "#b7791f", "#2f855a", "#805ad5",
    "#c0392b", "#0987a0", "#b83280", "#4a5568",
]


def categories_from_scope(scope) -> list[dict]:
    """Derive the CATEGORIES list (SPEC §3d) from a scope dict/JSON.

    Each scope cluster becomes one scoring category; ``id`` is its 0-based index
    (matching the ``cats`` indices the ranker emits). ``sources`` is left empty
    here — the caller (which holds the raw INPUTS) can populate it.
    """
    if isinstance(scope, str):
        try:
            scope = json.loads(scope)
        except json.JSONDecodeError:
            scope = {}
    clusters = (scope or {}).get("clusters", []) if isinstance(scope, dict) else []
    return [
        {
            "id": i,
            "label": (c.get("name") if isinstance(c, dict) else str(c)) or f"Cluster {i}",
            "sources": [],
            "color": _CAT_COLORS[i % len(_CAT_COLORS)],
        }
        for i, c in enumerate(clusters)
    ]


def build_payload(
    queue: ReviewQueue,
    profile: str = "",
    generated: str = "",
    categories: list[dict] | None = None,
) -> dict:
    """Split a ReviewQueue into the SPEC §3 JS structures (single source of truth).

    Returned dict is consumed both by :func:`build_html` (baked into the offline
    file) and by the live ``/review`` JSON endpoint (re-rendered client-side).
    """
    data = []
    tags = {}
    approach = {}
    for p in sorted(queue.papers, key=lambda x: x.rank):
        cats = list(getattr(p, "cats", []) or [])
        data.append(
            {
                "rank": p.rank,
                "title": p.title,
                "authors": p.authors,
                "institution": p.institution,
                "venue": p.venue,
                "date": p.date,
                "url": p.url,
                "insights": p.insights,
                "results": list(getattr(p, "results", []) or []),
                "limitation": p.limitation,
                "resources": p.resources,
                "comments": p.comments,
                "relevance": p.relevance,
                "impact": p.impact,
                "rel": p.rel,
                "cats": cats,
                "cover": len(cats),
                # Optional: set by the LangChain variant for user-added papers.
                "manual": bool(getattr(p, "manual", False)),
                "evidence": getattr(p, "evidence", "") or "",
                "in_library": bool(getattr(p, "in_library", False)),
                "origin": getattr(p, "origin", "") or "",
                "doi": getattr(p, "doi", "") or "",
            }
        )
        tags[str(p.rank)] = {"app": p.tags.app, "method": p.tags.method}
        approach[str(p.rank)] = {
            "algo": p.approach.algo,
            "nov": p.approach.nov,
            "aim": p.approach.aim,
            "question": p.approach.question,
            "data": p.approach.data,
            "model": p.approach.model,
            "method": p.approach.method,
        }

    return {
        "data": data,
        "tags": tags,
        "approach": approach,
        "categories": categories or [],
        "profile": profile,
        "generated": generated,
    }


def build_html(
    queue: ReviewQueue,
    profile: str = "",
    generated: str = "",
    categories: list[dict] | None = None,
    inputs: list[dict] | None = None,
) -> str:
    """Render the review queue into the single-file HTML app.

    ``categories`` seeds the coverage chips; ``inputs`` seeds the interests panel
    (empty in the offline file — the user edits it and regenerates via /review).
    """
    payload = build_payload(
        queue, profile=profile, generated=generated, categories=categories
    )
    return (
        _TEMPLATE.replace("__DATA__", _json_for_script(payload["data"]))
        .replace("__TAGS__", _json_for_script(payload["tags"]))
        .replace("__APPROACH__", _json_for_script(payload["approach"]))
        .replace("__CATEGORIES__", _json_for_script(payload["categories"]))
        .replace("__INPUTS__", _json_for_script(inputs or []))
        .replace("__PROFILE__", _esc(profile))
        .replace("__GENERATED__", _esc(generated))
    )


async def assemble_html_callback(callback_context: CallbackContext):
    """after_agent_callback for the last stage: render the HTML and save it.

    Reads the ranked queue from ``state['review_queue']`` (with the post-rank
    labels: added, Paperpile origin, full text, in library) and the profile
    from ``state['scope']``. Saves the result as
    an ADK artifact (portable to Cloud Run / Agent Runtime) and, when the local
    filesystem is writable, also drops ``literature-reviewer.html`` in the cwd.
    Overrides the agent's raw-JSON message with a friendly summary.
    """
    raw = callback_context.state.get("review_queue")
    if raw is None:
        return None  # nothing to render; let the agent's own output stand
    if isinstance(raw, str):
        raw = json.loads(raw)
    queue = LabeledQueue.model_validate(raw)

    if not queue.papers:
        # Nothing to review — usually means no CV URL or topics were given.
        return genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    text="No papers to review yet. Provide a CV/website URL or a "
                    "few research topics/keywords to start the review."
                )
            ],
        )

    profile = ""
    scope = callback_context.state.get("scope")
    if isinstance(scope, str):
        try:
            profile = json.loads(scope).get("profile", "")
        except json.JSONDecodeError:
            profile = ""
    elif isinstance(scope, dict):
        profile = scope.get("profile", "")

    html = build_html(
        queue, profile=profile, categories=categories_from_scope(scope)
    )
    html_bytes = html.encode("utf-8")

    # Portable path: save as an artifact the user can download.
    part = genai_types.Part(
        inline_data=genai_types.Blob(mime_type="text/html", data=html_bytes)
    )
    try:
        await callback_context.save_artifact("literature-reviewer.html", part)
    except Exception:
        pass  # no artifact service configured (e.g. bare `agents-cli run`)

    # Convenience for local dev: also write to disk if we can. Never for a
    # hosted request: the server's literature-reviewer.html is the public page.
    disk_note = ""
    if not callback_context.state.get("hosted"):
        try:
            with open("literature-reviewer.html", "wb") as fh:
                fh.write(html_bytes)
            disk_note = " and written to ./literature-reviewer.html"
        except OSError:
            pass

    n = len(queue.papers)
    n_added = sum(p.manual for p in queue.papers)
    n_full = sum(p.evidence == "fulltext" for p in queue.papers)
    n_lib = sum(p.in_library and not p.origin for p in queue.papers)
    details = [f"{n_full} read in full text"]
    if n_added:
        details.append(f"{n_added} added by you")
    if n_lib:
        details.append(f"{n_lib} already in your Paperpile")
    if failed := callback_context.state.get("added_failed"):
        details.append(f"not found: {', '.join(failed)}")
    summary = (
        f"Reviewed and ranked {n} paper{'s' if n != 1 else ''} "
        f"({'; '.join(details)}). "
        f"Saved the self-contained review app as artifact 'literature-reviewer.html'"
        f"{disk_note}. Open it in a browser to tag papers and copy notes."
    )
    return genai_types.Content(role="model", parts=[genai_types.Part(text=summary)])


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_for_script(obj) -> str:
    """JSON-encode for embedding inside a <script> block, XSS-safe.

    Paper text is untrusted (it comes from web/API search results). json.dumps
    does NOT escape ``</script>`` or ``<!--``, so a crafted title could break out
    of the ``<script type="application/json">`` tag and execute. We escape ``<``,
    ``>`` and ``&`` as \\uXXXX — still valid JSON (the browser's JSON.parse decodes
    the escapes), but no HTML tag can be formed.
    """
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


# ---------------------------------------------------------------------------
# The template. `__DATA__` / `__TAGS__` / `__APPROACH__` are replaced with JSON;
# `__PROFILE__` / `__GENERATED__` with escaped text. All CSS/JS is inlined so the
# file opens from file:// with no network or build step (SPEC §1, §10).
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Triage</title>
<style>
:root{
  --bg:#f6f7f9; --card:#fff; --ink:#1a1d21; --muted:#5b6570; --line:#e3e7ec;
  --accent:#2b6cb0; --accent-soft:#e8f0f8;
  --important:#c0392b; --further:#b7791f; --done:#2f855a;
  --pp:#7b3fb3; --pp-soft:#f6f0fb;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.5;overflow-x:hidden}
a{color:var(--accent)}
header.top{padding:20px 24px 12px;border-bottom:1px solid var(--line);background:var(--card)}
header.top h1{margin:0 0 4px;font-size:22px}
header.top .sub{color:var(--muted);margin:0 0 8px}
header.top .crit{color:var(--muted);font-size:13px;max-width:70ch}
.interests{background:var(--card);border-bottom:1px solid var(--line);padding:12px 24px}
.interests>summary{cursor:pointer;font-weight:600;font-size:14px;color:var(--ink);list-style:none}
.interests>summary::-webkit-details-marker{display:none}
.interests>summary::before{content:"\25B8";display:inline-block;margin-right:6px;color:var(--muted);transition:transform .15s}
.interests[open]>summary::before{transform:rotate(90deg)}
.interests>summary .hint{color:var(--muted);font-weight:400;font-size:12px;margin-left:6px}
.igroups{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:10px 16px}
.igroup{min-width:0}
.igroup h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.ichip{display:inline-flex;align-items:center;gap:6px;font-size:12px;background:var(--accent-soft);
  color:var(--ink);border:1px solid var(--line);border-radius:16px;padding:3px 6px 3px 10px;margin:0 6px 6px 0;max-width:340px}
.ichip .val{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ichip .x{cursor:pointer;color:var(--muted);border:none;background:none;font-size:13px;line-height:1;padding:0 2px}
.ichip .x:hover{color:var(--important)}
.inone{font-size:12px;color:var(--muted)}
.iadd{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.iadd select,.iadd input{font:inherit;padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:#fff}
.iadd input{min-width:0;flex:1 1 240px}
.btn.ppadd.nokey{opacity:.6}
.keyrow{display:flex;gap:6px;align-items:center;min-width:0}
.keyrow input{flex:1 1 auto;min-width:0}
.keyrow .btn{white-space:nowrap}
.iactions{display:flex;gap:12px;align-items:center;margin-top:10px;flex-wrap:wrap}
.profilebox{display:grid;gap:4px;margin:6px 0 8px;max-width:90ch}
.plabel{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
.profilebox textarea{font:inherit;font-style:italic;color:var(--ink);padding:6px 8px;border:1px dashed var(--line);
  border-radius:8px;resize:vertical;width:100%;background:transparent}
.profilebox textarea:focus{border-style:solid;border-color:var(--accent);outline:none;background:#fff}
.phint{margin:0;font-size:12px;color:var(--muted)}
.isections{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:16px;margin-top:12px;align-items:stretch}
@media(max-width:980px){.isections{grid-template-columns:1fr}}
.isec{display:grid;gap:10px;align-content:start}
.imain{gap:18px;grid-template-rows:auto auto 1fr}
.imain .igen{align-self:end}
.isub{display:grid;gap:8px;min-width:0}
.isub + .isub{border-top:1px solid var(--line);padding-top:16px}
.isec h3{margin:0}
.ihelp{margin:0}
.field{display:grid;gap:4px;min-width:0;font-size:12px;color:var(--muted)}
.field label{font-weight:600}
.field input,.field select{font:inherit;font-size:14px;color:var(--ink);padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:#fff;width:100%;min-width:0}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr));gap:10px 12px}
.pprow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px}
.pfoot{border-top:1px solid var(--line);padding-top:8px;font-size:11.5px}
.igen{display:flex;align-items:center;gap:12px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:14px;margin-top:0}
.isec{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--bg);min-width:0}
.isec h3{margin:0 0 6px;font-size:14px;display:flex;align-items:center;gap:8px}
.isec .step{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;
  background:var(--accent);color:#fff;font-size:11px;font-weight:700}
.isec .opt{font-size:11px;font-weight:400;color:var(--muted)}
.ihelp{font-size:12px;color:var(--muted);margin:0 0 8px}
.ihelp p{margin:0 0 6px}
.ihelp ul{margin:0 0 6px;padding-left:18px}
.ihelp .sel{user-select:all;color:var(--ink)}
.ippsave{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.check{display:inline-flex;gap:6px;align-items:center}
[hidden]{display:none !important}  /* `hidden` always wins over display rules */
.gp{display:inline-flex;align-items:center;gap:8px;font-size:13px;color:var(--ink);min-width:0;flex:1 1 260px;cursor:help}
.gp-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.gp-time{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.gp-spin{width:14px;height:14px;border-radius:50%;border:2px solid var(--accent-soft);border-top-color:var(--accent);
  animation:gp-spin .9s linear infinite;flex:0 0 auto}
.gp-bar{position:relative;flex:0 0 90px;height:4px;border-radius:2px;background:var(--accent-soft);overflow:hidden}
.gp-bar span{position:absolute;top:0;bottom:0;width:35%;border-radius:2px;background:var(--accent);animation:gp-slide 1.6s ease-in-out infinite}
@keyframes gp-spin{to{transform:rotate(360deg)}}
@keyframes gp-slide{0%{left:-35%}100%{left:100%}}
@media (prefers-reduced-motion:reduce){.gp-spin,.gp-bar span{animation:none}.gp-bar span{left:0;width:100%;opacity:.4}}
.btn.regen{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600;padding:7px 16px}
.btn.regen:hover{background:#255e91}
.btn.regen[disabled]{opacity:.6;cursor:progress}
.istatus{font-size:13px;color:var(--muted)}
.controls{position:sticky;top:0;z-index:10;display:flex;gap:12px;align-items:center;
  flex-wrap:wrap;padding:10px 24px;background:var(--card);border-bottom:1px solid var(--line)}
.controls label{font-size:13px;color:var(--muted)}
.controls select,.controls input[type=search]{
  font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:#fff}
.controls input[type=search]{min-width:200px}
.btn{font:inherit;cursor:pointer;padding:5px 10px;border:1px solid var(--line);
  border-radius:8px;background:#fff;color:var(--ink)}
.btn:hover{background:var(--accent-soft)}
.counts{color:var(--muted);font-size:13px;margin-left:auto}
.tagfilter{font-size:13px;color:var(--ink)}
.tagfilter .clear{cursor:pointer;color:var(--accent);margin-left:6px}
.layout{display:flex;align-items:flex-start;gap:20px;padding:16px 24px;max-width:1400px}
.sidebar{width:280px;flex:0 0 280px;position:sticky;top:56px;max-height:calc(100vh - 72px);
  overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px}
.sidebar h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:4px 6px 8px}
.sidebar ol{list-style:none;margin:0;padding:0}
.sidebar li{display:flex;gap:8px;align-items:flex-start;padding:6px;border-radius:8px;cursor:pointer}
.sidebar li:hover{background:var(--accent-soft)}
.sidebar .s-title{font-size:13px;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sidebar .s-sub{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sidebar li>span:last-child{min-width:0}
.sidetop{display:flex;align-items:center;justify-content:space-between;gap:6px}
.sidetoggle{font-size:11px;padding:2px 8px;margin:0 4px 6px 0}
.sidebar.full{width:420px;flex-basis:420px}
.sidebar.full .s-title{white-space:normal;overflow:visible}
.main{flex:1 1 auto;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid transparent;
  border-radius:12px;padding:16px 18px;margin:0 0 16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card.st-important{border-left-color:var(--important)}
.card.st-further{border-left-color:var(--further)}
.card.st-done{border-left-color:var(--done);opacity:.6}
.chip{display:inline-block;font-size:11px;font-weight:700;color:#fff;background:var(--accent);
  border-radius:6px;padding:1px 7px;vertical-align:middle}
.hrow{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.hrow h3{margin:0;font-size:17px}
.hrow h3 a{text-decoration:none}
.meta{color:var(--muted);font-size:13px}
.badge{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:6px;
  padding:1px 6px;color:var(--muted)}
.badge.manual{border-color:var(--accent);color:var(--accent);font-weight:700}
.badge.library{border-style:dashed}
.badge.origin{border-color:var(--pp);color:var(--pp);font-weight:700}
.card.from-pp{background:var(--pp-soft)}
.card.from-pp .hrow .chip{background:var(--pp)}
.scores{font-size:13px;color:var(--muted);margin-top:2px}
.tagrow{margin:10px 0 4px;display:flex;gap:6px;flex-wrap:wrap}
.tag{cursor:pointer;font-size:11px;border-radius:20px;padding:2px 9px;border:1px solid transparent}
.tag.app{background:#e8f0f8;color:#255e91}
.tag.method{background:#efeafc;color:#5b3fa8}
.tag.cat{color:#fff}
.tag.sel{outline:2px solid var(--accent);outline-offset:1px}
.coverhi{color:var(--accent);font-weight:700}
.statusbar{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}
.sbtn{font:inherit;font-size:12px;cursor:pointer;border:1px solid var(--line);border-radius:20px;
  padding:3px 10px;background:#fff;color:var(--ink)}
.sbtn.on-important{background:var(--important);color:#fff;border-color:var(--important)}
.sbtn.on-further{background:var(--further);color:#fff;border-color:var(--further)}
.sbtn.on-done{background:var(--done);color:#fff;border-color:var(--done)}
.copy{margin-left:auto}
.why{background:#eef8f0;border:1px solid #cdebd4;border-radius:8px;padding:8px 12px;margin:6px 0 4px;
  font-size:14px}
.why b{color:var(--done)}
dl{display:grid;grid-template-columns:150px 1fr;gap:6px 16px;margin:12px 0 0}
dl dt{font-weight:700;color:var(--muted);font-size:13px}
dl dd{margin:0}
dl dd ul{margin:4px 0;padding-left:18px}
.algo{display:inline-block;background:#1f2733;color:#fff;font-size:12px;border-radius:6px;padding:2px 9px}
.nboxes{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 2px}
.nbox{font-size:12px;padding:2px 10px;border:1px solid var(--line);border-radius:6px;color:var(--muted);background:transparent}
.nbox.on{border-color:var(--ink);color:var(--ink);background:var(--accent-soft);font-weight:600}
.gauge{display:inline-flex;align-items:center;gap:8px;vertical-align:middle}
.glabel{font-size:12px;color:var(--muted)}
.foot{text-align:center;font-size:12px;color:var(--muted);padding:16px 24px 24px;border-top:1px solid var(--line)}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1f2733;color:#fff;
  padding:10px 16px;border-radius:8px;opacity:0;transition:opacity .2s;pointer-events:none;z-index:50}
.toast.show{opacity:1}
@keyframes flashring{0%{box-shadow:0 0 0 0 rgba(43,108,176,.6)}100%{box-shadow:0 0 0 8px rgba(43,108,176,0)}}
.flash{animation:flashring .9s ease-out 2}
@media(max-width:900px){.sidebar{display:none}}
@media(max-width:560px){dl{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="top">
  <h1>Paper Triage</h1>
  <div class="profilebox">
    <label for="profile" class="plabel">Tailored to</label>
    <textarea id="profile" rows="2" aria-describedby="profilehint">__PROFILE__</textarea>
    <p class="phint" id="profilehint">This researcher profile steers the next review. Edit it to use your own wording, or leave it and the model writes one from your interests.</p>
  </div>
  <p class="crit">Ranked by a blend of field impact, relevance to the stated interests, and author/venue
    credibility. Scores are editorial judgments, not computed. Tag papers and copy clean notes
    into your note app. Generated __GENERATED__ &middot;
    <a href="https://github.com/jykr/paper_triage" target="_blank" rel="noopener">See the source</a></p>
</header>

<details class="interests" id="interests" open>
  <summary>Set up the next review<span class="hint">&mdash; add interests, pick a model, then regenerate; Paperpile is optional</span></summary>
  <div class="isections">
    <section class="isec imain" aria-label="Interests and model">
      <div class="isub" role="group" aria-labelledby="sec-interests">
        <h3 id="sec-interests"><span class="step">1</span>Interests</h3>
        <p class="ihelp">A CV or lab page, papers you care about, or research keywords.</p>
        <div class="iadd">
          <select id="ikind" aria-label="Interest kind">
            <option value="cv">CV / profile</option>
            <option value="paper">Paper</option>
            <option value="keyword">Keyword</option>
          </select>
          <input id="ivalue" type="text" placeholder="Paste a CV URL, a paper title/DOI/URL, or a keyword&hellip;" aria-label="Interest">
          <button class="btn" id="iaddbtn">&#65291; Add</button>
        </div>
        <div class="igroups" id="igroups"></div>
      </div>

      <div class="isub" role="group" aria-labelledby="sec-model">
        <h3 id="sec-model"><span class="step">2</span>Model</h3>
        <div class="mgrid">
          <div class="field">
            <label for="provider">Select model API</label>
            <select id="provider">
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="gemini">Google Gemini (AI Studio)</option>
              <option value="openai">OpenAI</option>
              <option value="openrouter">OpenRouter (sign in; Claude, GPT, Gemini&hellip;)</option>
            </select>
          </div>
          <div class="field">
            <label for="model">Model ID</label>
            <input id="model" type="text" list="modelids" placeholder="model id">
            <datalist id="modelids"></datalist>
          </div>
          <div class="field">
            <label for="apikey">API key</label>
            <span class="keyrow">
              <input id="apikey" type="password" placeholder="Your API key" autocomplete="off">
              <button class="btn" id="orsignin" type="button" hidden>Sign in with OpenRouter</button>
            </span>
          </div>
        </div>
        <p class="ihelp" id="modelhelp"></p>
      </div>

      <div class="iactions igen">
        <button class="btn regen" id="regen">&#8635; Regenerate review</button>
        <span class="gp" id="gp" hidden aria-live="polite">
          <span class="gp-spin" aria-hidden="true"></span>
          <span class="gp-label" id="gp-label">Starting&hellip;</span>
          <span class="gp-bar" aria-hidden="true"><span></span></span>
          <span class="gp-time" id="gp-time">0:00</span>
        </span>
        <span class="istatus" id="istatus" role="status"></span>
      </div>
    </section>

    <section class="isec ipp-block" aria-labelledby="sec-pp">
      <h3 id="sec-pp">Paperpile <span class="opt">optional &middot; beta</span></h3>
      <p class="ihelp">Connect your library to flag papers you already have, read the PDFs you saved
        when a paper shows <b>Abstract only</b>, include a label's papers, and add reviewed papers with the agent's notes.</p>
      <div class="field">
        <label for="ppkey">Paperpile API key</label>
        <input id="ppkey" type="password" placeholder="Paste your Paperpile API key" autocomplete="off">
      </div>
      <div class="pprow">
        <div class="field">
          <label for="pplabel">Include a label <span class="opt">optional</span></label>
          <input id="pplabel" type="text" placeholder="e.g. to read" aria-describedby="pplabelhint">
        </div>
        <div class="field">
          <label for="ppsince">Added</label>
          <select id="ppsince">
            <option value="">any time</option>
            <option value="1m">last month</option>
            <option value="3m">last 3 months</option>
            <option value="6m">last 6 months</option>
            <option value="1y">last year</option>
          </select>
        </div>
      </div>
      <p class="ihelp" id="pplabelhint">A label adds up to its 10 most recently added papers to the review. Leave it empty to review search results only.</p>
      <label class="check"><input id="ppnotes" type="checkbox" checked> Attach the agent's notes when adding papers</label>
      <div class="ippsave">
        <button class="btn" id="ppcheck" disabled title="Mark the papers already in your library; the review itself is not regenerated">&#10003; Check library</button>
        <button class="btn" id="ppaddall" disabled title="Add every paper shown to your Paperpile library">&#65291; Add all</button>
        <button class="btn" id="ppreread" disabled title="Re-read papers marked Abstract only from the PDFs saved in your Paperpile; uses your model key">&#8635; Re-read abstracts</button>
      </div>
      <p class="ihelp pfoot"><b>Check library</b> marks papers you already own; <b>Re-read abstracts</b> analyzes
        <b>Abstract only</b> papers from PDFs you saved in Paperpile. Neither regenerates the review.
        API access is in beta: email <span class="sel">support@paperpile.com</span>, then create a key in your account settings
        (<a href="https://paperpile.com/h/agent-plugins-mcp-beta/" target="_blank" rel="noopener">details</a>). Keys are never stored.</p>
    </section>
  </div>
</details>

<div class="controls">
  <label>Show
    <select id="filter">
      <option value="all">All</option>
      <option value="important">Important</option>
      <option value="further">Read further</option>
      <option value="done">Done</option>
      <option value="untagged">Untagged</option>
    </select>
  </label>
  <label>Sort
    <select id="sort">
      <option value="cover">Coverage</option>
      <option value="rank">Rank</option>
      <option value="impact">Field impact</option>
      <option value="rel">Relevance to me</option>
      <option value="venue">Venue A&ndash;Z</option>
      <option value="status">Status</option>
    </select>
  </label>
  <input id="search" type="search" placeholder="Search title, author, venue, tags&hellip;">
  <button class="btn" id="copyview">&#8681; Copy view for OneNote</button>
  <span class="tagfilter" id="tagfilter"></span>
  <span class="counts" id="counts"></span>
</div>

<div class="layout">
  <aside class="sidebar">
    <div class="sidetop"><h2 id="sidehead">Papers</h2>
      <button class="btn sidetoggle" id="sidetoggle" aria-pressed="false" title="Show full titles">&#8596; Full titles</button></div>
    <ol id="sidelist"></ol>
  </aside>
  <main class="main" id="list"></main>
</div>

<footer class="foot">&copy; 2026 Jayoung Ryu</footer>

<div class="toast" id="toast"></div>

<script id="data" type="application/json">__DATA__</script>
<script id="cats" type="application/json">__CATEGORIES__</script>
<script id="inputs" type="application/json">__INPUTS__</script>
<script>
let TAGS = __TAGS__;
let APPROACH = __APPROACH__;
let DATA = JSON.parse(document.getElementById('data').textContent);
let CATEGORIES = JSON.parse(document.getElementById('cats').textContent) || [];
// hydrate(): fold TAGS into each paper and normalise cats; re-run after regenerate.
function hydrate(){
  DATA.forEach(p => { const t = TAGS[p.rank] || {app:[],method:[]};
    p.app = t.app||[]; p.method = t.method||[]; p.cats = p.cats||[]; });
}
hydrate();

const EVAL_LABELS = ['New metric:','New eval data:','Design & novelty:','Eval limits:'];
const STORE = 'litreview-status-v1';
let status = {};
try { status = JSON.parse(localStorage.getItem(STORE)) || {}; } catch(e){ status = {}; }
function saveStatus(){ localStorage.setItem(STORE, JSON.stringify(status)); }

// ----- Interests (SPEC §2.0/§3d): INPUTS persisted; edited then regenerated -----
const ISTORE = 'litreview-inputs-v1';
let iseq = 0;
function ensureIds(){ INPUTS.forEach(i => { if(i.id==null) i.id = 'i'+(++iseq); }); }
let INPUTS;
try { INPUTS = JSON.parse(localStorage.getItem(ISTORE)); } catch(e){ INPUTS = null; }
if(!Array.isArray(INPUTS)) INPUTS = JSON.parse(document.getElementById('inputs').textContent) || [];
ensureIds();
function saveInputs(){ localStorage.setItem(ISTORE, JSON.stringify(INPUTS)); }
const IKIND_LABEL = {cv:'CV / profile', paper:'Paper', keyword:'Research topic/keyword'};

let tagFilter = null; // {val, kind}  kind: 'app' | 'method' | 'cat' (val is the category id)

const esc = s => (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function linkify(s){
  return esc(s).replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

function tagChips(p){
  const a = (p.app||[]).map(v=>chip(v,'app'));
  const m = (p.method||[]).map(v=>chip(v,'method'));
  return a.concat(m).join('');
}
function chip(val, kind){
  const sel = tagFilter && tagFilter.val===val && tagFilter.kind===kind ? ' sel':'';
  return `<span class="tag ${kind}${sel}" data-tag="${esc(val)}" data-kind="${kind}">${esc(val)}</span>`;
}
function catChips(p){
  return (p.cats||[]).map(id=>{
    const c = CATEGORIES.find(x=>x.id===id); if(!c) return '';
    const sel = tagFilter && tagFilter.kind==='cat' && tagFilter.val===id ? ' sel':'';
    return `<span class="tag cat${sel}" data-cat="${id}" style="background:${esc(c.color||'#4a5568')}">${esc(c.label)}</span>`;
  }).join('');
}
function coverBadge(p){
  const M = CATEGORIES.length; if(!M) return '';
  const cover = (p.cats||[]).length;
  return ` &middot; <span class="${cover>=2?'coverhi':''}">Matches ${cover} of ${M} interests</span>`;
}
// Novelty: three kinds of contribution, each a box that is highlighted when it
// applies (one or more). Same neutral style for all, so no kind reads as "better".
const NOVELTY = [
  ['Application','Application / resource','Existing methods applied to new data or a new problem, or a new dataset, benchmark or tool'],
  ['Adaptation','Adaptation','Existing components modified or combined'],
  ['Algorithm','Algorithm','A fundamentally new learning algorithm, objective or architectural principle'],
];
const LEGACY_NOV = {'New method':'Algorithm', Low:'Application', Medium:'Adaptation', High:'Algorithm'};
function noveltyOf(a){
  let v = a.nov != null && a.nov !== '' && !(Array.isArray(a.nov) && !a.nov.length) ? a.nov : a.nov_ml;  // older pages
  const one = x => typeof x === 'number' ? ['Application','Adaptation','Algorithm'][x < 34 ? 0 : x < 67 ? 1 : 2] : (LEGACY_NOV[x] || x);
  const kinds = (Array.isArray(v) ? v : (v == null || v === '' ? [] : [v])).map(one);
  return NOVELTY.map(([k])=>k).filter(k=>kinds.includes(k));
}
function noveltyHTML(a){
  const on = noveltyOf(a);
  if(!on.length) return '';
  return `<div class="nboxes" role="list" aria-label="Kind of contribution">`
    + NOVELTY.map(([k,label,help])=>`<span role="listitem" class="nbox${on.includes(k)?' on':''}" `
      + `title="${label}: ${help}" aria-label="${label}${on.includes(k)?' (applies)':''}">${label}</span>`).join('')
    + `</div>`;
}
function noveltyText(a){
  const on = noveltyOf(a);
  return on.length ? 'Contribution: ' + NOVELTY.filter(([k])=>on.includes(k)).map(([,l])=>l).join(' + ') : '';
}
function aimOf(p){
  const a = APPROACH[p.rank]; return a && a.aim ? a.aim : (p.aim||'');
}
function approachDD(p){
  const a = APPROACH[p.rank] || {};
  const algo = a.algo ? `<span class="algo">${esc(a.algo)}</span>` : '';
  return `${algo}${noveltyHTML(a)}`
    + `<ul>`
    + (a.question ? `<li><b>Key question:</b> ${esc(a.question)}</li>`:'')
    + (a.data ? `<li><b>Data:</b> ${esc(a.data)}</li>`:'')
    + (a.model ? `<li><b>Model:</b> ${esc(a.model)}</li>`:'')
    + (a.method ? `<li><b>Methodological novelty:</b> ${esc(a.method)}</li>`:'')
    + `</ul>`;
}
function evalDD(p){
  const txt = p.insights||'';
  // split on the inline labels, keep order
  const idx = EVAL_LABELS.map(l => ({l, i: txt.indexOf(l)})).filter(o=>o.i>=0).sort((x,y)=>x.i-y.i);
  if(!idx.length) return esc(txt);
  let out = '<ul>';
  for(let k=0;k<idx.length;k++){
    const start = idx[k].i + idx[k].l.length;
    const end = k+1<idx.length ? idx[k+1].i : txt.length;
    out += `<li><b>${esc(idx[k].l.replace(/:$/,''))}:</b> ${esc(txt.slice(start,end).trim())}</li>`;
  }
  return out + '</ul>';
}
function resultsDD(p){
  const r = (p.results||[]).filter(x=>x!=null && String(x).trim());
  if(!r.length) return '';
  return '<ul>' + r.map(x=>`<li>${esc(x)}</li>`).join('') + '</ul>';
}

function cardHTML(p){
  const st = status[p.rank];
  const cls = (st ? ' st-'+st : '') + (p.origin ? ' from-pp' : '');
  const btn = (key,label)=>`<button class="sbtn${st===key?' on-'+key:''}" data-status="${key}" data-rank="${p.rank}">${label}</button>`;
  return `<article class="card${cls}" id="card-${p.rank}" data-rank="${p.rank}">
    <div class="hrow">
      <span class="chip">${p.rank}</span>
      <h3><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
    </div>
    <div class="meta">${esc(p.authors)} &mdash; ${esc(p.institution)}</div>
    <div class="meta"><span class="badge">${esc(p.venue)}</span> ${esc(p.date)}${p.origin?` <span class="badge origin">${esc(p.origin)}</span>`:p.manual?' <span class="badge manual">Manually added</span>':''}${p.evidence==='abstract'?' <span class="badge" title="No open full text; analyzed from the abstract">Abstract only</span>':''}${p.in_library&&!p.origin?' <span class="badge library" title="Already in your Paperpile library">In your Paperpile</span>':''}</div>
    <div class="scores">Field impact ${p.impact}/10 &middot; Relevance ${p.rel}/10${coverBadge(p)}</div>
    <div class="tagrow">${tagChips(p)}${catChips(p)}</div>
    <div class="statusbar">
      ${btn('important','&#9733; Important')}
      ${btn('further','&#8599; Read further')}
      ${btn('done','&#10003; Done reviewing')}
      <button class="btn copy" data-copy="${p.rank}">&#8681; Copy for OneNote</button>
      ${ppButton(p)}
    </div>
    <div class="why"><b>Relevance:</b> ${esc(p.relevance)}</div>
    <dl>
      <dt>Aim</dt><dd>${esc(aimOf(p))}</dd>
      <dt>Main approach</dt><dd>${approachDD(p)}</dd>
      <dt>Evaluation</dt><dd>${evalDD(p)}</dd>
      ${resultsDD(p)?`<dt>Results</dt><dd>${resultsDD(p)}</dd>`:''}
      <dt>Limitation / next step</dt><dd>${esc(p.limitation)}</dd>
      <dt>Additional resources</dt><dd>${linkify(p.resources)||'&mdash;'}</dd>
      <dt>Comments</dt><dd>${esc(p.comments)}</dd>
    </dl>
  </article>`;
}

function currentView(){
  const f = document.getElementById('filter').value;
  const s = document.getElementById('sort').value;
  const q = document.getElementById('search').value.trim().toLowerCase();
  let rows = DATA.slice();
  if(f==='untagged') rows = rows.filter(p=>!status[p.rank]);
  else if(f!=='all') rows = rows.filter(p=>status[p.rank]===f);
  if(tagFilter){
    if(tagFilter.kind==='cat') rows = rows.filter(p => (p.cats||[]).includes(tagFilter.val));
    else rows = rows.filter(p => (tagFilter.kind==='app'?p.app:p.method).includes(tagFilter.val));
  }
  if(q) rows = rows.filter(p => (
    [p.title,p.authors,p.institution,p.venue,(p.app||[]).join(' '),(p.method||[]).join(' '),
     (p.cats||[]).map(id=>{const c=CATEGORIES.find(x=>x.id===id);return c?c.label:'';}).join(' ')]
    .join(' ').toLowerCase().includes(q)));
  const rank = (r)=>({important:0,further:1,undefined:2,done:3}[String(status[r]||'undefined')]);
  const cmp = {
    cover:(a,b)=>((b.cats||[]).length-(a.cats||[]).length)||a.rank-b.rank,
    rank:(a,b)=>a.rank-b.rank,
    impact:(a,b)=>b.impact-a.impact||a.rank-b.rank,
    rel:(a,b)=>b.rel-a.rel||a.rank-b.rank,
    venue:(a,b)=>a.venue.localeCompare(b.venue)||a.rank-b.rank,
    status:(a,b)=>rank(a.rank)-rank(b.rank)||a.rank-b.rank,
  }[s];
  rows.sort(cmp);
  return rows;
}

function render(){
  const rows = currentView();
  document.getElementById('list').innerHTML = rows.map(cardHTML).join('') ||
    '<p class="meta">No papers match the current filters.</p>';
  document.getElementById('sidehead').textContent = `Papers (${rows.length})`;
  document.getElementById('sidelist').innerHTML = rows.map(p=>{
    const st = status[p.rank];
    const bg = st==='important'?'var(--important)':st==='further'?'var(--further)':st==='done'?'var(--done)':p.origin?'var(--pp)':'var(--accent)';
    const sub = (p.origin?esc(p.origin)+' &middot; ':p.manual?'Manually added &middot; ':'') + ((p.app||[])[0]||'') + (p.venue? ' &middot; '+esc(p.venue):'');
    return `<li data-goto="${p.rank}"><span class="chip" style="background:${bg}">${p.rank}</span>
      <span><span class="s-title">${esc(p.title)}</span><br><span class="s-sub">${sub}</span></span></li>`;
  }).join('');
  const tf = document.getElementById('tagfilter');
  const flabel = tagFilter ? (tagFilter.kind==='cat'
      ? ((CATEGORIES.find(c=>c.id===tagFilter.val)||{}).label || '?')
      : tagFilter.val) : '';
  tf.innerHTML = tagFilter ? `Filter: <b>${esc(flabel)}</b> (${tagFilter.kind})
    <span class="clear" id="cleartag">clear &#10005;</span>` : '';
  document.getElementById('counts').textContent =
    `${rows.length} of ${DATA.length} shown`;
}

// ----- OneNote export (§7): minimal inline-styled HTML -----
function oneNoteHTML(p){
  const a = APPROACH[p.rank] || {};
  const evalItems = (()=>{ const d = document.createElement('div'); d.innerHTML = evalDD(p);
    return Array.from(d.querySelectorAll('li')).map(li=>`<li>${li.innerHTML}</li>`).join(''); })();
  const bul = [];
  if(a.question) bul.push(`<li><b>Key question:</b> ${esc(a.question)}</li>`);
  if(a.data) bul.push(`<li><b>Data:</b> ${esc(a.data)}</li>`);
  if(a.model) bul.push(`<li><b>Model:</b> ${esc(a.model)}</li>`);
  if(a.method) bul.push(`<li><b>Methodological novelty:</b> ${esc(a.method)}</li>`);
  const resItems = (p.results||[]).filter(x=>x!=null && String(x).trim())
    .map(x=>`<li>${esc(x)}</li>`).join('');
  return `<div style="font-family:Calibri,sans-serif">
    <h2>${esc(p.title)}</h2>
    <p>${esc(p.authors)} &mdash; ${esc(p.institution)}</p>
    <p><b>${esc(p.venue)}</b>, ${esc(p.date)} &middot; Field impact ${p.impact}/10 &middot;
       Relevance ${p.rel}/10 &middot; <a href="${esc(p.url)}">${esc(p.url)}</a></p>
    <p><b>Application:</b> ${esc((p.app||[]).join(', '))} &nbsp; <b>Method:</b> ${esc((p.method||[]).join(', '))}</p>
    <p><b>Aim:</b> ${esc(aimOf(p))}</p>
    <p><b>Main approach:</b> ${esc(a.algo||'')}${noveltyText(a)?` (${esc(noveltyText(a))})`:''}</p>
    <ul>${bul.join('')}</ul>
    <p><b>Evaluation:</b></p><ul>${evalItems}</ul>
    ${resItems?`<p><b>Results:</b></p><ul>${resItems}</ul>`:''}
    <p><b>Limitation / next step:</b> ${esc(p.limitation)}</p>
    <p><b>Resources:</b> ${esc(p.resources)}</p>
    <p><b>Comments:</b> ${esc(p.comments)}</p>
    <p><b>Relevance:</b> ${esc(p.relevance)}</p>
  </div>`;
}

async function copyHTML(html, msg){
  try{
    const item = new ClipboardItem({
      'text/html': new Blob([html],{type:'text/html'}),
      'text/plain': new Blob([html.replace(/<[^>]+>/g,'')],{type:'text/plain'})
    });
    await navigator.clipboard.write([item]);
    toast(msg);
  }catch(e){
    const el = document.createElement('div');
    el.contentEditable = 'true'; el.innerHTML = html;
    el.style.position='fixed'; el.style.left='-9999px';
    document.body.appendChild(el);
    const r = document.createRange(); r.selectNodeContents(el);
    const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
    try{ document.execCommand('copy'); toast(msg); }catch(_){ toast('Copy failed'); }
    sel.removeAllRanges(); document.body.removeChild(el);
  }
}
let toastT;
function toast(msg, ms){
  const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(()=>t.classList.remove('show'), ms || 1600);
}

// ----- Event delegation (§9) -----
document.getElementById('list').addEventListener('click', e=>{
  const s = e.target.closest('[data-status]');
  if(s){ const r=s.dataset.rank, k=s.dataset.status;
    status[r] = status[r]===k ? undefined : k;
    if(!status[r]) delete status[r];
    saveStatus(); render(); return; }
  const c = e.target.closest('[data-copy]');
  if(c){ const p = DATA.find(x=>String(x.rank)===c.dataset.copy);
    copyHTML(oneNoteHTML(p), 'Copied paper for OneNote'); return; }
  const t = e.target.closest('[data-tag]');
  if(t){ const val=t.dataset.tag, kind=t.dataset.kind;
    tagFilter = (tagFilter && tagFilter.val===val && tagFilter.kind===kind) ? null : {val,kind};
    render(); return; }
  const cat = e.target.closest('[data-cat]');
  if(cat){ const val = +cat.dataset.cat;
    tagFilter = (tagFilter && tagFilter.kind==='cat' && tagFilter.val===val) ? null : {val,kind:'cat'};
    render(); return; }
});
document.getElementById('sidelist').addEventListener('click', e=>{
  const li = e.target.closest('[data-goto]'); if(!li) return;
  const card = document.getElementById('card-'+li.dataset.goto);
  if(card){ card.scrollIntoView({behavior:'smooth',block:'start'});
    card.classList.remove('flash'); void card.offsetWidth; card.classList.add('flash'); }
});
document.getElementById('tagfilter').addEventListener('click', e=>{
  if(e.target.id==='cleartag'){ tagFilter=null; render(); }
});
document.getElementById('copyview').addEventListener('click', ()=>{
  const rows = currentView();
  const html = `<div style="font-family:Calibri,sans-serif"><h1>Literature Review Queue (${rows.length} papers)</h1>`
    + rows.map(oneNoteHTML).join('<hr>') + '</div>';
  copyHTML(html, `Copied ${rows.length} papers for OneNote`);
});
['filter','sort','search'].forEach(id=>{
  document.getElementById(id).addEventListener('input', render);
});

// ----- Interests panel: render chips, add/remove, regenerate (SPEC §5) -----
function renderInputs(){
  const groups = {cv:'CV / profile', paper:'Papers', keyword:'Keywords'};
  document.getElementById('igroups').innerHTML = Object.keys(groups).map(k=>{
    const items = INPUTS.filter(i=>i.kind===k);
    const chips = items.map(i=>`<span class="ichip"><span class="val" title="${esc(i.value)}">${esc(i.value)}</span>`
      + `<button class="x" data-remove="${esc(String(i.id))}" title="Remove" aria-label="Remove">&#10005;</button></span>`
    ).join('') || '<span class="inone">none yet</span>';
    return `<div class="igroup"><h4>${groups[k]} (${items.length})</h4>${chips}</div>`;
  }).join('');
}
function addInput(){
  const kind = document.getElementById('ikind').value;
  const inp = document.getElementById('ivalue');
  const v = inp.value.trim();
  if(!v){ inp.focus(); return; }
  INPUTS.push({kind, value:v, id:'i'+(++iseq)});
  inp.value=''; saveInputs(); renderInputs();
  setStatus(`${INPUTS.length} interest${INPUTS.length===1?'':'s'} — regenerate to update the review.`);
}
document.getElementById('iaddbtn').addEventListener('click', addInput);
document.getElementById('ivalue').addEventListener('keydown', e=>{
  if(e.key==='Enter'){ e.preventDefault(); addInput(); }
});
document.getElementById('igroups').addEventListener('click', e=>{
  const b = e.target.closest('[data-remove]'); if(!b) return;
  INPUTS = INPUTS.filter(i=>String(i.id)!==b.dataset.remove);
  saveInputs(); renderInputs();
});

function setStatus(m){ document.getElementById('istatus').textContent = m || ''; }
async function copyText(text, msg){
  try{ await navigator.clipboard.writeText(text); toast(msg); }
  catch(e){ toast('Copy failed'); }
}
// Profile: the header field holds the review's profile; it is sent only when
// the user changed it (otherwise the model writes a fresh one).
let PROFILE_BASE = document.getElementById('profile').value.trim();
function profileEdited(){
  const v = document.getElementById('profile').value.trim();
  return !!v && v !== PROFILE_BASE;
}
function sinceDate(span){
  const m = /^(\d+)([my])$/.exec(span||''); if(!m) return '';
  const d = new Date(); const n = +m[1];
  if(m[2]==='y') d.setFullYear(d.getFullYear()-n); else d.setMonth(d.getMonth()-n);
  return d.toISOString().slice(0,10);
}
function applyPayload(pl){
  DATA = pl.data || [];
  TAGS = pl.tags || {};
  APPROACH = pl.approach || {};
  CATEGORIES = pl.categories || [];
  if(Array.isArray(pl.inputs) && pl.inputs.length){
    INPUTS = pl.inputs.map(i=>({kind:i.kind, value:i.value}));
    ensureIds(); saveInputs();
  }
  if(pl.profile){
    PROFILE_BASE = pl.profile.trim();
    document.getElementById('profile').value = pl.profile;
  }
  hydrate(); markAddedFromSite(); tagFilter = null; renderInputs(); render();
}
function offlineFallback(){
  // No live backend (opened via file://, or server down): fall back to a
  // prompt the user can paste into the agent chat (Plan A behaviour).
  const lines = INPUTS.map(i=>`- ${IKIND_LABEL[i.kind]||i.kind}: ${i.value}`);
  const prompt = 'Build my recent literature review from these interest inputs:\n'
    + lines.join('\n');
  copyText(prompt, 'No live backend — prompt copied to clipboard');
  setStatus('No live backend reachable. Copied a prompt to paste into the agent chat.');
}
function doneMsg(){
  return `Updated — ${DATA.length} paper${DATA.length===1?'':'s'} across `
    + `${CATEGORIES.length} categor${CATEGORIES.length===1?'y':'ies'}.`;
}
// ----- Progress panel while a review is generated (or papers are re-read) -----
const GP_STAGES = [
  ['intake_agent','Reading the request'], ['scope_agent','Defining interests'],
  ['researcher','Searching 5 sources'], ['added_papers','Adding papers'],
  ['rank_agent','Ranking'], ['deepen_agent','Reading full texts'],
  ['library_agent','Finishing up'],
];
const gp = {timer:null, start:0, idx:-1, stages:[]};
function gpTick(){
  const s = Math.floor((Date.now()-gp.start)/1000);
  document.getElementById('gp-time').textContent = `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
}
function gpLabel(text){
  const n = gp.stages.length;
  document.getElementById('gp-label').textContent = text + (n && gp.idx >= 0 ? ` (${gp.idx + 1} of ${n})` : '');
}
function gpStart(label, note, stages){
  gp.start = Date.now(); gp.idx = stages.length ? 0 : -1; gp.stages = stages;
  gp.text = label; gpLabel(label);
  const el = document.getElementById('gp'); el.title = note; el.hidden = false;
  document.getElementById('istatus').textContent = '';
  gpTick(); clearInterval(gp.timer); gp.timer = setInterval(gpTick, 1000);
}
function gpStage(stage, label){
  const i = gp.stages.findIndex(([k])=>k===stage);
  if(i > gp.idx) gp.idx = i;
  if(label) gp.text = label;
  gpLabel(gp.text);
}
function gpStop(){ clearInterval(gp.timer); gp.timer = null; document.getElementById('gp').hidden = true; }

async function regenerate(){
  const ppKey = (document.getElementById('ppkey').value||'').trim();
  const ppLabel = (document.getElementById('pplabel').value||'').trim();
  if(!INPUTS.length && !ppLabel){ setStatus('Add at least one interest or a Paperpile label first.'); return; }
  if(ppLabel && !ppKey){ setStatus('Enter your Paperpile key to include a label.'); document.getElementById('ppkey').focus(); return; }
  const keyEl = document.getElementById('apikey');
  const key = (keyEl ? keyEl.value : '').trim();
  if(!key){ setStatus('Enter your API key first.'); if(keyEl) keyEl.focus(); return; }
  const provider = (document.getElementById('provider')||{}).value || 'gemini';
  const model = ((document.getElementById('model')||{}).value || '').trim();
  const btn = document.getElementById('regen');
  btn.disabled = true;
  setStatus('');
  gpStart('Starting…', 'A full review usually takes 3–5 minutes: it searches five sources, ranks the papers, '
    + 'then reads the selected ones in full. Keep this tab open; the review appears here when it is ready.', GP_STAGES);
  try{
    const res = await fetch('/review/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({inputs: INPUTS.map(i=>({kind:i.kind, value:i.value})), provider, model, apiKey:key,
        paperpileKey: ppKey, paperpileLabel: ppLabel, paperpileAddedSince: sinceDate(document.getElementById('ppsince').value),
        profile: profileEdited() ? document.getElementById('profile').value.trim() : ''})
    });
    if(!res.ok || !res.body){
      let detail=''; try{ detail=(await res.json()).detail||''; }catch(_){}
      throw new Error(detail || ('HTTP '+res.status));
    }
    // Parse the SSE stream (POST rules out EventSource): read data: frames.
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf='', done=false, errored=null;
    for(;;){
      const {value, done:rdone} = await reader.read();
      if(rdone) break;
      buf += dec.decode(value, {stream:true});
      let sep;
      while((sep = buf.indexOf('\n\n')) >= 0){
        const frame = buf.slice(0, sep); buf = buf.slice(sep+2);
        const line = frame.split('\n').find(l=>l.startsWith('data:'));
        if(!line) continue;
        let msg; try{ msg = JSON.parse(line.slice(5).trim()); }catch(_){ continue; }
        if(msg.type==='progress'){ gpStage(msg.stage, msg.label); }
        else if(msg.type==='done'){ applyPayload(msg.payload); done=true; setStatus(doneMsg()); }
        else if(msg.type==='error'){ errored = msg.detail || 'Could not regenerate.'; }
      }
    }
    if(errored) setStatus('Could not regenerate: ' + errored);
    else if(!done) setStatus('Regeneration ended without a result.');
  }catch(err){
    if(err instanceof TypeError) offlineFallback();          // fetch failed = no backend
    else setStatus('Could not regenerate: ' + err.message);  // backend reachable but errored
  }finally{ btn.disabled = false; gpStop(); }
}
document.getElementById('regen').addEventListener('click', regenerate);

// BYO model settings: remember provider/model, default the model per provider.
// (The API and Paperpile keys are deliberately NOT persisted — they live only
// in their fields.)
const PROVIDERS = {
  anthropic: {models:['claude-sonnet-5','claude-sonnet-5-5','claude-opus-5-5','claude-haiku-5-5'],
    docs:'https://docs.anthropic.com/en/docs/about-claude/models/overview',
    keys:'https://console.anthropic.com/settings/keys'},
  gemini: {models:['gemini-flash-latest'],
    docs:'https://ai.google.dev/gemini-api/docs/models',
    keys:'https://aistudio.google.com/apikey'},
  openai: {models:['gpt-4o','gpt-5'],
    docs:'https://platform.openai.com/docs/models',
    keys:'https://platform.openai.com/api-keys'},
  openrouter: {models:['anthropic/claude-sonnet-5.5','anthropic/claude-opus-5.5','anthropic/claude-haiku-5.5','openai/gpt-4o','google/gemini-3.8-flash'],
    docs:'https://openrouter.ai/models',
    keys:'https://openrouter.ai/settings/keys'},
};
(function(){
  const prov = document.getElementById('provider'), mdl = document.getElementById('model');
  if(!prov || !mdl) return;
  const MS = 'byo-model-cfg-v1';
  const DEFAULTS = Object.values(PROVIDERS).map(p=>p.models[0]);
  try{ const c = JSON.parse(localStorage.getItem(MS)||'{}');
    if(c.provider && PROVIDERS[c.provider]) prov.value = c.provider; if(c.model) mdl.value = c.model; }catch(e){}
  function sync(){
    const p = PROVIDERS[prov.value] || PROVIDERS.anthropic;
    if(!mdl.value || DEFAULTS.includes(mdl.value)) mdl.value = p.models[0];
    document.getElementById('modelids').innerHTML = p.models.map(m=>`<option value="${m}">`).join('');
    const isOR = prov.value === 'openrouter';
    document.getElementById('orsignin').hidden = !isOR;
    document.getElementById('apikey').placeholder = isOR ? 'Sign in, or paste an OpenRouter key' : 'Your API key';
    document.getElementById('modelhelp').innerHTML = isOR
      ? `Sign in with your OpenRouter account instead of pasting a key: OpenRouter creates a key on your own credits `
        + `(you can set a spending limit and expiry when approving). One sign-in covers Claude, GPT, Gemini and other `
        + `models; pick one from the <a href="${p.docs}" target="_blank" rel="noopener">model list</a>. `
        + `The key stays in this page only and is never stored.`
      : `Pick a suggested model ID or paste any ID from the <a href="${p.docs}" target="_blank" rel="noopener">model list</a>. `
        + `<a href="${p.keys}" target="_blank" rel="noopener">Get an API key</a>. `
        + `Your API key is sent only with this request to generate the review; it is never stored.`;
  }
  const save = ()=>{ try{ localStorage.setItem(MS, JSON.stringify({provider:prov.value, model:mdl.value})); }catch(e){} };
  prov.addEventListener('change', ()=>{ sync(); save(); });
  mdl.addEventListener('change', save);
  sync();
})();

// ----- OpenRouter sign-in (OAuth PKCE) in a popup, so the page keeps its review.
// The verifier and the resulting key live in memory only (never in storage).
let orPending = null;  // {verifier, state}
function b64url(bytes){
  let s = ''; bytes.forEach(b=>{ s += String.fromCharCode(b); });
  return btoa(s).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');
}
async function openRouterSignIn(){
  if(!(window.crypto && crypto.subtle)){ toast('Sign-in needs https or localhost'); return; }
  const verifier = b64url(crypto.getRandomValues(new Uint8Array(32)));
  const challenge = b64url(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))));
  const state = b64url(crypto.getRandomValues(new Uint8Array(16)));
  orPending = {verifier, state};
  const q = new URLSearchParams({callback_url: location.origin + '/openrouter/callback',
    code_challenge: challenge, code_challenge_method: 'S256', state,
    key_label: 'Literature Reviewer', expiration: '30-days'});
  const w = window.open('https://openrouter.ai/auth?' + q, 'openrouter-auth', 'width=520,height=720');
  if(!w) toast('Allow pop-ups for this site to sign in with OpenRouter');
  else setStatus('Waiting for OpenRouter sign-in…');
}
window.addEventListener('message', async ev=>{
  if(ev.origin !== location.origin || !ev.data || ev.data.type !== 'openrouter-code') return;
  const pending = orPending; orPending = null;
  if(!pending || ev.data.state !== pending.state || !ev.data.code){ setStatus('OpenRouter sign-in was cancelled.'); return; }
  try{
    const res = await fetch('/openrouter/exchange', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code: ev.data.code, code_verifier: pending.verifier})});
    const out = await res.json();
    if(!res.ok) throw new Error(out.detail || ('HTTP '+res.status));
    document.getElementById('apikey').value = out.key;
    setStatus('Signed in with OpenRouter. The key is kept in this page only.');
  }catch(err){
    setStatus('OpenRouter sign-in failed: ' + (err instanceof TypeError ? 'needs the live server' : err.message));
  }
});
document.getElementById('orsignin').addEventListener('click', openRouterSignIn);

// ----- Paperpile: add reviewed papers to the user's library -----
// Papers added from this page are remembered per browser (DOI -> reference id)
// so only those can be removed again; anything else is left to the Paperpile app.
const PP_ADDED_KEY = 'litreview-pp-added';
let PP_ADDED = {};
try{ PP_ADDED = JSON.parse(localStorage.getItem(PP_ADDED_KEY) || '{}') || {}; }catch(e){ PP_ADDED = {}; }
function savePPAdded(){ try{ localStorage.setItem(PP_ADDED_KEY, JSON.stringify(PP_ADDED)); }catch(e){} }
function markAddedFromSite(){
  DATA.forEach(p=>{ const id = p.doi && PP_ADDED[p.doi.toLowerCase()];
    if(id){ p.pp_ref = id; p.in_library = true; } });
}
const PP_ALREADY = 'This paper was already in your library; you can exclude it from your library in the Paperpile app.';
function ppButton(p){
  if(p.in_library && p.pp_ref)
    return `<button class="btn ppadd pp-mine" data-ppremove="${p.rank}" title="Added from this page. Click to remove it from your Paperpile (moved to Paperpile's trash).">&#10003; In your Paperpile &middot; Remove</button>`;
  if(p.in_library)
    return `<button class="btn ppadd pp-had" data-ppinfo="${p.rank}" title="${PP_ALREADY}">&#10003; In your Paperpile</button>`;
  return `<button class="btn ppadd${ppKey()?'':' nokey'}" data-ppadd="${p.rank}" title="${ppKey()?'Add this paper to your Paperpile library':'Needs your Paperpile API key (Paperpile panel above)'}">&#65291; Add to Paperpile</button>`;
}
async function removeFromPaperpile(p){
  const key = ppKey();
  if(!key){ toast('Enter your Paperpile API key in the Paperpile panel to remove papers'); return; }
  try{
    const res = await fetch('/paperpile/remove', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({paperpileKey:key, refs:[{id:p.pp_ref, doi:p.doi, title:p.title}]})});
    const out = await res.json();
    if(!res.ok) throw new Error(out.detail || ('HTTP '+res.status));
    const r = (out.results||[])[0] || {};
    if(r.status === 'removed' || r.status === 'not found'){
      delete PP_ADDED[(p.doi||'').toLowerCase()]; savePPAdded();
      delete p.pp_ref; p.in_library = false; render();
      toast(r.status === 'removed' ? "Removed from your Paperpile (in Paperpile's trash)" : 'Already gone from your Paperpile');
    }
  }catch(err){
    toast(err instanceof TypeError ? 'Removing needs the live server' : 'Paperpile: ' + err.message);
  }
}
function ppKey(){ const el = document.getElementById('ppkey'); return el ? el.value.trim() : ''; }
function noteMarkdown(p){
  const a = APPROACH[p.rank] || {};
  const lines = [`**Literature review note** (${p.venue}, ${p.date})`, '',
    `**Relevance:** ${p.relevance}`, `**Aim:** ${aimOf(p)}`];
  if(noveltyText(a)) lines.push(`**${noveltyText(a).replace(': ', ':** ')}`);
  if(a.question) lines.push(`**Key question:** ${a.question}`);
  if(a.data) lines.push(`**Data:** ${a.data}`);
  if(a.model) lines.push(`**Model:** ${a.model}`);
  if(a.method) lines.push(`**Methodological novelty:** ${a.method}`);
  const res = (p.results||[]).filter(x=>x && String(x).trim());
  if(res.length) lines.push('', '**Results:**', ...res.map(r=>`- ${r}`));
  if(p.limitation) lines.push('', `**Limitation:** ${p.limitation}`);
  if(p.comments) lines.push(`**Comments:** ${p.comments}`);
  return lines.join('\n');
}
async function addToPaperpile(papers){
  const key = ppKey();
  if(!key){ toast('Enter your Paperpile API key first'); return; }
  const withNotes = document.getElementById('ppnotes').checked;
  try{
    const res = await fetch('/paperpile/save', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({paperpileKey:key, papers: papers.map(p=>({title:p.title, url:p.url, doi:p.doi||'',
        note: withNotes ? noteMarkdown(p) : ''}))})});
    const out = await res.json();
    if(!res.ok) throw new Error(out.detail || ('HTTP '+res.status));
    const by = {}, ids = {}; (out.results||[]).forEach(r=>{ by[r.title] = r.status; if(r.id) ids[r.title] = r; });
    let added = 0, had = 0, failed = 0;
    papers.forEach(p=>{
      const st = by[p.title];
      if(st==='added'){ added++; p.in_library = true;
        const r = ids[p.title];
        if(r){ p.pp_ref = r.id; p.doi = p.doi || r.doi; PP_ADDED[r.doi.toLowerCase()] = r.id; savePPAdded(); } }
      else if(st==='already in library'){ had++; p.in_library = true; }
      else failed++;
    });
    render();
    toast(`Paperpile: ${added} added, ${had} already there` + (failed?`, ${failed} not added`:''));
  }catch(err){
    toast(err instanceof TypeError ? 'Adding to Paperpile needs the live server' : 'Paperpile: ' + err.message);
  }
}
document.getElementById('ppkey').addEventListener('input', ()=>{
  document.getElementById('ppaddall').disabled = !ppKey();
  document.getElementById('ppcheck').disabled = !ppKey();
  document.getElementById('ppreread').disabled = !ppKey();
  render();
});
// Re-read Abstract-only papers from Paperpile PDFs: only their analysis changes.
async function rereadFromPaperpile(){
  const ppk = ppKey(), key = (document.getElementById('apikey').value||'').trim();
  const todo = DATA.filter(p=>p.evidence==='abstract');
  if(!todo.length){ toast('No papers are marked Abstract only'); return; }
  if(!key){ setStatus('Enter your model API key (step 2) to re-read papers.'); document.getElementById('apikey').focus(); return; }
  const btn = document.getElementById('ppreread'); btn.disabled = true;
  setStatus('');
  gpStart(`Re-reading ${Math.min(todo.length,20)} paper(s) from your Paperpile PDFs…`,
    'Each paper takes about 20–60 seconds. The rest of the review stays as it is.', []);
  try{
    const res = await fetch('/review/reanalyze', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({provider: document.getElementById('provider').value,
        model: (document.getElementById('model').value||'').trim(), apiKey: key, paperpileKey: ppk,
        profile: document.getElementById('profile').value.trim(),
        papers: todo.slice(0,20).map(p=>({rank:p.rank, title:p.title, url:p.url, doi:p.doi||'',
          insights:p.insights, results:p.results, limitation:p.limitation, resources:p.resources,
          comments:p.comments, approach: APPROACH[p.rank]||{}}))})});
    const out = await res.json();
    if(!res.ok) throw new Error(out.detail || ('HTTP '+res.status));
    (out.papers||[]).forEach(u=>{
      const p = DATA.find(x=>x.rank===u.rank); if(!p) return;
      ['insights','results','limitation','comments','resources','evidence'].forEach(k=>{ if(u[k]!=null) p[k]=u[k]; });
      if(u.approach) APPROACH[u.rank] = u.approach;
    });
    render();
    const missing = out.missing||[];
    setStatus(`Re-read ${out.read||0} paper(s) in full from Paperpile.`
      + (missing.length ? ` No saved PDF found for: ${missing.join('; ')}.` : ''));
  }catch(err){
    setStatus('Could not re-read: ' + (err instanceof TypeError ? 'needs the live server' : err.message));
  }finally{ btn.disabled = !ppKey(); gpStop(); }
}
document.getElementById('ppreread').addEventListener('click', rereadFromPaperpile);
// Mark papers already in the library on the current page (no regeneration).
async function checkPaperpile(){
  const key = ppKey(); if(!key || !DATA.length) return;
  const btn = document.getElementById('ppcheck'); btn.disabled = true;
  try{
    const res = await fetch('/paperpile/check', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({paperpileKey:key, papers: DATA.slice(0,100).map(p=>({title:p.title, url:p.url, doi:p.doi||''}))})});
    const out = await res.json();
    if(!res.ok) throw new Error(out.detail || ('HTTP '+res.status));
    const by = {}; (out.results||[]).forEach(r=>{ by[r.title] = r.in_library; });
    DATA.forEach(p=>{
      if(by[p.title]) p.in_library = true;
      else if(p.title in by && p.pp_ref){  // removed in the Paperpile app since
        delete PP_ADDED[(p.doi||'').toLowerCase()]; savePPAdded(); delete p.pp_ref; p.in_library = false;
      }
    });
    render();
    const n = DATA.filter(p=>p.in_library).length;
    toast(`${n} of ${DATA.length} papers already in your Paperpile`);
  }catch(err){
    toast(err instanceof TypeError ? 'Checking Paperpile needs the live server' : 'Paperpile: ' + err.message);
  }finally{ btn.disabled = !ppKey(); }
}
document.getElementById('ppcheck').addEventListener('click', checkPaperpile);
document.getElementById('ppaddall').addEventListener('click', ()=>{
  const rows = currentView().filter(p=>!p.in_library);
  if(rows.length) addToPaperpile(rows.slice(0, 50)); else toast('All shown papers are already in your Paperpile');
});
document.getElementById('list').addEventListener('click', e=>{
  const info = e.target.closest('[data-ppinfo]');
  if(info){ toast(PP_ALREADY, 5000); return; }
  const rm = e.target.closest('[data-ppremove]');
  if(rm){
    const p = DATA.find(x=>String(x.rank)===rm.dataset.ppremove); if(!p) return;
    if(rm.dataset.armed !== '1'){  // two clicks: the first one asks
      rm.dataset.armed = '1'; rm.textContent = 'Click again to remove from Paperpile';
      setTimeout(()=>{ if(rm.isConnected && rm.dataset.armed === '1'){ rm.dataset.armed = ''; rm.innerHTML = '&#10003; In your Paperpile &middot; Remove'; } }, 4000);
      return;
    }
    removeFromPaperpile(p); return;
  }
  const b = e.target.closest('[data-ppadd]'); if(!b) return;
  if(!ppKey()){
    toast('Enter your Paperpile API key in the Paperpile panel to add papers');
    const det = document.getElementById('interests'); det.open = true;
    const k = document.getElementById('ppkey');
    k.scrollIntoView({behavior:'smooth', block:'center'}); k.focus({preventScroll:true});
    return;
  }
  const p = DATA.find(x=>String(x.rank)===b.dataset.ppadd);
  if(p) addToPaperpile([p]);
});

// Sidebar: compact (truncated titles, the default) or unfolded (full titles,
// wider). The choice is remembered per browser.
(function(){
  const side = document.querySelector('.sidebar'), btn = document.getElementById('sidetoggle');
  if(!side || !btn) return;
  const KEY = 'litreview-sidebar-full';
  function set(full){
    side.classList.toggle('full', full);
    btn.setAttribute('aria-pressed', String(full));
    btn.innerHTML = full ? '&#8596; Compact' : '&#8596; Full titles';
    btn.title = full ? 'Show compact titles' : 'Show full titles';
  }
  let full = false; try{ full = localStorage.getItem(KEY) === '1'; }catch(e){}
  set(full);
  btn.addEventListener('click', ()=>{
    full = !full; set(full);
    try{ localStorage.setItem(KEY, full ? '1' : '0'); }catch(e){}
  });
})();

markAddedFromSite();
renderInputs();
render();
</script>
</body>
</html>
"""
