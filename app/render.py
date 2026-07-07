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

from app.schemas import ReviewQueue


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
            }
        )
        tags[str(p.rank)] = {"app": p.tags.app, "method": p.tags.method}
        approach[str(p.rank)] = {
            "algo": p.approach.algo,
            "nov": p.approach.nov,
            "aim": p.approach.aim,
            "data": p.approach.data,
            "model": p.approach.model,
            "question": p.approach.question,
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
    """after_agent_callback for the ranker: render the HTML and save it.

    Reads the ranked queue from ``state['review_queue']`` (set by the ranker's
    ``output_key``) and the profile from ``state['scope']``. Saves the result as
    an ADK artifact (portable to Cloud Run / Agent Runtime) and, when the local
    filesystem is writable, also drops ``literature-reviewer.html`` in the cwd.
    Overrides the agent's raw-JSON message with a friendly summary.
    """
    raw = callback_context.state.get("review_queue")
    if raw is None:
        return None  # nothing to render; let the agent's own output stand
    if isinstance(raw, str):
        raw = json.loads(raw)
    queue = ReviewQueue.model_validate(raw)

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

    # Convenience for local dev: also write to disk if we can.
    disk_note = ""
    try:
        with open("literature-reviewer.html", "wb") as fh:
            fh.write(html_bytes)
        disk_note = " and written to ./literature-reviewer.html"
    except OSError:
        pass

    n = len(queue.papers)
    summary = (
        f"Reviewed and ranked {n} paper{'s' if n != 1 else ''}. "
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
<title>Recent Literature Review Queue</title>
<style>
:root{
  --bg:#f6f7f9; --card:#fff; --ink:#1a1d21; --muted:#5b6570; --line:#e3e7ec;
  --accent:#2b6cb0; --accent-soft:#e8f0f8;
  --important:#c0392b; --further:#b7791f; --done:#2f855a;
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
.igroups{display:flex;flex-wrap:wrap;gap:18px;margin:12px 0 4px}
.igroup{min-width:180px}
.igroup h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.ichip{display:inline-flex;align-items:center;gap:6px;font-size:12px;background:var(--accent-soft);
  color:var(--ink);border:1px solid var(--line);border-radius:16px;padding:3px 6px 3px 10px;margin:0 6px 6px 0;max-width:340px}
.ichip .val{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ichip .x{cursor:pointer;color:var(--muted);border:none;background:none;font-size:13px;line-height:1;padding:0 2px}
.ichip .x:hover{color:var(--important)}
.inone{font-size:12px;color:var(--muted)}
.iadd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px}
.iadd select,.iadd input{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:#fff}
.iadd input{min-width:260px;flex:1 1 260px}
.mset{display:inline-flex;gap:8px;align-items:center;flex-wrap:wrap;margin-right:12px}
.mset select,.mset input{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:#fff}
.mset input{min-width:200px}
.iactions{display:flex;gap:12px;align-items:center;margin-top:10px;flex-wrap:wrap}
.regen{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
.regen:hover{background:#255e91}
.regen[disabled]{opacity:.6;cursor:progress}
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
.gauge{display:inline-flex;align-items:center;gap:8px;margin-left:8px;vertical-align:middle}
.gtrack{position:relative;width:120px;height:8px;border-radius:6px;
  background:linear-gradient(90deg,#c9ccd1,#7c5bd0)}
.gmark{position:absolute;top:-3px;width:14px;height:14px;background:#fff;border:2px solid #5b3fa8;
  border-radius:50%;transform:translateX(-50%)}
.glabel{font-size:12px;color:var(--muted)}
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
  <h1>Recent Literature Review Queue</h1>
  <p class="sub">Ranked, taggable triage of recent papers &mdash; tailored to: <em>__PROFILE__</em></p>
  <p class="crit">Ranked by a blend of field impact, relevance to your work, and author/venue
    credibility. Scores are editorial judgments, not computed. Tag papers and copy clean notes
    into your note app. Generated __GENERATED__.</p>
</header>

<details class="interests" id="interests" open>
  <summary>Your interests<span class="hint">&mdash; add a CV, paper(s), or keyword(s), then regenerate the review</span></summary>
  <div class="igroups" id="igroups"></div>
  <div class="iadd">
    <select id="ikind" aria-label="Interest kind">
      <option value="cv">CV / profile</option>
      <option value="paper">Paper</option>
      <option value="keyword">Keyword</option>
    </select>
    <input id="ivalue" type="text" placeholder="Paste a CV URL, a paper title/DOI/URL, or a keyword&hellip;">
    <button class="btn" id="iaddbtn">&#65291; Add interest</button>
  </div>
  <div class="iactions">
    <span class="mset">
      <select id="provider" aria-label="Model provider">
        <option value="gemini">Gemini (AI Studio)</option>
        <option value="anthropic">Anthropic (Claude)</option>
        <option value="openai">OpenAI</option>
      </select>
      <input id="model" type="text" placeholder="model id" aria-label="Model id">
      <input id="apikey" type="password" placeholder="Your API key (not stored)" autocomplete="off" aria-label="Your API key">
    </span>
    <button class="btn regen" id="regen">&#8635; Regenerate review</button>
    <span class="istatus" id="istatus"></span>
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
    <h2 id="sidehead">Papers</h2>
    <ol id="sidelist"></ol>
  </aside>
  <main class="main" id="list"></main>
</div>

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
function gaugeHTML(nov){
  const n = Math.max(0,Math.min(100, nov==null?0:nov));
  return `<span class="gauge"><span class="gtrack"><span class="gmark" style="left:${n}%"></span></span>`
    + `<span class="glabel">Novelty ${n}/100</span></span>`;
}
function aimOf(p){
  const a = APPROACH[p.rank]; return a && a.aim ? a.aim : (p.aim||'');
}
function approachDD(p){
  const a = APPROACH[p.rank] || {};
  const algo = a.algo ? `<span class="algo">${esc(a.algo)}</span>` : '';
  return `${algo}${gaugeHTML(a.nov)}`
    + `<ul>`
    + (a.data ? `<li><b>Data:</b> ${esc(a.data)}</li>`:'')
    + (a.model ? `<li><b>Model:</b> ${esc(a.model)}</li>`:'')
    + (a.question ? `<li><b>Key question:</b> ${esc(a.question)}</li>`:'')
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
  const cls = st ? ' st-'+st : '';
  const btn = (key,label)=>`<button class="sbtn${st===key?' on-'+key:''}" data-status="${key}" data-rank="${p.rank}">${label}</button>`;
  return `<article class="card${cls}" id="card-${p.rank}" data-rank="${p.rank}">
    <div class="hrow">
      <span class="chip">${p.rank}</span>
      <h3><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
    </div>
    <div class="meta">${esc(p.authors)} &mdash; ${esc(p.institution)}</div>
    <div class="meta"><span class="badge">${esc(p.venue)}</span> ${esc(p.date)}</div>
    <div class="scores">Field impact ${p.impact}/10 &middot; Relevance to you ${p.rel}/10${coverBadge(p)}</div>
    <div class="tagrow">${tagChips(p)}${catChips(p)}</div>
    <div class="statusbar">
      ${btn('important','&#9733; Important')}
      ${btn('further','&#8599; Read further')}
      ${btn('done','&#10003; Done reviewing')}
      <button class="btn copy" data-copy="${p.rank}">&#8681; Copy for OneNote</button>
    </div>
    <div class="why"><b>Why it's on your list:</b> ${esc(p.relevance)}</div>
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
    const bg = st==='important'?'var(--important)':st==='further'?'var(--further)':st==='done'?'var(--done)':'var(--accent)';
    const sub = ((p.app||[])[0]||'') + (p.venue? ' &middot; '+esc(p.venue):'');
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
  if(a.data) bul.push(`<li><b>Data:</b> ${esc(a.data)}</li>`);
  if(a.model) bul.push(`<li><b>Model:</b> ${esc(a.model)}</li>`);
  if(a.question) bul.push(`<li><b>Key question:</b> ${esc(a.question)}</li>`);
  const resItems = (p.results||[]).filter(x=>x!=null && String(x).trim())
    .map(x=>`<li>${esc(x)}</li>`).join('');
  return `<div style="font-family:Calibri,sans-serif">
    <h2>${esc(p.title)}</h2>
    <p>${esc(p.authors)} &mdash; ${esc(p.institution)}</p>
    <p><b>${esc(p.venue)}</b>, ${esc(p.date)} &middot; Field impact ${p.impact}/10 &middot;
       Relevance ${p.rel}/10 &middot; <a href="${esc(p.url)}">${esc(p.url)}</a></p>
    <p><b>Application:</b> ${esc((p.app||[]).join(', '))} &nbsp; <b>Method:</b> ${esc((p.method||[]).join(', '))}</p>
    <p><b>Aim:</b> ${esc(aimOf(p))}</p>
    <p><b>Main approach:</b> ${esc(a.algo||'')} (Novelty ${a.nov==null?'?':a.nov}/100)</p>
    <ul>${bul.join('')}</ul>
    <p><b>Evaluation:</b></p><ul>${evalItems}</ul>
    ${resItems?`<p><b>Results:</b></p><ul>${resItems}</ul>`:''}
    <p><b>Limitation / next step:</b> ${esc(p.limitation)}</p>
    <p><b>Resources:</b> ${esc(p.resources)}</p>
    <p><b>Comments:</b> ${esc(p.comments)}</p>
    <p><b>Why relevant:</b> ${esc(p.relevance)}</p>
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
function toast(msg){
  const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(()=>t.classList.remove('show'), 1600);
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
function applyPayload(pl){
  DATA = pl.data || [];
  TAGS = pl.tags || {};
  APPROACH = pl.approach || {};
  CATEGORIES = pl.categories || [];
  if(Array.isArray(pl.inputs) && pl.inputs.length){
    INPUTS = pl.inputs.map(i=>({kind:i.kind, value:i.value}));
    ensureIds(); saveInputs();
  }
  hydrate(); tagFilter = null; renderInputs(); render();
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
async function regenerate(){
  if(!INPUTS.length){ setStatus('Add at least one interest first.'); return; }
  const keyEl = document.getElementById('apikey');
  const key = (keyEl ? keyEl.value : '').trim();
  if(!key){ setStatus('Enter your API key first.'); if(keyEl) keyEl.focus(); return; }
  const provider = (document.getElementById('provider')||{}).value || 'gemini';
  const model = ((document.getElementById('model')||{}).value || '').trim();
  const btn = document.getElementById('regen');
  btn.disabled = true;
  setStatus('Starting…');
  try{
    const res = await fetch('/review/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({inputs: INPUTS.map(i=>({kind:i.kind, value:i.value})), provider, model, apiKey:key})
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
        if(msg.type==='progress'){ setStatus(msg.label); }
        else if(msg.type==='done'){ applyPayload(msg.payload); done=true; setStatus(doneMsg()); }
        else if(msg.type==='error'){ errored = msg.detail || 'Could not regenerate.'; }
      }
    }
    if(errored) setStatus('Could not regenerate: ' + errored);
    else if(!done) setStatus('Regeneration ended without a result.');
  }catch(err){
    if(err instanceof TypeError) offlineFallback();          // fetch failed = no backend
    else setStatus('Could not regenerate: ' + err.message);  // backend reachable but errored
  }finally{ btn.disabled = false; }
}
document.getElementById('regen').addEventListener('click', regenerate);

// BYO model settings: remember provider/model, default the model per provider.
// (The API key is deliberately NOT persisted — it lives only in the field.)
(function(){
  const prov = document.getElementById('provider'), mdl = document.getElementById('model');
  if(!prov || !mdl) return;
  const DEF = {gemini:'gemini-flash-latest', anthropic:'claude-sonnet-5', openai:'gpt-4o'};
  const MS = 'byo-model-cfg-v1';
  try{ const c = JSON.parse(localStorage.getItem(MS)||'{}');
    if(c.provider) prov.value = c.provider; if(c.model) mdl.value = c.model; }catch(e){}
  if(!mdl.value) mdl.value = DEF[prov.value] || '';
  const save = ()=> localStorage.setItem(MS, JSON.stringify({provider:prov.value, model:mdl.value}));
  prov.addEventListener('change', ()=>{
    if(!mdl.value || Object.values(DEF).includes(mdl.value)) mdl.value = DEF[prov.value] || '';
    save();
  });
  mdl.addEventListener('change', save);
})();

renderInputs();
render();
</script>
</body>
</html>
"""
