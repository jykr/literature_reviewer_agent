# Specification — "Recent Computational Biology Review Queue"

A single-file, offline web app that presents a ranked, taggable, note-exportable
review of recent computational-biology papers, tailored to one or more
**interest inputs** — any mix of CV(s), seed paper(s), and keyword(s).
This document is complete enough to approximately reproduce the app from scratch.

---

## 1. Purpose & audience

- **Goal:** help a researcher triage recent papers: read structured summaries, tag them, and paste clean notes into their note-taking app of choice.
- **Interest inputs:** the researcher supplies one or more inputs, each of any
  supported kind — **CV** (URL or pasted text), **seed paper** (title/DOI/URL),
  or **keyword(s)** (free text). Inputs may be mixed freely. Each input becomes a
  research **cluster** (a CV may fan into several subfield sub-clusters); every
  cluster is also a scoring **category**, and papers that match the most
  categories are prioritized (see §2, §3d).
- **Scope window:** papers from ~mid-2025 to the build date (max ~1 year old).
- **Deliverable:** one self-contained `literature-reviewer.html` file. No build
  step, no server, no external network calls, no dependencies. Opens with
  `file://`. All state persists in `localStorage`.

## 2. Content-generation pipeline (how the data is produced)

The data is **researched, not hardcoded from memory**. To reproduce:
0. **Collect the interest inputs** — one or more items, each tagged with its kind
   (`cv` | `paper` | `keyword`) and its raw value (URL / DOI / text). The user may
   add / edit / remove inputs using an interface. Zero inputs is invalid; a single
   input is fine.

1. **Normalize each input into one or more clusters.** Each cluster is `{ id,
   label, source: <inputId>, terms:[...] }`.
   - **CV** → WebFetch (or read pasted text); extract the researcher's subfields
     and keywords, and split into one **sub-cluster per distinct subfield** (all
     back-referencing the same CV input).
   - **Paper** → resolve the title/DOI/URL; derive a cluster from its topic(s),
     method, and key terms.
   - **Keyword(s)** → one cluster per coherent keyword or phrase group, verbatim.
   Deduplicate/merge near-identical clusters, but keep each cluster's `source` list
   so a paper matching a merged cluster counts toward every contributing input.
   Every cluster is a **category** used for coverage scoring (§3d).
2. **Fan out parallel web research** (one agent per cluster), each asked for 4–6
   real, verifiable papers with: title, authors (1st + last/PI), institution, venue,
   date, URL, and draft summary fields. Deduplicate across clusters — when the same
   paper surfaces under multiple clusters, **merge** it and record *all* clusters/
   categories it hit (this drives coverage scoring, so do not drop the duplicates
   silently).
3. **Second pass per paper** to deepen two axes:
   - *Main approach* → is the method off-the-shelf or novel; name the algorithm.
   - *Evaluation* → new metric? new eval data? design novelty? eval limitations.
4. **Rank** by a blended judgment of, in priority order:
   - **Category coverage** — how many distinct input categories (clusters) the
     paper matches; **more categories ranks higher** (a paper answering several of
     the researcher's interests at once is the most valuable). This is the primary
     sort key.
   - then **field impact · relevance to the inputs · author/institution
     credibility · venue**, as tie-breakers within a coverage level.
   Optionally ground impact with real bibliometrics (e.g. OpenAlex citations + FWCI
   via a DOI bulk lookup).
5. **Integrity rule:** only include verified papers; where a figure can't be
   confirmed from primary text, hedge it in prose rather than invent. Prefer
   characterizing *how* a paper evaluated over quoting exact result numbers.

## 3. Data model

Two sources merged at load, both keyed by integer `rank` (1 = top):

### 3a. `DATA` — JSON array in `<script id="data" type="application/json">`
One object per paper:

| field | type | meaning |
|---|---|---|
| `rank` | int | 1-based rank; primary key & default sort |
| `title` | string | paper title (card heading, links to `url`) |
| `authors` | string | "First A., …, Senior Z." |
| `institution` | string | lead institution(s) |
| `venue` | string | journal / conference / "bioRxiv preprint" |
| `date` | string | e.g. "Jan 2026 (bioRxiv Jun 2025)" |
| `url` | string | canonical link (DOI / journal / preprint) |
| `insights` | string | **Evaluation** text, written with 4 inline labels: `New metric:` … `New eval data:` … `Design & novelty:` … `Eval limits:` … (parsed into bullets at render) |
| `limitation` | string | limitation / next step |
| `resources` | string | code/data/blog links as bare URLs (auto-linkified) |
| `comments` | string | reviewer's significance note |
| `relevance` | string | why it matters to *this* researcher (name which input(s) it serves) |
| `impact` | int 1–10 | field impact score |
| `rel` | int 1–10 | relevance-to-inputs score (best match across all inputs) |
| `cats` | int[] | ids of the input categories/clusters this paper matches (see §3d) |
| `cover` | int | `cats.length` — distinct categories hit; primary rank driver |
| `topic` | string | legacy single label (superseded by tags; may be unused) |
| `aim`, `approach` | string | legacy fallbacks; superseded by the `APPROACH` map |

### 3b. `TAGS` — JS object `{ rank: { app:[...], method:[...] } }`
- `app` = **Application** (biological problem): e.g. *Variant effect, Perturbation
  prediction, Spatial domains, GRN inference, Multiomics integration, Drug response,
  Cell annotation, Genome annotation, Clinical genetics, CRISPR screens*.
- `method` = **Method** (computational class): e.g. *Optimal transport, DNA language
  model, Foundation model, Graph neural network, CNN + Transformer, Benchmark /
  metric, Dataset, Software toolkit/pipeline, GAN, Gradient-boosted trees,
  Saturation genome editing (MAVE), Prime editing, Generative / ODE, RetNet, U-Net*.

Merged in with `DATA.forEach(p => { p.app = TAGS[p.rank].app; p.method = TAGS[p.rank].method; })`.

### 3c. `APPROACH` — JS object `{ rank: { algo, nov, aim, data, model, bio } }`
Holds the detailed narrative (single source of truth; JSON `aim`/`approach` are
fallbacks only):

- `algo` (string): short label of the **core method/algorithm** (e.g.
  "StripedHyena-2 (multi-hybrid convolution)"). Shown as a dark chip under *Main
  approach*.
- `nov` (int 0–100): position on the **Engineering ↔ Novel-algorithm** gauge.
  0 = pure engineering/integration (datasets, benchmarks, pipelines, off-the-shelf
  method applied); 100 = fundamentally new algorithm/architecture. Middle = modified
  existing architecture.
- `aim` (string): the goal **plus** the current limitation it addresses, phrased as
  "*Goal.* Limitation addressed: *the data + challenge / gap.*"
- `data` (string): what data the model/algorithm consumes (training + eval inputs, scale).
- `model` (string): the model/algorithm — if novel, *how*; if not, *what was done /
  how it was adapted to the data*.
- `bio` (string): the biological question the paper answers.

### 3d. `INPUTS` / `CATEGORIES` — the interest inputs and their clusters
- `INPUTS` — array `[{ id, kind:'cv'|'paper'|'keyword', value, label }]`, the raw
  items the user supplied (§2.0). Editable; persisted to
  `localStorage['compbio-review-inputs-v1']`.
- `CATEGORIES` — array `[{ id, label, sources:[inputId,…], color }]`, the clusters
  derived in §2.1 (one per subfield/paper-topic/keyword-group). Each is a scoring
  **category**; `sources` lets one category credit multiple inputs (e.g. a subfield
  shared by two CVs).
- A paper's `cats` (§3a) holds the ids of the categories it matched, and `cover =
  cats.length`. Coverage is the **primary rank key** (§2.4): sort by `cover`
  descending, then by the blended impact/relevance/credibility/venue judgment. A
  paper hitting 3 of the researcher's input categories outranks one hitting 1, even
  if the latter has a slightly higher impact score.

## 4. Card layout (rendered per paper, in view order)

1. **Header row:** rank chip · title (link) · authors · institution · venue badge +
   date · scores line ("Field impact N/10 · Relevance to you N/10 · **Matches N of M
   interests**"). The coverage figure uses `cover` / `INPUTS.length` (or
   `CATEGORIES.length`); highlight it when `cover ≥ 2`.
2. **Tag row:** `App` chips (blue) then `Method` chips (purple), followed by
   **Category chips** (one per entry in the paper's `cats`, colored per
   `CATEGORIES[i].color`, labeled with the input/cluster). Each chip is clickable →
   sets the active tag/category filter.
3. **Status buttons:** `★ Important` · `↗ Read further` · `✓ Done reviewing`
   (toggle, mutually exclusive) + right-aligned `⧉ Copy for OneNote`.
4. **"Why it's on your list"** callout (green) = `relevance`.
5. **Definition list** (`<dl>`), in order:
   - **Aim** = `APPROACH[rank].aim`
   - **Main approach** = dark **algo** chip + **novelty gauge**, then 3 bullets:
     **Data**, **Model**, **Biological question**.
   - **Evaluation** = `insights` auto-split into 4 bullets on its inline labels
     (bolded): New metric · New eval data · Design & novelty · Eval limits.
   - **Limitation / next step**, **Additional resources**, **Comments**.

## 5. Layout & navigation

- **Full-width header** (title, subtitle, criteria explainer).
- **Interests panel** (below the header, collapsible): lists the current `INPUTS`
  as removable chips grouped by kind (CV / paper / keyword), with an "＋ Add
  interest" control (pick kind, paste value). Shows each input's derived
  categories. Editing inputs is what re-scopes the review (§2).
- **Sticky controls bar** (`position:sticky; top:0`): Show filter, Sort, Search box,
  "Copy view for OneNote" button, active-tag indicator, live counts.
- **Two-column `.layout`** below the controls:
  - **Left sidebar** (`width:280px`, sticky, own scroll): header "Papers (N)" + an
    ordered list mirroring the current filtered/sorted view. Each item = rank chip
    (colored by status) + truncated title + subline (top app tag · venue). **Click
    → smooth-scroll to that card**, which flashes a blue ring (`.flash` animation).
  - **Main content** = the card list.
- **Responsive:** sidebar hidden below 900px; `<dl>` collapses to single column
  below 560px; horizontal overflow never allowed.

## 6. Interactions & state

- **Status tagging:** click a status button → set/clear `status[rank]`; persisted to
  `localStorage['compbio-review-status-v1']` (JSON `{rank: 'important'|'further'|'done'}`).
  "Done" cards dim; important/further/done get a colored left border; sidebar rank
  chip recolors.
- **Filter (`Show`):** All / Important / Read further / Done / Untagged.
- **Sort:** Coverage (`cover` desc — default, matches the `rank` order from §2.4) |
  Rank | Field impact | Relevance to me | Venue A–Z | Status
  (important→further→untagged→done), all tie-broken by rank.
- **Search:** substring match over title, authors, institution, venue, app+method
  tags, and category labels.
- **Tag filter:** clicking any App/Method/Category chip filters list + sidebar to
  that tag (Category chips filter to papers whose `cats` include that category);
  an indicator "Filter: <val> (kind) [clear ✕]" appears; clicking the active chip or
  ✕ clears it. Selected chip gets an outline.
- **Re-render** is a single `render()` that rebuilds list, sidebar, tag bar, and
  counts from `currentView()` (the filtered+sorted array). Everything stays in sync.

## 7. OneNote export

- Per-card `⧉ Copy for OneNote` and a global `⧉ Copy view for OneNote`
  (current filtered view).
- Builds **simple inline-styled HTML** (Calibri; `<h2>`, `<b>`, `<ul>/<li>`,
  `<a>`, `<p>`) — deliberately minimal CSS so OneNote paste stays clean. Includes:
  heading, authors/institution, venue + scores + link, Application/Method line,
  Aim, Main approach (algo chip + "Novelty N/100" + Data/Model/Biological-question
  bullets), Evaluation bullets, Limitation, Resources, Comments, Why-relevant.
- Copy uses the async Clipboard API with a `ClipboardItem` carrying both
  `text/html` and a `text/plain` fallback; if unavailable, falls back to a hidden
  `contenteditable` element + `document.execCommand('copy')`. A toast confirms.

## 8. Styling tokens

CSS variables: `--bg#f6f7f9 --card#fff --ink#1a1d21 --muted#5b6570 --line#e3e7ec
--accent#2b6cb0 --accent-soft#e8f0f8` and status colors `--important#c0392b
--further#b7791f --done#2f855a`. System font stack. Cards: white, 12px radius,
1px border, subtle shadow, colored left border when tagged. App chips blue
(`#e8f0f8/#255e91`), Method chips purple (`#efeafc/#5b3fa8`). Algo chip dark
(`#1f2733`, white text). Gauge = 120px track with a grey→purple gradient and a
white dot (`.gmark`) positioned at `left:{nov}%`.

## 9. Rendering functions (JS contract)

- `esc(s)` — HTML-escape. `linkify(s)` — escape then turn bare `https?://…` URLs
  into `<a target=_blank>`. (Because fields are escaped, **field text must not
  contain intended HTML**; structure/bold is added by the render functions.)
- `tagChips(p)` (App + Method + **Category** chips, the last from `p.cats` →
  `CATEGORIES`), `coverBadge(p)` ("Matches N of M interests"), `gaugeHTML(nov)`,
  `approachDD(p)` (algo chip + gauge + Data/Model/
  Bio bullets), `aimOf(p)`, `evalDD(p)` (splits `insights` on `EVAL_LABELS =
  ['New metric:','New eval data:','Design & novelty:','Eval limits:']`),
  `cardHTML(p)`, `currentView()`, `render()`, `oneNoteHTML(p)`, `copyHTML(html,msg)`,
  `toast(msg)`.
- Event delegation: one listener on the list (status / tag / copy), one on the
  sidebar (scroll-to), one on the tag bar (clear), plus `input` listeners on the
  three controls. Final line calls `render()`.

## 10. File structure

Single `.html`: `<head>` with one `<style>`; `<body>` = header, controls,
`.layout`(sidebar + main), toast div; then `<script id="data" type="application/json">`
(the `DATA` array) and a second `<script>` (the `TAGS` + `APPROACH` maps and all
logic). Everything inlined.

## 11. Reproduction checklist

0. Collect `INPUTS` (§2.0) and derive `CATEGORIES` (§2.1); persist inputs.
1. Generate/verify paper content per §2; fill `DATA`, `TAGS`, `APPROACH`, and each
   paper's `cats`/`cover`.
2. Assign `impact`/`rel` (1–10) and `nov` (0–100) per paper; set the rank order by
   `cover` desc, then the blended judgment (§2.4).
3. Implement the render/interaction contract (§6, §9) and layout (§4, §5).
4. Wire the OneNote export (§7) and `localStorage` persistence.
5. Validate: JSON parses; JS `node --check`; open in a browser; confirm tagging,
   sorting, tag-filter, sidebar scroll, and both copy buttons.

## 12. Known constraints / honesty notes

- Impact/relevance/novelty scores are **editorial judgments** (optionally grounded
  in bibliometrics), not computed.
- Some quantitative claims come from preprints/secondary sources where publisher
  full text was inaccessible; such cases are hedged in prose.
- The app is retrieval/summarization only — it does not run any of the papers'
  models or verify their results.
