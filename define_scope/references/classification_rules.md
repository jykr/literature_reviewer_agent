# Input classification rules

How `scripts/classify_input.py` tags each raw item. Kinds: `cv`, `paper`,
`keyword` (SPEC §2.0).

## Decision order

1. **Bare DOI** (`10.xxxx/...` with no URL scheme) → `paper`.
2. **URL on a known paper host** (`doi.org`, `arxiv.org`, `biorxiv.org`,
   `nature.com`, `pubmed`, `openreview.net`, publisher domains, …) → `paper`.
3. **Any other URL** → `cv` (personal site / hosted CV to fetch).
4. **Everything else** (free text) → `keyword`.

## Known limits (when to override by hand)

- A **pasted CV as plain text** (no URL) classifies as `keyword`. If the user
  clearly pasted a CV, treat it as `cv` and extract subfields directly.
- A **paper title with no DOI/URL** classifies as `keyword`, because there's no
  id to resolve. That's usually fine — the model can still derive a cluster — but
  if you can resolve the title to a DOI, reclassify it as `paper` for a tighter
  cluster.
- A **preprint on a personal domain** may look like a CV URL. If a fetch reveals
  it's a single paper, treat it as `paper`.

The classifier is a fast triage, not the final word: the normalization step
(SKILL.md step 2) can always re-decide once it has fetched the content.
