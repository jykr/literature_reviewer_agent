---
name: define-scope
description: >-
  Turn a researcher's interest inputs — any mix of CV/site URLs, seed papers
  (title/DOI/URL), and free-text keywords — into up to 5 deduplicated subfield
  clusters plus a short researcher profile, emitted as strict Scope JSON. Use at
  the start of a literature-review run to decide WHAT to research. If given no
  usable input, ask for a CV URL or a few topics instead of inventing a scope.
---

# Define Scope

Derives the research **scope** for the compbio paper-review pipeline: a 2–4
sentence researcher `profile` plus up to 5 subfield `clusters`, each with 3–8
search keywords. This is SPEC §2.0–2.1. Downstream, one research agent fans out
per cluster and every cluster doubles as a scoring category (§3d).

## When to use

- The user supplies one or more **interest inputs** of any kind (CV, paper,
  keyword), in any combination.
- You need a clean cluster set before fanning out web research.

Do **not** use for ranking, fetching papers, or rendering — only for producing
the `Scope` object.

## Inputs

The user's message may contain ANY combination of:

- **CV / personal-site URL** (or pasted CV text)
- **Seed paper** — title, DOI, or URL
- **Free-text topics / keywords / subfield names**
- ...or none of the above.

## Procedure

1. **Classify each input into its kind.** Run the classifier to split the raw
   message into typed items (`cv` | `paper` | `keyword`):

   ```bash
   echo '["https://jdoe.github.io/cv", "single-cell foundation models", "10.1101/2025.06.01.123456"]' \
     | python scripts/classify_input.py
   ```

   It emits `[{"kind": "...", "value": "..."}, ...]`. See
   `references/classification_rules.md` for the heuristics and how to override a
   misclassification.

2. **Normalize each input into one or more clusters** (SPEC §2.1):
   - **cv** → fetch the URL with your web tool (or read pasted text); extract
     subfields, recurring methods, and seniority. Split into **one sub-cluster
     per distinct subfield**. Record the profile sentence(s) here.
   - **paper** → resolve the title/DOI/URL; derive one cluster from its
     topic(s), method, and key terms. This is the "paper_fields" step — it is
     model + web work, not a script, because it needs to read the paper.
   - **keyword** → one cluster per coherent keyword/phrase group, verbatim. User
     keywords take priority and merge with anything the CV implies.

   Name clusters using the shared vocabulary in
   `references/subfield_taxonomy.md` so labels line up with the pipeline's tags.

3. **Consolidate.** Feed the raw cluster list through the merger to dedupe
   near-identical clusters, union their keywords, keep every contributing
   `source`, and cap to 5 clusters / 3–8 keywords each:

   ```bash
   cat raw_clusters.json | python scripts/consolidate_clusters.py
   ```

4. **Emit strict Scope JSON.** Fill `assets/scope_output_template.json` and
   output ONLY that object — no prose, no code fences. The contract (field types,
   caps, examples) is in `references/cluster_schema.md`.

## Edge case: no usable input

If the message has NEITHER a URL/paper NOR any usable topics (empty, a greeting,
or unrelated chit-chat), do **not** invent a scope. Reply in plain language
asking for a CV/site URL or a few research topics, and stop — emit no JSON.

## Output contract (summary)

```json
{
  "profile": "<2-4 sentences: subfields, methods, seniority; note if derived only from topics>",
  "clusters": [
    {"name": "<short subfield label>", "keywords": ["kw1", "kw2", "..."]}
  ]
}
```

Full schema, caps, and worked examples: `references/cluster_schema.md`.
