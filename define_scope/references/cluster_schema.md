# Scope output contract

Mirrors `app/schemas.py` (`Scope`, `Cluster`) and SPEC §2.1 / §3. The scope step
must emit exactly this shape — nothing else — as raw JSON (no code fences).

## Shape

```json
{
  "profile": "string, 2-4 sentences",
  "clusters": [
    { "name": "string", "keywords": ["string", "..."] }
  ]
}
```

## Field rules

| Field | Type | Rule |
|---|---|---|
| `profile` | string | 2–4 sentences: subfields, recurring methods, seniority. If derived only from keywords (no CV), say so. Drives per-paper relevance scoring. |
| `clusters` | array | **1–5** items. Minimise overlap between clusters. |
| `clusters[].name` | string | Short subfield label, e.g. `"Variant effect prediction"`. Prefer a label from `subfield_taxonomy.md`. |
| `clusters[].keywords` | string[] | **3–8** concrete search terms characterising the subfield. |

> The pipeline also tracks a `source` per cluster internally (which input it came
> from, SPEC §2.1) for coverage scoring. `app/schemas.py::Cluster` does **not**
> currently carry `source`, so the final emitted object drops it. Keep it during
> consolidation, strip it on output. If you later add `source` to the schema,
> stop stripping it.

## Worked example

Input: CV URL for a scientist doing single-cell + variant modelling, plus the
keyword `"optimal transport"`.

```json
{
  "profile": "Computational biologist (postdoc/PI level) working on single-cell foundation models and non-coding variant effect prediction, with recurring use of transformers and optimal-transport methods. Publishes benchmarks and open tooling. Profile derived from CV plus one user-supplied topic.",
  "clusters": [
    {"name": "Single-cell foundation models", "keywords": ["scRNA-seq", "foundation model", "cell embedding", "zero-shot annotation", "transformer"]},
    {"name": "Variant effect prediction", "keywords": ["non-coding variant", "DNA language model", "regulatory variant", "MAVE", "saturation mutagenesis"]},
    {"name": "Optimal transport", "keywords": ["optimal transport", "trajectory inference", "distribution alignment", "Wasserstein"]}
  ]
}
```

## Common mistakes

- Emitting prose or a code fence around the JSON. Output the bare object.
- More than 5 clusters, or heavily overlapping clusters (merge them).
- Fewer than 3 keywords in a cluster (top it up).
- Inventing a scope when the user gave no usable input (ask instead).
