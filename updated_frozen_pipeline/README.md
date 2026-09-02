# Updated Frozen Pipeline

This directory contains the validated frozen generation of the
hard-carbon literature-mining pipeline.

The historical public version is preserved separately in:

`../original_pipeline/`

## Validation status

The final represented benchmark contained:

- 33 represented papers
- 351 represented benchmark rows
- 439 benchmark-eligible extracted rows

Final validation results:

- Strict one-to-one matches: **323 / 351**
- Strict numerical recall: **92.02%**
- Source-temperature-compatible matches: **327 / 351**
- Source-temperature-compatible recall: **93.16%**
- Unresolved source/extraction failures: **0**
- Accounted represented benchmark rows: **351 / 351**

The 351 / 351 accounting result is not described as 100% recall or accuracy.
It means every represented benchmark row was assigned either to a validated
match or to a documented benchmark-side discrepancy class.

## Final discrepancy classes

The remaining 24 post-range benchmark rows were classified as:

- benchmark elemental inconsistency: 10
- benchmark temperature inconsistency: 6
- benchmark basis inconsistency: 5
- benchmark elemental row-shift inconsistency: 3
- unresolved source/extraction failure: 0

## Range-compatible validation

Four benchmark records associated with a reported processing interval were
validated as source-temperature-compatible rather than by inventing a midpoint
temperature.

No benchmark-derived temperature was inserted into the extracted source data.

## Directory structure

- `scripts/` — active frozen extraction, structural-recovery, semantic-recovery,
  benchmark-validation, and final-report scripts
- `results/final_validation/` — aggregate validation summaries
- `figures/final_validation/` — final validation figures
- `docs/` — pipeline documentation
- `data/` — data-structure documentation only

## Data safety

The protected benchmark workbook is not included.

Raw benchmark rows, raw papers, Markdown corpora, embeddings, raw LLM
responses, checkpoints, and private runtime artifacts are also excluded.

## Hard-carbon application phase

This frozen directory represents the completed validation pipeline.

The subsequent hard-carbon scientific application—precursor, carbonization
temperature, BET surface area, pore volume, d002, ID/IG, reversible capacity,
ICE, and related properties—is being developed separately and is not included
in this frozen snapshot.
