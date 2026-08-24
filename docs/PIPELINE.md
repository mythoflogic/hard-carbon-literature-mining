# Pipeline

## 1. Literature ingestion
Convert research publications to machine-readable text and create document chunks.

## 2. Retrieval
Retrieve relevant scientific passages and candidate tables from each paper.

## 3. Table processing
Normalize scientific tables, repair multi-level headers, recover transposed tables,
and identify elemental-composition rows.

## 4. Structural recovery
Recover sample identities, processing temperatures, and elemental values using
deterministic structural rules while preserving source provenance.

## 5. Semantic recovery
Use surrounding table, caption, methods, and paper context for rows that cannot
be resolved structurally.

## 6. Final dataset
Merge deterministic and semantic results into a provenance-aware final table.

## 7. Benchmark validation
Map records by DOI and perform conservative one-to-one comparison against the
manually curated benchmark.

## 8. Scientific analysis
Generate temperature overlays, parity plots, Van Krevelen diagrams, curve fits,
and benchmark-performance summaries.
