cat > "$REPO/README.md" <<'EOF'
# Hard-Carbon Literature-Mining Pipeline

A reproducible scientific literature-mining workflow for extracting structured
carbon-material data from research papers.

The pipeline combines:

- PDF/Markdown preprocessing
- document chunking
- semantic retrieval
- scientific-table detection
- multi-level header repair
- transposed-table recovery
- deterministic row classification
- structural recovery
- context-aware semantic recovery
- provenance-aware final table assembly
- DOI-constrained benchmark validation
- automated comparison and scientific plotting

The workflow was developed in the context of carbon materials and hard-carbon
anodes for sodium-ion batteries, with a controlled literature benchmark used
to evaluate extraction accuracy.

---

## Pipeline overview

```text
Research papers
      |
      v
PDF / Markdown preprocessing
      |
      v
Chunking + semantic retrieval
      |
      v
Candidate scientific tables
      |
      v
Table normalization
      |
      +--> multi-level header repair
      |
      +--> transposed-table recovery
      |
      v
Row classification
      |
      v
Deterministic structural recovery
      |
      v
Semantic/context recovery
      |
      v
Final provenance-aware dataset
      |
      v
DOI-scoped one-to-one benchmark validation
      |
      v
Trend plots + parity plots + Van Krevelen analysis
