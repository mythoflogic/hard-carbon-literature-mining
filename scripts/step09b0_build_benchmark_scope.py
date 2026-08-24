#!/usr/bin/env python3

from pathlib import Path
import re

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

FINAL_PATH = (
    ROOT
    / "processed_tables"
    / "final_table_dataset.csv"
)

BENCHMARK_PATH = (
    ROOT
    / "data"
    / "feedstock.xlsx"
)

MAPPING_PATH = (
    ROOT
    / "outputs"
    / "paper_to_doi_matching_report.xlsx"
)

OUT_DIR = (
    ROOT
    / "processed_tables"
    / "benchmark_final_table_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

DOI_RE = re.compile(
    r"10\.\d{4,9}/"
    r"[-._;()/:a-z0-9]+",
    flags=re.I,
)


def clean_doi(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        text,
    )

    text = re.sub(
        r"^doi:\s*",
        "",
        text,
    )

    match = DOI_RE.search(text)

    if not match:
        return ""

    doi = match.group(0)

    doi = doi.rstrip(
        ".,;:)]}"
    )

    return doi


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# Read frozen extraction + protected benchmark
# ============================================================

final = pd.read_csv(
    FINAL_PATH,
    low_memory=False,
)

manual = pd.read_excel(
    BENCHMARK_PATH,
)

mapping = pd.read_excel(
    MAPPING_PATH,
)


# ============================================================
# Validate frozen final table
# ============================================================

if len(final) != 521:
    raise RuntimeError(
        "Frozen final table no longer has "
        f"521 rows: {len(final)}"
    )

if final["source_key"].nunique() != len(final):
    raise RuntimeError(
        "Final source_key is not unique."
    )


# ============================================================
# Paper -> DOI bridge
# ============================================================

mapping = mapping.copy()

mapping["paper_id"] = (
    mapping["Markdown_filename"]
    .astype(str)
    .str.replace(
        r"\.md$",
        "",
        regex=True,
    )
)


mapping["doi_normalized"] = (
    mapping[
        "Expected_DOIs_detected"
    ]
    .apply(clean_doi)
)


# Fall back to front DOI candidate only when the expected
# detected DOI field did not yield a DOI.
fallback = (
    mapping[
        "All_front_DOI_candidates"
    ]
    .apply(clean_doi)
)

mapping[
    "doi_normalized"
] = (
    mapping["doi_normalized"]
    .where(
        mapping[
            "doi_normalized"
        ].ne(""),
        fallback,
    )
)


bridge_source = (
    mapping[
        [
            "paper_id",
            "doi_normalized",
            "Status",
            "Markdown_filename",
        ]
    ]
    .copy()
)


# One markdown file must not map to conflicting DOIs.
conflicts = (
    bridge_source[
        bridge_source[
            "doi_normalized"
        ].ne("")
    ]
    .groupby("paper_id")[
        "doi_normalized"
    ]
    .nunique()
)

conflicts = conflicts[
    conflicts > 1
]

if len(conflicts):

    raise RuntimeError(
        "Conflicting DOI mapping for paper_id:\n"
        + conflicts.to_string()
    )


bridge_source = (
    bridge_source
    .drop_duplicates(
        subset=["paper_id"],
        keep="first",
    )
)


final_papers = pd.DataFrame(
    {
        "paper_id":
            sorted(
                final["paper_id"]
                .dropna()
                .astype(str)
                .unique()
            )
    }
)


bridge = final_papers.merge(
    bridge_source,
    on="paper_id",
    how="left",
    validate="one_to_one",
)


bridge[
    "doi_mapping_ok"
] = (
    bridge["doi_normalized"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)


# ============================================================
# Attach DOI to all 521 final rows
# ============================================================

final_with_doi = final.merge(
    bridge[
        [
            "paper_id",
            "doi_normalized",
            "doi_mapping_ok",
        ]
    ],
    on="paper_id",
    how="left",
    validate="many_to_one",
)


# ============================================================
# Manual benchmark normalization
# ============================================================

manual = manual.copy()

manual.insert(
    0,
    "benchmark_row_id",
    [
        f"MANUAL_{i:05d}"
        for i in range(
            1,
            len(manual) + 1,
        )
    ],
)


manual[
    "doi_normalized"
] = manual[
    "DOI"
].apply(
    clean_doi
)


manual[
    "temperature_C"
] = numeric(
    manual["T (°C)"]
)

manual[
    "C_value"
] = numeric(
    manual["C_char(wt%)"]
)

manual[
    "H_value"
] = numeric(
    manual["H_char(wt%)"]
)

manual[
    "N_value"
] = numeric(
    manual["N_char(wt%)"]
)

manual[
    "O_value"
] = numeric(
    manual["O_char(wt%)"]
)


# ============================================================
# Scope benchmark to final-table papers
# ============================================================

represented_dois = set(
    bridge.loc[
        bridge["doi_mapping_ok"],
        "doi_normalized",
    ]
    .astype(str)
)


benchmark_scope = manual[
    manual["doi_normalized"]
    .astype(str)
    .isin(
        represented_dois
    )
].copy()


# ============================================================
# Define extracted rows comparable to manual benchmark
# ============================================================

final_with_doi[
    "benchmark_eligible"
] = (
    final_with_doi[
        "doi_mapping_ok"
    ]
    .fillna(False)
    .astype(bool)
    &
    final_with_doi[
        "temperature_status"
    ]
    .eq("exact")
    &
    numeric(
        final_with_doi[
            "temperature_C"
        ]
    ).notna()
    &
    (
        numeric(
            final_with_doi[
                "reported_element_count"
            ]
        )
        >= 2
    )
)


eligible = final_with_doi[
    final_with_doi[
        "benchmark_eligible"
    ]
].copy()


# ============================================================
# Hard DOI invariants
# ============================================================

missing_bridge = bridge[
    ~bridge["doi_mapping_ok"]
].copy()


duplicate_final_doi_mapping = (
    bridge.loc[
        bridge["doi_mapping_ok"],
        "doi_normalized",
    ]
    .duplicated()
    .sum()
)


# Do not assume duplicate DOI is impossible:
# if present, report it rather than silently collapsing papers.

# ============================================================
# Per-paper accounting
# ============================================================

total_counts = (
    final_with_doi
    .groupby(
        [
            "paper_id",
            "doi_normalized",
        ],
        dropna=False,
    )
    .size()
    .rename(
        "final_rows_total"
    )
    .reset_index()
)


eligible_counts = (
    eligible
    .groupby(
        [
            "paper_id",
            "doi_normalized",
        ],
        dropna=False,
    )
    .size()
    .rename(
        "final_rows_benchmark_eligible"
    )
    .reset_index()
)


manual_counts = (
    benchmark_scope
    .groupby(
        "doi_normalized",
        dropna=False,
    )
    .size()
    .rename(
        "benchmark_rows"
    )
    .reset_index()
)


paper_counts = total_counts.merge(
    eligible_counts,
    on=[
        "paper_id",
        "doi_normalized",
    ],
    how="left",
)

paper_counts[
    "final_rows_benchmark_eligible"
] = (
    paper_counts[
        "final_rows_benchmark_eligible"
    ]
    .fillna(0)
    .astype(int)
)


paper_counts = paper_counts.merge(
    manual_counts,
    on="doi_normalized",
    how="left",
)

paper_counts[
    "benchmark_rows"
] = (
    paper_counts[
        "benchmark_rows"
    ]
    .fillna(0)
    .astype(int)
)


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "metric":
                "final_rows_total",
            "count":
                len(final),
        },
        {
            "metric":
                "final_papers",
            "count":
                final[
                    "paper_id"
                ].nunique(),
        },
        {
            "metric":
                "final_papers_mapped_to_doi",
            "count":
                bridge[
                    "doi_mapping_ok"
                ].sum(),
        },
        {
            "metric":
                "final_papers_missing_doi",
            "count":
                len(missing_bridge),
        },
        {
            "metric":
                "duplicate_doi_across_final_papers",
            "count":
                int(
                    duplicate_final_doi_mapping
                ),
        },
        {
            "metric":
                "full_benchmark_rows",
            "count":
                len(manual),
        },
        {
            "metric":
                "full_benchmark_dois",
            "count":
                manual[
                    "doi_normalized"
                ].nunique(),
        },
        {
            "metric":
                "represented_benchmark_rows",
            "count":
                len(
                    benchmark_scope
                ),
        },
        {
            "metric":
                "represented_benchmark_dois",
            "count":
                benchmark_scope[
                    "doi_normalized"
                ].nunique(),
        },
        {
            "metric":
                "final_benchmark_eligible_rows",
            "count":
                len(eligible),
        },
        {
            "metric":
                "final_exact_temperature_rows",
            "count":
                final[
                    "temperature_status"
                ]
                .eq("exact")
                .sum(),
        },
        {
            "metric":
                "final_nonbenchmark_rows",
            "count":
                (
                    ~final_with_doi[
                        "benchmark_eligible"
                    ]
                ).sum(),
        },
    ]
)


# ============================================================
# Save derived audit files only
# ============================================================

bridge.to_csv(
    OUT_DIR
    / "paper_doi_bridge.csv",
    index=False,
)

final_with_doi.to_csv(
    OUT_DIR
    / "final_table_with_doi.csv",
    index=False,
)

benchmark_scope.to_csv(
    OUT_DIR
    / "benchmark_scope.csv",
    index=False,
)

eligible.to_csv(
    OUT_DIR
    / "final_benchmark_eligible.csv",
    index=False,
)

paper_counts.to_csv(
    OUT_DIR
    / "per_paper_scope_counts.csv",
    index=False,
)

summary.to_csv(
    OUT_DIR
    / "benchmark_scope_summary.csv",
    index=False,
)


# ============================================================
# Console report
# ============================================================

print("=" * 100)
print("STEP 09B0 — BENCHMARK SCOPE")
print("=" * 100)

print()
print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("MISSING DOI MAPPINGS")
print("=" * 100)

if len(missing_bridge):

    print(
        missing_bridge.to_string(
            index=False
        )
    )

else:

    print("None")


print()
print("=" * 100)
print("PER-PAPER COUNTS")
print("=" * 100)

print(
    paper_counts
    .sort_values(
        [
            "doi_normalized",
            "paper_id",
        ]
    )
    .to_string(
        index=False,
        max_colwidth=55,
    )
)


print()
print("=" * 100)
print("INVARIANTS")
print("=" * 100)

print(
    "Final rows remain 521:",
    len(final) == 521,
)

print(
    "Unique final source keys:",
    final["source_key"]
    .nunique(),
)

print(
    "Protected benchmark rows remain:",
    len(manual),
)

print(
    "Protected benchmark modified: False"
)

print()
print("Saved derived audit files:")
print("-", OUT_DIR)
