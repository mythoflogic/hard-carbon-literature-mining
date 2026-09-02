#!/usr/bin/env python3

"""
STEP 07F3
=========

Repair Elemental Means Anchored by ± in Multiline Cells

Purpose
-------
Repair a narrowly defined parsing failure where a table cell contains
multiple vertically stacked numeric fragments and the reported elemental
mean is the unique fragment marked with the ± symbol.

Example:

    0.42<br>82.48±

must yield:

    82.48

whereas:

    85.56±<br>0.30

already correctly yields:

    85.56

Safety principles
-----------------
- operate only on C/H/N/O candidate cells;
- require exactly one line fragment containing ±;
- require exactly one numeric value in that ± fragment;
- require multiple numeric values in the complete raw cell;
- only repair rows where at least two elemental fields independently
  show the same problem;
- never modify sample, temperature, classification, or source text;
- preserve all upstream columns;
- record row-level and cell-level provenance;
- no LLM is used.
"""

from pathlib import Path
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

project_dir = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

processed_tables_dir = (
    project_dir
    / "processed_tables"
)

input_path = (
    processed_tables_dir
    / "validated_table_rows.csv"
)

output_path = (
    processed_tables_dir
    / "validated_table_rows_pm_repaired.csv"
)

audit_path = (
    processed_tables_dir
    / "pm_mean_cell_repair_audit.csv"
)


# ============================================================
# Helpers
# ============================================================

ELEMENTS = [
    "C",
    "H",
    "N",
    "O",
]


def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def numeric_tokens(value):

    text = clean_text(
        value
    )

    text = (
        text
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    matches = re.findall(
        r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)",
        text,
    )

    return [
        float(match)
        for match in matches
    ]


def numeric_value(value):

    if value is None:
        return None

    if pd.isna(value):
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def pm_mean_candidate(raw_value):
    """
    Return the numeric value belonging to the unique line fragment
    containing ±, but only for a genuinely multi-number cell.
    """

    text = clean_text(
        raw_value
    )

    if not text:
        return None

    if "±" not in text:
        return None

    all_numbers = numeric_tokens(
        text
    )

    # This repair is only relevant when the cell contains
    # multiple numeric fragments.
    if len(all_numbers) < 2:
        return None

    fragments = re.split(
        r"(?:<br\s*/?>|\n)",
        text,
        flags=re.IGNORECASE,
    )

    fragments = [
        fragment.strip()
        for fragment in fragments
        if fragment.strip()
    ]

    pm_fragments = [
        fragment
        for fragment in fragments
        if "±" in fragment
    ]

    # Ambiguous cells remain untouched.
    if len(pm_fragments) != 1:
        return None

    pm_numbers = numeric_tokens(
        pm_fragments[0]
    )

    if len(pm_numbers) != 1:
        return None

    proposed = pm_numbers[0]

    if not (
        0 <= proposed <= 100
    ):
        return None

    return proposed


# ============================================================
# Load
# ============================================================

if not input_path.exists():

    raise FileNotFoundError(
        f"Input file not found:\n"
        f"{input_path}"
    )


df = pd.read_csv(
    input_path,
    low_memory=False,
)


print()
print("=" * 78)
print(
    "STEP 07F3 — ±-ANCHORED "
    "ELEMENTAL MEAN REPAIR"
)
print("=" * 78)
print()

print(
    "Rows loaded:",
    len(df),
)


# ============================================================
# Provenance columns
# ============================================================

df[
    "pm_mean_repaired_07f3"
] = False

df[
    "pm_mean_repaired_elements_07f3"
] = None

df[
    "pm_mean_repair_rule_07f3"
] = None


# ============================================================
# Discover proposed repairs without modifying anything
# ============================================================

row_proposals = {}

for idx, row in df.iterrows():

    proposals = {}

    for element in ELEMENTS:

        raw_column = (
            f"{element}_candidate_raw"
        )

        value_column = (
            f"{element}_value"
        )

        if (
            raw_column
            not in df.columns
        ):
            continue

        raw_value = row.get(
            raw_column
        )

        proposed = pm_mean_candidate(
            raw_value
        )

        if proposed is None:
            continue

        current = numeric_value(
            row.get(
                value_column
            )
        )

        if current is None:
            continue

        # Current parsed value must itself occur in the
        # original candidate cell. This prevents replacing
        # values that came from another source.
        tokens = numeric_tokens(
            raw_value
        )

        current_is_source_token = any(
            abs(
                current - token
            ) <= 1e-12
            for token in tokens
        )

        if not current_is_source_token:
            continue

        if (
            abs(
                current - proposed
            )
            <= 1e-12
        ):
            continue

        proposals[
            element
        ] = {
            "raw":
                clean_text(
                    raw_value
                ),
            "old":
                current,
            "new":
                proposed,
        }

    if proposals:
        row_proposals[
            idx
        ] = proposals


# ============================================================
# Conservative row-level coherence gate
# ============================================================

accepted_rows = {
    idx: proposals
    for idx, proposals
    in row_proposals.items()
    if len(proposals) >= 2
}


# ============================================================
# Apply accepted repairs
# ============================================================

audit_records = []

for idx, proposals in (
    accepted_rows.items()
):

    repaired_elements = []

    for element, info in (
        proposals.items()
    ):

        value_column = (
            f"{element}_value"
        )

        df.at[
            idx,
            value_column,
        ] = info[
            "new"
        ]

        repaired_elements.append(
            element
        )

        audit_records.append(
            {
                "row_index":
                    idx,

                "paper_id":
                    df.at[
                        idx,
                        "paper_id",
                    ]
                    if "paper_id"
                    in df.columns
                    else None,

                "table_id":
                    df.at[
                        idx,
                        "table_id",
                    ],

                "source_row_index":
                    df.at[
                        idx,
                        "source_row_index",
                    ],

                "sample_raw":
                    df.at[
                        idx,
                        "sample_raw",
                    ]
                    if "sample_raw"
                    in df.columns
                    else None,

                "semantic_class":
                    df.at[
                        idx,
                        "semantic_class",
                    ]
                    if "semantic_class"
                    in df.columns
                    else None,

                "element":
                    element,

                "candidate_raw":
                    info[
                        "raw"
                    ],

                "old_value":
                    info[
                        "old"
                    ],

                "new_value":
                    info[
                        "new"
                    ],

                "recovery_rule":
                    (
                        "UNIQUE_PM_FRAGMENT_"
                        "IN_MULTILINE_CELL"
                    ),

                "raw_source_row":
                    df.at[
                        idx,
                        "raw_source_row",
                    ]
                    if "raw_source_row"
                    in df.columns
                    else None,
            }
        )

    df.at[
        idx,
        "pm_mean_repaired_07f3",
    ] = True

    df.at[
        idx,
        "pm_mean_repaired_elements_07f3",
    ] = ",".join(
        repaired_elements
    )

    df.at[
        idx,
        "pm_mean_repair_rule_07f3",
    ] = (
        "UNIQUE_PM_FRAGMENT_"
        "IN_MULTILINE_CELL"
    )


# ============================================================
# Audit dataframe
# ============================================================

audit_df = pd.DataFrame(
    audit_records
)

if (
    audit_df.empty
    and len(
        audit_df.columns
    ) == 0
):

    audit_df = pd.DataFrame(
        columns=[
            "row_index",
            "paper_id",
            "table_id",
            "source_row_index",
            "sample_raw",
            "semantic_class",
            "element",
            "candidate_raw",
            "old_value",
            "new_value",
            "recovery_rule",
            "raw_source_row",
        ]
    )


# ============================================================
# Save
# ============================================================

df.to_csv(
    output_path,
    index=False,
)

audit_df.to_csv(
    audit_path,
    index=False,
)


# ============================================================
# Summary
# ============================================================

repaired_row_count = (
    df[
        "pm_mean_repaired_07f3"
    ]
    .fillna(False)
    .astype(str)
    .str.lower()
    .eq("true")
    .sum()
)


print(
    "Rows repaired:",
    int(
        repaired_row_count
    ),
)

print(
    "Element cells repaired:",
    len(audit_df),
)


if len(audit_df):

    print()
    print(
        "Repaired rows by table:"
    )

    print(
        audit_df[
            [
                "table_id",
                "source_row_index",
            ]
        ]
        .drop_duplicates()
        ["table_id"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Repaired elements:"
    )

    print(
        audit_df[
            "element"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


print()
print(
    "Generated files:"
)

print(
    "-",
    output_path,
)

print(
    "-",
    audit_path,
)
