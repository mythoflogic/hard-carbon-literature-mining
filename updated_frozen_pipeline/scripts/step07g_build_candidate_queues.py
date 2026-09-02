#!/usr/bin/env python3

"""
STEP 07G
========

Build Deterministic and Semantic-Recovery Queues

Purpose
-------
Separate validated table rows into:

1. deterministic candidates whose numeric structure is already clear;
2. rows requiring semantic/context interpretation;
3. rejected non-data rows.

No scientific values are modified.
No LLM is used in this step.
"""

from pathlib import Path
import hashlib
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

outputs_dir = (
    project_dir
    / "outputs"
)

input_path = (
    processed_tables_dir
    / "validated_table_rows_pm_repaired.csv"
)

deterministic_path = (
    processed_tables_dir
    / "deterministic_candidates.csv"
)

recovery_path = (
    processed_tables_dir
    / "semantic_recovery_queue.csv"
)

rejected_path = (
    processed_tables_dir
    / "rejected_table_rows.csv"
)

summary_path = (
    processed_tables_dir
    / "candidate_queue_summary.csv"
)

excel_path = (
    outputs_dir
    / "candidate_queues.xlsx"
)


# ============================================================
# Load validated rows
# ============================================================

if not input_path.exists():
    raise FileNotFoundError(
        f"Step 07F output not found: "
        f"{input_path}"
    )

df = pd.read_csv(
    input_path
)

print(
    "Validated rows loaded:",
    len(df),
)
# ============================================================
# Stable source provenance key
# ============================================================

def make_source_key(row):

    parts = [
        str(row.get("paper_id", "")),
        str(row.get("table_id", "")),
        str(row.get("source_row_index", "")),
        str(row.get("raw_source_row", "")),
    ]

    payload = "\x1f".join(
        parts
    )

    digest = hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return f"SRC_{digest}"


df["source_key"] = df.apply(
    make_source_key,
    axis=1,
)

duplicate_source_keys = (
    df[
        "source_key"
    ]
    .duplicated(
        keep=False
    )
)

if duplicate_source_keys.any():

    duplicate_rows = df.loc[
        duplicate_source_keys,
        [
            "source_key",
            "paper_id",
            "table_id",
            "source_row_index",
            "raw_source_row",
        ],
    ]

    raise RuntimeError(
        "Duplicate stable source_key values detected:\n"
        + duplicate_rows.to_string(
            index=False
        )
    )


# ============================================================
# Queue definitions
# ============================================================

DETERMINISTIC_CLASSES = {
    "DIRECT_COMPLETE_CHNO",
    "DIRECT_PARTIAL_CHNO",
}

RECOVERY_CLASSES = {
    "SEMANTIC_RECOVERY_REQUIRED",
    "NEEDS_TEMPERATURE_INTERPRETATION",
    "NEEDS_SAMPLE_INTERPRETATION",
    "TEMPERATURE_RANGE",
}

REJECT_CLASSES = {
    "REJECT_NON_DATA_ROW",
}


# ============================================================
# Split rows
# ============================================================

deterministic_df = df[
    df["semantic_class"].isin(
        DETERMINISTIC_CLASSES
    )
].copy()

recovery_df = df[
    df["semantic_class"].isin(
        RECOVERY_CLASSES
    )
].copy()

rejected_df = df[
    df["semantic_class"].isin(
        REJECT_CLASSES
    )
].copy()


# ============================================================
# Safety gate: every row must be routed exactly once
# ============================================================

known_classes = (
    DETERMINISTIC_CLASSES
    | RECOVERY_CLASSES
    | REJECT_CLASSES
)

unrouted_df = df[
    ~df[
        "semantic_class"
    ].isin(
        known_classes
    )
].copy()

if len(unrouted_df):

    print()
    print(
        "UNROUTED SEMANTIC CLASSES:"
    )
    print(
        unrouted_df[
            "semantic_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    raise RuntimeError(
        f"{len(unrouted_df)} validated rows "
        "were not assigned to any queue."
    )


routed_count = (
    len(deterministic_df)
    + len(recovery_df)
    + len(rejected_df)
)

if routed_count != len(df):
    raise RuntimeError(
        "Queue row-count mismatch: "
        f"{routed_count} routed vs "
        f"{len(df)} input rows."
    )


# ============================================================
# Add candidate identifiers
# ============================================================

deterministic_df = (
    deterministic_df
    .reset_index(drop=True)
)

recovery_df = (
    recovery_df
    .reset_index(drop=True)
)

deterministic_df[
    "candidate_id"
] = [
    f"DET_{number:05d}"
    for number in range(
        1,
        len(deterministic_df) + 1,
    )
]

recovery_df[
    "recovery_id"
] = [
    f"REC_{number:05d}"
    for number in range(
        1,
        len(recovery_df) + 1,
    )
]


# ============================================================
# Candidate completeness
# ============================================================

element_columns = [
    "C_value",
    "H_value",
    "N_value",
    "O_value",
]

deterministic_df[
    "reported_element_count"
] = (
    deterministic_df[
        element_columns
    ]
    .notna()
    .sum(axis=1)
)

deterministic_df[
    "is_complete_CHNO"
] = (
    deterministic_df[
        "reported_element_count"
    ]
    .eq(4)
)


# ============================================================
# Preserve provenance status
# ============================================================

deterministic_df[
    "provenance_status"
] = "NOT_YET_VERIFIED"

recovery_df[
    "provenance_status"
] = "NOT_YET_VERIFIED"


# ============================================================
# Save queues
# ============================================================

deterministic_df.to_csv(
    deterministic_path,
    index=False,
)

recovery_df.to_csv(
    recovery_path,
    index=False,
)

rejected_df.to_csv(
    rejected_path,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "queue": "deterministic",
            "row_count": len(
                deterministic_df
            ),
        },
        {
            "queue": "semantic_recovery",
            "row_count": len(
                recovery_df
            ),
        },
        {
            "queue": "rejected",
            "row_count": len(
                rejected_df
            ),
        },
    ]
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Excel report
# ============================================================

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

    deterministic_df.to_excel(
        writer,
        sheet_name="Deterministic",
        index=False,
    )

    recovery_df.to_excel(
        writer,
        sheet_name="Semantic_recovery",
        index=False,
    )

    rejected_df.to_excel(
        writer,
        sheet_name="Rejected",
        index=False,
    )


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 70)
print("STEP 07G — CANDIDATE QUEUES")
print("=" * 70)
print()

print(
    summary_df.to_string(
        index=False
    )
)

print()

print(
    "Complete deterministic CHNO:",
    int(
        deterministic_df[
            "is_complete_CHNO"
        ]
        .sum()
    ),
)

print(
    "Partial deterministic CHNO:",
    int(
        (
            ~deterministic_df[
                "is_complete_CHNO"
            ]
        )
        .sum()
    ),
)

print()
print("Generated files:")
print("-", deterministic_path)
print("-", recovery_path)
print("-", rejected_path)
print("-", summary_path)
print("-", excel_path)

