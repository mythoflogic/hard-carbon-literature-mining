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

PT = ROOT / "processed_tables"

DET_PATH = (
    PT
    / "deterministic_candidates_structural.csv"
)

SEM_PATH = (
    PT
    / "semantic_recovery_final_with_manual.csv"
)

QUEUE_PATH = (
    PT
    / "semantic_recovery_queue_structural.csv"
)

OUT_PATH = (
    PT
    / "final_table_dataset.csv"
)

SUMMARY_PATH = (
    PT
    / "final_table_dataset_summary.csv"
)


# ============================================================
# Helpers
# ============================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def bool_series(series):
    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


# ============================================================
# Load frozen inputs
# ============================================================

det = pd.read_csv(
    DET_PATH,
    low_memory=False,
)

sem = pd.read_csv(
    SEM_PATH,
    low_memory=False,
)

queue = pd.read_csv(
    QUEUE_PATH,
    low_memory=False,
)


# ============================================================
# Basic input validation
# ============================================================

for name, df in (
    ("deterministic", det),
    ("semantic", sem),
    ("queue", queue),
):

    if "source_key" not in df.columns:
        raise RuntimeError(
            f"{name}: source_key missing."
        )

    if df["source_key"].isna().any():
        raise RuntimeError(
            f"{name}: missing source_key."
        )

    if (
        df["source_key"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"{name}: duplicate source_key."
        )


if not (
    sem["final_bucket"]
    .astype(str)
    .eq("ACCEPTED")
    .all()
):
    raise RuntimeError(
        "Semantic input contains "
        "non-accepted rows."
    )


ranges = queue[
    queue["semantic_class"]
    .astype(str)
    .eq("TEMPERATURE_RANGE")
].copy()


# ============================================================
# Source-row metadata for semantic rows
# ============================================================

queue_meta = queue[
    [
        "source_key",
        "source_row_index",
    ]
].copy()

sem = sem.merge(
    queue_meta,
    on="source_key",
    how="left",
    validate="one_to_one",
)

if sem["source_row_index"].isna().any():

    missing = sem[
        sem["source_row_index"]
        .isna()
    ]

    raise RuntimeError(
        "Semantic rows missing current "
        "source_row_index:\n"
        + missing[
            [
                "recovery_id",
                "source_key",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Deterministic branch
# ============================================================

det_temp = numeric(
    det["temperature_exact_C"]
).combine_first(
    numeric(
        det["temperature_C"]
    )
)


temp_na = (
    det["semantic_class"]
    .astype(str)
    .str.endswith("_TEMP_NA")
)

if (
    "temperature_not_applicable_07g2"
    in det.columns
):

    temp_na = (
        temp_na
        |
        bool_series(
            det[
                "temperature_not_applicable_07g2"
            ]
        )
    )


det_status = pd.Series(
    "not_resolved",
    index=det.index,
    dtype="object",
)

det_status.loc[
    det_temp.notna()
] = "exact"

det_status.loc[
    temp_na
] = "not_applicable"


# TEMP_NA must never retain numeric temperature.
if (
    temp_na
    & det_temp.notna()
).any():

    bad = det[
        temp_na
        & det_temp.notna()
    ]

    raise RuntimeError(
        "TEMP_NA deterministic rows "
        "contain numeric temperature:\n"
        + bad[
            [
                "source_key",
                "sample_raw",
                "temperature_C",
                "temperature_exact_C",
            ]
        ].to_string(
            index=False
        )
    )


det_exact = det_temp.where(
    det_status.eq("exact")
)


det_final = pd.DataFrame(
    {
        "final_record_id":
            det["candidate_id_07g2"],

        "branch":
            "TABLE_DETERMINISTIC",

        "source_key":
            det["source_key"],

        "paper_id":
            det["paper_id"],

        "table_id":
            det["table_id"],

        "source_row_index":
            det["source_row_index"],

        "sample":
            det["sample_raw"],

        "temperature_status":
            det_status,

        "temperature_C":
            det_exact,

        "temperature_exact_C":
            det_exact,

        "temperature_low_C":
            pd.NA,

        "temperature_high_C":
            pd.NA,

        "temperature_type":
            det["temperature_type"],

        "C_value":
            numeric(
                det["C_value"]
            ),

        "H_value":
            numeric(
                det["H_value"]
            ),

        "N_value":
            numeric(
                det["N_value"]
            ),

        "O_value":
            numeric(
                det["O_value"]
            ),

        "semantic_class":
            det["semantic_class"],

        "final_status":
            "DETERMINISTIC_ACCEPTED",

        "upstream_provenance":
            det["provenance_status"],

        "raw_source_row":
            det["raw_source_row"],
    }
)


# ============================================================
# Accepted semantic branch
# ============================================================

sem_temp = numeric(
    sem["resolved_temperature_C"]
)

sem_status = pd.Series(
    "not_resolved",
    index=sem.index,
    dtype="object",
)

sem_status.loc[
    sem_temp.notna()
] = "exact"


resolved_sample = (
    sem["resolved_sample"]
    .combine_first(
        sem["sample_raw"]
    )
)


sem_final = pd.DataFrame(
    {
        "final_record_id":
            sem["recovery_id"],

        "branch":
            "TABLE_SEMANTIC",

        "source_key":
            sem["source_key"],

        "paper_id":
            sem["paper_id"],

        "table_id":
            sem["table_id"],

        "source_row_index":
            sem["source_row_index"],

        "sample":
            resolved_sample,

        "temperature_status":
            sem_status,

        "temperature_C":
            sem_temp,

        "temperature_exact_C":
            sem_temp,

        "temperature_low_C":
            pd.NA,

        "temperature_high_C":
            pd.NA,

        "temperature_type":
            sem[
                "resolved_temperature_type"
            ],

        "C_value":
            numeric(
                sem["C_value"]
            ),

        "H_value":
            numeric(
                sem["H_value"]
            ),

        "N_value":
            numeric(
                sem["N_value"]
            ),

        "O_value":
            numeric(
                sem["O_value"]
            ),

        "semantic_class":
            sem["semantic_class"],

        "final_status":
            sem["final_status"],

        "upstream_provenance":
            sem["provenance"],

        "raw_source_row":
            sem["raw_source_row"],
    }
)


# ============================================================
# Explicit range branch
# ============================================================

range_low = numeric(
    ranges["temperature_low_C"]
)

range_high = numeric(
    ranges["temperature_high_C"]
)


if (
    range_low.isna()
    | range_high.isna()
).any():

    raise RuntimeError(
        "Temperature range with "
        "missing bound."
    )


if (
    range_low
    > range_high
).any():

    raise RuntimeError(
        "Temperature range has "
        "low > high."
    )


range_final = pd.DataFrame(
    {
        "final_record_id":
            ranges["recovery_id"],

        "branch":
            "TABLE_RANGE",

        "source_key":
            ranges["source_key"],

        "paper_id":
            ranges["paper_id"],

        "table_id":
            ranges["table_id"],

        "source_row_index":
            ranges["source_row_index"],

        "sample":
            ranges["sample_raw"],

        # IMPORTANT:
        # A range is not an exact temperature.
        "temperature_status":
            "range",

        "temperature_C":
            pd.NA,

        "temperature_exact_C":
            pd.NA,

        "temperature_low_C":
            range_low,

        "temperature_high_C":
            range_high,

        "temperature_type":
            "range",

        "C_value":
            numeric(
                ranges["C_value"]
            ),

        "H_value":
            numeric(
                ranges["H_value"]
            ),

        "N_value":
            numeric(
                ranges["N_value"]
            ),

        "O_value":
            numeric(
                ranges["O_value"]
            ),

        "semantic_class":
            ranges["semantic_class"],

        "final_status":
            "VALID_TEMPERATURE_RANGE",

        "upstream_provenance":
            "SOURCE_TABLE_REPORTED_RANGE",

        "raw_source_row":
            ranges["raw_source_row"],
    }
)


# ============================================================
# Combine branches
# ============================================================

final = pd.concat(
    [
        det_final,
        sem_final,
        range_final,
    ],
    ignore_index=True,
)


# ============================================================
# Final identity invariants
# ============================================================

if final["source_key"].isna().any():
    raise RuntimeError(
        "Final dataset has missing source_key."
    )


if (
    final["source_key"]
    .astype(str)
    .duplicated()
    .any()
):

    bad = final[
        final["source_key"]
        .astype(str)
        .duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Duplicate final source_key:\n"
        + bad[
            [
                "branch",
                "source_key",
                "final_record_id",
                "sample",
            ]
        ].to_string(
            index=False
        )
    )


if final["final_record_id"].isna().any():
    raise RuntimeError(
        "Missing final_record_id."
    )


if (
    final["final_record_id"]
    .astype(str)
    .duplicated()
    .any()
):
    raise RuntimeError(
        "Duplicate final_record_id."
    )


expected_total = (
    len(det)
    + len(sem)
    + len(ranges)
)

if len(final) != expected_total:

    raise RuntimeError(
        "Final row-count mismatch: "
        f"{len(final)} != "
        f"{expected_total}"
    )


# ============================================================
# Temperature invariants
# ============================================================

exact_mask = (
    final["temperature_status"]
    .eq("exact")
)

range_mask = (
    final["temperature_status"]
    .eq("range")
)

na_mask = (
    final["temperature_status"]
    .eq("not_applicable")
)

unresolved_mask = (
    final["temperature_status"]
    .eq("not_resolved")
)


if (
    exact_mask
    & final["temperature_C"].isna()
).any():

    raise RuntimeError(
        "Exact temperature row "
        "missing temperature_C."
    )


if (
    range_mask
    & final["temperature_C"].notna()
).any():

    raise RuntimeError(
        "Range row incorrectly has "
        "exact temperature_C."
    )


if (
    range_mask
    & (
        final[
            "temperature_low_C"
        ].isna()
        |
        final[
            "temperature_high_C"
        ].isna()
    )
).any():

    raise RuntimeError(
        "Range row missing bounds."
    )


if (
    na_mask
    & final["temperature_C"].notna()
).any():

    raise RuntimeError(
        "Not-applicable temperature row "
        "contains numeric temperature."
    )


if (
    unresolved_mask
    & final["temperature_C"].notna()
).any():

    raise RuntimeError(
        "Not-resolved temperature row "
        "contains numeric temperature."
    )


# ============================================================
# Element-count metadata
# ============================================================

element_cols = [
    "C_value",
    "H_value",
    "N_value",
    "O_value",
]

final[
    "reported_element_count"
] = (
    final[element_cols]
    .notna()
    .sum(axis=1)
)

final[
    "complete_CHNO"
] = (
    final[
        "reported_element_count"
    ]
    .eq(4)
)


# ============================================================
# Temperature schema normalization
# ============================================================

# Historical upstream "temperature_type" mixes two concepts:
#
#   exact / missing / range
#
# with process labels such as:
#
#   pyrolysis / hydrothermal
#
# Keep the historical value for provenance, but expose a
# separate clean process field.

final = final.rename(
    columns={
        "temperature_type":
            "temperature_type_legacy"
    }
)


def normalize_temperature_process(row):

    legacy = str(
        row.get(
            "temperature_type_legacy",
            "",
        )
    ).strip().lower()

    raw = str(
        row.get(
            "raw_source_row",
            "",
        )
    ).strip().lower()

    if legacy == "pyrolysis":
        return "pyrolysis"

    if legacy in {
        "hydrothermal",
        "htc",
        "hydrothermal carbonization",
    }:
        return "hydrothermal"

    if legacy in {
        "carbonization",
        "carbonisation",
    }:
        return "carbonization"

    # Range rows preserve an explicitly reported process
    # when it is visible in the source row.
    if (
        row.get("temperature_status")
        == "range"
    ):

        if re.search(
            r"\bpyrolysis\b",
            raw,
        ):
            return "pyrolysis"

        if re.search(
            r"\bhtc\b"
            r"|hydrothermal",
            raw,
        ):
            return "hydrothermal"

        if re.search(
            r"\bcarbonization\b"
            r"|\bcarbonisation\b",
            raw,
        ):
            return "carbonization"

    return pd.NA


final[
    "temperature_process"
] = final.apply(
    normalize_temperature_process,
    axis=1,
)


# ============================================================
# CHNO source-quality flags
# ============================================================

# Do NOT normalize or repair source-reported values here.
#
# A total above 100 can occur because of source reporting,
# analytical basis differences, rounding, transcription,
# or upstream OCR. These rows remain source-faithful and
# are merely flagged for downstream review.

final[
    "CHNO_sum_reported"
] = (
    final[
        [
            "C_value",
            "H_value",
            "N_value",
            "O_value",
        ]
    ]
    .sum(
        axis=1,
        min_count=4,
    )
)


final[
    "chno_sum_quality_flag"
] = pd.NA


mask_gt_105 = (
    final["CHNO_sum_reported"]
    > 105
)

mask_gt_120 = (
    final["CHNO_sum_reported"]
    > 120
)


final.loc[
    mask_gt_105,
    "chno_sum_quality_flag",
] = (
    "REVIEW_CHNO_SUM_GT_105"
)


final.loc[
    mask_gt_120,
    "chno_sum_quality_flag",
] = (
    "REVIEW_EXTREME_CHNO_SUM_GT_120"
)


# ============================================================
# Stable assembly identifier
# ============================================================

final.insert(
    0,
    "assembly_id",
    [
        f"TABLE_{i:05d}"
        for i in range(
            1,
            len(final) + 1,
        )
    ],
)


# ============================================================
# Summary
# ============================================================

summary_rows = [
    {
        "metric":
            "deterministic_rows",
        "count":
            len(det_final),
    },
    {
        "metric":
            "semantic_rows",
        "count":
            len(sem_final),
    },
    {
        "metric":
            "range_rows",
        "count":
            len(range_final),
    },
    {
        "metric":
            "final_rows",
        "count":
            len(final),
    },
    {
        "metric":
            "unique_source_keys",
        "count":
            final[
                "source_key"
            ].nunique(),
    },
    {
        "metric":
            "exact_temperature_rows",
        "count":
            exact_mask.sum(),
    },
    {
        "metric":
            "temperature_not_applicable_rows",
        "count":
            na_mask.sum(),
    },
    {
        "metric":
            "temperature_range_rows",
        "count":
            range_mask.sum(),
    },
    {
        "metric":
            "temperature_not_resolved_rows",
        "count":
            unresolved_mask.sum(),
    },
    {
        "metric":
            "complete_CHNO_rows",
        "count":
            final[
                "complete_CHNO"
            ].sum(),
    },
]

summary_rows.extend(
    [
        {
            "metric":
                "chno_sum_gt_105_rows",
            "count":
                mask_gt_105.sum(),
        },
        {
            "metric":
                "chno_sum_gt_120_rows",
            "count":
                mask_gt_120.sum(),
        },
        {
            "metric":
                "temperature_process_known_rows",
            "count":
                final[
                    "temperature_process"
                ].notna().sum(),
        },
    ]
)

summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Save
# ============================================================

final.to_csv(
    OUT_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Console report
# ============================================================

print("=" * 90)
print("STEP 08D4 — FINAL TABLE DATASET")
print("=" * 90)

print()

print(
    summary.to_string(
        index=False
    )
)

print()
print("BRANCH COUNTS")

print(
    final["branch"]
    .value_counts()
    .to_string()
)

print()
print("TEMPERATURE STATUS")

print(
    final[
        "temperature_status"
    ]
    .value_counts()
    .to_string()
)

print()
print("SEMANTIC / FINAL STATUS")

print(
    final[
        "final_status"
    ]
    .value_counts()
    .to_string()
)

print()
print("Saved:")
print("-", OUT_PATH)
print("-", SUMMARY_PATH)
