#!/usr/bin/env python3

"""
STEP 08C — FINALIZE SEMANTIC RECOVERY

Purpose
-------
Account for all Step 08B recovery candidates without forcing
ambiguous scientific rows to become false LLM successes.

This script does NOT overwrite Step 08B results.

It:
1. keeps valid LLM-resolved records;
2. flags known scientifically incorrect accepted records;
3. classifies unresolved structural problems;
4. records deterministic metadata recoveries where justified;
5. sends genuinely ambiguous rows to manual review;
6. rejects fragmented/non-independent rows.
"""

from pathlib import Path
import json
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

processed_dir = (
    project_dir
    / "processed_tables"
)

outputs_dir = (
    project_dir
    / "outputs"
)

packages_path = (
    processed_dir
    / "semantic_recovery_packages_enriched.jsonl"
)

results_path = (
    processed_dir
    / "semantic_recovery_llm"
    / "semantic_recovery_results.jsonl"
)

failures_path = (
    processed_dir
    / "semantic_recovery_llm"
    / "semantic_recovery_failures.jsonl"
)

accounting_path = (
    processed_dir
    / "semantic_recovery_final_accounting.csv"
)

review_path = (
    processed_dir
    / "semantic_recovery_manual_review.csv"
)

deterministic_path = (
    processed_dir
    / "semantic_recovery_deterministic_exceptions.csv"
)

excel_path = (
    outputs_dir
    / "semantic_recovery_final_accounting.xlsx"
)

grouped_repairs_path = (
    processed_dir
    / "grouped_row_repairs.csv"
)

# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):
    records = []

    if not path.exists():
        return records

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# Load data
# ============================================================

packages = load_jsonl(
    packages_path
)

results = load_jsonl(
    results_path
)

failures = load_jsonl(
    failures_path
)

grouped_repairs = {}

if grouped_repairs_path.exists():

    grouped_df = pd.read_csv(
        grouped_repairs_path
    )

    for _, row in grouped_df.iterrows():

        if (
            row.get("repair_status")
            == "DETERMINISTIC_GROUPED_ROW_RECOVERY"
        ):
            grouped_repairs[
                str(row["recovery_id"])
            ] = {
                "resolved_sample": (
                    row[
                        "resolved_sample"
                    ]
                ),
                "resolved_temperature_C": (
                    float(
                        row[
                            "resolved_temperature_C"
                        ]
                    )
                ),
                "resolved_temperature_type": (
                    row[
                        "resolved_temperature_type"
                    ]
                ),
            }

# ============================================================
# Eligible packages
# ============================================================

eligible_packages = {
    str(x["recovery_id"]): x
    for x in packages
    if x.get("semantic_class")
    != "TEMPERATURE_RANGE"
}

print(
    "Eligible recovery packages:",
    len(eligible_packages),
)


# ============================================================
# Latest successful result per recovery ID
# ============================================================

success_by_id = {}

for result in results:

    recovery_id = str(
        result["recovery_id"]
    )

    success_by_id[
        recovery_id
    ] = result


# ============================================================
# Latest failure per recovery ID
# ============================================================

failure_by_id = {}

for failure in failures:

    recovery_id = str(
        failure["recovery_id"]
    )

    failure_by_id[
        recovery_id
    ] = failure


# ============================================================
# Scientifically known false-positive success
# ============================================================

# REC_00030 passed JSON/schema validation but the LLM associated
# the 750 C row with the wrong sample.
#
# Do not count it as scientifically resolved.

false_positive_successes = {
    "REC_00030",
}


# ============================================================
# Explicit final classifications discovered during validation
# ============================================================

structural_mapping_error = {
    "REC_00217",
}

grouped_row_repair_required = {
    "REC_00369",
    "REC_00373",
    "REC_00377",
    "REC_00381",
}

manual_temperature_review = {
    "REC_00418",
    "REC_00422",
    "REC_00423",
}

fragmented_rows = {
    "REC_00486",
}


# ============================================================
# Deterministic metadata recoveries
# ============================================================

# These sample codes are explicitly present in the target row.
#
# The numeric suffix is retained as treatment temperature only
# for these validated sample-series exceptions.
#
# These do NOT imply that missing elemental columns have been
# scientifically recovered.

deterministic_exceptions = {
    "REC_00484": {
        "resolved_sample": "GH600",
        "resolved_temperature_C": 600.0,
        "resolved_temperature_type": "pyrolysis",
    },

    "REC_00488": {
        "resolved_sample": "CM400",
        "resolved_temperature_C": 400.0,
        "resolved_temperature_type": "pyrolysis",
    },

    "REC_00491": {
        "resolved_sample": "PM600",
        "resolved_temperature_C": 600.0,
        "resolved_temperature_type": "pyrolysis",
    },

    "REC_00492": {
        "resolved_sample": "PC400",
        "resolved_temperature_C": 400.0,
        "resolved_temperature_type": "pyrolysis",
    },
}


# ============================================================
# Build final accounting
# ============================================================

rows = []

for recovery_id, package in (
    eligible_packages.items()
):

    record = {
        "recovery_id": recovery_id,
        "paper_id": package.get(
            "paper_id"
        ),
        "table_id": package.get(
            "table_id"
        ),
        "semantic_class": package.get(
            "semantic_class"
        ),
        "raw_source_row": package.get(
            "raw_source_row"
        ),
        "sample_raw": package.get(
            "sample_raw"
        ),
        "temperature_original": (
            package.get(
                "temperature_original"
            )
        ),
        "C_value": package.get(
            "C_value"
        ),
        "H_value": package.get(
            "H_value"
        ),
        "N_value": package.get(
            "N_value"
        ),
        "O_value": package.get(
            "O_value"
        ),
        "final_status": None,
        "resolved_sample": None,
        "resolved_temperature_C": None,
        "resolved_temperature_type": None,
        "reason": None,
    }

    # --------------------------------------------------------
    # False-positive LLM success
    # --------------------------------------------------------

    if (
        recovery_id
        in false_positive_successes
    ):
        record[
            "final_status"
        ] = (
            "MANUAL_REVIEW_FALSE_POSITIVE"
        )

        record[
            "reason"
        ] = (
            "LLM result passed schema validation "
            "but target-row sample association "
            "was scientifically incorrect."
        )

    # --------------------------------------------------------
    # Structural mapping problem
    # --------------------------------------------------------

    elif (
        recovery_id
        in structural_mapping_error
    ):
        record[
            "final_status"
        ] = (
            "STRUCTURAL_MAPPING_ERROR"
        )

        record[
            "reason"
        ] = (
            "Elemental columns are misaligned. "
            "Return row to table/header repair."
        )

    elif (
        recovery_id
        in grouped_repairs
    ):
        resolution = grouped_repairs[
            recovery_id
        ]

        record[
            "final_status"
        ] = (
            "DETERMINISTIC_GROUPED_ROW_RECOVERY"
        )

        record[
            "resolved_sample"
        ] = resolution[
            "resolved_sample"
        ]

        record[
            "resolved_temperature_C"
        ] = resolution[
            "resolved_temperature_C"
        ]

        record[
            "resolved_temperature_type"
        ] = resolution[
            "resolved_temperature_type"
        ]

        record[
            "reason"
        ] = (
            "Parent sample identity recovered "
            "deterministically from grouped table "
            "structure."
        )

    elif (
        recovery_id
        in grouped_row_repair_required
    ):
        record[
            "final_status"
        ] = (
            "GROUPED_ROW_REPAIR_REQUIRED"
        )

        record[
            "reason"
        ] = (
            "Temperature row is visible but "
            "parent sample identity was lost "
            "during merged/grouped-row parsing."
        )
    # --------------------------------------------------------
    # Missing parent sample/grouped rows
    # --------------------------------------------------------

    elif (
        recovery_id
        in grouped_row_repair_required
    ):
        record[
            "final_status"
        ] = (
            "GROUPED_ROW_REPAIR_REQUIRED"
        )

        record[
            "reason"
        ] = (
            "Temperature row is visible but "
            "parent sample identity was lost "
            "during merged/grouped-row parsing."
        )

    # --------------------------------------------------------
    # Genuine ambiguity
    # --------------------------------------------------------

    elif (
        recovery_id
        in manual_temperature_review
    ):
        record[
            "final_status"
        ] = (
            "MANUAL_REVIEW_TEMPERATURE"
        )

        record[
            "resolved_sample"
        ] = package.get(
            "sample_raw"
        )

        record[
            "reason"
        ] = (
            "Sample identity is known but "
            "available evidence does not map "
            "one unique processing temperature "
            "to this sample."
        )

    # --------------------------------------------------------
    # Fragmented row
    # --------------------------------------------------------

    elif (
        recovery_id
        in fragmented_rows
    ):
        record[
            "final_status"
        ] = (
            "REJECT_FRAGMENTED_ROW"
        )

        record[
            "reason"
        ] = (
            "Row is a continuation/error-value "
            "fragment without independent sample "
            "identity."
        )

    # --------------------------------------------------------
    # Deterministic exception
    # --------------------------------------------------------

    elif (
        recovery_id
        in deterministic_exceptions
    ):
        resolution = (
            deterministic_exceptions[
                recovery_id
            ]
        )

        record[
            "final_status"
        ] = (
            "DETERMINISTIC_METADATA_RECOVERY"
        )

        record[
            "resolved_sample"
        ] = resolution[
            "resolved_sample"
        ]

        record[
            "resolved_temperature_C"
        ] = resolution[
            "resolved_temperature_C"
        ]

        record[
            "resolved_temperature_type"
        ] = resolution[
            "resolved_temperature_type"
        ]

        record[
            "reason"
        ] = (
            "Target-row sample code explicitly "
            "contains the validated treatment "
            "temperature suffix."
        )

    # --------------------------------------------------------
    # Successful Step 08B result
    # --------------------------------------------------------

    elif (
        recovery_id
        in success_by_id
    ):
        result = success_by_id[
            recovery_id
        ]

        record[
            "final_status"
        ] = (
            "LLM_RESOLVED"
        )

        record[
            "resolved_sample"
        ] = result.get(
            "resolved_sample"
        )

        record[
            "resolved_temperature_C"
        ] = result.get(
            "resolved_temperature_C"
        )

        record[
            "resolved_temperature_type"
        ] = result.get(
            "resolved_temperature_type"
        )

        record[
            "reason"
        ] = (
            "Passed Step 08B validation."
        )

    # --------------------------------------------------------
    # Still unresolved
    # --------------------------------------------------------

    else:
        failure = failure_by_id.get(
            recovery_id,
            {},
        )

        record[
            "final_status"
        ] = (
            "UNRESOLVED"
        )

        record[
            "reason"
        ] = failure.get(
            "error",
            "No final resolution available.",
        )

    rows.append(record)


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    rows
)

summary = (
    df[
        "final_status"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "final_status"
    )
    .reset_index(
        name="row_count"
    )
)


# ============================================================
# Manual-review queue
# ============================================================

manual_review_df = df[
    df[
        "final_status"
    ].isin(
        {
            "MANUAL_REVIEW_FALSE_POSITIVE",
            "STRUCTURAL_MAPPING_ERROR",
            "GROUPED_ROW_REPAIR_REQUIRED",
            "MANUAL_REVIEW_TEMPERATURE",
            "REJECT_FRAGMENTED_ROW",
            "UNRESOLVED",
        }
    )
].copy()


# ============================================================
# Deterministic exceptions
# ============================================================

deterministic_df = df[
    df[
        "final_status"
    ].eq(
        "DETERMINISTIC_METADATA_RECOVERY"
    )
].copy()


# ============================================================
# Save
# ============================================================

df.to_csv(
    accounting_path,
    index=False,
)

manual_review_df.to_csv(
    review_path,
    index=False,
)

deterministic_df.to_csv(
    deterministic_path,
    index=False,
)

with pd.ExcelWriter(
    excel_path
) as writer:

    df.to_excel(
        writer,
        sheet_name="all_accounting",
        index=False,
    )

    summary.to_excel(
        writer,
        sheet_name="summary",
        index=False,
    )

    manual_review_df.to_excel(
        writer,
        sheet_name="manual_review",
        index=False,
    )

    deterministic_df.to_excel(
        writer,
        sheet_name="deterministic",
        index=False,
    )


# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print(
    "STEP 08C — FINAL SEMANTIC RECOVERY ACCOUNTING"
)
print("=" * 70)
print()

print(
    summary.to_string(
        index=False
    )
)

print()

print(
    "Total accounted:",
    len(df),
)

print()
print("Generated files:")
print("-", accounting_path)
print("-", review_path)
print("-", deterministic_path)
print("-", excel_path)
