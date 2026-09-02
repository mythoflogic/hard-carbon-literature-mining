from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

extraction_path = Path(
    "processed_extraction/"
    "final_merged_extractions_with_doi.jsonl"
)

recovery_path = Path(
    "processed_tables/"
    "semantic_recovery_final_with_manual.csv"
)

out_all = Path(
    "processed_tables/"
    "final_extractions_with_recovery_all.csv"
)

out_usable = Path(
    "processed_tables/"
    "final_extractions_with_recovery_usable.csv"
)

out_rejected = Path(
    "processed_tables/"
    "final_extractions_with_recovery_rejected.csv"
)

summary_path = Path(
    "processed_tables/"
    "final_extractions_with_recovery_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

extractions = pd.read_json(
    extraction_path,
    lines=True,
)

recovery = pd.read_csv(
    recovery_path
)


print("=" * 72)
print("STEP 08D — MERGE FINAL SEMANTIC RECOVERY")
print("=" * 72)

print("\nExtraction rows:", len(extractions))
print("Recovery rows:", len(recovery))

print("\nExtraction columns:")
print(extractions.columns.tolist())

print("\nRecovery columns:")
print(recovery.columns.tolist())


# ============================================================
# IDENTIFY SHARED RECORD KEY
# ============================================================

possible_keys = [
    "recovery_id",
    "record_id",
    "extraction_id",
]

merge_key = None

for key in possible_keys:

    if (
        key in extractions.columns
        and key in recovery.columns
    ):
        merge_key = key
        break


if merge_key is None:

    print(
        "\nNo direct recovery key exists in both datasets."
    )

    print(
        "Available extraction columns:",
        extractions.columns.tolist(),
    )

    print(
        "Available recovery columns:",
        recovery.columns.tolist(),
    )

    raise SystemExit(
        "\nSTOP: inspect identifiers before merging. "
        "Do not guess a join key."
    )


print(
    "\nUsing merge key:",
    merge_key,
)


# ============================================================
# VALIDATE IDENTIFIERS
# ============================================================

if recovery[merge_key].duplicated().any():

    duplicates = recovery[
        recovery[merge_key].duplicated(
            keep=False
        )
    ]

    print(
        "\nDuplicate recovery IDs found:"
    )

    print(
        duplicates[
            [merge_key, "final_status"]
        ].to_string(index=False)
    )

    raise SystemExit(
        "STOP: recovery IDs are not unique."
    )


# ============================================================
# PRESERVE ORIGINAL EXTRACTION VALUES
# ============================================================

sample_candidates = [
    "sample",
    "sample_name",
    "material",
    "feedstock",
]

temperature_candidates = [
    "temperature_C",
    "temperature",
    "processing_temperature_C",
    "pyrolysis_temperature_C",
    "carbonization_temperature_C",
]

original_sample_col = next(
    (
        c
        for c in sample_candidates
        if c in extractions.columns
    ),
    None,
)

original_temperature_col = next(
    (
        c
        for c in temperature_candidates
        if c in extractions.columns
    ),
    None,
)


if original_sample_col is not None:

    extractions[
        "original_extracted_sample"
    ] = extractions[
        original_sample_col
    ]


if original_temperature_col is not None:

    extractions[
        "original_extracted_temperature_C"
    ] = extractions[
        original_temperature_col
    ]


# ============================================================
# SELECT RECOVERY FIELDS
# ============================================================

recovery_fields = [
    merge_key,
    "final_status",
    "final_bucket",
    "resolved_sample",
    "resolved_temperature_C",
    "resolved_temperature_type",
    "provenance",
    "validation_reason",
    "automatic_final_status",
    "manual_decision",
]

recovery_fields = [
    c
    for c in recovery_fields
    if c in recovery.columns
]


recovery_small = recovery[
    recovery_fields
].copy()


# ============================================================
# MERGE
# ============================================================

merged = extractions.merge(
    recovery_small,
    on=merge_key,
    how="left",
    validate="one_to_one",
)


print(
    "\nMerged rows:",
    len(merged),
)


# ============================================================
# CHECK MERGE COVERAGE
# ============================================================

matched = merged[
    "final_status"
].notna().sum()

unmatched = merged[
    "final_status"
].isna().sum()


print(
    "Rows with recovery decision:",
    matched,
)

print(
    "Rows without recovery decision:",
    unmatched,
)


# ============================================================
# BUILD FINAL VALUES
# ============================================================

if original_sample_col is not None:

    merged[
        "final_sample"
    ] = merged[
        "resolved_sample"
    ].combine_first(
        merged[
            "original_extracted_sample"
        ]
    )

else:

    merged[
        "final_sample"
    ] = merged[
        "resolved_sample"
    ]


if original_temperature_col is not None:

    merged[
        "final_temperature_C"
    ] = merged[
        "resolved_temperature_C"
    ].combine_first(
        merged[
            "original_extracted_temperature_C"
        ]
    )

else:

    merged[
        "final_temperature_C"
    ] = merged[
        "resolved_temperature_C"
    ]


# ============================================================
# IMPORTANT PARTIAL-RECORD SAFEGUARD
# ============================================================

# A manually/automatically accepted partial record with
# unresolved temperature must stay null.
#
# Do NOT refill it from an unreliable original extraction.

partial_statuses = {
    "LLM_RESOLVED_PARTIAL",
    "MANUAL_ACCEPTED_PARTIAL",
}

partial_mask = merged[
    "final_status"
].isin(
    partial_statuses
)

merged.loc[
    partial_mask,
    "final_temperature_C",
] = None


# ============================================================
# OUTPUT BUCKETS
# ============================================================

usable = merged[
    merged[
        "final_bucket"
    ].eq(
        "ACCEPTED"
    )
].copy()


rejected = merged[
    merged[
        "final_bucket"
    ].eq(
        "REJECTED"
    )
].copy()


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    out_all,
    index=False,
)

usable.to_csv(
    out_usable,
    index=False,
)

rejected.to_csv(
    out_rejected,
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "metric": "extraction_rows",
            "count": len(extractions),
        },
        {
            "metric": "recovery_rows",
            "count": len(recovery),
        },
        {
            "metric": "matched_recovery_rows",
            "count": matched,
        },
        {
            "metric": "unmatched_extraction_rows",
            "count": unmatched,
        },
        {
            "metric": "usable_rows",
            "count": len(usable),
        },
        {
            "metric": "rejected_rows",
            "count": len(rejected),
        },
    ]
)

summary.to_csv(
    summary_path,
    index=False,
)


print("\nFINAL MERGE SUMMARY")
print("-" * 72)

print(
    summary.to_string(
        index=False
    )
)

print("\nSaved:")
print("-", out_all)
print("-", out_usable)
print("-", out_rejected)
print("-", summary_path)
