#!/usr/bin/env python3

"""
STEP 07H
========

Recover Temperature from Sample Identifiers

Purpose
-------
Recover exact processing temperatures that are explicitly encoded
inside sample identifiers.

Examples:
    CF-200      -> 200 C
    PW−400      -> 400 C
    PM-500 °C   -> 500 C
    BC350       -> 350 C
    SG-600C     -> 600 C
    HW450       -> 450 C
    SP 350-10-0 -> 350 C

This step:
- only fills temperature_C when it is currently missing;
- never changes sample identity;
- never changes C/H/N/O values;
- never overwrites an existing temperature;
- does not use an LLM.

The output is consumed by semantic validation.
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
    / "classified_extractable_rows_role_enriched.csv"
)

output_path = (
    processed_tables_dir
    / "classified_extractable_rows_role_enriched_07h.csv"
)

audit_path = (
    processed_tables_dir
    / "temperature_from_sample_recovery_audit.csv"
)


# ============================================================
# Helpers
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_existing_temperature(value):

    text = clean_text(value)

    if not text:
        return None

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)",
        text,
    )

    if not match:
        return None

    try:
        return float(
            match.group(0)
        )

    except ValueError:
        return None


def extract_single_temperature_integer(sample):

    text = clean_text(sample)

    if not text:
        return None

    matches = re.findall(
        r"""
        (?<![\d.])
        (\d{3,4})
        (?![\d.])
        """,
        text,
        flags=re.VERBOSE,
    )

    plausible = [
        float(value)
        for value in matches
        if 100 <= float(value) <= 3000
    ]

    if len(plausible) != 1:
        return None

    return plausible[0]


def sample_temperature_stem(sample):

    """
    Return a normalized non-temperature sample-code stem.

    Examples:
        BC-400   -> bc
        BC500    -> bc
        PHC-600  -> phc
        SP 350-10-0 -> sp-10-0

    Bare numbers and prose-like labels are rejected.
    """

    text = clean_text(sample)

    if not text:
        return None

    temperature = (
        extract_single_temperature_integer(
            text
        )
    )

    if temperature is None:
        return None

    # Require some alphabetic sample-code information.
    if not re.search(
        r"[A-Za-z]",
        text,
    ):
        return None

    # Reject long prose-like labels. 07H is intended for
    # compact identifiers, not arbitrary descriptions.
    words = re.findall(
        r"[A-Za-z]+",
        text,
    )

    if (
        len(words) > 4
        or len(text) > 40
    ):
        return None

    stem = re.sub(
        r"(?<![\d.])\d{3,4}(?![\d.])",
        "",
        text,
        count=1,
    )

    stem = stem.lower()

    stem = re.sub(
        r"[^a-z0-9]+",
        "-",
        stem,
    )

    stem = stem.strip("-")

    if not stem:
        return None

    return stem


def recover_temperature_from_sample(
    sample,
    structurally_supported=False,
):

    """
    Return:

        (temperature, recovery_rule)

    or:

        (None, None)
    """

    text = clean_text(sample)

    if not text:
        return None, None


    # ========================================================
    # Rule 1:
    # Explicit Celsius notation.
    #
    # Examples:
    #   PM-500 °C
    #   CR-300 °C
    #   SG-600C
    # ========================================================

    explicit_matches = re.findall(
        r"""
        (?<!\d)
        (\d{3,4}(?:\.\d+)?)
        \s*
        °?
        \s*
        C
        \b
        """,
        text,
        flags=(
            re.IGNORECASE
            | re.VERBOSE
        ),
    )

    explicit_temperatures = [
        float(value)
        for value in explicit_matches
        if (
            100
            <= float(value)
            <= 3000
        )
    ]

    if len(
        explicit_temperatures
    ) == 1:

        return (
            explicit_temperatures[0],
            "SAMPLE_EXPLICIT_C",
        )


    # ========================================================
    # Rule 2:
    # Exactly one plausible 3-4 digit integer embedded in
    # the sample identifier.
    #
    # Examples:
    #   CF-200
    #   PW−400
    #   BC350
    #   SD700
    #   HW450
    #   SP 350-10-0
    #
    # Small decimal identifiers such as LSD0.05 are ignored.
    # ========================================================

    if structurally_supported:

        candidate = (
            extract_single_temperature_integer(
                text
            )
        )

        if candidate is not None:

            return (
                candidate,
                "SAMPLE_STRUCTURALLY_SUPPORTED_INTEGER",
            )


    return None, None


# ============================================================
# Load
# ============================================================

if not input_path.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{input_path}"
    )


df = pd.read_csv(
    input_path
)

print()
print("=" * 78)
print(
    "STEP 07H — TEMPERATURE RECOVERY FROM SAMPLE IDENTIFIERS"
)
print("=" * 78)
print()

print(
    "Rows loaded:",
    len(df),
)


required_columns = [
    "sample_raw",
    "temperature_C",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise KeyError(
        "Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# Recover temperature
# ============================================================

audit_records = []

# Preserve 07H recovery provenance directly in the main
# row-level output so downstream stages do not need to
# reconstruct it from the separate audit file.
df[
    "temperature_recovered_07h"
] = False

df[
    "temperature_recovery_rule_07h"
] = None

df[
    "temperature_structural_support_07h"
] = False


# ============================================================
# Build support for temperature-coded sample series
# ============================================================

eligible_classes = {
    "NEEDS_TEMPERATURE_INTERPRETATION",
    "NEEDS_SAMPLE_AND_TEMPERATURE",
}

series_members = {}

for idx, row in df.iterrows():

    if row.get(
        "classification"
    ) not in eligible_classes:
        continue

    if parse_existing_temperature(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    stem = sample_temperature_stem(
        row.get(
            "sample_raw"
        )
    )

    candidate = (
        extract_single_temperature_integer(
            row.get(
                "sample_raw"
            )
        )
    )

    if (
        stem is None
        or candidate is None
    ):
        continue

    key = (
        row.get("paper_id"),
        row.get("table_id"),
        stem,
    )

    series_members.setdefault(
        key,
        []
    ).append(
        (
            idx,
            candidate,
        )
    )


structurally_supported_indices = set()

for members in series_members.values():

    distinct_temperatures = {
        temperature
        for _, temperature
        in members
    }

    # Require at least two rows and at least two distinct
    # plausible temperatures with the same sample-code stem.
    if (
        len(members) >= 2
        and len(
            distinct_temperatures
        ) >= 2
    ):
        structurally_supported_indices.update(
            idx
            for idx, _
            in members
        )


for idx, row in df.iterrows():

    if row.get(
        "classification"
    ) not in eligible_classes:
        continue

    current_temperature = (
        parse_existing_temperature(
            row.get(
                "temperature_C"
            )
        )
    )

    # --------------------------------------------------------
    # Never overwrite an existing plausible temperature.
    # --------------------------------------------------------

    if (
        current_temperature is not None
        and 100 <= current_temperature <= 3000
    ):
        continue


    recovered_temperature, recovery_rule = (
        recover_temperature_from_sample(
            row.get(
                "sample_raw"
            ),
            structurally_supported=(
                idx
                in structurally_supported_indices
            ),
        )
    )


    if recovered_temperature is None:
        continue


    old_temperature = row.get(
        "temperature_C"
    )


    # --------------------------------------------------------
    # Only modify temperature_C.
    # --------------------------------------------------------

    df.at[
        idx,
        "temperature_C"
    ] = recovered_temperature

    df.at[
        idx,
        "temperature_recovered_07h"
    ] = True

    df.at[
        idx,
        "temperature_recovery_rule_07h"
    ] = recovery_rule

    df.at[
        idx,
        "temperature_structural_support_07h"
    ] = (
        idx
        in structurally_supported_indices
    )


    audit_records.append(
        {
            "row_index":
                idx,

            "paper_id":
                row.get(
                    "paper_id"
                ),

            "table_id":
                row.get(
                    "table_id"
                ),

            "source_row_index":
                row.get(
                    "source_row_index"
                ),

            "sample_raw":
                row.get(
                    "sample_raw"
                ),

            "old_temperature_C":
                old_temperature,

            "new_temperature_C":
                recovered_temperature,

            "recovery_rule":
                recovery_rule,

            "structural_series_support":
                (
                    idx
                    in structurally_supported_indices
                ),

            "classification_before_07h":
                row.get(
                    "classification"
                ),

            "C_value":
                row.get(
                    "C_value"
                ),

            "H_value":
                row.get(
                    "H_value"
                ),

            "N_value":
                row.get(
                    "N_value"
                ),

            "O_value":
                row.get(
                    "O_value"
                ),

            "raw_source_row":
                row.get(
                    "raw_source_row"
                ),
        }
    )


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
            "old_temperature_C",
            "new_temperature_C",
            "recovery_rule",
            "structural_series_support",
            "classification_before_07h",
            "C_value",
            "H_value",
            "N_value",
            "O_value",
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

print(
    "Temperatures recovered:",
    len(audit_df),
)

if len(audit_df):

    print()
    print(
        "Recovery rules:"
    )

    print(
        audit_df[
            "recovery_rule"
        ]
        .value_counts()
        .to_string()
    )


print()
print(
    "Rows still missing temperature_C:",
    df[
        "temperature_C"
    ]
    .isna()
    .sum(),
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
