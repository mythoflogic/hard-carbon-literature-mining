#!/usr/bin/env python3

"""
STEP 07F
========

Validate Table Semantics

Purpose
-------
Take the classified table rows from Step 07E and apply a stricter
quality gate before any row is accepted for deterministic extraction.

This step checks:
- whether the row contains CHNO values;
- whether temperature is exact or a range;
- whether the row contains missing-value markers;
- whether the elemental mapping is complete or partial;
- whether the row is safe for direct extraction;
- whether semantic interpretation is still required.

This step does NOT:
- change source values;
- use an LLM;
- write final benchmark records.
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

outputs_dir = (
    project_dir
    / "outputs"
)

input_path = (
    processed_tables_dir
    / "classified_extractable_rows_role_enriched_07i.csv"
)

output_path = (
    processed_tables_dir
    / "validated_table_rows.csv"
)

summary_path = (
    processed_tables_dir
    / "validated_table_row_summary.csv"
)

excel_path = (
    outputs_dir
    / "validated_table_rows.xlsx"
)

# ============================================================
# Helpers
# ============================================================

MISSING_MARKERS = {
    "",
    "-",
    "–",
    "—",
    "na",
    "n/a",
    "nan",
    "nd",
    "n.d.",
    "not detected",
    "not reported",
}


def clean_text(value):
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def normalize_missing_marker(value):
    text = clean_text(
        value
    ).lower()

    return text in MISSING_MARKERS


def parse_numeric(value):
    text = clean_text(
        value
    )

    if not text:
        return None

    if normalize_missing_marker(
        text
    ):
        return None

    text = (
        text
        .replace(",", "")
        .replace("−", "-")
    )

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


def valid_element_value(value):

    number = parse_numeric(
        value
    )

    if number is None:
        return None

    if 0 <= number <= 100:
        return number

    return None


def extract_temperature_information(
    raw_temperature,
    current_temperature,
):
    """
    Detect whether the source row contains an exact temperature
    or a range such as 300-350 or 180–230.
    """

    text = clean_text(
        raw_temperature
    )

    range_match = re.search(
        r"""
        (?<!\d)
        (\d{2,4}(?:\.\d+)?)
        \s*
        [\-–—]
        \s*
        (\d{2,4}(?:\.\d+)?)
        (?!\d)
        """,
        text,
        flags=re.VERBOSE,
    )

    if range_match:
        low = float(
            range_match.group(1)
        )

        high = float(
            range_match.group(2)
        )

        if (
            100 <= low <= 3000
            and 100 <= high <= 3000
        ):
            return {
                "temperature_type": "range",
                "temperature_exact_C": None,
                "temperature_low_C": low,
                "temperature_high_C": high,
            }

    exact = parse_numeric(
        current_temperature
    )

    if (
        exact is not None
        and 100 <= exact <= 3000
    ):
        return {
            "temperature_type": "exact",
            "temperature_exact_C": exact,
            "temperature_low_C": None,
            "temperature_high_C": None,
        }

    return {
        "temperature_type": "missing",
        "temperature_exact_C": None,
        "temperature_low_C": None,
        "temperature_high_C": None,
    }

# ============================================================
# Row-quality classification
# ============================================================

def classify_row_quality(row):
    sample_raw = clean_text(
        row.get("sample_raw")
    )

    raw_source_row = clean_text(
        row.get("raw_source_row")
    )

    temperature_info = (
        extract_temperature_information(
            row.get(
                "temperature_candidate_raw"
            ),
            row.get(
                "temperature_C"
            ),
        )
    )

    element_values = {
        "C": valid_element_value(
            row.get("C_value")
        ),
        "H": valid_element_value(
            row.get("H_value")
        ),
        "N": valid_element_value(
            row.get("N_value")
        ),
        "O": valid_element_value(
            row.get("O_value")
        ),
    }

    valid_element_count = sum(
        value is not None
        for value in element_values.values()
    )

    complete_chno = (
        valid_element_count == 4
    )

    partial_chno = (
        0 < valid_element_count < 4
    )

    sample_present = bool(
        sample_raw
    )

    temperature_type = (
        temperature_info[
            "temperature_type"
        ]
    )

    # --------------------------------------------------------
    # Final semantic class
    # --------------------------------------------------------

    original_classification = clean_text(
        row.get(
            "classification"
        )
    )

    # An upstream structural/prose rejection must not be
    # resurrected merely because a footnote happens to contain
    # a parsable number such as "n = 3".
    if (
        original_classification
        == "NOT_CHNO_ROW"
    ):
        semantic_class = (
            "REJECT_NON_DATA_ROW"
        )

    elif valid_element_count == 0:
        semantic_class = (
            "REJECT_NON_DATA_ROW"
        )

    elif temperature_type == "range":
        semantic_class = (
            "TEMPERATURE_RANGE"
        )

    elif (
        complete_chno
        and sample_present
        and temperature_type == "exact"
    ):
        semantic_class = (
            "DIRECT_COMPLETE_CHNO"
        )

    elif (
        partial_chno
        and sample_present
        and temperature_type == "exact"
    ):
        semantic_class = (
            "DIRECT_PARTIAL_CHNO"
        )

    elif (
        not sample_present
        and temperature_type == "exact"
    ):
        semantic_class = (
            "NEEDS_SAMPLE_INTERPRETATION"
        )

    elif (
        sample_present
        and temperature_type == "missing"
    ):
        semantic_class = (
            "NEEDS_TEMPERATURE_INTERPRETATION"
        )

    else:
        semantic_class = (
            "SEMANTIC_RECOVERY_REQUIRED"
        )

    return {
        "sample_present": (
            sample_present
        ),
        "valid_element_count": (
            valid_element_count
        ),
        "complete_chno": (
            complete_chno
        ),
        "partial_chno": (
            partial_chno
        ),
        "temperature_type": (
            temperature_type
        ),
        "temperature_exact_C": (
            temperature_info[
                "temperature_exact_C"
            ]
        ),
        "temperature_low_C": (
            temperature_info[
                "temperature_low_C"
            ]
        ),
        "temperature_high_C": (
            temperature_info[
                "temperature_high_C"
            ]
        ),
        "semantic_class": (
            semantic_class
        ),
    }

# ============================================================
# Load Step 07E rows
# ============================================================

if not input_path.exists():
    raise FileNotFoundError(
        f"Step 07E output not found: "
        f"{input_path}"
    )

df = pd.read_csv(
    input_path
)

print(
    "Rows loaded:",
    len(df),
)


# ============================================================
# Apply semantic validation
# ============================================================

validated_records = []

for _, row in df.iterrows():

    quality = classify_row_quality(
        row
    )

    # Preserve the complete upstream row, including all
    # 07E / 07F / 07F2 / 07H / 07I provenance.
    record = row.to_dict()

    # Preserve the upstream values explicitly before semantic
    # validation adds/recomputes its own fields.
    record[
        "original_classification"
    ] = row.get(
        "classification"
    )

    record[
        "valid_element_count_07e"
    ] = row.get(
        "valid_element_count"
    )

    record[
        "temperature_original"
    ] = row.get(
        "temperature_C"
    )


    record.update(
        quality
    )

    validated_records.append(
        record
    )

# ============================================================
# Save outputs
# ============================================================

validated_df = pd.DataFrame(
    validated_records
)

validated_df.to_csv(
    output_path,
    index=False,
)

validated_df.to_excel(
    excel_path,
    index=False,
)

summary_df = (
    validated_df[
        "semantic_class"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "semantic_class"
    )
    .reset_index(
        name="row_count"
    )
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("STEP 07F — TABLE SEMANTIC VALIDATION")
print("=" * 70)
print()

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Generated files:")
print("-", output_path)
print("-", summary_path)
print("-", excel_path)
