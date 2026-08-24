#!/usr/bin/env python3

"""
STEP 07E
========

Classify Extractable Table Rows

Purpose
-------
Classify normalized table rows according to whether they can be
converted into candidate elemental-analysis records directly by
Python or whether they require semantic interpretation.

This step does not modify source values and does not use an LLM.
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

rows_path = (
    processed_tables_dir
    / "repaired_candidate_table_rows_v2.csv"
)

tables_path = (
    processed_tables_dir
    / "repaired_candidate_tables.csv"
)

classified_rows_path = (
    processed_tables_dir
    / "classified_extractable_rows.csv"
)

classified_rows_xlsx = (
    outputs_dir
    / "classified_extractable_rows.xlsx"
)

summary_path = (
    processed_tables_dir
    / "extractable_row_summary.csv"
)

# ============================================================
# Helpers
# ============================================================

def clean_text(value):
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def parse_numeric(value):
    text = clean_text(
        value
    )

    if not text:
        return None

    text = (
        text
        .replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
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


def parse_temperature(value):
    number = parse_numeric(
        value
    )

    if number is None:
        return None

    if 100 <= number <= 3000:
        return number

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

def looks_like_prose(value):
    """
    Detect substantial prose in a cell.

    This is intentionally conservative:
    ordinary scientific numeric forms such as
    '500 ± 10', '<0.5', 'ca. 500', etc. should
    not be treated as prose.
    """

    text = clean_text(
        value
    )

    if not text:
        return False

    # Remove HTML line-break artifacts.
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    words = re.findall(
        r"[A-Za-z]{2,}",
        text,
    )

    return (
        len(words) >= 2
        and sum(
            len(word)
            for word in words
        ) >= 8
    )

# ============================================================
# Load normalized rows and table metadata
# ============================================================

if not rows_path.exists():
    raise FileNotFoundError(
        f"Row file not found: {rows_path}"
    )

if not tables_path.exists():
    raise FileNotFoundError(
        f"Table metadata not found: {tables_path}"
    )

rows_df = pd.read_csv(
    rows_path
)

tables_df = pd.read_csv(
    tables_path
)

table_metadata = tables_df[
    [
        "table_id",
        "sample_column",
        "temperature_column",
        "C_column",
        "H_column",
        "N_column",
        "O_column",
    ]
].copy()

rows_df = rows_df.merge(
    table_metadata,
    on="table_id",
    how="left",
)

print(
    "Source rows loaded:",
    len(rows_df),
)

# ============================================================
# Classify each source row
# ============================================================

classified_records = []

for _, row in rows_df.iterrows():

    sample_raw = clean_text(
        row.get(
            "sample_candidate"
        )
    )

    temperature = parse_temperature(
        row.get(
            "temperature_candidate"
        )
    )

    carbon = valid_element_value(
        row.get(
            "C_candidate"
        )
    )

    hydrogen = valid_element_value(
        row.get(
            "H_candidate"
        )
    )

    nitrogen = valid_element_value(
        row.get(
            "N_candidate"
        )
    )

    oxygen = valid_element_value(
        row.get(
            "O_candidate"
        )
    )

    element_values = {
        "C": carbon,
        "H": hydrogen,
        "N": nitrogen,
        "O": oxygen,
    }

    valid_element_count = sum(
        value is not None
        for value in element_values.values()
    )

    sample_known = bool(
        sample_raw
    )

    temperature_known = (
        temperature is not None
    )
    # --------------------------------------------------------
    # Reject flattened footnotes / prose rows.
    #
    # A genuine missing-temperature row may have textual
    # temperature context, so that alone is NOT enough.
    #
    # We require substantial prose contamination in at least
    # two fields that structurally should contain temperature
    # or elemental numeric values.
    # --------------------------------------------------------

    numeric_field_candidates = [
        row.get(
            "temperature_candidate"
        ),
        row.get(
            "C_candidate"
        ),
        row.get(
            "H_candidate"
        ),
        row.get(
            "N_candidate"
        ),
        row.get(
            "O_candidate"
        ),
    ]

    prose_numeric_field_count = sum(
        looks_like_prose(
            value
        )
        for value in numeric_field_candidates
    )

    prose_pseudorow = (
        looks_like_prose(
            sample_raw
        )
        and prose_numeric_field_count >= 2
    )
    if prose_pseudorow:
        classification = (
            "NOT_CHNO_ROW"
        )

    elif valid_element_count == 0:
        classification = (
            "NOT_CHNO_ROW"
        )
    elif (
        sample_known
        and temperature_known
    ):
        classification = (
            "DIRECTLY_EXTRACTABLE"
        )

    elif (
        not sample_known
        and temperature_known
    ):
        classification = (
            "NEEDS_SAMPLE_INTERPRETATION"
        )

    elif (
        sample_known
        and not temperature_known
    ):
        classification = (
            "NEEDS_TEMPERATURE_INTERPRETATION"
        )

    else:
        classification = (
            "NEEDS_SAMPLE_AND_TEMPERATURE"
        )

    classified_records.append(
        {
            "paper_id": row.get(
                "paper_id"
            ),
            "table_id": row.get(
                "table_id"
            ),
            "source_row_index": row.get(
                "source_row_index"
            ),
            # Preserve structural provenance from 07D4.
            "row_orientation": row.get(
                "row_orientation",
                "STANDARD",
            ),

            # Preserve exact source candidates before numeric parsing.
            "sample_candidate_raw": row.get(
                "sample_candidate"
            ),
            "temperature_candidate_raw": row.get(
                "temperature_candidate"
            ),
            "C_candidate_raw": row.get(
                "C_candidate"
            ),
            "H_candidate_raw": row.get(
                "H_candidate"
            ),
            "N_candidate_raw": row.get(
                "N_candidate"
            ),
            "O_candidate_raw": row.get(
                "O_candidate"
            ),

            # Parsed / normalized values used for classification.
            "sample_raw": sample_raw,
            "temperature_C": temperature,
            "C_value": carbon,
            "H_value": hydrogen,
            "N_value": nitrogen,
            "O_value": oxygen,

            "valid_element_count": (
                valid_element_count
            ),
            "classification": (
                classification
            ),

            # Preserve complete original flattened source row.
            "raw_source_row": row.get(
                "raw_source_row"
            ),
        }
    )

# ============================================================
# Save outputs
# ============================================================

classified_df = pd.DataFrame(
    classified_records
)

classified_df.to_csv(
    classified_rows_path,
    index=False,
)

classified_df.to_excel(
    classified_rows_xlsx,
    index=False,
)

summary_df = (
    classified_df[
        "classification"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "classification"
    )
    .reset_index(
        name="row_count"
    )
)

summary_df.to_csv(
    summary_path,
    index=False,
)

print()
print("=" * 70)
print("STEP 07E — EXTRACTABLE ROW CLASSIFICATION")
print("=" * 70)
print()

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Generated files:")
print("-", classified_rows_path)
print("-", classified_rows_xlsx)
print("-", summary_path)
