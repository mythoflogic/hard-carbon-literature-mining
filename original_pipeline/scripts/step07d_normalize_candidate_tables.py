#!/usr/bin/env python3

"""
STEP 07D
========

Normalize Candidate Elemental Tables

Purpose
-------
Take the likely and possible elemental-analysis tables identified
in Step 07C and convert them into a normalized structural form.

This step:
- identifies likely header rows;
- detects possible sample and temperature columns;
- detects C/H/N/O columns;
- preserves every original row;
- does not modify any scientific values;
- does not use an LLM.
"""

from pathlib import Path

import json
import math
import re
import unicodedata

import pandas as pd


# ============================================================
# Project paths
# ============================================================

project_dir = (
    Path.home()
    / "Scratch"
    / "hardcarbon_project"
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
    / "classified_elemental_tables.jsonl"
)

normalized_jsonl_path = (
    processed_tables_dir
    / "normalized_candidate_tables.jsonl"
)

normalized_csv_path = (
    processed_tables_dir
    / "normalized_candidate_tables.csv"
)

normalized_xlsx_path = (
    outputs_dir
    / "normalized_candidate_tables.xlsx"
)

row_output_path = (
    processed_tables_dir
    / "normalized_candidate_table_rows.csv"
)

# ============================================================
# Configuration
# ============================================================

TARGET_CLASSIFICATIONS = {
    "LIKELY_ELEMENTAL_TABLE",
    "POSSIBLE_ELEMENTAL_TABLE",
}

ELEMENT_ALIASES = {
    "C": [
        "c",
        "carbon",
        "total carbon",
        "organic carbon",
        "c total",
    ],
    "H": [
        "h",
        "hydrogen",
        "total hydrogen",
        "h total",
    ],
    "N": [
        "n",
        "nitrogen",
        "total nitrogen",
        "n total",
    ],
    "O": [
        "o",
        "oxygen",
        "total oxygen",
        "o total",
    ],
}

TEMPERATURE_ALIASES = [
    "temperature",
    "temp",
    "pyrolysis temperature",
    "carbonization temperature",
    "carbonisation temperature",
    "heat treatment temperature",
    "htt",
    "t",
]

SAMPLE_ALIASES = [
    "sample",
    "samples",
    "sample id",
    "sample name",
    "sample type",
    "feedstock",
    "feedstocks",
    "biomass",
    "biomass type",
    "biomass sample",
    "material",
    "biochar",
    "biochar source",
    "biochar sample",
    "biochar samples",
    "char",
    "code",
    "treatment",
]


# ============================================================
# General helpers
# ============================================================

def load_jsonl(path):
    records = []

    with path.open(
        encoding="utf-8"
    ) as source:

        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {error}"
                ) from error

    return records


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = text.lower()

    text = re.sub(
        r"[*_`]",
        "",
        text,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\([^)]*\)",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9/%+.-]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def is_numeric_cell(value):
    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    text = text.replace(
        ",",
        "",
    )

    match = re.fullmatch(
        r"""
        [<>≤≥~≈]?
        \s*
        [-+]?
        (?:
            \d+(?:\.\d+)?
            |
            \.\d+
        )
        (?:\s*±\s*\d+(?:\.\d+)?)?
        """,
        text,
        flags=re.VERBOSE,
    )

    return match is not None


def numeric_fraction(row):
    if not row:
        return 0.0

    nonempty = [
        cell
        for cell in row
        if str(cell).strip()
    ]

    if not nonempty:
        return 0.0

    numeric_count = sum(
        is_numeric_cell(cell)
        for cell in nonempty
    )

    return (
        numeric_count
        / len(nonempty)
    )

# ============================================================
# Header detection
# ============================================================

def header_token_score(cell):
    normalized = normalize_text(
        cell
    )

    score = 0

    for element, aliases in (
        ELEMENT_ALIASES.items()
    ):
        if normalized in aliases:
            score += 3

    if any(
        normalized == alias
        for alias in TEMPERATURE_ALIASES
    ):
        score += 2

    if any(
        normalized == alias
        for alias in SAMPLE_ALIASES
    ):
        score += 2

    if "%" in normalized:
        score += 1

    if "wt" in normalized:
        score += 1

    return score


def score_header_row(row):
    if not row:
        return -999

    text_score = sum(
        header_token_score(cell)
        for cell in row
    )

    number_fraction = numeric_fraction(
        row
    )

    # Header rows should usually contain more text than numbers.
    penalty = (
        number_fraction * 3
    )

    return (
        text_score
        - penalty
    )


def find_best_header_row(
    parsed_rows,
):
    if not parsed_rows:
        return None

    best_index = None
    best_score = -math.inf

    # We only inspect the first few rows because table headers
    # almost always appear near the top.
    for index, row in enumerate(
        parsed_rows[:6]
    ):
        score = score_header_row(
            row
        )

        if score > best_score:
            best_score = score
            best_index = index

    return best_index

# ============================================================
# Column identification
# ============================================================

def identify_element_columns(
    header_row,
):
    column_map = {}

    for index, cell in enumerate(
        header_row
    ):
        normalized = normalize_text(
            cell
        )

        # Reject ratio columns.
        if any(
            ratio in normalized
            for ratio in [
                "h/c",
                "o/c",
                "c/n",
            ]
        ):
            continue

        for element, aliases in (
            ELEMENT_ALIASES.items()
        ):
            if normalized in aliases:
                if element not in column_map:
                    column_map[
                        element
                    ] = index

    return column_map


def identify_temperature_column(
    header_row,
):
    for index, cell in enumerate(
        header_row
    ):
        normalized = normalize_text(
            cell
        )

        compact = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        ).strip()

        # Explicit aliases
        if compact in TEMPERATURE_ALIASES:
            return index

        # Full word
        if "temperature" in compact:
            return index

        # Common abbreviation
        if re.search(
            r"\btemp\b",
            compact,
        ):
            return index

        # HTT = heat treatment temperature
        if re.search(
            r"\bhtt\b",
            compact,
        ):
            return index

        # Temperature expressed as degrees Celsius.
        # Require temperature-like wording so ordinary columns
        # containing units are not selected accidentally.
        if (
            "c" in compact.split()
            and (
                "temp" in compact
                or "temperature" in compact
            )
        ):
            return index

    return None


def identify_sample_column(
    header_row,
):
    for index, cell in enumerate(
        header_row
    ):
        normalized = normalize_text(
            cell
        )

        # Exact recognized sample heading.
        if normalized in SAMPLE_ALIASES:
            return index

        # Compound heading, e.g.
        #
        # feedstock-method
        # sample-treatment
        # biomass / condition
        #
        # Require the alias to be a distinct component rather
        # than an arbitrary substring.
        parts = [
            normalize_text(part)
            for part in re.split(
                r"\s*(?:/|-)\s*",
                str(cell),
            )
        ]

        parts = [
            part
            for part in parts
            if part
        ]

        if any(
            part in SAMPLE_ALIASES
            for part in parts
        ):
            return index

    # Do not guess the sample column.
    return None
def infer_structural_sample_column(
    header_row,
    data_rows,
    element_columns,
    temperature_column,
):
    """
    Conservatively infer an unnamed first-column row label.

    This handles tables whose first header cell is blank but whose
    first data column contains sample/material identifiers.

    The rule is intentionally conservative:
    - only consider column 0;
    - its header must be blank;
    - it must not already be an elemental or temperature column;
    - most populated cells must be nonnumeric;
    - most populated cells must contain alphabetic characters.
    """

    if not header_row or not data_rows:
        return None

    # Structural sample inference is only justified when the table
    # already has substantial independent elemental-column evidence.
    if len(element_columns) < 2:
        return None

    if normalize_text(
        header_row[0]
    ):
        return None

    excluded = set(
        element_columns.values()
    )

    if temperature_column is not None:
        excluded.add(
            temperature_column
        )

    if 0 in excluded:
        return None

    values = []

    for row in data_rows:
        if not row:
            continue

        value = str(
            row[0]
        ).strip()

        if value:
            values.append(
                value
            )

    # Require repeated evidence rather than one isolated cell.
    if len(values) < 3:
        return None

    nonnumeric_fraction = (
        sum(
            not is_numeric_cell(value)
            for value in values
        )
        / len(values)
    )

    alphabetic_fraction = (
        sum(
            bool(
                re.search(
                    r"[A-Za-z]",
                    value,
                )
            )
            for value in values
        )
        / len(values)
    )

    if (
        nonnumeric_fraction >= 0.70
        and alphabetic_fraction >= 0.70
    ):
        return 0

    return None
# ============================================================
# Table normalization
# ============================================================

def normalize_table(
    record,
):
    parsed_rows = record.get(
        "parsed_rows",
        [],
    )

    if not parsed_rows:
        return None

    header_index = (
        find_best_header_row(
            parsed_rows
        )
    )

    if header_index is None:
        return None

    header_row = parsed_rows[
        header_index
    ]

    element_columns = (
        identify_element_columns(
            header_row
        )
    )

    temperature_column = (
        identify_temperature_column(
            header_row
        )
    )

    sample_column = (
        identify_sample_column(
            header_row
        )
    )

    data_rows = parsed_rows[
        header_index + 1:
    ]
    # Generic fallback for tables with an unnamed row-label column.
    if sample_column is None:
        sample_column = (
            infer_structural_sample_column(
                header_row=header_row,
                data_rows=data_rows,
                element_columns=element_columns,
                temperature_column=temperature_column,
            )
        )


    normalized = {
        "paper_id": record.get(
            "paper_id"
        ),
        "source_filename": record.get(
            "source_filename"
        ),
        "table_id": record.get(
            "table_id"
        ),
        "classification": record.get(
            "classification"
        ),
        "elemental_score": record.get(
            "elemental_score"
        ),
        "caption": record.get(
            "caption",
            "",
        ),
        "preceding_context": record.get(
            "preceding_context",
            "",
        ),
        "following_context": record.get(
            "following_context",
            "",
        ),
        "header_row_index": (
            header_index
        ),
        "header_row": (
            header_row
        ),
        "data_rows": (
            data_rows
        ),
        "sample_column": (
            sample_column
        ),
        "temperature_column": (
            temperature_column
        ),
        "element_columns": (
            element_columns
        ),
        "C_column": (
            element_columns.get(
                "C"
            )
        ),
        "H_column": (
            element_columns.get(
                "H"
            )
        ),
        "N_column": (
            element_columns.get(
                "N"
            )
        ),
        "O_column": (
            element_columns.get(
                "O"
            )
        ),
        "data_row_count": (
            len(data_rows)
        ),
        "column_count": record.get(
            "column_count"
        ),
        "raw_table_text": record.get(
            "raw_table_text",
            "",
        ),
    }

    return normalized

# ============================================================
# Load candidate tables
# ============================================================

if not input_path.exists():
    raise FileNotFoundError(
        f"Step 07C output not found: "
        f"{input_path}"
    )

all_records = load_jsonl(
    input_path
)

candidate_records = [
    record
    for record in all_records
    if record.get(
        "classification"
    )
    in TARGET_CLASSIFICATIONS
]

print(
    "Candidate tables loaded:",
    len(candidate_records),
)


# ============================================================
# Normalize candidate tables
# ============================================================

normalized_records = []

for record in candidate_records:
    normalized = normalize_table(
        record
    )

    if normalized is None:
        continue

    normalized_records.append(
        normalized
    )

print(
    "Tables normalized:",
    len(normalized_records),
)

# ============================================================
# Save normalized JSONL
# ============================================================

with normalized_jsonl_path.open(
    "w",
    encoding="utf-8",
) as output_file:

    for record in normalized_records:
        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Build table-level report
# ============================================================

table_rows = []

for record in normalized_records:
    table_rows.append(
        {
            "paper_id": (
                record[
                    "paper_id"
                ]
            ),
            "table_id": (
                record[
                    "table_id"
                ]
            ),
            "classification": (
                record[
                    "classification"
                ]
            ),
            "elemental_score": (
                record[
                    "elemental_score"
                ]
            ),
            "caption": (
                record[
                    "caption"
                ]
            ),
            "header_row_index": (
                record[
                    "header_row_index"
                ]
            ),
            "header_row": (
                " | ".join(
                    str(value)
                    for value in record[
                        "header_row"
                    ]
                )
            ),
            "sample_column": (
                record[
                    "sample_column"
                ]
            ),
            "temperature_column": (
                record[
                    "temperature_column"
                ]
            ),
            "C_column": (
                record[
                    "C_column"
                ]
            ),
            "H_column": (
                record[
                    "H_column"
                ]
            ),
            "N_column": (
                record[
                    "N_column"
                ]
            ),
            "O_column": (
                record[
                    "O_column"
                ]
            ),
            "data_row_count": (
                record[
                    "data_row_count"
                ]
            ),
            "column_count": (
                record[
                    "column_count"
                ]
            ),
        }
    )

normalized_df = pd.DataFrame(
    table_rows
)

normalized_df.to_csv(
    normalized_csv_path,
    index=False,
)

normalized_df.to_excel(
    normalized_xlsx_path,
    index=False,
)


# ============================================================
# Flatten every source data row
# ============================================================

flat_rows = []

for record in normalized_records:

    for source_row_index, row in enumerate(
        record[
            "data_rows"
        ],
        start=1,
    ):

        def get_cell(column_index):
            if column_index is None:
                return None

            if column_index >= len(row):
                return None

            return row[
                column_index
            ]

        flat_rows.append(
            {
                "paper_id": (
                    record[
                        "paper_id"
                    ]
                ),
                "table_id": (
                    record[
                        "table_id"
                    ]
                ),
                "source_row_index": (
                    source_row_index
                ),
                "sample_candidate": (
                    get_cell(
                        record[
                            "sample_column"
                        ]
                    )
                ),
                "temperature_candidate": (
                    get_cell(
                        record[
                            "temperature_column"
                        ]
                    )
                ),
                "C_candidate": (
                    get_cell(
                        record[
                            "C_column"
                        ]
                    )
                ),
                "H_candidate": (
                    get_cell(
                        record[
                            "H_column"
                        ]
                    )
                ),
                "N_candidate": (
                    get_cell(
                        record[
                            "N_column"
                        ]
                    )
                ),
                "O_candidate": (
                    get_cell(
                        record[
                            "O_column"
                        ]
                    )
                ),
                "raw_source_row": (
                    " | ".join(
                        str(value)
                        for value in row
                    )
                ),
            }
        )

row_df = pd.DataFrame(
    flat_rows
)

row_df.to_csv(
    row_output_path,
    index=False,
)


# ============================================================
# Final audit
# ============================================================

tables_with_C = int(
    normalized_df[
        "C_column"
    ]
    .notna()
    .sum()
)

tables_with_H = int(
    normalized_df[
        "H_column"
    ]
    .notna()
    .sum()
)

tables_with_N = int(
    normalized_df[
        "N_column"
    ]
    .notna()
    .sum()
)

tables_with_O = int(
    normalized_df[
        "O_column"
    ]
    .notna()
    .sum()
)

tables_with_all_CHNO = int(
    normalized_df[
        [
            "C_column",
            "H_column",
            "N_column",
            "O_column",
        ]
    ]
    .notna()
    .all(axis=1)
    .sum()
)

tables_with_temperature = int(
    normalized_df[
        "temperature_column"
    ]
    .notna()
    .sum()
)

tables_with_sample = int(
    normalized_df[
        "sample_column"
    ]
    .notna()
    .sum()
)


print()
print("=" * 70)
print("STEP 07D — NORMALIZED CANDIDATE TABLES")
print("=" * 70)
print()

print(
    "Candidate tables:",
    len(candidate_records),
)

print(
    "Normalized tables:",
    len(normalized_records),
)

print(
    "Tables with sample column:",
    tables_with_sample,
)

print(
    "Tables with temperature column:",
    tables_with_temperature,
)

print(
    "Tables with C column:",
    tables_with_C,
)

print(
    "Tables with H column:",
    tables_with_H,
)

print(
    "Tables with N column:",
    tables_with_N,
)

print(
    "Tables with O column:",
    tables_with_O,
)

print(
    "Tables with all CHNO columns:",
    tables_with_all_CHNO,
)

print(
    "Flattened source rows:",
    len(row_df),
)

print()
print("Generated files:")
print("-", normalized_jsonl_path)
print("-", normalized_csv_path)
print("-", normalized_xlsx_path)
print("-", row_output_path)
