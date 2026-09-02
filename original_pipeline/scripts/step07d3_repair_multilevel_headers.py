#!/usr/bin/env python3

"""
STEP 07D3
=========

Repair Multi-Level Scientific Table Headers

Purpose
-------
Repair candidate tables where column alignment is broken because
scientific tables contain multiple header rows, merged cells, or
blank cells carried across header levels.

This step:
- preserves original source values;
- reconstructs aligned headers;
- remaps sample / temperature / C / H / N / O columns;
- regenerates flattened candidate rows;
- does NOT use an LLM.
"""

from pathlib import Path

import json
import math
import re
import unicodedata

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
    / "classified_elemental_tables.jsonl"
)

normalized_input_path = (
    processed_tables_dir
    / "structural_edge_repaired_candidate_tables.jsonl"
)

repaired_jsonl_path = (
    processed_tables_dir
    / "repaired_candidate_tables.jsonl"
)

repaired_table_csv = (
    processed_tables_dir
    / "repaired_candidate_tables.csv"
)

repaired_rows_csv = (
    processed_tables_dir
    / "repaired_candidate_table_rows.csv"
)

repaired_xlsx = (
    outputs_dir
    / "repaired_candidate_tables.xlsx"
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
    ],
    "H": [
        "h",
        "hydrogen",
        "total hydrogen",
    ],
    "N": [
        "n",
        "nitrogen",
        "total nitrogen",
    ],
    "O": [
        "o",
        "oxygen",
        "total oxygen",
    ],
}

SAMPLE_ALIASES = [
    "sample",
    "samples",
    "sample id",
    "sample name",
    "feedstock",
    "feedstocks",
    "biomass",
    "biochar",
    "biochar source",
    "biochar samples",
    "material",
]

TEMPERATURE_ALIASES = [
    "temperature",
    "temp",
    "pyrolysis temperature",
    "carbonization temperature",
    "carbonisation temperature",
    "heat treatment temperature",
    "htt",
]

# ============================================================
# Helpers
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
        r"<sup>.*?</sup>",
        "",
        text,
     )

    text = re.sub(
        r"<sub>.*?</sub>",
        "",
        text,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )
    text = re.sub(
        r"[*_`]",
        "",
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


def is_separator_row(row):
    if not row:
        return False

    meaningful = [
        str(cell).strip()
        for cell in row
        if str(cell).strip()
    ]

    if not meaningful:
        return False

    return all(
        re.fullmatch(
            r":?-{3,}:?",
            cell.replace(" ", ""),
        )
        is not None
        for cell in meaningful
    )


def is_numeric_cell(value):
    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    text = text.replace(",", "")

    return (
        re.fullmatch(
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
        is not None
    )


def row_numeric_fraction(row):
    nonempty = [
        cell
        for cell in row
        if str(cell).strip()
    ]

    if not nonempty:
        return 0.0

    return (
        sum(
            is_numeric_cell(cell)
            for cell in nonempty
        )
        / len(nonempty)
    )

# ============================================================
# Header-row detection
# ============================================================

def header_signal_score(row):
    score = 0

    for cell in row:
        normalized = normalize_text(
            cell
        )

        if not normalized:
            continue

        for aliases in (
            ELEMENT_ALIASES.values()
        ):
            if normalized in aliases:
                score += 3

        if any(
            normalized == alias
            for alias in SAMPLE_ALIASES
        ):
            score += 2

        if (
            normalized in TEMPERATURE_ALIASES
            or "temperature" in normalized
            or re.search(
                r"\btemp\b",
                normalized,
            )
        ):
            score += 2

        if "%" in normalized:
            score += 1

        if "analysis" in normalized:
            score += 1

    score -= (
        row_numeric_fraction(row)
        * 3
    )

    return score


def detect_header_rows(parsed_rows):
    """
    Detect consecutive header rows near the top of a table.

    Scientific tables commonly contain:
    - one group-header row;
    - one actual column-name row.
    """

    useful_rows = [
        row
        for row in parsed_rows
        if not is_separator_row(row)
    ]

    if not useful_rows:
        return [], []

    candidate_indices = []

    for index, row in enumerate(
        useful_rows[:5]
    ):
        score = header_signal_score(
            row
        )

        if (
            score > 0
            or row_numeric_fraction(row) < 0.25
        ):
            candidate_indices.append(
                index
            )

        else:
            break

    if not candidate_indices:
        candidate_indices = [0]

    header_rows = [
        useful_rows[index]
        for index in candidate_indices
    ]

    data_rows = useful_rows[
        max(candidate_indices) + 1:
    ]

    return (
        header_rows,
        data_rows,
    )

# ============================================================
# Multi-level header alignment
# ============================================================

def pad_row(
    row,
    width,
):
    padded = list(row)

    if len(padded) < width:
        padded.extend(
            [""] * (
                width - len(padded)
            )
        )

    return padded[:width]


def infer_table_width(
    header_rows,
    data_rows,
):
    lengths = [
        len(row)
        for row in (
            header_rows
            + data_rows[:20]
        )
        if row
    ]

    if not lengths:
        return 0

    # Data rows are usually the most trustworthy width.
    data_lengths = [
        len(row)
        for row in data_rows[:20]
        if row
    ]

    if data_lengths:
        return max(
            set(data_lengths),
            key=data_lengths.count,
        )

    return max(lengths)


def repair_header_row_alignment(
    header_rows,
    data_rows,
):
    """
    Construct one aligned header across multiple header rows.

    Critical rule:
    preserve blank cells because they represent merged-header
    positions and maintain column alignment.
    """

    width = infer_table_width(
        header_rows,
        data_rows,
    )

    if width == 0:
        return [], []

    aligned_headers = [
        pad_row(
            row,
            width,
        )
        for row in header_rows
    ]

    aligned_data_rows = [
        pad_row(
            row,
            width,
        )
        for row in data_rows
    ]

    # Important special case:
    # lower header row can be one column shorter because its
    # leading blank "Sample" position disappeared during parsing.
    for index, row in enumerate(
        aligned_headers
    ):
        original_nonempty = [
            cell
            for cell in row
            if str(cell).strip()
        ]

        if (
            len(row) == width
            and original_nonempty
        ):
            continue

    combined_header = []

    for column_index in range(width):
        pieces = []

        for header_row in aligned_headers:
            value = str(
                header_row[
                    column_index
                ]
            ).strip()

            if value:
                pieces.append(
                    value
                )

        # Keep unique header levels in order.
        unique_pieces = []

        for piece in pieces:
            if piece not in unique_pieces:
                unique_pieces.append(
                    piece
                )

        combined_header.append(
            " / ".join(
                unique_pieces
            )
        )

    return (
        combined_header,
        aligned_data_rows,
    )

# ============================================================
# Leading-column shift repair
# ============================================================

def repair_leading_header_shift(
    header_rows,
    data_rows,
):
    """
    Repair cases where the first header row contains Sample,
    while the second header row begins at column 1 and has lost
    its leading blank cell.

    Example
    -------
    Row 1:
    Sample | Proximate analysis ... | Ultimate analysis ...

    Row 2:
    Volatile matter | Fixed carbon | Ash | N | C | H | S | O ...

    Data row:
    CF-200 | 78.26 | 15.04 | 6.70 | 1.35 | 50.42 | ...

    Row 2 must be shifted right by one column.
    """

    if len(header_rows) < 2:
        return header_rows

    data_widths = [
        len(row)
        for row in data_rows[:20]
        if row
    ]

    if not data_widths:
        return header_rows

    width = max(
        set(data_widths),
        key=data_widths.count,
    )

    repaired = []

    first_header = list(
        header_rows[0]
    )

    first_normalized = [
        normalize_text(cell)
        for cell in first_header
    ]

    first_has_sample = any(
        value in SAMPLE_ALIASES
        for value in first_normalized
    )

    for index, row in enumerate(
        header_rows
    ):
        current = list(row)

        if (
            index > 0
            and first_has_sample
            and len(current) == width - 1
        ):
            current = (
                [""]
                + current
            )

        repaired.append(
            current
        )

    return repaired

# ============================================================
# Column detection
# ============================================================

def identify_element_columns(
    combined_header,
):
    mapping = {}

    # --------------------------------------------------------
    # Reject equation/statistical tables before attempting
    # elemental-column mapping.
    # --------------------------------------------------------

    full_header = normalize_text(
        " ".join(
            str(cell)
            for cell in combined_header
        )
    )

    table_level_reject_terms = [
        "independent variable",
        "spearman",
        "correlation coefficient",
        "rank correlation",
        "regression coefficient",
        "contribution to overall",
        "equation",
    ]

    if any(
        term in full_header
        for term in table_level_reject_terms
    ):
        return mapping

    forbidden_element_context = [
        "temperature",
        "temp",
        "peak",
        "dtg",
        "mass loss",
        "residue",
        "stage",
        "interval",
        "combustion",
        "decomposition",
        "equation",
        "coefficient",
        "regression",
        "correlation",
        "r2",
        "r squared",
        "hhv",
        "spearman",
    ]

    ratio_patterns = [
        "h/c",
        "o/c",
        "c/n",
        "n/c",
        "c/h",
        "c/o",
    ]

    for index, cell in enumerate(
        combined_header
    ):

        raw_parts = str(
            cell
        ).split(" / ")

        for raw_part in raw_parts:

            raw_lower = str(
                raw_part
            ).lower()

            # Detect Celsius BEFORE normalize_text()
            # removes symbols such as ° or broken �.
            if (
                "°c" in raw_lower
                or "ºc" in raw_lower
                or "℃" in raw_lower
                or "�c" in raw_lower
            ):
                continue

            part = normalize_text(
                raw_part
            )

            if not part:
                continue

            # Reject thermal/statistical context at the
            # individual header level.
            if any(
                phrase in part
                for phrase in forbidden_element_context
            ):
                continue

            # Written Celsius forms such as "degree C".
            if re.search(
                r"\b(?:deg|degree|degrees)\s+c\b",
                part,
            ):
                continue

            # Atomic ratios are not elemental wt% columns.
            if any(
                ratio in part
                for ratio in ratio_patterns
            ):
                continue

            for element, aliases in (
                ELEMENT_ALIASES.items()
            ):
                if part in aliases:
                    if element not in mapping:
                        mapping[
                            element
                        ] = index

    return mapping


def identify_sample_column(
    combined_header,
):
    for index, cell in enumerate(
        combined_header
    ):
        normalized = normalize_text(
            cell
        )

        if normalized in SAMPLE_ALIASES:
            return index

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

    return None

def identify_temperature_column(
    combined_header,
):
    """
    Identify a processing-temperature column from reconstructed
    multilevel headers.

    Handles both clean and parser-damaged temperature headings.
    """

    for index, cell in enumerate(
        combined_header
    ):
        normalized = normalize_text(
            cell
        )

        compact = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        ).strip()

        # ----------------------------------------------------
        # Exact aliases
        # ----------------------------------------------------

        if compact in TEMPERATURE_ALIASES:
            return index

        # ----------------------------------------------------
        # Clean temperature wording
        # ----------------------------------------------------

        if "temperature" in compact:
            return index

        if re.search(
            r"\btemp\b",
            compact,
        ):
            return index

        if re.search(
            r"\bhtt\b",
            compact,
        ):
            return index

        # ----------------------------------------------------
        # OCR loss of initial "t":
        #
        # emperature
        # ----------------------------------------------------

        if re.search(
            r"\bemperature\b",
            compact,
        ):
            return index

        # ----------------------------------------------------
        # Joined split fragments:
        #
        # temper ature -> temperature
        # ----------------------------------------------------

        joined = re.sub(
            r"\s+",
            "",
            compact,
        )

        if "temperature" in joined:
            return index

        # ----------------------------------------------------
        # Processing context plus damaged fragment.
        # ----------------------------------------------------

        processing_context = (
            "pyrolysis" in compact
            or "carbonization" in compact
            or "carbonisation" in compact
            or "heat treatment" in compact
            or "treatment" in compact
        )

        damaged_temperature_fragment = (
            re.search(
                r"\btemper\b",
                compact,
            )
            is not None
            or re.search(
                r"\bature\b",
                compact,
            )
            is not None
        )

        if (
            processing_context
            and damaged_temperature_fragment
        ):
            return index

    return None

# ============================================================
# Process candidate tables
# ============================================================

records = load_jsonl(
    input_path
)

normalized_records = load_jsonl(
    normalized_input_path
)


# ------------------------------------------------------------
# Join normalized 07D metadata back onto the richer classified
# records. The classified records retain parsed_rows, while the
# normalized records contain the column mappings discovered by
# 07D.
# ------------------------------------------------------------

def record_key(record):
    return (
        record.get("paper_id"),
        record.get("table_id"),
    )


normalized_lookup = {
    record_key(record): record
    for record in normalized_records
}


candidate_records = [
    record
    for record in records
    if record.get(
        "classification"
    )
    in TARGET_CLASSIFICATIONS
]


normalized_matches = 0

for record in candidate_records:

    normalized_record = normalized_lookup.get(
        record_key(record)
    )

    if normalized_record is None:
        continue

    normalized_matches += 1

    # Preserve the 07D mapping as explicit upstream metadata.
    # 07D3 can still improve it when reconstruction gives
    # stronger evidence.
    record["sample_column"] = (
        normalized_record.get(
            "sample_column"
        )
    )


print(
    "Candidate tables loaded:",
    len(candidate_records),
)

print(
    "Matched to normalized 07D records:",
    normalized_matches,
)

repaired_records = []
flat_rows = []

for record in candidate_records:

    parsed_rows = record.get(
        "parsed_rows",
        [],
    )

    (
        header_rows,
        data_rows,
    ) = detect_header_rows(
        parsed_rows
    )

    repaired_header_rows = (
        repair_leading_header_shift(
            header_rows,
            data_rows,
        )
    )

    (
        combined_header,
        aligned_data_rows,
    ) = repair_header_row_alignment(
        repaired_header_rows,
        data_rows,
    )

    element_columns = (
        identify_element_columns(
            combined_header
        )
    )

    # --------------------------------------------------------
    # Sample-column detection
    #
    # 07D may already contain a valid sample_column.
    # 07D3 reconstructs multi-level headers, so it may discover
    # a better sample column. However, failure to rediscover the
    # word "Sample" must not erase a valid upstream column.
    # --------------------------------------------------------

    upstream_sample_column = record.get(
        "sample_column"
    )

    # Normalize the upstream index conservatively.
    if isinstance(
        upstream_sample_column,
        bool,
    ):
        upstream_sample_column = None

    elif upstream_sample_column is not None:

        try:
            numeric_sample_column = float(
                upstream_sample_column
            )

            if numeric_sample_column.is_integer():
                upstream_sample_column = int(
                    numeric_sample_column
                )
            else:
                upstream_sample_column = None

        except (
            TypeError,
            ValueError,
        ):
            upstream_sample_column = None

    # An inherited index is only valid if it still lies inside
    # the reconstructed table width.
    if (
        upstream_sample_column is not None
        and not (
            0
            <= upstream_sample_column
            < len(combined_header)
        )
    ):
        upstream_sample_column = None


    detected_sample_column = (
        identify_sample_column(
            combined_header
        )
    )


    if detected_sample_column is not None:

        sample_column = (
            detected_sample_column
        )

        sample_column_source = (
            "DETECTED_07D3"
        )

    elif upstream_sample_column is not None:

        sample_column = (
            upstream_sample_column
        )

        sample_column_source = (
            "PRESERVED_07D"
        )

    else:

        sample_column = None

        sample_column_source = (
            "NONE"
        )

    temperature_column = (
        identify_temperature_column(
            combined_header
        )
    )

    repaired_record = {
        "paper_id": record.get(
            "paper_id"
        ),
        "table_id": record.get(
            "table_id"
        ),
        "classification": record.get(
            "classification"
        ),
        "caption": record.get(
            "caption",
            "",
        ),
        "header_rows": (
            repaired_header_rows
        ),
        "combined_header": (
            combined_header
        ),
        "sample_column": (
            sample_column
        ),
        "sample_column_upstream": (
            upstream_sample_column
        ),
        "sample_column_detected_07d3": (
            detected_sample_column
        ),
        "sample_column_source": (
            sample_column_source
        ),
        "temperature_column": (
            temperature_column
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
            len(
                aligned_data_rows
            )
        ),
        "raw_table_text": record.get(
            "raw_table_text",
            "",
        ),
    }

    repaired_records.append(
        repaired_record
    )

    for source_row_index, row in enumerate(
        aligned_data_rows,
        start=1,
    ):

        def get_cell(
            column_index,
        ):
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
                    repaired_record[
                        "paper_id"
                    ]
                ),
                "table_id": (
                    repaired_record[
                        "table_id"
                    ]
                ),
                "source_row_index": (
                    source_row_index
                ),
                "sample_candidate": (
                    get_cell(
                        sample_column
                    )
                ),
                "temperature_candidate": (
                    get_cell(
                        temperature_column
                    )
                ),
                "C_candidate": (
                    get_cell(
                        repaired_record[
                            "C_column"
                        ]
                    )
                ),
                "H_candidate": (
                    get_cell(
                        repaired_record[
                            "H_column"
                        ]
                    )
                ),
                "N_candidate": (
                    get_cell(
                        repaired_record[
                            "N_column"
                        ]
                    )
                ),
                "O_candidate": (
                    get_cell(
                        repaired_record[
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

# ============================================================
# Save outputs
# ============================================================

with repaired_jsonl_path.open(
    "w",
    encoding="utf-8",
) as output_file:

    for record in repaired_records:
        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


table_df = pd.DataFrame(
    [
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
            "combined_header": (
                " | ".join(
                    str(value)
                    for value in record[
                        "combined_header"
                    ]
                )
            ),
            "sample_column": (
                record[
                    "sample_column"
                ]
            ),
            "sample_column_upstream": (
                record[
                    "sample_column_upstream"
                ]
            ),
            "sample_column_detected_07d3": (
                record[
                    "sample_column_detected_07d3"
                ]
            ),
            "sample_column_source": (
                record[
                    "sample_column_source"
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
        }
        for record in repaired_records
    ]
)

row_df = pd.DataFrame(
    flat_rows
)

table_df.to_csv(
    repaired_table_csv,
    index=False,
)

row_df.to_csv(
    repaired_rows_csv,
    index=False,
)

with pd.ExcelWriter(
    repaired_xlsx,
    engine="openpyxl",
) as writer:

    table_df.to_excel(
        writer,
        sheet_name="Tables",
        index=False,
    )

    row_df.to_excel(
        writer,
        sheet_name="Rows",
        index=False,
    )


print()
print("=" * 70)
print("STEP 07D3 — MULTI-LEVEL HEADER REPAIR")
print("=" * 70)
print()

print(
    "Tables repaired:",
    len(table_df),
)

print(
    "Rows regenerated:",
    len(row_df),
)

print(
    "Tables with sample column:",
    int(
        table_df[
            "sample_column"
        ]
        .notna()
        .sum()
    ),
)

print(
    "Tables with temperature column:",
    int(
        table_df[
            "temperature_column"
        ]
        .notna()
        .sum()
    ),
)

print(
    "Tables with complete CHNO mapping:",
    int(
        table_df[
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
    ),
)

print()
print("Generated files:")
print("-", repaired_jsonl_path)
print("-", repaired_table_csv)
print("-", repaired_rows_csv)
print("-", repaired_xlsx)


