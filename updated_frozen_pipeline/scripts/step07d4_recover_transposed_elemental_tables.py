#!/usr/bin/env python3

"""
STEP 07D4
=========

Recover Transposed Elemental Tables

Purpose
-------
Detect scientific tables where samples are columns and elemental
properties (C, H, N, O) are rows.

Example source:

        VP      OP      AB      PL
C       40.86   42.62   35.10   35.70
H        5.27    6.36    4.00    5.27
N        1.03    2.08    1.05    9.61
O       48.07   44.78   38.10   40.98

Recovered rows:

VP  C=40.86 H=5.27 N=1.03 O=48.07
OP  C=42.62 H=6.36 N=2.08 O=44.78
AB  C=35.10 H=4.00 N=1.05 O=38.10
PL  C=35.70 H=5.27 N=9.61 O=40.98

The script replaces the incorrectly flattened rows for detected
transposed tables while leaving all other Step 07D3 rows unchanged.

No LLM is used.
No scientific values are modified.
"""

from pathlib import Path

import json
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

classified_path = (
    processed_tables_dir
    / "classified_elemental_tables.jsonl"
)

base_rows_path = (
    processed_tables_dir
    / "repaired_candidate_table_rows.csv"
)

output_rows_path = (
    processed_tables_dir
    / "repaired_candidate_table_rows_v2.csv"
)

repaired_tables_path = (
    processed_tables_dir
    / "repaired_candidate_tables.csv"
)

left_shift_audit_path = (
    processed_tables_dir
    / "left_shifted_row_repair_audit.csv"
)

grouped_sample_audit_path = (
    processed_tables_dir
    / "grouped_sample_block_repair_audit.csv"
)

combined_sample_temperature_audit_path = (
    processed_tables_dir
    / "combined_sample_temperature_repair_audit.csv"
)

recovered_rows_path = (
    processed_tables_dir
    / "recovered_transposed_elemental_rows.csv"
)

inventory_path = (
    processed_tables_dir
    / "transposed_elemental_table_inventory.csv"
)

excel_path = (
    outputs_dir
    / "transposed_elemental_recovery.xlsx"
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):

    records = []

    with path.open(
        encoding="utf-8"
    ) as source:

        for line in source:

            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

    return records


def clean_html(value):

    if value is None:
        return ""

    text = str(value)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_label(value):

    return (
        clean_html(value)
        .strip()
        .upper()
    )


def parse_numeric(value):

    if value is None:
        return None

    text = clean_html(
        value
    )

    text = (
        text
        .replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("_._", ".")
        .replace("_.", ".")
        .replace("._", ".")
        .replace("_", "")
    )

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)",
        text,
    )

    if not match:
        return None

    return float(
        match.group(0)
    )

def is_separator_row(row):

    if not row:
        return False

    cells = [
        str(cell).strip()
        for cell in row
    ]

    meaningful = [
        cell
        for cell in cells
        if cell
    ]

    if not meaningful:
        return False

    return all(
        re.fullmatch(
            r":?-{3,}:?",
            cell,
        )
        is not None
        for cell in meaningful
    )


# ============================================================
# Detect transposed CHNO structure
# ============================================================

def recover_transposed_table(record):

    parsed_rows = record.get(
        "parsed_rows",
        [],
    )

    if not parsed_rows:
        return None

    # ========================================================
    # Find elemental rows
    # ========================================================

    element_source_rows = {}
    element_row_indices = {}

    # Accept both elemental symbols and exact full names.
    #
    # This remains deliberately conservative: only explicit
    # elemental labels are accepted here. Generic lowercase
    # parameter names such as adsorption-model "n" are not
    # involved in transposed-row detection.
    element_label_aliases = {
        "C": "C",
        "CARBON": "C",
        "H": "H",
        "HYDROGEN": "H",
        "N": "N",
        "NITROGEN": "N",
        "O": "O",
        "OXYGEN": "O",
    }

    for row_index, row in enumerate(
        parsed_rows
    ):

        if not row:
            continue

        label = normalize_label(
            row[0]
        )

        element = (
            element_label_aliases.get(
                label
            )
        )

        if element is not None:
            element_source_rows[
                element
            ] = list(row)

            element_row_indices[
                element
            ] = row_index

    # Require complete CHNO structure.
    if not all(
        element in element_source_rows
        for element in [
            "C",
            "H",
            "N",
            "O",
        ]
    ):
        return None

    # ========================================================
    # Detect whether column 1 is a unit column
    # ========================================================

    def looks_like_unit(value):

        text = (
            clean_html(value)
            .strip()
            .lower()
        )

        if not text:
            return False

        unit_patterns = [
            "%",
            "wt%",
            "wt.%",
            "mass%",
            "g kg",
            "mg kg",
            "g/kg",
            "mg/kg",
        ]

        return any(
            pattern in text
            for pattern in unit_patterns
        )

    second_cells = []

    for element in [
        "C",
        "H",
        "N",
        "O",
    ]:

        row = element_source_rows[
            element
        ]

        if len(row) > 1:
            second_cells.append(
                row[1]
            )

    has_unit_column = (
        len(second_cells) == 4
        and all(
            looks_like_unit(value)
            for value in second_cells
        )
    )

    if has_unit_column:
        value_start_index = 2
    else:
        value_start_index = 1

    # ========================================================
    # Extract element value arrays
    # ========================================================

    element_rows = {}

    for element in [
        "C",
        "H",
        "N",
        "O",
    ]:

        source_row = (
            element_source_rows[
                element
            ]
        )

        values = source_row[
            value_start_index:
        ]

        element_rows[
            element
        ] = values

    value_counts = [
        len(
            element_rows[element]
        )
        for element in [
            "C",
            "H",
            "N",
            "O",
        ]
    ]

    if len(
        set(value_counts)
    ) != 1:
        return None

    sample_count = (
        value_counts[0]
    )

    if sample_count < 2:
        return None

    first_element_index = min(
        element_row_indices.values()
    )

    # ========================================================
    # Determine elemental-analysis unit
    # ========================================================

    element_unit = "wt_percent"
    element_scale_to_wt_percent = 1.0

    for context_index in range(
        first_element_index - 1,
        -1,
        -1,
    ):

        context_row = parsed_rows[
            context_index
        ]

        if not context_row:
            continue

        context_text = (
            " ".join(
                clean_html(cell)
                for cell in context_row
            )
            .lower()
        )

        if (
            "ultimate analysis"
            not in context_text
            and "elemental analysis"
            not in context_text
        ):
            continue

        # g kg-1 = one tenth of wt%
        if (
            "g kg" in context_text
            or "gkg" in context_text
            or "g/kg" in context_text
        ):
            element_unit = "g_per_kg"
            element_scale_to_wt_percent = 0.1

        elif (
            "wt%" in context_text
            or "wt %" in context_text
            or "wt.%" in context_text
            or "%" in context_text
        ):
            element_unit = "wt_percent"
            element_scale_to_wt_percent = 1.0

        break

    # ========================================================
    # Header helpers
    # ========================================================

    section_terms = [
        "ultimate analysis",
        "proximate analysis",
        "ash analysis",
        "elemental analysis",
        "biochemical analysis",
        "structural properties",
        "plant nutrients",
    ]

    descriptor_terms = {
        "property",
        "sample",
        "samples",
        "feedstock",
        "feedstocks",
        "material",
        "materials",
    }

    def clean_row(row):

        return [
            clean_html(cell)
            for cell in row
        ]

    def nonempty_row(row):

        return [
            value
            for value in clean_row(row)
            if value
        ]

    def looks_like_section(row):

        text = (
            " ".join(
                nonempty_row(row)
            )
            .lower()
        )

        return any(
            term in text
            for term in section_terms
        )

    # ========================================================
    # Find sample names
    # ========================================================

    sample_names = None
    sample_header_index = None

    # --------------------------------------------------------
    # Strategy 1:
    # Prefer an explicit descriptor row such as:
    #
    # Property | Sawdust | BC-400 | BC-500 ...
    #
    # This prevents caption fragments from being selected.
    # --------------------------------------------------------

    for candidate_index in range(
        first_element_index - 1,
        -1,
        -1,
    ):

        row = parsed_rows[
            candidate_index
        ]

        if (
            not row
            or is_separator_row(row)
            or looks_like_section(row)
        ):
            continue

        cleaned = clean_row(
            row
        )

        if not cleaned:
            continue

        first_cell = (
            cleaned[0]
            .strip()
            .lower()
        )

        if first_cell not in descriptor_terms:
            continue

        # With a unit column:
        # Property | Unit | Sample1 | Sample2 ...
        if has_unit_column:

            if (
                len(cleaned)
                >= sample_count + 2
            ):

                candidates = cleaned[
                    2:
                    2 + sample_count
                ]

                if all(
                    value.strip()
                    for value in candidates
                ):
                    sample_names = (
                        candidates
                    )

                    sample_header_index = (
                        candidate_index
                    )

                    break

        # Without a unit column:
        # Property | Sample1 | Sample2 ...
        else:

            if (
                len(cleaned)
                >= sample_count + 1
            ):

                candidates = cleaned[
                    1:
                    1 + sample_count
                ]

                if all(
                    value.strip()
                    for value in candidates
                ):
                    sample_names = (
                        candidates
                    )

                    sample_header_index = (
                        candidate_index
                    )

                    break

    # --------------------------------------------------------
    # Strategy 2:
    # Simple transposed table:
    #
    # VP | OP | AB | PL
    # C  | ...
    # H  | ...
    # --------------------------------------------------------

    if sample_names is None:

        for candidate_index in range(
            first_element_index - 1,
            -1,
            -1,
        ):

            row = parsed_rows[
                candidate_index
            ]

            if (
                not row
                or is_separator_row(row)
                or looks_like_section(row)
            ):
                continue

            cleaned = nonempty_row(
                row
            )

            if len(cleaned) != sample_count:
                continue

            # Reject obvious caption fragments.
            joined = (
                " ".join(cleaned)
                .lower()
            )

            caption_terms = [
                "table ",
                "physico-chemical",
                "properties of",
                "used in",
                "experiment",
            ]

            if any(
                term in joined
                for term in caption_terms
            ):
                continue

            sample_names = cleaned
            sample_header_index = (
                candidate_index
            )

            break

    # --------------------------------------------------------
    # Strategy 3:
    # Multi-level header with unit column, e.g.
    #
    # Property | Unit | Raw residue |
    #            Biochar | temperature
    #
    #                    200 | 350 | 500
    #
    # Reconstruct:
    # Raw residue
    # Biochar 200
    # Biochar 350
    # Biochar 500
    # --------------------------------------------------------

    temperature_candidates = [
        None
        for _ in range(
            sample_count
        )
    ]

    if (
        sample_names is None
        and has_unit_column
    ):

        descriptor_row = None
        descriptor_index = None

        for candidate_index in range(
            first_element_index - 1,
            -1,
            -1,
        ):

            row = parsed_rows[
                candidate_index
            ]

            if not row:
                continue

            cleaned = clean_row(
                row
            )

            if len(cleaned) < 3:
                continue

            first_cell = (
                cleaned[0]
                .strip()
                .lower()
            )

            second_cell = (
                cleaned[1]
                .strip()
                .lower()
            )

            if (
                first_cell in descriptor_terms
                and (
                    second_cell == "unit"
                    or "unit" in second_cell
                )
            ):
                descriptor_row = cleaned
                descriptor_index = (
                    candidate_index
                )
                break

        if descriptor_row is not None:

            numeric_header = None

            for candidate_index in range(
                descriptor_index + 1,
                first_element_index,
            ):

                row = parsed_rows[
                    candidate_index
                ]

                if not row:
                    continue

                values = nonempty_row(
                    row
                )

                if not values:
                    continue

                numeric_values = []

                all_numeric = True

                for value in values:

                    parsed = parse_numeric(
                        value
                    )

                    if parsed is None:
                        all_numeric = False
                        break

                    numeric_values.append(
                        parsed
                    )

                if (
                    all_numeric
                    and numeric_values
                ):
                    numeric_header = (
                        numeric_values
                    )
                    break

            if numeric_header is not None:

                # Number of processing-temperature samples
                # should be one fewer than total sample columns:
                # one feedstock + several processed materials.
                if (
                    len(numeric_header)
                    == sample_count - 1
                ):

                    source_name = (
                        descriptor_row[2]
                        if len(
                            descriptor_row
                        ) > 2
                        else "Feedstock"
                    )

                    processed_name = (
                        descriptor_row[3]
                        if len(
                            descriptor_row
                        ) > 3
                        else "Biochar"
                    )

                    # Remove overly generic continuation text.
                    if (
                        not processed_name
                        or "temperature" in
                        processed_name.lower()
                    ):
                        processed_name = (
                            "Biochar"
                        )

                    sample_names = [
                        source_name
                    ]

                    temperature_candidates = [
                        None
                    ]

                    for temperature in numeric_header:

                        sample_names.append(
                            (
                                f"{processed_name} "
                                f"{temperature:g}"
                            )
                        )

                        temperature_candidates.append(
                            temperature
                        )

                    sample_header_index = (
                        descriptor_index
                    )

    if sample_names is None:
        return None

    if len(
        sample_names
    ) != sample_count:
        return None

    # ========================================================
    # Optional deterministic temperature from sample code
    # ========================================================

    if not any(
        value is not None
        for value in temperature_candidates
    ):

        temperature_candidates = []

        for sample_name in sample_names:

            match = re.search(
                r"(\d{3,4})"
                r"(?:\s*°?\s*c)?$",
                str(
                    sample_name
                ).strip(),
                flags=re.IGNORECASE,
            )

            if match:
                temperature_candidates.append(
                    float(
                        match.group(1)
                    )
                )
            else:
                temperature_candidates.append(
                    None
                )

    # ========================================================
    # Recover sample records
    # ========================================================

    recovered = []

    for sample_index, sample_name in enumerate(
        sample_names
    ):

        recovered.append(
            {
                "paper_id": record.get(
                    "paper_id"
                ),
                "table_id": record.get(
                    "table_id"
                ),
                "source_row_index": (
                    sample_index + 1
                ),
                "sample_candidate": (
                    sample_name
                ),
                "temperature_candidate": (
                    temperature_candidates[
                        sample_index
                    ]
                ),

                "C_candidate": (
                    parse_numeric(
                        element_rows["C"][
                            sample_index
                        ]
                    )
                    * element_scale_to_wt_percent
                    if parse_numeric(
                        element_rows["C"][
                            sample_index
                        ]
                    )
                    is not None
                    else None
                ),
                "H_candidate": (
                    parse_numeric(
                        element_rows["H"][
                            sample_index
                        ]
                    )
                    * element_scale_to_wt_percent
                    if parse_numeric(
                        element_rows["H"][
                            sample_index
                        ]
                    )
                    is not None
                    else None
                ),
                "N_candidate": (
                    parse_numeric(
                        element_rows["N"][
                            sample_index
                        ]
                    )
                    * element_scale_to_wt_percent
                    if parse_numeric(
                        element_rows["N"][
                            sample_index
                        ]
                    )
                    is not None
                    else None
                ),
                "O_candidate": (
                    parse_numeric(
                        element_rows["O"][
                            sample_index
                        ]
                    )
                    * element_scale_to_wt_percent
                    if parse_numeric(
                        element_rows["O"][
                            sample_index
                        ]
                    )
                    is not None
                    else None
                ),
                "raw_source_row": (
                    f"{sample_name} | "
                    f"{element_rows['C'][sample_index]} | "
                    f"{element_rows['H'][sample_index]} | "
                    f"{element_rows['N'][sample_index]} | "
                    f"{element_rows['O'][sample_index]}"
                ),
                "row_orientation": (
                    "TRANSPOSED_RECOVERED"
                ),
                "element_source_unit": (
                    element_unit
                ),
                "element_scale_to_wt_percent": (
                    element_scale_to_wt_percent
                ),
            }
        )

    return {
        "table_id": record.get(
            "table_id"
        ),
        "paper_id": record.get(
            "paper_id"
        ),
        "sample_count": sample_count,
        "samples": sample_names,
        "sample_header_index": (
            sample_header_index
        ),
        "has_unit_column": (
            has_unit_column
        ),
        "recovered_rows": recovered,
    }
# ============================================================
# Structural grouped-row repair
# ============================================================

def first_numeric_value(value):

    if pd.isna(value):
        return None

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        str(value),
    )

    if not match:
        return None

    try:
        return float(
            match.group()
        )
    except Exception:
        return None


def valid_text_sample(value):

    if pd.isna(value):
        return False

    text = str(value).strip()

    if not text:
        return False

    if re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    ):
        return False

    return True


def repair_grouped_elemental_rows(
    rows_df,
    tables_df,
):

    rows_df = rows_df.copy()

    eligible = tables_df[
        tables_df[
            "classification"
        ].eq(
            "LIKELY_ELEMENTAL_TABLE"
        )
        &
        tables_df[
            "temperature_column"
        ].notna()
        &
        tables_df[
            "C_column"
        ].notna()
        &
        tables_df[
            "H_column"
        ].notna()
        &
        tables_df[
            "N_column"
        ].notna()
        &
        tables_df[
            "O_column"
        ].notna()
    ].copy()

    eligible_by_id = {
        row["table_id"]: row
        for _, row in eligible.iterrows()
    }

    left_shift_audit = []

    # --------------------------------------------------------
    # Phase 1:
    # Repair rows whose leading grouped sample cell is absent.
    #
    # TEMPORARILY DISABLED:
    # The previous logic shifted raw cells one position left
    # and could overwrite already-correct CHNO mappings from
    # Step 07D3.
    #
    # Step 07D4 should preserve the CHNO mapping produced by
    # Step 07D3 unless a table is explicitly identified and
    # reconstructed as a transposed table.
    # --------------------------------------------------------

    ENABLE_LEFT_SHIFT_REPAIR = False

    if ENABLE_LEFT_SHIFT_REPAIR:

        for idx, row in rows_df.iterrows():

            table_id = row.get(
                "table_id"
            )

            if table_id not in eligible_by_id:
                continue

            sample_num = first_numeric_value(
                row.get(
                    "sample_candidate"
                )
            )

            temp_num = first_numeric_value(
                row.get(
                    "temperature_candidate"
                )
            )

            # Structural signature:
            #
            # parsed "sample" is actually a plausible temperature,
            # while parsed "temperature" looks like a percentage.
            if not (
                sample_num is not None
                and 100 <= sample_num <= 3000
                and temp_num is not None
                and 0 <= temp_num <= 100
            ):
                continue

            raw = str(
                row.get(
                    "raw_source_row",
                    ""
                )
            )

            cells = [
                cell.strip()
                for cell in raw.split("|")
            ]

            table = eligible_by_id[
                table_id
            ]

            positions = {
                "temperature_candidate":
                    int(
                        table[
                            "temperature_column"
                        ]
                    ),

                "C_candidate":
                    int(
                        table[
                            "C_column"
                        ]
                    ),

                "H_candidate":
                    int(
                        table[
                            "H_column"
                        ]
                    ),

                "N_candidate":
                    int(
                        table[
                            "N_column"
                        ]
                    ),

                "O_candidate":
                    int(
                        table[
                            "O_column"
                        ]
                    ),
            }

            repaired = {}

            for column, original_pos in (
                positions.items()
            ):

                shifted_pos = (
                    original_pos - 1
                )

                if (
                    0 <= shifted_pos
                    < len(cells)
                ):
                    repaired[
                        column
                    ] = cells[
                        shifted_pos
                    ]
                else:
                    repaired[
                        column
                    ] = None

            old_sample = row.get(
                "sample_candidate"
            )

            old_temperature = row.get(
                "temperature_candidate"
            )

            rows_df.at[
                idx,
                "sample_candidate"
            ] = None

            for column, value in repaired.items():

                rows_df.at[
                    idx,
                    column
                ] = value

            left_shift_audit.append(
                {
                    "paper_id":
                        row.get(
                            "paper_id"
                        ),

                    "table_id":
                        table_id,

                    "source_row_index":
                        row.get(
                            "source_row_index"
                        ),

                    "repair_type":
                        "LEFT_SHIFT_MISSING_SAMPLE_CELL",

                    "old_sample_candidate":
                        old_sample,

                    "old_temperature_candidate":
                        old_temperature,

                    "new_temperature_candidate":
                        repaired[
                            "temperature_candidate"
                        ],

                    "new_C_candidate":
                        repaired[
                            "C_candidate"
                        ],

                    "new_H_candidate":
                        repaired[
                            "H_candidate"
                        ],

                    "new_N_candidate":
                        repaired[
                            "N_candidate"
                        ],

                    "new_O_candidate":
                        repaired[
                            "O_candidate"
                        ],

                    "raw_source_row":
                        raw,
                }
            )
    # --------------------------------------------------------
    # Phase 1B:
    # Recover combined sample-temperature identifiers.
    #
    # Examples:
    #   VP-250 -> sample=VP, temperature=250
    #   OP-300 -> sample=OP, temperature=300
    #
    # IMPORTANT:
    # This repair ONLY fills sample_candidate and
    # temperature_candidate. It never modifies C/H/N/O.
    # --------------------------------------------------------

    combined_sample_temperature_audit = []

    for idx, row in rows_df.iterrows():

        # Only repair rows where both fields are currently absent.
        current_sample = row.get("sample_candidate")
        current_temperature = row.get("temperature_candidate")

        sample_missing = (
            pd.isna(current_sample)
            or str(current_sample).strip() == ""
        )

        temperature_missing = (
            pd.isna(current_temperature)
            or str(current_temperature).strip() == ""
        )

        if not (sample_missing and temperature_missing):
            continue

        raw = str(
            row.get(
                "raw_source_row",
                ""
            )
        )

        if not raw.strip():
            continue

        # The first raw table cell contains strings such as:
        # VP-250, OP-300, AB-350, PL-600.
        first_cell = raw.split("|")[0].strip()

        match = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9_]*)\s*[-–—]\s*(\d{2,4}(?:\.\d+)?)",
            first_cell,
        )

        if match is None:
            continue

        recovered_sample = match.group(1).strip()
        recovered_temperature = float(match.group(2))

        # Conservative temperature sanity check.
        if not (100 <= recovered_temperature <= 3000):
            continue

        # Require at least one elemental value so that unrelated
        # hyphenated labels are not interpreted as sample-temperature
        # data rows.
        elemental_values = [
            first_numeric_value(row.get("C_candidate")),
            first_numeric_value(row.get("H_candidate")),
            first_numeric_value(row.get("N_candidate")),
            first_numeric_value(row.get("O_candidate")),
        ]

        if not any(value is not None for value in elemental_values):
            continue

        # ONLY fill sample and temperature.
        # Never touch the elemental candidates here.
        rows_df.at[
            idx,
            "sample_candidate"
        ] = recovered_sample

        rows_df.at[
            idx,
            "temperature_candidate"
        ] = recovered_temperature

        combined_sample_temperature_audit.append(
            {
                "paper_id":
                    row.get("paper_id"),

                "table_id":
                    row.get("table_id"),

                "source_row_index":
                    row.get("source_row_index"),

                "repair_type":
                    "COMBINED_SAMPLE_TEMPERATURE_IDENTIFIER",

                "raw_first_cell":
                    first_cell,

                "recovered_sample_candidate":
                    recovered_sample,

                "recovered_temperature_candidate":
                    recovered_temperature,

                "C_candidate":
                    row.get("C_candidate"),

                "H_candidate":
                    row.get("H_candidate"),

                "N_candidate":
                    row.get("N_candidate"),

                "O_candidate":
                    row.get("O_candidate"),
            }
        )
    # --------------------------------------------------------
    # Phase 2:
    # Reconstruct grouped sample blocks using exactly one
    # explicit sample anchor inside a monotonic temperature run.
    # --------------------------------------------------------

    grouped_audit = []

    for table_id in eligible_by_id:

        indices = rows_df.index[
            rows_df[
                "table_id"
            ].eq(
                table_id
            )
        ].tolist()

        indices = sorted(
            indices,
            key=lambda i:
                rows_df.at[
                    i,
                    "source_row_index"
                ],
        )

        blocks = []
        current = []
        previous_temp = None

        for idx in indices:

            temperature = (
                first_numeric_value(
                    rows_df.at[
                        idx,
                        "temperature_candidate"
                    ]
                )
            )

            if temperature is None:

                if current:
                    blocks.append(
                        current
                    )

                current = []
                previous_temp = None
                continue

            if (
                current
                and previous_temp is not None
                and temperature <= previous_temp
            ):

                blocks.append(
                    current
                )

                current = []

            current.append(
                idx
            )

            previous_temp = temperature

        if current:
            blocks.append(
                current
            )

        for block_number, block in enumerate(
            blocks,
            start=1,
        ):

            if len(block) < 2:
                continue

            anchors = [
                str(
                    rows_df.at[
                        idx,
                        "sample_candidate"
                    ]
                ).strip()
                for idx in block
                if valid_text_sample(
                    rows_df.at[
                        idx,
                        "sample_candidate"
                    ]
                )
            ]

            unique_anchors = list(
                dict.fromkeys(
                    anchors
                )
            )

            # Exactly one explicit identity in the block.
            if len(
                unique_anchors
            ) != 1:
                continue

            temperatures = [
                first_numeric_value(
                    rows_df.at[
                        idx,
                        "temperature_candidate"
                    ]
                )
                for idx in block
            ]

            # Do not collapse repeated-condition rows.
            if len(
                set(
                    temperatures
                )
            ) != len(
                temperatures
            ):
                continue

            missing = [
                idx
                for idx in block
                if not valid_text_sample(
                    rows_df.at[
                        idx,
                        "sample_candidate"
                    ]
                )
            ]

            if not missing:
                continue

            anchor = (
                unique_anchors[0]
            )

            for idx in missing:

                rows_df.at[
                    idx,
                    "sample_candidate"
                ] = anchor

                grouped_audit.append(
                    {
                        "paper_id":
                            rows_df.at[
                                idx,
                                "paper_id"
                            ],

                        "table_id":
                            table_id,

                        "source_row_index":
                            rows_df.at[
                                idx,
                                "source_row_index"
                            ],

                        "block_number":
                            block_number,

                        "block_temperatures":
                            " | ".join(
                                str(x)
                                for x in temperatures
                            ),

                        "explicit_anchor":
                            anchor,

                        "new_sample_candidate":
                            anchor,

                        "temperature_candidate":
                            rows_df.at[
                                idx,
                                "temperature_candidate"
                            ],

                        "repair_type":
                            "GROUPED_SAMPLE_BLOCK_RECONSTRUCTION",

                        "raw_source_row":
                            rows_df.at[
                                idx,
                                "raw_source_row"
                            ],
                    }
                )

    return (
        rows_df,
        pd.DataFrame(
            left_shift_audit
        ),
        pd.DataFrame(
            grouped_audit
        ),
        pd.DataFrame(
            combined_sample_temperature_audit
        ),
    )

# ============================================================
# Load data
# ============================================================

records = load_jsonl(
    classified_path
)

base_df = pd.read_csv(
    base_rows_path
)

print(
    "Classified tables loaded:",
    len(records),
)

print(
    "Step 07D3 rows loaded:",
    len(base_df),
)


# ============================================================
# Recover transposed tables
# ============================================================

recoveries = []

for record in records:

    result = recover_transposed_table(
        record
    )

    if result is not None:
        recoveries.append(
            result
        )


transposed_table_ids = {
    result["table_id"]
    for result in recoveries
}

recovered_rows = [
    row
    for result in recoveries
    for row in result[
        "recovered_rows"
    ]
]

recovered_df = pd.DataFrame(
    recovered_rows
)


# ============================================================
# Remove incorrectly flattened versions
# ============================================================

base_kept_df = base_df[
    ~base_df[
        "table_id"
    ].isin(
        transposed_table_ids
    )
].copy()

if (
    "row_orientation"
    not in base_kept_df.columns
):
    base_kept_df[
        "row_orientation"
    ] = "STANDARD"


# ============================================================
# Combine
# ============================================================

combined_df = pd.concat(
    [
        base_kept_df,
        recovered_df,
    ],
    ignore_index=True,
    sort=False,
)

# ============================================================
# Repair grouped / merged-cell elemental rows
# ============================================================

repaired_tables_df = pd.read_csv(
    repaired_tables_path
)

(
    combined_df,
    left_shift_audit_df,
    grouped_sample_audit_df,
    combined_sample_temperature_audit_df,
) = repair_grouped_elemental_rows(
    combined_df,
    repaired_tables_df,
)

print(
    "Left-shifted elemental rows repaired:",
    len(
        left_shift_audit_df
    ),
)

print(
    "Grouped sample cells reconstructed:",
    len(
        grouped_sample_audit_df
    ),
)

# Ensure audit outputs retain a stable schema even when
# no repairs were made.
if (
    combined_sample_temperature_audit_df.empty
    and len(
        combined_sample_temperature_audit_df.columns
    ) == 0
):
    combined_sample_temperature_audit_df = pd.DataFrame(
        columns=[
            "paper_id",
            "table_id",
            "source_row_index",
            "repair_type",
            "raw_first_cell",
            "recovered_sample_candidate",
            "recovered_temperature_candidate",
            "C_candidate",
            "H_candidate",
            "N_candidate",
            "O_candidate",
        ]
    )

if (
    left_shift_audit_df.empty
    and len(
        left_shift_audit_df.columns
    ) == 0
):
    left_shift_audit_df = pd.DataFrame(
        columns=[
            "paper_id",
            "table_id",
            "source_row_index",
            "repair_type",
            "old_sample_candidate",
            "old_temperature_candidate",
            "new_temperature_candidate",
            "new_C_candidate",
            "new_H_candidate",
            "new_N_candidate",
            "new_O_candidate",
            "raw_source_row",
        ]
    )

print(
    "Combined sample-temperature IDs repaired:",
    len(
        combined_sample_temperature_audit_df
    ),
)

# ============================================================
# Inventory
# ============================================================

inventory_df = pd.DataFrame(
    [
        {
            "paper_id": result[
                "paper_id"
            ],
            "table_id": result[
                "table_id"
            ],
            "sample_count": result[
                "sample_count"
            ],
            "samples": " | ".join(
                result[
                    "samples"
                ]
            ),
            "recovered_row_count": len(
                result[
                    "recovered_rows"
                ]
            ),
        }
        for result in recoveries
    ]
)


# ============================================================
# Save
# ============================================================

combined_df.to_csv(
    output_rows_path,
    index=False,
)
left_shift_audit_df.to_csv(
    left_shift_audit_path,
    index=False,
)

grouped_sample_audit_df.to_csv(
    grouped_sample_audit_path,
    index=False,
)

combined_sample_temperature_audit_df.to_csv(
    combined_sample_temperature_audit_path,
    index=False,
)

recovered_df.to_csv(
    recovered_rows_path,
    index=False,
)

inventory_df.to_csv(
    inventory_path,
    index=False,
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:

    inventory_df.to_excel(
        writer,
        sheet_name="Recovered_tables",
        index=False,
    )

    recovered_df.to_excel(
        writer,
        sheet_name="Recovered_rows",
        index=False,
    )

    combined_df.to_excel(
        writer,
        sheet_name="Combined_rows",
        index=False,
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("STEP 07D4 — TRANSPOSED ELEMENTAL RECOVERY")
print("=" * 70)
print()

print(
    "Transposed elemental tables detected:",
    len(
        transposed_table_ids
    ),
)

print(
    "Recovered sample rows:",
    len(
        recovered_df
    ),
)

print(
    "Original 07D3 rows removed:",
    len(base_df)
    - len(base_kept_df),
)

print(
    "Combined output rows:",
    len(
        combined_df
    ),
)

print()
print("Generated files:")
print("-", output_rows_path)
print("-", recovered_rows_path)
print("-", inventory_path)
print("-", excel_path)
