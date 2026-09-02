#!/usr/bin/env python3

"""
STEP 08C2 — GENERIC GROUPED / MERGED ROW REPAIR

Purpose
-------
Detect and repair rows where a sample/feedstock name appears only
once for a group and subsequent rows inherit that sample.

Example:

Sample | Temperature | C | H | N | O
AB     | 400         | ...
       | 500         | ...
       | 600         | ...

The 500 and 600 rows inherit AB.

Important
---------
- No recovery IDs are hard-coded.
- No sample names are hard-coded.
- No LLM is used.
- Detection uses table structure.
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
    / "semantic_recovery_packages.jsonl"
)

tables_path = (
    processed_dir
    / "all_tables.jsonl"
)

output_path = (
    processed_dir
    / "grouped_row_repairs.csv"
)

excel_path = (
    outputs_dir
    / "grouped_row_repairs.xlsx"
)


# ============================================================
# Helpers
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    text = str(value)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = (
        text
        .replace("**", "")
        .replace("−", "-")
        .replace("–", "-")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def parse_number(value):

    text = clean_text(
        value
    )

    if not text:
        return None

    match = re.search(
        r"[-+]?"
        r"(?:\d+(?:\.\d+)?|\.\d+)",
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


def contains_letters(value):

    return (
        re.search(
            r"[A-Za-z]",
            clean_text(value),
        )
        is not None
    )


def header_supports_grouped_structure(
    package
):

    header = package.get(
        "header_row",
        []
    )

    if isinstance(
        header,
        list,
    ):
        text = " ".join(
            clean_text(x)
            for x in header
        ).lower()

    else:
        text = clean_text(
            header
        ).lower()

    has_sample_axis = (
        "sample" in text
        or "feedstock" in text
        or "material" in text
        or "biomass" in text
        or "precursor" in text
    )

    has_temperature_axis = (
        "temperature" in text
        or "temp" in text
        or "°c" in text
        or "pyrolysis" in text
        or "carbonization" in text
        or "carbonisation" in text
    )

    return (
        has_sample_axis
        and has_temperature_axis
    )


def first_cell_is_parent_sample(
    value
):

    text = clean_text(
        value
    )

    if not text:
        return False

    if not contains_letters(
        text
    ):
        return False

    forbidden = {
        "sample",
        "samples",
        "feedstock",
        "precursor",
        "material",
        "temperature",
        "emperature",
        "temp",
        "property",
        "c",
        "h",
        "n",
        "o",
        "s",
        "ash",
        "fixed carbon",
        "volatile matter",
    }
    lower = text.lower()

    if (
        "temperature" in lower
        or "emperature" in lower
        or "°c" in lower
    ):
        return False

    if lower in forbidden:
        return False

    return True

# ============================================================
# Load packages
# ============================================================

packages = []

with packages_path.open(
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        obj = json.loads(
            line
        )

        if (
            obj.get(
                "semantic_class"
            )
            == "TEMPERATURE_RANGE"
        ):
            continue

        packages.append(
            obj
        )


print(
    "Eligible packages loaded:",
    len(packages),
)


# ============================================================
# Load complete source tables
# ============================================================

tables = {}

with tables_path.open(
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        obj = json.loads(
            line
        )

        table_id = obj.get(
            "table_id"
        )

        if table_id:
            tables[
                str(table_id)
            ] = obj


print(
    "Source tables loaded:",
    len(tables),
)


# ============================================================
# Group packages by table
# ============================================================

packages_by_table = {}

for package in packages:

    table_id = str(
        package.get(
            "table_id"
        )
    )

    packages_by_table.setdefault(
        table_id,
        []
    ).append(
        package
    )


# ============================================================
# Discover grouped structures
# ============================================================

repairs = []


for table_id, table_packages in (
    packages_by_table.items()
):

    # Require at least one package whose header says this
    # is a sample/feedstock + temperature table.
    if not any(
        header_supports_grouped_structure(
            package
        )
        for package in table_packages
    ):
        continue

    table = tables.get(
        table_id
    )

    if table is None:
        continue

    parsed_rows = table.get(
        "parsed_rows",
        []
    )

    if not parsed_rows:
        continue

    # --------------------------------------------------------
    # Reconstruct parent sample through the table
    # --------------------------------------------------------

    current_sample = None

    reconstructed = {}

    for row_index, row in enumerate(
        parsed_rows
    ):

        if not row:
            continue

        cells = [
            clean_text(cell)
            for cell in row
        ]

        if not cells:
            continue

        first = cells[0]

        # ----------------------------------------------------
        # New parent sample row
        # ----------------------------------------------------

        if first_cell_is_parent_sample(
            first
        ):

            if len(cells) < 2:
                continue

            candidate_temperature = (
                parse_number(
                    cells[1]
                )
            )

            # A parent sample row must itself establish
            # a plausible processing temperature.
            #
            # This prevents scientific values such as
            # 0.10, 0.26, ratios, pH, etc. from being
            # mistaken for temperatures.
            if (
                candidate_temperature
                is None
                or not (
                    50
                    <= candidate_temperature
                    <= 3000
                )
            ):
                continue

            current_sample = first

            temperature = (
                candidate_temperature
            )

            scientific_values = (
                cells[2:]
            )

        # ----------------------------------------------------
        # Continuation row
        #
        # Because the blank sample cell was dropped by the
        # parser, the first visible cell is temperature.
        # ----------------------------------------------------

        else:

            if current_sample is None:
                continue

            temperature = parse_number(
                first
            )
            if (
                temperature is None
                or not (
                    50
                    <= temperature
                    <= 3000
                )
            ):
                continue
            scientific_values = (
                cells[1:]
            )

        if temperature is None:
            continue

        reconstructed[
            row_index
        ] = {
            "sample": (
                current_sample
            ),
            "temperature_C": (
                float(
                    temperature
                )
            ),
            "values": (
                scientific_values
            ),
        }

    # --------------------------------------------------------
    # Resolve packages belonging to this table
    # --------------------------------------------------------

    for package in table_packages:

        recovery_id = str(
            package[
                "recovery_id"
            ]
        )

        # ----------------------------------------------------
        # Locate the actual row in parsed_rows by CONTENT,
        # not source_row_index.
        #
        # source_row_index may be offset because different
        # preprocessing stages remove headers/separator rows.
        # ----------------------------------------------------

        raw_source_row = str(
            package.get(
                "raw_source_row",
                "",
            )
        )

        target_cells = [
            clean_text(cell)
            for cell in raw_source_row.split("|")
        ]

        # Preserve internal blank cells, but remove trailing
        # blanks introduced by markdown formatting.
        while (
            target_cells
            and not target_cells[-1]
        ):
            target_cells.pop()

        if not target_cells:
            continue

        matching_row_indices = []

        for row_index, parsed_row in enumerate(
            parsed_rows
        ):

            parsed_cells = [
                clean_text(cell)
                for cell in parsed_row
            ]

            while (
                parsed_cells
                and not parsed_cells[-1]
            ):
                parsed_cells.pop()

            # Exact normalized row match.
            if parsed_cells == target_cells:
                matching_row_indices.append(
                    row_index
                )

        # ----------------------------------------------------
        # Fallback:
        # match using the complete sequence of visible values.
        # This handles cases where the source row lost the
        # original blank sample cell.
        # ----------------------------------------------------

        if not matching_row_indices:

            target_visible = [
                cell
                for cell in target_cells
                if cell
            ]

            for row_index, parsed_row in enumerate(
                parsed_rows
            ):

                parsed_visible = [
                    clean_text(cell)
                    for cell in parsed_row
                    if clean_text(cell)
                ]

                if (
                    parsed_visible
                    == target_visible
                ):
                    matching_row_indices.append(
                        row_index
                    )

        # We only repair when the source row maps uniquely.
        if len(
            matching_row_indices
        ) != 1:
            continue

        actual_row_index = (
            matching_row_indices[0]
        )

        reconstructed_row = (
            reconstructed.get(
                actual_row_index
            )
        )

        if reconstructed_row is None:
            continue


        raw_source_row = str(
            package.get(
                "raw_source_row",
                "",
            )
        )

        raw_cells = [
            clean_text(cell)
            for cell in raw_source_row.split(
                "|"
            )
        ]

        raw_cells = [
            cell
            for cell in raw_cells
            if cell
        ]

        if not raw_cells:
            continue

        first_visible_cell = (
            raw_cells[0]
        )

        # ----------------------------------------------------
        # Only treat this as a grouped continuation row if:
        #
        # - package currently thinks first cell is sample;
        # - first visible cell is numeric;
        # - reconstructed table says this row inherited a
        #   textual parent sample.
        # ----------------------------------------------------

        numeric_first = parse_number(
            first_visible_cell
        )

        if numeric_first is None:
            continue

        if contains_letters(
            first_visible_cell
        ):
            continue

        parent_sample = (
            reconstructed_row[
                "sample"
            ]
        )

        if not parent_sample:
            continue

        # The first numeric visible value in a continuation
        # row should correspond to temperature.
        if abs(
            numeric_first
            - reconstructed_row[
                "temperature_C"
            ]
        ) > 1e-6:
            continue

        repairs.append(
            {
                "recovery_id": (
                    recovery_id
                ),
                "paper_id": package.get(
                    "paper_id"
                ),
                "table_id": (
                    table_id
                ),
                "source_row_index": (
                    actual_row_index
                ),
                "original_sample_raw": (
                    package.get(
                        "sample_raw"
                    )
                ),
                "resolved_sample": (
                    parent_sample
                ),
                "resolved_temperature_C": (
                    reconstructed_row[
                        "temperature_C"
                    ]
                ),
                "resolved_temperature_type": (
                    None
                ),
                "repair_status": (
                    "DETERMINISTIC_"
                    "GROUPED_ROW_RECOVERY"
                ),
                "repair_reason": (
                    "Numeric continuation row "
                    "inherits nearest preceding "
                    "textual sample/feedstock label."
                ),
            }
        )


# ============================================================
# Deduplicate
# ============================================================

repair_df = pd.DataFrame(
    repairs
)


if not repair_df.empty:

    repair_df = (
        repair_df
        .drop_duplicates(
            subset=[
                "recovery_id",
            ],
            keep="last",
        )
        .sort_values(
            [
                "table_id",
                "source_row_index",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Ensure output always has columns
# ============================================================

if repair_df.empty:

    repair_df = pd.DataFrame(
        columns=[
            "recovery_id",
            "paper_id",
            "table_id",
            "source_row_index",
            "original_sample_raw",
            "resolved_sample",
            "resolved_temperature_C",
            "resolved_temperature_type",
            "repair_status",
            "repair_reason",
        ]
    )


# ============================================================
# Save
# ============================================================

repair_df.to_csv(
    output_path,
    index=False,
)


with pd.ExcelWriter(
    excel_path
) as writer:

    repair_df.to_excel(
        writer,
        sheet_name="grouped_repairs",
        index=False,
    )

    summary = (
        repair_df[
            "repair_status"
        ]
        .value_counts()
        .rename_axis(
            "repair_status"
        )
        .reset_index(
            name="row_count"
        )
    )

    summary.to_excel(
        writer,
        sheet_name="summary",
        index=False,
    )


# ============================================================
# Report
# ============================================================

print()
print("=" * 72)
print(
    "STEP 08C2 — GENERIC GROUPED ROW REPAIR"
)
print("=" * 72)
print()

print(
    "Grouped rows recovered:",
    len(repair_df),
)

print()

if not repair_df.empty:

    print(
        repair_df[
            [
                "recovery_id",
                "resolved_sample",
                "resolved_temperature_C",
                "repair_status",
            ]
        ].to_string(
            index=False
        )
    )

print()
print(
    "Output:",
    output_path,
)

print(
    "Excel:",
    excel_path,
)

