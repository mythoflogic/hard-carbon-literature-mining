#!/usr/bin/env python3

"""
STEP 07I
========

Recover Sample Names from Grouped Table Rows

Purpose
-------
Recover sample names that are visually represented once for a
temperature-series block but omitted from neighboring rows.

Examples:

    Rice husk | 350
              | 450
              | 550

and:

              | 500
    OPT       | 550
              | 600

Method
------
Within each table:

1. sort rows by source_row_index;
2. divide rows into temperature blocks;
3. start a new block whenever temperature resets downward;
4. require exactly one explicit sample name inside the block;
5. fill only rows currently classified as
   NEEDS_SAMPLE_INTERPRETATION;
6. never modify temperature or CHNO values.

This step does not use an LLM.
"""

from pathlib import Path

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
    / "classified_extractable_rows_role_enriched_07h.csv"
)

output_path = (
    processed_tables_dir
    / "classified_extractable_rows_role_enriched_07i.csv"
)

audit_path = (
    processed_tables_dir
    / "grouped_sample_recovery_audit.csv"
)


# ============================================================
# Helpers
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
    }:
        return ""

    return text


def numeric_temperature(value):

    if value is None:
        return None

    if pd.isna(value):
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


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
    "STEP 07I — GROUPED SAMPLE RECOVERY"
)
print("=" * 78)
print()

print(
    "Rows loaded:",
    len(df),
)


required_columns = [
    "table_id",
    "source_row_index",
    "sample_raw",
    "temperature_C",
    "classification",
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
# Recover grouped samples
# ============================================================

audit_records = []

# Preserve 07I provenance in the main row-level output.
df[
    "sample_recovered_07i"
] = False

df[
    "sample_recovery_rule_07i"
] = None

df[
    "sample_recovery_anchor_07i"
] = None

df[
    "sample_recovery_anchor_row_07i"
] = None

df[
    "sample_recovery_block_07i"
] = None


group_columns = (
    [
        "paper_id",
        "table_id",
    ]
    if "paper_id" in df.columns
    else [
        "table_id",
    ]
)

for group_key, table_indices in (
    df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ).groups.items()
):

    if (
        "paper_id" in df.columns
    ):
        (
            group_paper_id,
            table_id,
        ) = group_key
    else:
        group_paper_id = None

        if isinstance(
            group_key,
            tuple,
        ):
            table_id = group_key[0]
        else:
            table_id = group_key

    table = (
        df.loc[
            list(table_indices)
        ]
        .sort_values(
            "source_row_index"
        )
        .copy()
    )


    # --------------------------------------------------------
    # Build temperature-sequence blocks.
    #
    # A new block begins when the temperature decreases.
    #
    # Examples:
    #
    # 350 450 550 650 750 | 550 650 750
    #
    # 400 500 600 700 800 | 400 500 600 ...
    #
    # 500 550 600 | 500 550 600
    # --------------------------------------------------------

    blocks = []
    current_block = []

    previous_temperature = None


    for idx, row in table.iterrows():

        temperature = numeric_temperature(
            row.get(
                "temperature_C"
            )
        )


        if (
            current_block
            and temperature is not None
            and previous_temperature is not None
            and temperature < previous_temperature
        ):

            blocks.append(
                current_block
            )

            current_block = []


        current_block.append(
            idx
        )


        if temperature is not None:

            previous_temperature = (
                temperature
            )


    if current_block:

        blocks.append(
            current_block
        )


    # --------------------------------------------------------
    # Evaluate each block.
    # --------------------------------------------------------

    for block_number, block_indices in enumerate(
        blocks,
        start=1,
    ):

        block = df.loc[
            block_indices
        ]


        explicit_cells = []

        for block_idx, value in zip(
            block.index,
            block[
                "sample_raw"
            ],
        ):

            sample = clean_text(
                value
            )

            if sample:
                explicit_cells.append(
                    (
                        block_idx,
                        sample,
                    )
                )

        explicit_samples = []

        for _, sample in explicit_cells:

            if sample not in explicit_samples:
                explicit_samples.append(
                    sample
                )


        # ----------------------------------------------------
        # Conservative structural safety gate.
        #
        # Require:
        #   - exactly one explicit sample CELL;
        #   - exactly one sample identity;
        #   - no missing temperatures;
        #   - at least two temperatures;
        #   - a strictly increasing temperature sequence.
        #
        # Equal/repeated temperatures or missing temperatures
        # remain unresolved rather than being forward-filled.
        # ----------------------------------------------------

        block_temperatures = [
            numeric_temperature(
                value
            )
            for value in block[
                "temperature_C"
            ]
        ]

        all_temperatures_known = all(
            value is not None
            for value in block_temperatures
        )

        strictly_increasing = (
            len(block_temperatures) >= 2
            and all_temperatures_known
            and all(
                later > earlier
                for earlier, later
                in zip(
                    block_temperatures,
                    block_temperatures[1:],
                )
            )
        )

        if not (
            len(explicit_cells) == 1
            and len(explicit_samples) == 1
            and strictly_increasing
        ):
            continue


        (
            anchor_idx,
            recovered_sample,
        ) = explicit_cells[0]

        anchor_source_row_index = (
            df.at[
                anchor_idx,
                "source_row_index",
            ]
        )


        # ----------------------------------------------------
        # Fill ONLY rows that explicitly require sample
        # interpretation.
        #
        # Rejected rows and already-valid rows are untouched.
        # ----------------------------------------------------

        for idx in block_indices:

            current_sample = clean_text(
                df.at[
                    idx,
                    "sample_raw",
                ]
            )

            classification = clean_text(
                df.at[
                    idx,
                    "classification",
                ]
            )


            if current_sample:

                continue


            if (
                classification
                != "NEEDS_SAMPLE_INTERPRETATION"
            ):

                continue


            df.at[
                idx,
                "sample_raw",
            ] = recovered_sample

            df.at[
                idx,
                "sample_recovered_07i",
            ] = True

            df.at[
                idx,
                "sample_recovery_rule_07i",
            ] = (
                "UNIQUE_SAMPLE_IN_STRICT_"
                "TEMPERATURE_BLOCK"
            )

            df.at[
                idx,
                "sample_recovery_anchor_07i",
            ] = recovered_sample

            df.at[
                idx,
                "sample_recovery_anchor_row_07i",
            ] = anchor_source_row_index

            df.at[
                idx,
                "sample_recovery_block_07i",
            ] = block_number


            audit_records.append(
                {
                    "row_index":
                        idx,

                    "paper_id":
                        df.at[
                            idx,
                            "paper_id",
                        ]
                        if "paper_id" in df.columns
                        else None,

                    "table_id":
                        table_id,

                    "source_row_index":
                        df.at[
                            idx,
                            "source_row_index",
                        ],

                    "block_number":
                        block_number,

                    "old_sample_raw":
                        None,

                    "new_sample_raw":
                        recovered_sample,

                    "anchor_source_row_index":
                        anchor_source_row_index,

                    "temperature_C":
                        df.at[
                            idx,
                            "temperature_C",
                        ],

                    "classification_before_07i":
                        classification,

                    "recovery_rule":
                        (
                            "UNIQUE_SAMPLE_IN_STRICT_"
                            "TEMPERATURE_BLOCK"
                        ),

                    "raw_source_row":
                        df.at[
                            idx,
                            "raw_source_row",
                        ]
                        if "raw_source_row" in df.columns
                        else None,
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
            "block_number",
            "old_sample_raw",
            "new_sample_raw",
            "anchor_source_row_index",
            "temperature_C",
            "classification_before_07i",
            "recovery_rule",
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
    "Samples recovered:",
    len(audit_df),
)


if len(audit_df):

    print()

    print(
        "Recovered rows by table:"
    )

    print(
        audit_df[
            "table_id"
        ]
        .value_counts()
        .to_string()
    )


remaining = (
    (
        df["classification"]
        == "NEEDS_SAMPLE_INTERPRETATION"
    )
    & (
        df["sample_raw"]
        .apply(
            clean_text
        )
        .eq("")
    )
).sum()


print()

print(
    "Sample-interpretation rows still missing sample:",
    remaining,
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
