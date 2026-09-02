from pathlib import Path
import re
import pandas as pd


ROWS_PATH = Path(
    "processed_tables/"
    "repaired_candidate_table_rows_v3.csv"
)

TABLES_PATH = Path(
    "processed_tables/"
    "normalized_candidate_tables.csv"
)

OUT_PATH = Path(
    "processed_tables/"
    "repaired_candidate_table_rows_v4.csv"
)

AUDIT_PATH = Path(
    "processed_tables/"
    "grouped_sample_block_repair_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def first_number(value):

    if pd.isna(value):
        return None

    m = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        str(value),
    )

    if not m:
        return None

    try:
        return float(m.group())
    except Exception:
        return None


def valid_text_sample(value):

    if pd.isna(value):
        return False

    text = str(value).strip()

    if not text:
        return False

    # A pure number is not a sample label.
    if re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    ):
        return False

    return True


# ============================================================
# LOAD
# ============================================================

rows = pd.read_csv(
    ROWS_PATH
)

tables = pd.read_csv(
    TABLES_PATH
)


# Same conservative table eligibility as 08F1.
eligible = tables[
    tables["classification"].eq(
        "LIKELY_ELEMENTAL_TABLE"
    )
    &
    tables["temperature_column"].notna()
    &
    tables["C_column"].notna()
    &
    tables["H_column"].notna()
    &
    tables["N_column"].notna()
    &
    tables["O_column"].notna()
].copy()


eligible_ids = set(
    eligible["table_id"]
)


audit = []


# ============================================================
# PROCESS EACH TABLE IN SOURCE-ROW ORDER
# ============================================================

for table_id in sorted(eligible_ids):

    idxs = rows.index[
        rows["table_id"].eq(
            table_id
        )
    ].tolist()

    if not idxs:
        continue


    # Sort by actual source order.
    idxs = sorted(
        idxs,
        key=lambda i:
            rows.at[
                i,
                "source_row_index"
            ],
    )


    # --------------------------------------------------------
    # Build contiguous temperature blocks.
    #
    # A new block begins when temperature decreases or resets.
    #
    # Example:
    #
    # 500, 550, 600, 500, 550, 600
    #
    # becomes:
    #
    # [500,550,600]
    # [500,550,600]
    # --------------------------------------------------------

    blocks = []
    current = []

    previous_temp = None


    for idx in idxs:

        temp = first_number(
            rows.at[
                idx,
                "temperature_candidate"
            ]
        )

        # Rows without interpretable temperature are
        # not used for structural block reconstruction.
        if temp is None:

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
            and temp <= previous_temp
        ):

            blocks.append(
                current
            )

            current = []


        current.append(
            idx
        )

        previous_temp = temp


    if current:
        blocks.append(
            current
        )


    # ========================================================
    # CONSERVATIVE BLOCK FILL
    # ========================================================

    for block_number, block in enumerate(
        blocks,
        start=1,
    ):

        if len(block) < 2:
            continue


        anchors = []

        for idx in block:

            sample = rows.at[
                idx,
                "sample_candidate"
            ]

            if valid_text_sample(
                sample
            ):
                anchors.append(
                    str(sample).strip()
                )


        unique_anchors = list(
            dict.fromkeys(
                anchors
            )
        )


        # ----------------------------------------------------
        # Safety condition:
        #
        # Exactly ONE explicit sample identity must occur
        # anywhere in the temperature block.
        #
        # If zero or >1 identities occur, do nothing.
        # ----------------------------------------------------

        if len(unique_anchors) != 1:
            continue


        anchor = unique_anchors[0]


        # Need at least one actually missing label.
        missing = [
            idx
            for idx in block
            if not valid_text_sample(
                rows.at[
                    idx,
                    "sample_candidate"
                ]
            )
        ]


        if not missing:
            continue


        # ----------------------------------------------------
        # Additional safety:
        #
        # Temperatures in a block must be unique.
        # We don't want to collapse repeated-condition rows.
        # ----------------------------------------------------

        temps = [
            first_number(
                rows.at[
                    idx,
                    "temperature_candidate"
                ]
            )
            for idx in block
        ]


        if len(set(temps)) != len(temps):
            continue


        # ----------------------------------------------------
        # Fill only the missing sample cells.
        # ----------------------------------------------------

        block_temps = " | ".join(
            str(x)
            for x in temps
        )


        for idx in missing:

            old_sample = rows.at[
                idx,
                "sample_candidate"
            ]

            rows.at[
                idx,
                "sample_candidate"
            ] = anchor


            audit.append(
                {
                    "paper_id":
                        rows.at[
                            idx,
                            "paper_id"
                        ],

                    "table_id":
                        table_id,

                    "source_row_index":
                        rows.at[
                            idx,
                            "source_row_index"
                        ],

                    "block_number":
                        block_number,

                    "block_temperatures":
                        block_temps,

                    "explicit_anchor":
                        anchor,

                    "old_sample_candidate":
                        old_sample,

                    "new_sample_candidate":
                        anchor,

                    "temperature_candidate":
                        rows.at[
                            idx,
                            "temperature_candidate"
                        ],

                    "repair_type":
                        "GROUPED_SAMPLE_BLOCK_RECONSTRUCTION",

                    "raw_source_row":
                        rows.at[
                            idx,
                            "raw_source_row"
                        ],
                }
            )


# ============================================================
# SAVE
# ============================================================

audit_df = pd.DataFrame(
    audit
)


rows.to_csv(
    OUT_PATH,
    index=False,
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 72)
print("STEP 08F2 — GROUPED-SAMPLE BLOCK RECONSTRUCTION")
print("=" * 72)

print(
    "\nRows assigned a grouped sample:",
    len(audit_df),
)


if len(audit_df):

    print(
        "Tables affected:",
        audit_df[
            "table_id"
        ].nunique(),
    )

    print(
        "Papers affected:",
        audit_df[
            "paper_id"
        ].nunique(),
    )


    print("\nBY TABLE:")

    print(
        audit_df.groupby(
            "table_id"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


    print("\nREPAIRS:\n")

    print(
        audit_df[
            [
                "table_id",
                "source_row_index",
                "block_temperatures",
                "explicit_anchor",
                "temperature_candidate",
                "new_sample_candidate",
            ]
        ]
        .to_string(
            index=False
        )
    )


print("\nSaved:")
print("-", OUT_PATH)
print("-", AUDIT_PATH)

