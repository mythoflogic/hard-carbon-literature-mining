from pathlib import Path
import pandas as pd
import re


ROWS_PATH = Path(
    "processed_tables/"
    "repaired_candidate_table_rows_v2.csv"
)

TABLES_PATH = Path(
    "processed_tables/"
    "normalized_candidate_tables.csv"
)

OUT_PATH = Path(
    "processed_tables/"
    "repaired_candidate_table_rows_v3.csv"
)

AUDIT_PATH = Path(
    "processed_tables/"
    "left_shifted_row_repair_audit.csv"
)


def first_number(value):

    if pd.isna(value):
        return None

    text = str(value)

    m = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )

    if not m:
        return None

    try:
        return float(m.group())
    except Exception:
        return None


rows = pd.read_csv(
    ROWS_PATH
)

tables = pd.read_csv(
    TABLES_PATH
)


# ------------------------------------------------------------
# Restrict repair to tables with an explicit temperature
# column and explicit C/H/N/O mapping.
# ------------------------------------------------------------

eligible_tables = tables[
    tables["temperature_column"].notna()
    &
    tables["C_column"].notna()
    &
    tables["H_column"].notna()
    &
    tables["N_column"].notna()
    &
    tables["O_column"].notna()
    &
    tables["classification"].eq(
        "LIKELY_ELEMENTAL_TABLE"
    )
].copy()


eligible_ids = set(
    eligible_tables["table_id"]
)


audit = []


for idx, row in rows.iterrows():

    if row["table_id"] not in eligible_ids:
        continue

    sample_num = first_number(
        row.get(
            "sample_candidate"
        )
    )

    temp_num = first_number(
        row.get(
            "temperature_candidate"
        )
    )


    # --------------------------------------------------------
    # Signature of a missing leading sample/group cell:
    #
    # parsed sample looks like temperature
    # parsed temperature looks like elemental %
    #
    # Do not infer the missing sample yet.
    # --------------------------------------------------------

    if not (
        sample_num is not None
        and 100 <= sample_num <= 3000
        and temp_num is not None
        and 0 <= temp_num <= 100
    ):
        continue


    old = {
        "sample_candidate":
            row.get("sample_candidate"),

        "temperature_candidate":
            row.get("temperature_candidate"),

        "C_candidate":
            row.get("C_candidate"),

        "H_candidate":
            row.get("H_candidate"),

        "N_candidate":
            row.get("N_candidate"),

        "O_candidate":
            row.get("O_candidate"),
    }


    # --------------------------------------------------------
    # Reparse directly from raw row.
    #
    # Since the leading sample/group cell is absent,
    # fields correspond to:
    #
    # temp | C | H | N | ... | O | ...
    #
    # We use the table's known column positions shifted by -1.
    # --------------------------------------------------------

    raw = str(
        row.get(
            "raw_source_row",
            ""
        )
    )

    cells = [
        x.strip()
        for x in raw.split("|")
    ]


    tinfo = eligible_tables[
        eligible_tables[
            "table_id"
        ].eq(
            row["table_id"]
        )
    ].iloc[0]


    original_positions = {
        "temperature_candidate":
            int(
                tinfo[
                    "temperature_column"
                ]
            ),

        "C_candidate":
            int(
                tinfo[
                    "C_column"
                ]
            ),

        "H_candidate":
            int(
                tinfo[
                    "H_column"
                ]
            ),

        "N_candidate":
            int(
                tinfo[
                    "N_column"
                ]
            ),

        "O_candidate":
            int(
                tinfo[
                    "O_column"
                ]
            ),
    }


    repaired = {}

    for col, pos in original_positions.items():

        shifted_pos = pos - 1

        if (
            shifted_pos >= 0
            and shifted_pos < len(cells)
        ):
            repaired[col] = (
                cells[shifted_pos]
            )

        else:
            repaired[col] = None


    rows.at[
        idx,
        "sample_candidate"
    ] = None

    for col, value in repaired.items():
        rows.at[
            idx,
            col
        ] = value


    audit.append(
        {
            "paper_id":
                row["paper_id"],

            "table_id":
                row["table_id"],

            "source_row_index":
                row["source_row_index"],

            "repair_type":
                "LEFT_SHIFT_MISSING_SAMPLE_CELL",

            "old_sample_candidate":
                old[
                    "sample_candidate"
                ],

            "new_sample_candidate":
                None,

            "old_temperature_candidate":
                old[
                    "temperature_candidate"
                ],

            "new_temperature_candidate":
                repaired[
                    "temperature_candidate"
                ],

            "old_C_candidate":
                old[
                    "C_candidate"
                ],

            "new_C_candidate":
                repaired[
                    "C_candidate"
                ],

            "old_H_candidate":
                old[
                    "H_candidate"
                ],

            "new_H_candidate":
                repaired[
                    "H_candidate"
                ],

            "old_N_candidate":
                old[
                    "N_candidate"
                ],

            "new_N_candidate":
                repaired[
                    "N_candidate"
                ],

            "old_O_candidate":
                old[
                    "O_candidate"
                ],

            "new_O_candidate":
                repaired[
                    "O_candidate"
                ],

            "raw_source_row":
                raw,
        }
    )


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


print("=" * 72)
print("STEP 08F1 — LEFT-SHIFTED ELEMENTAL ROW REPAIR")
print("=" * 72)

print(
    "\nEligible elemental tables:",
    len(eligible_ids),
)

print(
    "Rows repaired:",
    len(audit_df),
)

if len(audit_df):

    print(
        "Tables repaired:",
        audit_df[
            "table_id"
        ].nunique(),
    )

    print(
        "Papers repaired:",
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

    print(
        "\nFIRST 20 REPAIRS:\n"
    )

    print(
        audit_df[
            [
                "table_id",
                "source_row_index",
                "old_sample_candidate",
                "new_temperature_candidate",
                "new_C_candidate",
                "new_H_candidate",
                "new_N_candidate",
                "new_O_candidate",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


print("\nSaved:")
print("-", OUT_PATH)
print("-", AUDIT_PATH)

