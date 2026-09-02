import json
import re
from pathlib import Path

import pandas as pd


PROJECT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ROWS_PATH = (
    PROJECT
    / "processed_tables"
    / "classified_extractable_rows_enriched.csv"
)

TABLES_PATH = (
    PROJECT
    / "processed_tables"
    / "normalized_candidate_tables.csv"
)

REPAIRED_TABLES_PATH = (
    PROJECT
    / "processed_tables"
    / "repaired_candidate_tables.csv"
)

OUT_PATH = (
    PROJECT
    / "processed_tables"
    / "classified_extractable_rows_role_enriched.csv"
)

AUDIT_PATH = (
    PROJECT
    / "processed_tables"
    / "caption_component_role_assignment_audit.csv"
)


def clean_text(x):

    if x is None or pd.isna(x):
        return ""

    x = re.sub(
        r"<[^>]+>",
        " ",
        str(x),
    )

    x = re.sub(
        r"\*+",
        "",
        x,
    )

    x = re.sub(
        r"\s+",
        " ",
        x,
    )

    return x.strip()


def is_true(value):

    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def split_components(x):

    x = clean_text(
        x
    )

    if not x:
        return []

    return [
        clean_text(part)
        for part in re.split(
            r"\s*[-/]\s*",
            x,
        )
        if clean_text(part)
    ]


# ============================================================
# LOAD
# ============================================================

rows = pd.read_csv(
    ROWS_PATH
)

tables = pd.read_csv(
    TABLES_PATH
)

repaired_tables = pd.read_csv(
    REPAIRED_TABLES_PATH
)

if repaired_tables[
    "table_id"
].duplicated().any():
    raise RuntimeError(
        "Duplicate table_id values in "
        "repaired_candidate_tables.csv"
    )

repaired_sample_by_table = (
    repaired_tables
    .set_index("table_id")[
        "sample_column"
    ]
    .to_dict()
)

# Keep the original 07D value for provenance, but use the
# authoritative 07D3 value whenever that table exists there.
tables[
    "sample_column_07d"
] = tables[
    "sample_column"
]

tables[
    "sample_column_source_for_roles"
] = "07D"

matched_07d3 = tables[
    "table_id"
].isin(
    repaired_sample_by_table
)

tables.loc[
    matched_07d3,
    "sample_column",
] = tables.loc[
    matched_07d3,
    "table_id",
].map(
    repaired_sample_by_table
)

tables.loc[
    matched_07d3,
    "sample_column_source_for_roles",
] = "07D3"


# ============================================================
# EXTRACT ACTUAL SAMPLE HEADER CELL
# ============================================================

sample_header_by_table = {}

for _, row in tables.iterrows():

    if pd.isna(
        row.get(
            "sample_column"
        )
    ):
        continue

    cells = [
        str(cell).strip()
        for cell in str(
            row.get(
                "header_row",
                ""
            )
        ).split("|")
    ]

    try:
        index = int(
            row["sample_column"]
        )
    except Exception:
        continue

    if not (
        0 <= index < len(cells)
    ):
        continue

    sample_header_by_table[
        row["table_id"]
    ] = cells[index]


# ============================================================
# OUTPUT COLUMNS
# ============================================================

rows["sample_role_1"] = None
rows["sample_role_1_code"] = None
rows["sample_role_1_value"] = None

rows["sample_role_2"] = None
rows["sample_role_2_code"] = None
rows["sample_role_2_value"] = None

rows[
    "structural_role_assignment"
] = False


audit = []


# ============================================================
# ASSIGN ROLES ONLY WHEN STRUCTURE IS EXPLICIT
# ============================================================

for idx, row in rows.iterrows():

    if row.get(
        "classification"
    ) == "NOT_CHNO_ROW":
        continue

    if not is_true(
        row.get(
            "caption_abbreviation_resolution"
        )
    ):
        continue

    table_id = row.get(
        "table_id"
    )

    sample_header = (
        sample_header_by_table.get(
            table_id,
            ""
        )
    )

    header_parts = (
        split_components(
            sample_header
        )
    )

    # Must genuinely be a compound sample heading.
    if len(header_parts) < 2:
        continue

    sample_codes = [
        clean_text(x)
        for x in str(
            row.get(
                "sample_component_codes",
                ""
            )
        ).split("|")
        if clean_text(x)
        and clean_text(x).lower() != "nan"
    ]

    expansions = [
        clean_text(x)
        for x in str(
            row.get(
                "sample_component_expansions",
                ""
            )
        ).split("|")
        if clean_text(x)
        and clean_text(x).lower() != "nan"
    ]

    # Require exact structural correspondence.
    #
    # feedstock-method
    # CC-Slow
    #
    # => 2 header roles, 2 source components,
    #    2 caption expansions.

    # Require source-defined components to align positionally
    # with the LEFT side of the compound header.
    #
    # Complete:
    #   feedstock-method
    #   CC-Slow
    #
    # Partial:
    #   feedstock-method
    #   MS
    #
    # In the partial case we assign only the observed leading
    # role and do not infer the missing trailing role.
    if not (
        len(sample_codes)
        == len(expansions)
        and 1 <= len(sample_codes) <= len(header_parts)
    ):
        continue

    rows.at[
        idx,
        "sample_role_1"
    ] = header_parts[0]

    rows.at[
        idx,
        "sample_role_1_code"
    ] = sample_codes[0]

    rows.at[
        idx,
        "sample_role_1_value"
    ] = expansions[0]

    if len(sample_codes) >= 2:

        rows.at[
            idx,
            "sample_role_2"
        ] = header_parts[1]

        rows.at[
            idx,
            "sample_role_2_code"
        ] = sample_codes[1]

        rows.at[
            idx,
            "sample_role_2_value"
        ] = expansions[1]

        assignment_type = (
            "POSITIONAL_COMPOUND_HEADER_MATCH"
        )

    else:

        assignment_type = (
            "PARTIAL_POSITIONAL_COMPOUND_HEADER_MATCH"
        )

    rows.at[
        idx,
        "structural_role_assignment"
    ] = True

    audit.append(
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

            "sample_raw":
                row.get(
                    "sample_raw"
                ),

            "sample_header":
                clean_text(
                    sample_header
                ),

            "role_1":
                header_parts[0],

            "role_1_code":
                sample_codes[0],

            "role_1_value":
                expansions[0],

            "role_2":
                (
                    header_parts[1]
                    if len(sample_codes) >= 2
                    else None
                ),

            "role_2_code":
                (
                    sample_codes[1]
                    if len(sample_codes) >= 2
                    else None
                ),

            "role_2_value":
                (
                    expansions[1]
                    if len(sample_codes) >= 2
                    else None
                ),

            "assignment_type":
                assignment_type,
        }
    )


audit_df = pd.DataFrame(
    audit
)

if (
    audit_df.empty
    and len(
        audit_df.columns
    ) == 0
):
    audit_df = pd.DataFrame(
        columns=[
            "paper_id",
            "table_id",
            "source_row_index",
            "sample_raw",
            "sample_header",
            "role_1",
            "role_1_code",
            "role_1_value",
            "role_2",
            "role_2_code",
            "role_2_value",
            "assignment_type",
        ]
    )


# ============================================================
# SAVE
# ============================================================

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
print("STEP 07F2 — STRUCTURAL COMPONENT ROLE ASSIGNMENT")
print("=" * 72)

print(
    "\nRows assigned roles:",
    len(audit_df),
)

print(
    "Tables affected:",
    (
        audit_df[
            "table_id"
        ].nunique()
        if len(audit_df)
        else 0
    ),
)

print(
    "Papers affected:",
    (
        audit_df[
            "paper_id"
        ].nunique()
        if len(audit_df)
        else 0
    ),
)

if len(audit_df):

    print("\nROLE COMBINATIONS:")

    print(
        audit_df.groupby(
            [
                "role_1",
                "role_2",
            ]
        )
        .size()
        .to_string()
    )

    print("\nFIRST 30 ASSIGNMENTS:\n")

    print(
        audit_df[
            [
                "source_row_index",
                "sample_raw",
                "sample_header",
                "role_1",
                "role_1_code",
                "role_1_value",
                "role_2",
                "role_2_code",
                "role_2_value",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

print("\nSaved:")
print("-", OUT_PATH)
print("-", AUDIT_PATH)
