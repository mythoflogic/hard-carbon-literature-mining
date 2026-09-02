#!/usr/bin/env python3

"""
STEP 08A
========

Build Semantic Recovery Packages

Purpose
-------
Create compact evidence packages for rows that could not be resolved
deterministically in Steps 07D–07G.

Each package contains:
- the ambiguous source row;
- the full source table;
- detected header information;
- table caption;
- preceding and following context;
- current parsed CHNO values;
- the specific reason semantic recovery is required.

No LLM is used in this step.
No scientific values are modified.
"""

from pathlib import Path

import json

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

recovery_queue_path = (
    processed_tables_dir
    / "semantic_recovery_queue_structural.csv"
)

normalized_tables_path = (
    processed_tables_dir
    / "repaired_candidate_tables.jsonl"
)

classified_tables_path = (
    processed_tables_dir
    / "classified_elemental_tables.jsonl"
)

output_jsonl_path = (
    processed_tables_dir
    / "semantic_recovery_packages.jsonl"
)

output_csv_path = (
    processed_tables_dir
    / "semantic_recovery_packages.csv"
)

output_xlsx_path = (
    outputs_dir
    / "semantic_recovery_packages.xlsx"
)

summary_path = (
    processed_tables_dir
    / "semantic_recovery_package_summary.csv"
)


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


def clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return value


# ============================================================
# Load recovery queue
# ============================================================

if not recovery_queue_path.exists():
    raise FileNotFoundError(
        f"Recovery queue not found: "
        f"{recovery_queue_path}"
    )

recovery_df = pd.read_csv(
    recovery_queue_path
)

print(
    "Recovery rows loaded:",
    len(recovery_df),
)


# ============================================================
# Load normalized table records
# ============================================================

normalized_records = load_jsonl(
    normalized_tables_path
)

normalized_lookup = {
    str(record["table_id"]): record
    for record in normalized_records
}

print(
    "Normalized tables loaded:",
    len(normalized_lookup),
)


# ============================================================
# Load richer Step 07C records
# ============================================================

classified_records = load_jsonl(
    classified_tables_path
)

classified_lookup = {
    str(record["table_id"]): record
    for record in classified_records
}

print(
    "Classified table records loaded:",
    len(classified_lookup),
)


# ============================================================
# Build semantic recovery packages
# ============================================================

packages = []

missing_table_metadata = 0

for _, row in recovery_df.iterrows():

    recovery_id = str(
        row["recovery_id"]
    )

    paper_id = str(
        row["paper_id"]
    )

    table_id = str(
        row["table_id"]
    )

    normalized_record = (
        normalized_lookup.get(
            table_id
        )
    )

    classified_record = (
        classified_lookup.get(
            table_id
        )
    )

    if (
        normalized_record is None
        and classified_record is None
    ):
        missing_table_metadata += 1

    # Prefer normalized structure when available.
    header_row = []

    if normalized_record is not None:
        header_row = (
            normalized_record.get(
                "combined_header",
		normalized_record.get(
		    "header_row",
                    [],
		),
            )
            or []
        )

    raw_table_text = ""
    caption = ""
    preceding_context = ""
    following_context = ""

    if classified_record is not None:

        raw_table_text = (
            classified_record.get(
                "raw_table_text",
                ""
            )
            or ""
        )

        caption = (
            classified_record.get(
                "caption",
                ""
            )
            or ""
        )

        preceding_context = (
            classified_record.get(
                "preceding_context",
                ""
            )
            or ""
        )

        following_context = (
            classified_record.get(
                "following_context",
                ""
            )
            or ""
        )

    package = {
        "recovery_id": recovery_id,
        "source_key": clean_value(
            row.get(
                "source_key"
            )
        ),
        "paper_id": paper_id,
        "table_id": table_id,

        "source_row_index": clean_value(
            row.get(
                "source_row_index"
            )
        ),

        "semantic_class": clean_value(
            row.get(
                "semantic_class"
            )
        ),

        "original_classification": clean_value(
            row.get(
                "original_classification"
            )
        ),

        "sample_raw": clean_value(
            row.get(
                "sample_raw"
            )
        ),
        "sample_component_codes": clean_value(
            row.get(
                "sample_component_codes"
            )
        ),

        "sample_component_expansions": clean_value(
            row.get(
                "sample_component_expansions"
            )
        ),

        "caption_abbreviation_resolution": clean_value(
            row.get(
                "caption_abbreviation_resolution"
            )
        ),

        "sample_role_1": clean_value(
            row.get(
                "sample_role_1"
            )
        ),

        "sample_role_1_code": clean_value(
            row.get(
                "sample_role_1_code"
            )
        ),

        "sample_role_1_value": clean_value(
            row.get(
                "sample_role_1_value"
            )
        ),

        "sample_role_2": clean_value(
            row.get(
                "sample_role_2"
            )
        ),

        "sample_role_2_code": clean_value(
            row.get(
                "sample_role_2_code"
            )
        ),

        "sample_role_2_value": clean_value(
            row.get(
                "sample_role_2_value"
            )
        ),

        "structural_role_assignment": clean_value(
            row.get(
                "structural_role_assignment"
            )
        ),
        "temperature_original": clean_value(
            row.get(
                "temperature_original"
            )
        ),

        "temperature_type": clean_value(
            row.get(
                "temperature_type"
            )
        ),

        "temperature_exact_C": clean_value(
            row.get(
                "temperature_exact_C"
            )
        ),

        "temperature_low_C": clean_value(
            row.get(
                "temperature_low_C"
            )
        ),

        "temperature_high_C": clean_value(
            row.get(
                "temperature_high_C"
            )
        ),

        "C_value": clean_value(
            row.get("C_value")
        ),

        "H_value": clean_value(
            row.get("H_value")
        ),

        "N_value": clean_value(
            row.get("N_value")
        ),

        "O_value": clean_value(
            row.get("O_value")
        ),

        "raw_source_row": clean_value(
            row.get(
                "raw_source_row"
            )
        ),

        "header_row": header_row,

        "caption": caption,

        "preceding_context": (
            preceding_context
        ),

        "following_context": (
            following_context
        ),

        "raw_table_text": (
            raw_table_text
        ),

        "provenance_status": clean_value(
            row.get(
                "provenance_status"
            )
        ),
    }

    packages.append(
        package
    )


# ============================================================
# Save JSONL
# ============================================================

with output_jsonl_path.open(
    "w",
    encoding="utf-8",
) as output_file:

    for package in packages:

        output_file.write(
            json.dumps(
                package,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Flat report
# ============================================================

flat_rows = []

for package in packages:

    flat_rows.append(
        {
            "recovery_id": (
                package[
                    "recovery_id"
                ]
            ),
            "source_key": (
                package[
                    "source_key"
                ]
            ),
            "paper_id": (
                package[
                    "paper_id"
                ]
            ),
            "table_id": (
                package[
                    "table_id"
                ]
            ),
            "semantic_class": (
                package[
                    "semantic_class"
                ]
            ),
            "source_row_index": (
                package[
                    "source_row_index"
                ]
            ),
            "sample_raw": (
                package[
                    "sample_raw"
                ]
            ),
            "sample_component_codes": (
                package[
                    "sample_component_codes"
                ]
            ),

            "sample_component_expansions": (
                package[
                    "sample_component_expansions"
                ]
            ),

            "caption_abbreviation_resolution": (
                package[
                    "caption_abbreviation_resolution"
                ]
            ),

            "sample_role_1": (
                package[
                    "sample_role_1"
                ]
            ),

            "sample_role_1_code": (
                package[
                    "sample_role_1_code"
                ]
            ),

            "sample_role_1_value": (
                package[
                    "sample_role_1_value"
                ]
            ),

            "sample_role_2": (
                package[
                    "sample_role_2"
                ]
            ),

            "sample_role_2_code": (
                package[
                    "sample_role_2_code"
                ]
            ),

            "sample_role_2_value": (
                package[
                    "sample_role_2_value"
                ]
            ),

            "structural_role_assignment": (
                package[
                    "structural_role_assignment"
                ]
            ),
            "temperature_original": (
                package[
                    "temperature_original"
                ]
            ),
            "C_value": (
                package[
                    "C_value"
                ]
            ),
            "H_value": (
                package[
                    "H_value"
                ]
            ),
            "N_value": (
                package[
                    "N_value"
                ]
            ),
            "O_value": (
                package[
                    "O_value"
                ]
            ),
            "header_row": (
                " | ".join(
                    str(value)
                    for value in package[
                        "header_row"
                    ]
                )
            ),
            "caption": (
                package[
                    "caption"
                ]
            ),
            "preceding_context": (
                package[
                    "preceding_context"
                ]
            ),
            "following_context": (
                package[
                    "following_context"
                ]
            ),
            "raw_source_row": (
                package[
                    "raw_source_row"
                ]
            ),
            "table_character_count": (
                len(
                    package[
                        "raw_table_text"
                    ]
                )
            ),
        }
    )

flat_df = pd.DataFrame(
    flat_rows
)

flat_df.to_csv(
    output_csv_path,
    index=False,
)

flat_df.to_excel(
    output_xlsx_path,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary_df = (
    flat_df[
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
# Audit
# ============================================================

packages_with_table = sum(
    bool(
        package[
            "raw_table_text"
        ]
    )
    for package in packages
)

packages_with_header = sum(
    bool(
        package[
            "header_row"
        ]
    )
    for package in packages
)

packages_with_caption = sum(
    bool(
        package[
            "caption"
        ]
    )
    for package in packages
)

packages_with_context = sum(
    bool(
        package[
            "preceding_context"
        ]
        or package[
            "following_context"
        ]
    )
    for package in packages
)


print()
print("=" * 70)
print("STEP 08A — SEMANTIC RECOVERY PACKAGES")
print("=" * 70)
print()

print(
    "Recovery packages:",
    len(packages),
)

print(
    "Packages with complete table:",
    packages_with_table,
)

print(
    "Packages with detected header:",
    packages_with_header,
)

print(
    "Packages with caption:",
    packages_with_caption,
)

print(
    "Packages with nearby context:",
    packages_with_context,
)

print(
    "Missing table metadata:",
    missing_table_metadata,
)

print()
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Generated files:")
print("-", output_jsonl_path)
print("-", output_csv_path)
print("-", output_xlsx_path)
print("-", summary_path)

