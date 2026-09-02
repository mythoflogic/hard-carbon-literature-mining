#!/usr/bin/env python3

"""
STEP 07B
========

Complete Table Inventory

Purpose
-------
Scan every Markdown paper and preserve complete Markdown tables
as standalone structured objects.

This step does NOT:
- extract C/H/N/O values;
- use an LLM;
- modify the existing chunking pipeline;
- modify the manual benchmark.

Outputs will be used by the new table-aware extraction branch.
"""

from pathlib import Path

import json
import re

import pandas as pd


# ============================================================
# Project paths
# ============================================================

project_dir = (
    Path.home()
    / "Scratch"
    / "hardcarbon_project"
)

markdown_dir = (
    project_dir
    / "papers_markdown"
)

processed_tables_dir = (
    project_dir
    / "processed_tables"
)

outputs_dir = (
    project_dir
    / "outputs"
)

processed_tables_dir.mkdir(
    parents=True,
    exist_ok=True,
)

outputs_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

tables_jsonl_path = (
    processed_tables_dir
    / "all_tables.jsonl"
)

table_inventory_csv = (
    processed_tables_dir
    / "table_inventory.csv"
)

table_inventory_xlsx = (
    outputs_dir
    / "table_inventory.xlsx"
)

paper_summary_csv = (
    processed_tables_dir
    / "table_inventory_by_paper.csv"
)

paper_summary_xlsx = (
    outputs_dir
    / "table_inventory_by_paper.xlsx"
)

# ============================================================
# Table-detection helpers
# ============================================================

def is_markdown_table_row(line):
    """
    Return True when a line looks like part of a Markdown table.
    """

    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith("```"):
        return False

    return (
        stripped.startswith("|")
        and stripped.count("|") >= 2
    )


def split_markdown_row(line):
    """
    Convert a Markdown table line into cells.
    """
    stripped = line.strip()

    if not is_markdown_table_row(
        stripped
    ):
        return []

    # Remove exactly one Markdown boundary pipe from each side.
    # Preserve empty positional cells inside the table.
    if stripped.startswith("|"):
        stripped = stripped[1:]

    if stripped.endswith("|"):
        stripped = stripped[:-1]

    return [
        cell.strip()
        for cell in stripped.split("|")
    
]

def is_separator_row(cells):
    """
    Detect Markdown separator rows such as:

    |---|---|---|
    |:---|---:|---|
    """

    if not cells:
        return False

    meaningful = [
        cell.strip()
        for cell in cells
        if cell.strip()
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


def looks_like_table_caption(line):
    """
    Detect likely captions immediately before a table.
    """

    stripped = line.strip()

    if not stripped:
        return False

    return bool(
        re.match(
            r"^(table|tab\.)\s*\d+",
            stripped,
            flags=re.IGNORECASE,
        )
    )

# ============================================================
# Extract complete tables from one Markdown document
# ============================================================

def extract_tables_from_markdown(
    markdown_text,
):
    """
    Find complete Markdown table blocks.

    Returns one dictionary per detected table.
    """

    lines = markdown_text.splitlines()

    tables = []

    current_table_lines = []
    current_start_line = None

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        if is_markdown_table_row(line):

            if not current_table_lines:
                current_start_line = (
                    line_number
                )

            current_table_lines.append(
                line
            )

        else:

            if current_table_lines:

                tables.append(
                    {
                        "start_line": (
                            current_start_line
                        ),
                        "end_line": (
                            current_start_line
                            + len(
                                current_table_lines
                            )
                            - 1
                        ),
                        "raw_lines": (
                            current_table_lines.copy()
                        ),
                    }
                )

                current_table_lines = []
                current_start_line = None

    if current_table_lines:

        tables.append(
            {
                "start_line": (
                    current_start_line
                ),
                "end_line": (
                    current_start_line
                    + len(
                        current_table_lines
                    )
                    - 1
                ),
                "raw_lines": (
                    current_table_lines.copy()
                ),
            }
        )

    return tables

# ============================================================
# Table context and structure helpers
# ============================================================

def get_nonempty_line_before(
    lines,
    start_index,
):
    """
    Return the closest non-empty line before start_index.
    """

    for index in range(
        start_index - 1,
        -1,
        -1,
    ):
        value = lines[index].strip()

        if value:
            return value

    return ""


def get_context_before(
    lines,
    start_index,
    max_lines=3,
):
    """
    Collect up to max_lines non-empty lines before a table.
    """

    collected = []

    index = start_index - 1

    while (
        index >= 0
        and len(collected) < max_lines
    ):
        value = lines[index].strip()

        if value:
            collected.append(
                value
            )

        index -= 1

    collected.reverse()

    return "\n".join(
        collected
    )


def get_context_after(
    lines,
    end_index,
    max_lines=3,
):
    """
    Collect up to max_lines non-empty lines after a table.
    """

    collected = []

    index = end_index + 1

    while (
        index < len(lines)
        and len(collected) < max_lines
    ):
        value = lines[index].strip()

        if value:
            collected.append(
                value
            )

        index += 1

    return "\n".join(
        collected
    )


def parse_table_rows(
    raw_lines,
):
    """
    Convert raw Markdown table lines into parsed cell rows.
    """

    parsed_rows = []

    for line in raw_lines:
        cells = split_markdown_row(
            line
        )

        if not cells:
            continue

        parsed_rows.append(
            cells
        )

    return parsed_rows


def calculate_table_dimensions(
    parsed_rows,
):
    """
    Return row and column counts.

    Separator rows are excluded from the data-row count.
    """

    if not parsed_rows:
        return 0, 0, 0

    column_count = max(
        len(row)
        for row in parsed_rows
    )

    separator_count = sum(
        is_separator_row(row)
        for row in parsed_rows
    )

    non_separator_rows = (
        len(parsed_rows)
        - separator_count
    )

    return (
        len(parsed_rows),
        non_separator_rows,
        column_count,
    )


def identify_caption(
    lines,
    table_start_line,
):
    """
    Identify a likely table caption immediately above a table.

    table_start_line uses one-based indexing.
    """

    start_index = (
        table_start_line - 1
    )

    previous_line = (
        get_nonempty_line_before(
            lines,
            start_index,
        )
    )

    if looks_like_table_caption(
        previous_line
    ):
        return previous_line

    return ""


# ============================================================
# Process all Markdown papers
# ============================================================

if not markdown_dir.exists():
    raise FileNotFoundError(
        f"Markdown directory not found: "
        f"{markdown_dir}"
    )

markdown_files = sorted(
    markdown_dir.glob("*.md")
)

if not markdown_files:
    raise FileNotFoundError(
        f"No Markdown files found in: "
        f"{markdown_dir}"
    )

print(
    "Markdown papers found:",
    len(markdown_files),
)

table_records = []
paper_records = []

global_table_number = 0

for paper_number, markdown_path in enumerate(
    markdown_files,
    start=1,
):
    markdown_text = (
        markdown_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    lines = markdown_text.splitlines()

    paper_id = (
        markdown_path.stem
    )

    detected_tables = (
        extract_tables_from_markdown(
            markdown_text
        )
    )

    for paper_table_number, table in enumerate(
        detected_tables,
        start=1,
    ):
        global_table_number += 1

        raw_lines = table[
            "raw_lines"
        ]

        raw_table_text = "\n".join(
            raw_lines
        )

        parsed_rows = parse_table_rows(
            raw_lines
        )

        (
            total_parsed_rows,
            non_separator_rows,
            column_count,
        ) = calculate_table_dimensions(
            parsed_rows
        )

        caption = identify_caption(
            lines,
            table["start_line"],
        )

        preceding_context = (
            get_context_before(
                lines,
                table["start_line"] - 1,
                max_lines=3,
            )
        )

        following_context = (
            get_context_after(
                lines,
                table["end_line"] - 1,
                max_lines=3,
            )
        )

        table_id = (
            f"{paper_id}_T"
            f"{paper_table_number:03d}"
        )

        table_records.append(
            {
                "global_table_number": (
                    global_table_number
                ),
                "paper_number": (
                    paper_number
                ),
                "paper_id": paper_id,
                "source_filename": (
                    markdown_path.name
                ),
                "table_id": table_id,
                "paper_table_number": (
                    paper_table_number
                ),
                "start_line": (
                    table["start_line"]
                ),
                "end_line": (
                    table["end_line"]
                ),
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
                "parsed_rows": (
                    parsed_rows
                ),
                "total_parsed_rows": (
                    total_parsed_rows
                ),
                "non_separator_rows": (
                    non_separator_rows
                ),
                "column_count": (
                    column_count
                ),
                "character_count": (
                    len(raw_table_text)
                ),
            }
        )

    paper_records.append(
        {
            "paper_number": (
                paper_number
            ),
            "paper_id": paper_id,
            "source_filename": (
                markdown_path.name
            ),
            "markdown_characters": (
                len(markdown_text)
            ),
            "tables_detected": (
                len(detected_tables)
            ),
        }
    )

    print(
        f"{paper_number}/"
        f"{len(markdown_files)} "
        f"{markdown_path.name}: "
        f"{len(detected_tables)} tables"
    )

# ============================================================
# Save complete table objects
# ============================================================

with tables_jsonl_path.open(
    "w",
    encoding="utf-8",
) as jsonl_file:

    for record in table_records:
        jsonl_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Build table inventory
# ============================================================

inventory_rows = []

for record in table_records:
    inventory_rows.append(
        {
            "global_table_number": (
                record[
                    "global_table_number"
                ]
            ),
            "paper_number": (
                record[
                    "paper_number"
                ]
            ),
            "paper_id": (
                record[
                    "paper_id"
                ]
            ),
            "source_filename": (
                record[
                    "source_filename"
                ]
            ),
            "table_id": (
                record[
                    "table_id"
                ]
            ),
            "paper_table_number": (
                record[
                    "paper_table_number"
                ]
            ),
            "start_line": (
                record[
                    "start_line"
                ]
            ),
            "end_line": (
                record[
                    "end_line"
                ]
            ),
            "caption": (
                record[
                    "caption"
                ]
            ),
            "total_parsed_rows": (
                record[
                    "total_parsed_rows"
                ]
            ),
            "non_separator_rows": (
                record[
                    "non_separator_rows"
                ]
            ),
            "column_count": (
                record[
                    "column_count"
                ]
            ),
            "character_count": (
                record[
                    "character_count"
                ]
            ),
            "raw_table_preview": (
                record[
                    "raw_table_text"
                ][:500]
            ),
        }
    )

inventory_df = pd.DataFrame(
    inventory_rows
)

inventory_df.to_csv(
    table_inventory_csv,
    index=False,
)

inventory_df.to_excel(
    table_inventory_xlsx,
    index=False,
)


# ============================================================
# Build paper-level summary
# ============================================================

paper_df = pd.DataFrame(
    paper_records
)

paper_df.to_csv(
    paper_summary_csv,
    index=False,
)

paper_df.to_excel(
    paper_summary_xlsx,
    index=False,
)


# ============================================================
# Quality checks
# ============================================================

total_tables = len(
    table_records
)

papers_with_tables = int(
    paper_df[
        "tables_detected"
    ]
    .gt(0)
    .sum()
)

papers_without_tables = int(
    paper_df[
        "tables_detected"
    ]
    .eq(0)
    .sum()
)

empty_tables = 0
single_row_tables = 0
zero_column_tables = 0
tables_without_caption = 0

if not inventory_df.empty:

    empty_tables = int(
        inventory_df[
            "character_count"
        ]
        .eq(0)
        .sum()
    )

    single_row_tables = int(
        inventory_df[
            "non_separator_rows"
        ]
        .le(1)
        .sum()
    )

    zero_column_tables = int(
        inventory_df[
            "column_count"
        ]
        .eq(0)
        .sum()
    )

    tables_without_caption = int(
        inventory_df[
            "caption"
        ]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("STEP 07B — COMPLETE TABLE INVENTORY")
print("=" * 70)
print()

print(
    "Markdown papers:",
    len(markdown_files),
)

print(
    "Tables detected:",
    total_tables,
)

print(
    "Papers with at least one table:",
    papers_with_tables,
)

print(
    "Papers with no detected tables:",
    papers_without_tables,
)

print(
    "Empty tables:",
    empty_tables,
)

print(
    "Tables with <= 1 non-separator row:",
    single_row_tables,
)

print(
    "Tables with zero columns:",
    zero_column_tables,
)

print(
    "Tables without detected caption:",
    tables_without_caption,
)

print()
print("Generated files:")

print(
    "-",
    tables_jsonl_path,
)

print(
    "-",
    table_inventory_csv,
)

print(
    "-",
    table_inventory_xlsx,
)

print(
    "-",
    paper_summary_csv,
)

print(
    "-",
    paper_summary_xlsx,
)


# ============================================================
# Completion status
# ============================================================

step07b_complete = (
    len(markdown_files) > 0
    and total_tables > 0
    and empty_tables == 0
    and zero_column_tables == 0
)

print()

if step07b_complete:
    print("STEP 07B COMPLETE")
else:
    print("STEP 07B NEEDS REVIEW")
