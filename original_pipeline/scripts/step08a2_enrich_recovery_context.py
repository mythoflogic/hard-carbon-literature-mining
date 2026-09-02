#!/usr/bin/env python3

"""
STEP 08A2
=========

Enrich Semantic-Recovery Packages With Paper Context

Purpose
-------
For each ambiguous table row, search the source Markdown paper
for passages that may define:

- sample codes;
- feedstock abbreviations;
- preparation temperatures;
- pyrolysis/carbonization conditions.

The table remains the numerical source of truth.

This step does NOT:
- modify C/H/N/O values;
- use an LLM;
- perform extraction from arbitrary paper chunks.
"""

from pathlib import Path

import json
import re


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

markdown_dir = (
    project_dir
    / "papers_markdown"
)

input_path = (
    processed_tables_dir
    / "semantic_recovery_packages.jsonl"
)

output_path = (
    processed_tables_dir
    / "semantic_recovery_packages_enriched.jsonl"
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:

        for line in source:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def derive_search_terms(package):
    """
    Build conservative search terms from the target row.
    """

    terms = []

    sample = clean_text(
        package.get(
            "sample_raw"
        )
    )

    raw_row = clean_text(
        package.get(
            "raw_source_row"
        )
    )

    # First cell is frequently the sample code.
    if raw_row:
        first_cell = (
            raw_row.split("|")[0]
            .strip()
        )

        if first_cell:
            terms.append(
                first_cell
            )

            # CF-200 -> CF
            prefix_match = re.match(
                r"^([A-Za-z]{1,10})"
                r"[-_ ]?"
                r"\d{2,4}$",
                first_cell,
            )

            if prefix_match:
                terms.append(
                    prefix_match.group(1)
                )

    if sample:
        terms.append(
            sample
        )

    # Remove duplicates while preserving order.
    unique_terms = []

    for term in terms:
        term = term.strip()

        if (
            term
            and term not in unique_terms
        ):
            unique_terms.append(
                term
            )

    return unique_terms


def find_context_windows(
    text,
    search_terms,
    window_chars=1200,
    max_windows=6,
):
    """
    Retrieve text surrounding exact occurrences
    of the search terms.
    """

    windows = []

    lower_text = text.lower()

    for term in search_terms:

        lower_term = term.lower()

        start_position = 0

        while True:

            position = lower_text.find(
                lower_term,
                start_position,
            )

            if position == -1:
                break

            start = max(
                0,
                position - window_chars,
            )

            end = min(
                len(text),
                position
                + len(term)
                + window_chars,
            )

            window = (
                text[start:end]
                .strip()
            )

            if window not in windows:
                windows.append(
                    window
                )

            if len(windows) >= max_windows:
                return windows

            start_position = (
                position
                + len(term)
            )

    return windows

# ============================================================
# Load packages
# ============================================================

packages = load_jsonl(
    input_path
)

print(
    "Recovery packages loaded:",
    len(packages),
)


# ============================================================
# Find Markdown file for paper
# ============================================================

markdown_lookup = {
    path.stem: path
    for path in markdown_dir.glob("*.md")
}


# ============================================================
# Enrich packages
# ============================================================

enriched = []

packages_with_context = 0
missing_papers = 0

for package in packages:

    paper_id = str(
        package["paper_id"]
    )

    search_terms = (
        derive_search_terms(
            package
        )
    )

    markdown_path = (
        markdown_lookup.get(
            paper_id
        )
    )

    paper_context = []

    if markdown_path is None:
        missing_papers += 1

    else:
        paper_text = (
            markdown_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

        paper_context = (
            find_context_windows(
                paper_text,
                search_terms,
            )
        )

    if paper_context:
        packages_with_context += 1

    enriched_package = (
        package.copy()
    )

    enriched_package[
        "paper_search_terms"
    ] = search_terms

    enriched_package[
        "targeted_paper_context"
    ] = paper_context

    enriched.append(
        enriched_package
    )


# ============================================================
# Save
# ============================================================

with output_path.open(
    "w",
    encoding="utf-8",
) as output_file:

    for package in enriched:
        output_file.write(
            json.dumps(
                package,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("STEP 08A2 — TARGETED PAPER CONTEXT")
print("=" * 70)
print()

print(
    "Packages:",
    len(enriched),
)

print(
    "Packages with targeted paper context:",
    packages_with_context,
)

print(
    "Missing Markdown papers:",
    missing_papers,
)

print()
print(
    "Output:",
    output_path,
)
