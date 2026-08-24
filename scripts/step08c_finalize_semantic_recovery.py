#!/usr/bin/env python3

"""
STEP 08C — GENERALIZED FINAL SEMANTIC RECOVERY VALIDATION

Purpose
-------
Produce final accounting for all eligible semantic-recovery records.

This stage does NOT force every record to become a successful extraction.

It distinguishes:

1. LLM_RESOLVED
2. DETERMINISTIC_METADATA_RECOVERY
3. DETERMINISTIC_GROUPED_ROW_RECOVERY
4. STRUCTURAL_MAPPING_ERROR
5. MANUAL_REVIEW_TEMPERATURE
6. REJECT_FRAGMENTED_ROW
7. REJECT_NO_ELEMENTAL_DATA
8. UNRESOLVED

Important
---------
No recovery IDs are hard-coded.

The logic must generalize to unseen scientific papers.
"""

from pathlib import Path

import json
import math
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
    / "semantic_recovery_packages_enriched.jsonl"
)

results_path = (
    processed_dir
    / "semantic_recovery_llm"
    / "semantic_recovery_results_remapped.jsonl"
)
failures_path = (
    processed_dir
    / "semantic_recovery_llm"
    / "semantic_recovery_failures_remapped.jsonl"
)

grouped_repairs_path = (
    processed_dir
    / "grouped_row_repairs_remapped.csv"
)

accounting_path = (
    processed_dir
    / "semantic_recovery_final_accounting.csv"
)

manual_review_path = (
    processed_dir
    / "semantic_recovery_manual_review.csv"
)

accepted_path = (
    processed_dir
    / "semantic_recovery_accepted.csv"
)

rejected_path = (
    processed_dir
    / "semantic_recovery_rejected.csv"
)

summary_path = (
    processed_dir
    / "semantic_recovery_final_summary.csv"
)

excel_path = (
    outputs_dir
    / "semantic_recovery_final_accounting.xlsx"
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):

    records = []

    if not path.exists():
        return records

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

    return records


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
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def is_missing(value):

    if value is None:
        return True

    if isinstance(
        value,
        float,
    ):
        return math.isnan(
            value
        )

    text = str(
        value
    ).strip()

    return (
        text == ""
        or text.lower()
        in {
            "nan",
            "none",
            "null",
        }
    )


def parse_number(value):

    if is_missing(
        value
    ):
        return None

    text = clean_text(
        value
    )

    text = (
        text
        .replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
    )

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


def get_first_cell(
    package
):

    raw = str(
        package.get(
            "raw_source_row",
            "",
        )
    )

    first = (
        raw
        .split("|", 1)[0]
    )

    return clean_text(
        first
    )


def first_cell_is_textual(
    package
):

    first = get_first_cell(
        package
    )

    if not first:
        return False

    return (
        re.search(
            r"[A-Za-z]",
            first,
        )
        is not None
    )


def get_explicit_sample_from_row(
    package
):

    first = get_first_cell(
        package
    )

    if not first:
        return None

    if not re.search(
        r"[A-Za-z]",
        first,
    ):
        return None

    forbidden = {
        "c",
        "h",
        "n",
        "o",
        "s",
        "ash",
        "temperature",
        "property",
        "sample",
        "samples",
        "feedstock",
        "fixed carbon",
        "volatile matter",
    }

    if (
        first.lower()
        in forbidden
    ):
        return None

    return first


def get_sample_suffix_temperature(
    sample
):

    if not sample:
        return None

    text = str(
        sample
    ).strip()

    match = re.search(
        r"(?<!\d)"
        r"(\d{3,4})"
        r"(?:\s*°?\s*[Cc])?$",
        text,
    )

    if not match:
        return None

    value = float(
        match.group(1)
    )

    # Plausible processing range.
    if not (
        50
        <= value
        <= 3000
    ):
        return None

    return value

def get_composite_condition_temperature(
    package
):

    first = get_first_cell(
        package
    )

    if not first:
        return None

    # --------------------------------------------------------
    # Only interpret composite numeric conditions when
    # the table explicitly defines the first column as
    # temperature combined with another process variable.
    #
    # Examples:
    #   °C*min
    #   °C × min
    #   C/min
    #   temperature*time
    # --------------------------------------------------------

    header = header_text(
        package
    )

    raw_table = clean_text(
        package.get(
            "raw_table_text",
            ""
        )
    ).lower()

    evidence = (
        header
        + " "
        + raw_table
    )

    has_temperature_unit = (
        "°c" in evidence
        or "ºc" in evidence
        or "◦c" in evidence
        or "celsius" in evidence
        or "temperature" in evidence
    )

    has_composite_unit = (
        "*min" in evidence
        or "×min" in evidence
        or "x min" in evidence
        or "c*min" in evidence
        or "c × min" in evidence
        or "temperature*time" in evidence
        or "temperature * time" in evidence
    )

    if not (
        has_temperature_unit
        and has_composite_unit
    ):
        return None

    text = clean_text(
        first
    )

    match = re.match(
        r"^\s*"
        r"(\d{2,4}(?:\.\d+)?)"
        r"\s*[*×x]\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    temperature = float(
        match.group(1)
    )

    if not (
        50
        <= temperature
        <= 3000
    ):
        return None

    return temperature

def sample_suffix_temperature_is_supported(
    package,
    sample,
):

    suffix_temperature = (
        get_sample_suffix_temperature(
            sample
        )
    )

    if suffix_temperature is None:
        return False

    # --------------------------------------------------------
    # Require evidence that temperature is actually a
    # meaningful variable in this table/context.
    # --------------------------------------------------------

    evidence_parts = [
        package.get(
            "caption",
            ""
        ),
        package.get(
            "preceding_context",
            ""
        ),
        package.get(
            "following_context",
            ""
        ),
        " ".join(
            str(x)
            for x in package.get(
                "header_row",
                []
            )
        )
        if isinstance(
            package.get(
                "header_row",
                []
            ),
            list,
        )
        else str(
            package.get(
                "header_row",
                ""
            )
        ),
    ]

    evidence = clean_text(
        " ".join(
            str(x)
            for x in evidence_parts
            if x is not None
        )
    ).lower()

    temperature_terms = [
        "temperature",
        "temperatures",
        "°c",
        "ºc",
        "deg c",
        "degree c",
        "degrees c",
        "celsius",
        "pyrolysis",
        "carbonization",
        "carbonisation",
        "calcination",
        "annealing",
        "annealed",
        "heat treatment",
        "heat-treatment",
        "thermal treatment",
        "thermal-treatment",
    ]

    return any(
        term in evidence
        for term in temperature_terms
    )

# ============================================================
# Structural checks
# ============================================================

def is_fragmented_row(
    package
):

    raw = str(
        package.get(
            "raw_source_row",
            "",
        )
    ).strip()

    if not raw:
        return True

    first = get_first_cell(
        package
    )

    sample = package.get(
        "sample_raw"
    )

    has_textual_sample = (
        (
            not is_missing(
                sample
            )
        )
        and re.search(
            r"[A-Za-z]",
            str(sample),
        )
        is not None
    )

    if has_textual_sample:
        return False

    if first_cell_is_textual(
        package
    ):
        return False

    # Typical extracted uncertainty /
    # continuation fragments.
    if (
        raw.startswith("_(")
        or raw.startswith("(")
        or raw.count("_(") >= 2
    ):
        return True

    return False


def header_text(
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
            str(x)
            for x in header
        )

    else:

        text = str(
            header
        )

    return clean_text(
        text
    ).lower()


def mixed_proximate_ultimate_table(
    package
):

    text = header_text(
        package
    )

    return (
        "proximate" in text
        and "ultimate" in text
    )


def likely_structural_mapping_error(
    package
):

    semantic_class = package.get(
        "semantic_class"
    )

    raw = str(
        package.get(
            "raw_source_row",
            "",
        )
    )

    # Tables explicitly mixing proximate and ultimate
    # analyses need stronger scrutiny.
    if mixed_proximate_ultimate_table(
        package
    ):

        # If the deterministic parser has only partial
        # elemental values while the table clearly contains
        # multiple analysis blocks, treat the mapping as
        # suspicious rather than trusting shifted columns.
        elemental_present = sum(
            not is_missing(
                package.get(
                    key
                )
            )
            for key in [
                "C_value",
                "H_value",
                "N_value",
                "O_value",
            ]
        )

        if elemental_present < 2:
            return True

    # Generic semantic-recovery rows with no identifiable
    # sample and a long multi-column row can indicate
    # structural misalignment.
    if (
        semantic_class
        == "SEMANTIC_RECOVERY_REQUIRED"
        and not first_cell_is_textual(
            package
        )
        and raw.count("|") >= 5
    ):
        return True

    return False

def likely_structural_artifact(
    package
):

    raw = str(
        package.get(
            "raw_source_row",
            ""
        )
    )

    if not raw:
        return True

    cells = [
        clean_text(cell)
        for cell in raw.split("|")
    ]

    nonempty_cells = [
        cell
        for cell in cells
        if cell
    ]

    text = clean_text(
        raw
    ).lower()

    # --------------------------------------------------------
    # Explicit graphical / axis-like labels.
    # --------------------------------------------------------

    axis_terms = [
        "atomic ratio",
        "axis",
        "tick",
    ]

    if any(
        term in text
        for term in axis_terms
    ):
        return True

    # --------------------------------------------------------
    # Very sparse rows produced by figures, merged labels,
    # or broken table geometry.
    #
    # Only apply this after the row has already been judged
    # structurally suspicious.
    # --------------------------------------------------------

    if (
        len(cells) >= 6
        and len(nonempty_cells) <= 3
    ):
        return True

    return False

def no_usable_elemental_data(
    package
):

    values = [
        package.get(
            "C_value"
        ),
        package.get(
            "H_value"
        ),
        package.get(
            "N_value"
        ),
        package.get(
            "O_value"
        ),
    ]

    present_count = sum(
        not is_missing(
            value
        )
        for value in values
    )

    return (
        present_count == 0
    )


# ============================================================
# Metadata helpers
# ============================================================

def exact_temperature_from_package(
    package
):

    value = package.get(
        "temperature_exact_C"
    )

    number = parse_number(
        value
    )

    if number is not None:
        return number

    temperature_original = (
        package.get(
            "temperature_original"
        )
    )

    temperature_type = (
        package.get(
            "temperature_type"
        )
    )

    if (
        temperature_type
        == "exact"
    ):

        return parse_number(
            temperature_original
        )

    return None


def sample_from_package(
    package
):

    sample = package.get(
        "sample_raw"
    )

    if (
        not is_missing(
            sample
        )
        and re.search(
            r"[A-Za-z]",
            str(sample),
        )
    ):

        return clean_text(
            sample
        )

    return get_explicit_sample_from_row(
        package
    )

def targeted_context_text(
    package
):

    context = package.get(
        "targeted_paper_context",
        ""
    )

    if isinstance(
        context,
        list,
    ):
        return "\n".join(
            str(x)
            for x in context
            if x is not None
        )

    return str(
        context
        or ""
    )


def normalize_sample_token(
    value
):

    if value is None:
        return ""

    text = clean_text(
        value
    )

    text = re.sub(
        r"[*_`]+",
        "",
        text,
    )

    return re.sub(
        r"\s+",
        "",
        text,
    ).lower()


def contextual_temperature_from_mapping_table(
    package,
    sample,
):

    if not sample:
        return None

    target = normalize_sample_token(
        sample
    )

    if not target:
        return None

    context = targeted_context_text(
        package
    )

    lines = context.splitlines()

    # --------------------------------------------------------
    # Look for explicit markdown tables where one row gives
    # sample/material names and another aligned row gives
    # temperatures.
    # --------------------------------------------------------

    for i, line in enumerate(
        lines
    ):

        if "|" not in line:
            continue

        low = clean_text(
            line
        ).lower()

        if "temperature" not in low:
            continue

        if not (
            "°c" in low
            or "ºc" in low
            or "◦c" in low
            or "celsius" in low
        ):
            continue

        temp_cells = [
            clean_text(x)
            for x in line.strip().strip("|").split("|")
        ]

        # Search nearby preceding rows for a sample-name row.
        for j in range(
            max(0, i - 4),
            i,
        ):

            candidate = lines[j]

            if "|" not in candidate:
                continue

            # Skip markdown separator rows.
            if re.fullmatch(
                r"[\s|:\-]+",
                candidate,
            ):
                continue

            sample_cells = [
                clean_text(x)
                for x in candidate.strip().strip("|").split("|")
            ]

            if len(
                sample_cells
            ) != len(
                temp_cells
            ):
                continue

            for col in range(
                1,
                len(sample_cells),
            ):

                if (
                    normalize_sample_token(
                        sample_cells[col]
                    )
                    != target
                ):
                    continue

                number = parse_number(
                    temp_cells[col]
                )

                if (
                    number is not None
                    and 50
                    <= number
                    <= 3000
                ):
                    return number

    return None


def contextual_temperature_from_prose(
    package,
    sample,
):

    if not sample:
        return None

    context = clean_text(
        targeted_context_text(
            package
        )
    )

    if not context:
        return None

    sample_text = clean_text(
        sample
    )

    # Find occurrences of the exact sample label.
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9])"
            + re.escape(
                sample_text
            )
            + r"(?![A-Za-z0-9])",
            context,
            flags=re.IGNORECASE,
        )
    )

    candidates = set()

    for match in matches:

        start = max(
            0,
            match.start() - 150,
        )

        end = min(
            len(context),
            match.end() + 450,
        )

        window = context[
            start:end
        ]

        low = window.lower()

        process_terms = [
            "produced",
            "prepared",
            "treated",
            "heated",
            "pyrolysis",
            "pyrolysed",
            "pyrolyzed",
            "carbonized",
            "carbonised",
        ]

        if not any(
            term in low
            for term in process_terms
        ):
            continue

        temperatures = re.findall(
            r"(?<!\d)"
            r"(\d{2,4}(?:\.\d+)?)"
            r"\s*[°º◦]\s*[Cc]",
            window,
        )

        numeric = {
            float(x)
            for x in temperatures
            if 50
            <= float(x)
            <= 3000
        }

        # Only trust a prose window with one unique
        # processing-temperature candidate.
        if len(
            numeric
        ) == 1:
            candidates.update(
                numeric
            )

    if len(
        candidates
    ) == 1:
        return next(
            iter(
                candidates
            )
        )

    return None


def contextual_temperature_from_package(
    package,
    sample,
):

    # Only accept explicit aligned sample-to-temperature
    # mappings from structured tables.
    #
    # Free-text proximity matching is intentionally excluded
    # because nearby temperatures may refer to other samples,
    # processes, or measured properties.

    value = (
        contextual_temperature_from_mapping_table(
            package,
            sample,
        )
    )

    if value is not None:
        return (
            value,
            "context_mapping_table",
        )

    return (
        None,
        None,
    )

def temperature_is_ambiguous(
    package,
    llm_result=None,
):

    sample = sample_from_package(
        package
    )

    exact_temperature = (
        exact_temperature_from_package(
            package
        )
    )

    if exact_temperature is not None:
        return False

    if sample is None:
        return False

    if llm_result is None:
        return True

    temperature = llm_result.get(
        "resolved_temperature_C"
    )

    return (
        temperature is None
        or not isinstance(
            temperature,
            (int, float),
        )
    )


# ============================================================
# Load data
# ============================================================

packages = load_jsonl(
    packages_path
)

results = load_jsonl(
    results_path
)

failures = load_jsonl(
    failures_path
)


eligible_packages = {
    str(
        package[
            "recovery_id"
        ]
    ): package

    for package in packages

    if package.get(
        "semantic_class"
    )
    != "TEMPERATURE_RANGE"
}


print(
    "Eligible recovery packages:",
    len(
        eligible_packages
    ),
)


# ============================================================
# Latest LLM success / failure
# ============================================================

success_by_id = {}

for result in results:

    recovery_id = str(
        result.get(
            "recovery_id"
        )
    )

    success_by_id[
        recovery_id
    ] = result


failure_by_id = {}

for failure in failures:

    recovery_id = str(
        failure.get(
            "recovery_id"
        )
    )

    failure_by_id[
        recovery_id
    ] = failure


# ============================================================
# Load grouped repairs
# ============================================================

grouped_repairs = {}

if (
    grouped_repairs_path.exists()
    and grouped_repairs_path.stat().st_size > 0
):

    try:
        grouped_df = pd.read_csv(
            grouped_repairs_path
        )

    except pd.errors.EmptyDataError:
        grouped_df = pd.DataFrame()

    if not grouped_df.empty:

        for _, row in grouped_df.iterrows():

            if (
                row.get(
                    "repair_status"
                )
                !=
                "DETERMINISTIC_"
                "GROUPED_ROW_RECOVERY"
            ):
                continue

            recovery_id = str(
                row[
                    "recovery_id"
                ]
            )

            grouped_repairs[
                recovery_id
            ] = {
                "resolved_sample": (
                    row.get(
                        "resolved_sample"
                    )
                ),
                "resolved_temperature_C": (
                    parse_number(
                        row.get(
                            "resolved_temperature_C"
                        )
                    )
                ),
                "resolved_temperature_type": (
                    row.get(
                        "resolved_temperature_type"
                    )
                ),
            }


# ============================================================
# Final accounting
# ============================================================

rows = []


for recovery_id, package in (
    eligible_packages.items()
):

    llm_result = success_by_id.get(
        recovery_id
    )

    llm_failure = failure_by_id.get(
        recovery_id,
        {},
    )

    record = {
        "recovery_id": recovery_id,
        "paper_id": package.get(
            "paper_id"
        ),
        "table_id": package.get(
            "table_id"
        ),
        "semantic_class": package.get(
            "semantic_class"
        ),
        "raw_source_row": package.get(
            "raw_source_row"
        ),
        "sample_raw": package.get(
            "sample_raw"
        ),
        "temperature_original": (
            package.get(
                "temperature_original"
            )
        ),
        "C_value": package.get(
            "C_value"
        ),
        "H_value": package.get(
            "H_value"
        ),
        "N_value": package.get(
            "N_value"
        ),
        "O_value": package.get(
            "O_value"
        ),
        "final_status": None,
        "resolved_sample": None,
        "resolved_temperature_C": None,
        "resolved_temperature_type": None,
        "provenance": None,
        "confidence": None,
        "validation_reason": None,
        "requires_manual_review": False,
    }

    # --------------------------------------------------------
    # 1. Fragmented row
    # --------------------------------------------------------

    if is_fragmented_row(
        package
    ):

        record[
            "final_status"
        ] = (
            "REJECT_FRAGMENTED_ROW"
        )

        record[
            "validation_reason"
        ] = (
            "Row appears to be a continuation, "
            "uncertainty, or formatting fragment "
            "without independent sample identity."
        )

    # --------------------------------------------------------
    # 2. Generic grouped-row repair
    # --------------------------------------------------------

    elif (
        recovery_id
        in grouped_repairs
    ):

        repair = grouped_repairs[
            recovery_id
        ]

        record[
            "final_status"
        ] = (
            "DETERMINISTIC_"
            "GROUPED_ROW_RECOVERY"
        )

        record[
            "resolved_sample"
        ] = repair[
            "resolved_sample"
        ]

        record[
            "resolved_temperature_C"
        ] = repair[
            "resolved_temperature_C"
        ]

        record[
            "resolved_temperature_type"
        ] = repair[
            "resolved_temperature_type"
        ]

        record[
            "validation_reason"
        ] = (
            "Parent sample identity recovered "
            "from grouped/merged table structure."
        )
    # --------------------------------------------------------
    # 2.5. Composite processing-condition recovery
    # --------------------------------------------------------

    elif (
        get_composite_condition_temperature(
            package
        )
        is not None
    ):

        composite_temperature = (
            get_composite_condition_temperature(
                package
            )
        )

        record[
            "final_status"
        ] = (
            "DETERMINISTIC_"
            "COMPOSITE_CONDITION_RECOVERY"
        )

        record[
            "resolved_sample"
        ] = (
            sample_from_package(
                package
            )
        )

        record[
            "resolved_temperature_C"
        ] = (
            composite_temperature
        )

        record[
            "resolved_temperature_type"
        ] = None

        record[
            "provenance"
        ] = (
            "composite_condition_column"
        )

        record[
            "confidence"
        ] = (
            "high"
        )

        record[
            "validation_reason"
        ] = (
            "Processing temperature was recovered "
            "from an explicitly defined composite "
            "temperature-time condition column."
        )
   
    # --------------------------------------------------------
    # 3. Structural mapping problem
    # --------------------------------------------------------

    elif likely_structural_mapping_error(
        package
    ):

        if likely_structural_artifact(
            package
        ):

            record[
                "final_status"
            ] = (
                "REJECT_STRUCTURAL_ARTIFACT"
            )

            record[
                "validation_reason"
            ] = (
                "Structurally suspicious row is sparse "
                "or contains axis/graphical content rather "
                "than an independent experimental record."
            )

        else:

            record[
                "final_status"
            ] = (
                "STRUCTURAL_MAPPING_ERROR"
            )

            record[
                "requires_manual_review"
            ] = True

            record[
                "validation_reason"
            ] = (
                "Table structure indicates that "
                "column mapping may be shifted or "
                "scientifically ambiguous."
            )

    # --------------------------------------------------------
    # 4. No usable elemental data
    # --------------------------------------------------------

    elif no_usable_elemental_data(
        package
    ):

        record[
            "final_status"
        ] = (
            "REJECT_NO_ELEMENTAL_DATA"
        )

        record[
            "validation_reason"
        ] = (
            "No usable elemental C/H/N/O values "
            "remain after structural validation."
        )

    # --------------------------------------------------------
    # 5. Deterministic metadata already available
    # --------------------------------------------------------

    else:

        sample = sample_from_package(
            package
        )

        exact_temperature = (
            exact_temperature_from_package(
                package
            )
        )

        contextual_temperature = None
        contextual_temperature_source = None

        if (
            sample is not None
            and exact_temperature is None
        ):

            (
                contextual_temperature,
                contextual_temperature_source,
            ) = contextual_temperature_from_package(
                package,
                sample,
            )

        if contextual_temperature is not None:
            exact_temperature = (
                contextual_temperature
            )

        suffix_temperature = None
        if (
            sample is not None
            and exact_temperature is None
            and sample_suffix_temperature_is_supported(
                package,
                sample,
            )
        ):

            suffix_temperature = (
                get_sample_suffix_temperature(
                    sample
                )
            )

        if suffix_temperature is not None:
            exact_temperature = (
                suffix_temperature
            )
        if (
            sample is not None
            and exact_temperature
            is not None
        ):

            if contextual_temperature is not None:

                record[
                    "final_status"
                ] = (
                    "DETERMINISTIC_"
                    "CONTEXTUAL_TEMPERATURE_RECOVERY"
                )

                record[
                    "provenance"
                ] = (
                    contextual_temperature_source
                )

            elif suffix_temperature is not None:

                record[
                    "final_status"
                ] = (
                    "DETERMINISTIC_"
                    "SAMPLE_CODE_TEMPERATURE_RECOVERY"
                )

                record[
                    "provenance"
                ] = (
                    "sample_code_temperature"
                )

            else:

                record[
                    "final_status"
                ] = (
                    "DETERMINISTIC_METADATA_RECOVERY"
                )
            record[
                "resolved_sample"
            ] = sample

            record[
                "resolved_temperature_C"
            ] = exact_temperature

            if llm_result:

                record[
                    "resolved_temperature_type"
                ] = llm_result.get(
                    "resolved_temperature_type"
                )

                if (
                    suffix_temperature is None
                    and contextual_temperature is None
                ):
                    record[
                        "provenance"
                    ] = llm_result.get(
                        "provenance"
                    )

                record[
                    "confidence"
                ] = llm_result.get(
                    "confidence"
                )

            if contextual_temperature is not None:

                record[
                    "validation_reason"
                ] = (
                    "Processing temperature was recovered "
                    "from explicit contextual evidence "
                    "linking the sample/material to a "
                    "unique processing temperature."
                )

            elif suffix_temperature is not None:

                record[
                    "validation_reason"
                ] = (
                    "Processing temperature was encoded "
                    "in the sample/material label and "
                    "independently supported by "
                    "temperature-related table/context "
                    "evidence."
                )

            else:

                record[
                    "validation_reason"
                ] = (
                    "Sample and exact processing "
                    "temperature were available from "
                    "deterministic table metadata."
                )
        # ----------------------------------------------------
        # 6. Sample known but temperature ambiguous
        # ----------------------------------------------------

        elif (
            sample is not None
            and temperature_is_ambiguous(
                package,
                llm_result,
            )
        ):

            # A valid material/sample record can legitimately
            # have no recoverable processing temperature.
            #
            # Missing temperature is not automatically
            # an extraction failure.

            if (
                llm_result is not None
                and llm_result.get(
                    "row_decision"
                )
                == "resolve"
            ):

                record[
                    "final_status"
                ] = (
                    "LLM_RESOLVED_PARTIAL"
                )

                record[
                    "resolved_sample"
                ] = (
                    llm_result.get(
                        "resolved_sample"
                    )
                    or sample
                )

                record[
                    "resolved_temperature_C"
                ] = None

                record[
                    "resolved_temperature_type"
                ] = (
                    llm_result.get(
                        "resolved_temperature_type"
                    )
                )

                record[
                    "provenance"
                ] = llm_result.get(
                    "provenance"
                )

                record[
                    "confidence"
                ] = llm_result.get(
                    "confidence"
                )

                record[
                    "validation_reason"
                ] = (
                    "Valid material/sample record "
                    "was resolved, but no unique "
                    "processing temperature was "
                    "supported by the available "
                    "evidence."
                )

            else:

                record[
                    "final_status"
                ] = (
                    "MANUAL_REVIEW_TEMPERATURE"
                )

                record[
                    "resolved_sample"
                ] = sample

                record[
                    "requires_manual_review"
                ] = True

                record[
                    "validation_reason"
                ] = (
                    "Sample identity is available "
                    "but no successful validated "
                    "temperature resolution exists."
                )

        # ----------------------------------------------------
        # 7. Valid LLM result
        # ----------------------------------------------------

        elif llm_result is not None:

            llm_sample = (
                llm_result.get(
                    "resolved_sample"
                )
            )

            llm_temperature = (
                llm_result.get(
                    "resolved_temperature_C"
                )
            )

            # Ensure row-level output.
            llm_is_row_level = (
                (
                    llm_sample is None
                    or isinstance(
                        llm_sample,
                        str,
                    )
                )
                and (
                    llm_temperature is None
                    or isinstance(
                        llm_temperature,
                        (int, float),
                    )
                )
            )

            if not llm_is_row_level:

                record[
                    "final_status"
                ] = (
                    "MANUAL_REVIEW_"
                    "MULTI_VALUE_LLM_OUTPUT"
                )

                record[
                    "requires_manual_review"
                ] = True

                record[
                    "validation_reason"
                ] = (
                    "LLM returned multiple samples, "
                    "multiple temperatures, or a "
                    "non-row-level object."
                )

            else:

                record[
                    "final_status"
                ] = (
                    "LLM_RESOLVED"
                )

                record[
                    "resolved_sample"
                ] = llm_sample

                record[
                    "resolved_temperature_C"
                ] = llm_temperature

                record[
                    "resolved_temperature_type"
                ] = llm_result.get(
                    "resolved_temperature_type"
                )

                record[
                    "provenance"
                ] = llm_result.get(
                    "provenance"
                )

                record[
                    "confidence"
                ] = llm_result.get(
                    "confidence"
                )

                record[
                    "validation_reason"
                ] = (
                    "LLM returned a valid "
                    "single-row resolution after "
                    "structural checks."
                )

        # ----------------------------------------------------
        # 8. Nothing resolved
        # ----------------------------------------------------

        else:

            record[
                "final_status"
            ] = (
                "UNRESOLVED"
            )

            record[
                "requires_manual_review"
            ] = True

            record[
                "validation_reason"
            ] = (
                llm_failure.get(
                    "error",
                    "No safe deterministic or "
                    "LLM resolution available.",
                )
            )

    rows.append(
        record
    )


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    rows
)


summary = (
    df[
        "final_status"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "final_status"
    )
    .reset_index(
        name="row_count"
    )
)


# ============================================================
# Accepted / manual / rejected
# ============================================================

accepted_statuses = {
    "LLM_RESOLVED",
    "LLM_RESOLVED_PARTIAL",
    "DETERMINISTIC_METADATA_RECOVERY",
    "DETERMINISTIC_GROUPED_ROW_RECOVERY",
    "DETERMINISTIC_SAMPLE_CODE_TEMPERATURE_RECOVERY",
    "DETERMINISTIC_COMPOSITE_CONDITION_RECOVERY",
    "DETERMINISTIC_CONTEXTUAL_TEMPERATURE_RECOVERY",
}

rejected_statuses = {
    "REJECT_FRAGMENTED_ROW",
    "REJECT_NO_ELEMENTAL_DATA",
    "REJECT_STRUCTURAL_ARTIFACT",
}


accepted_df = df[
    df[
        "final_status"
    ].isin(
        accepted_statuses
    )
].copy()


rejected_df = df[
    df[
        "final_status"
    ].isin(
        rejected_statuses
    )
].copy()


manual_review_df = df[
    df[
        "requires_manual_review"
    ].eq(
        True
    )
].copy()


# ============================================================
# Save
# ============================================================

df.to_csv(
    accounting_path,
    index=False,
)

accepted_df.to_csv(
    accepted_path,
    index=False,
)

manual_review_df.to_csv(
    manual_review_path,
    index=False,
)

rejected_df.to_csv(
    rejected_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)


with pd.ExcelWriter(
    excel_path
) as writer:

    df.to_excel(
        writer,
        sheet_name="all_accounting",
        index=False,
    )

    summary.to_excel(
        writer,
        sheet_name="summary",
        index=False,
    )

    accepted_df.to_excel(
        writer,
        sheet_name="accepted",
        index=False,
    )

    manual_review_df.to_excel(
        writer,
        sheet_name="manual_review",
        index=False,
    )

    rejected_df.to_excel(
        writer,
        sheet_name="rejected",
        index=False,
    )


# ============================================================
# Report
# ============================================================

print()
print("=" * 72)
print(
    "STEP 08C — GENERALIZED FINAL "
    "SEMANTIC RECOVERY"
)
print("=" * 72)
print()

print(
    summary.to_string(
        index=False
    )
)

print()

print(
    "Total accounted:",
    len(df),
)

print(
    "Accepted:",
    len(
        accepted_df
    ),
)

print(
    "Manual review:",
    len(
        manual_review_df
    ),
)

print(
    "Rejected:",
    len(
        rejected_df
    ),
)

print()

print(
    "Generated files:"
)

print(
    "-",
    accounting_path,
)

print(
    "-",
    accepted_path,
)

print(
    "-",
    manual_review_path,
)

print(
    "-",
    rejected_path,
)

print(
    "-",
    summary_path,
)

print(
    "-",
    excel_path,
)

