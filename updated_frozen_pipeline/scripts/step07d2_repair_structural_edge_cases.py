#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_TABLES = ROOT / "processed_tables"
OUTPUTS = ROOT / "outputs"

DEFAULT_INPUT = (
    PROCESSED_TABLES
    / "normalized_candidate_tables.jsonl"
)

OUTPUT_JSONL = (
    PROCESSED_TABLES
    / "structural_edge_repaired_candidate_tables.jsonl"
)

REPORT_CSV = (
    OUTPUTS
    / "step07f3_structural_edge_case_repairs.csv"
)


# ============================================================
# POSSIBLE SCHEMA KEYS
# ============================================================

ROW_KEYS = (
    "normalized_rows",
    "rows",
    "table_rows",
    "data_rows",
    "records",
    "data",
)

ID_KEYS = (
    "table_id",
    "candidate_table_id",
    "id",
    "table_name",
)

SOURCE_KEYS = (
    "source_file",
    "paper_file",
    "markdown_file",
    "paper_id",
    "source",
)


# ============================================================
# REGEX
# ============================================================

BR_RE = re.compile(
    r"(?i)<br\s*/?>|\n+"
)

TAG_RE = re.compile(
    r"<[^>]+>"
)

NUMBER_FRAGMENT_RE = re.compile(
    r"""
    ^[<>~≈≤≥]?\s*
    [+\-−–]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+\-]?\d+)?
    \s*[A-Za-z*†‡]*
    \s*%?
    \s*$
    """,
    re.X,
)

COMPLETE_MEASUREMENT_RE = re.compile(
    r"""
    ^[<>~≈≤≥]?\s*
    [+\-−–]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+\-]?\d+)?
    \s*[A-Za-z*†‡]*
    \s*
    (?:
        (?:±|\+/-)
        \s*
        [+\-−–]?
        (?:
            \d+(?:\.\d*)?
            |
            \.\d+
        )
        (?:[eE][+\-]?\d+)?
        \s*[A-Za-z*†‡]*
    )?
    \s*%?
    \s*$
    """,
    re.X,
)

DANGLING_PM_RE = re.compile(
    r"""
    [+\-−–]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+\-]?\d+)?
    \s*[A-Za-z*†‡]*
    \s*(?:±|\+/-)\s*$
    """,
    re.X,
)

NUM_TOKEN_RE = re.compile(
    r"""
    (?<![A-Za-z])
    [-+−–]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    """,
    re.X,
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(value)

    text = TAG_RE.sub(
        " ",
        text,
    )

    text = (
        text
        .replace("**", "")
        .replace("__", "")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def split_parts(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    result = []

    for part in BR_RE.split(
        str(value)
    ):

        part = clean(
            part
        )

        if part:
            result.append(
                part
            )

    return result


def join_parts(
    parts: list[str],
) -> str:

    return "<br>".join(
        part
        for part in parts
        if clean(part)
    )


# ============================================================
# MEASUREMENT DETECTION
# ============================================================

def is_number_fragment(
    value: Any,
) -> bool:

    return bool(
        NUMBER_FRAGMENT_RE.fullmatch(
            clean(value)
        )
    )


def is_complete_measurement(
    value: Any,
) -> bool:

    return bool(
        COMPLETE_MEASUREMENT_RE.fullmatch(
            clean(value)
        )
    )


def ends_with_numeric_pm(
    value: Any,
) -> bool:

    text = clean(
        value
    )

    if not text:
        return False

    return bool(
        DANGLING_PM_RE.search(
            text
        )
    )


def looks_like_measurement_part(
    value: Any,
) -> bool:

    return (
        is_complete_measurement(value)
        or
        ends_with_numeric_pm(value)
    )


def cell_has_measurement(
    value: Any,
) -> bool:

    parts = split_parts(
        value
    )

    if not parts:
        return False

    return any(
        looks_like_measurement_part(
            part
        )
        for part in parts
    )


# ============================================================
# SAMPLE LABEL DETECTION
# ============================================================

def looks_like_sample_label(
    value: Any,
) -> bool:

    text = clean(
        value
    )

    if not text:
        return False

    # Reject measurements such as:
    #
    # 0.730*
    # −0.078
    # 6.45a ± 1.24

    if looks_like_measurement_part(
        text
    ):
        return False

    # Real sample names/codes should contain
    # at least one letter.

    if not re.search(
        r"[A-Za-z]",
        text,
    ):
        return False

    # Avoid treating long prose/citations
    # as sample identifiers.

    if len(text) > 80:
        return False

    return True


# ============================================================
# TABLE SCHEMA HELPERS
# ============================================================

def find_rows_key(
    record: dict,
) -> str | None:

    for key in ROW_KEYS:

        value = record.get(
            key
        )

        if not isinstance(
            value,
            list,
        ):
            continue

        if not value:
            continue

        sample = value[:5]

        row_like = sum(
            isinstance(
                row,
                (list, dict),
            )
            for row in sample
        )

        if row_like >= max(
            1,
            len(sample) // 2,
        ):
            return key

    return None


def get_table_id(
    record: dict,
    fallback: str,
) -> str:

    for key in ID_KEYS:

        value = record.get(
            key
        )

        if value not in (
            None,
            "",
        ):
            return str(
                value
            )

    return fallback


def get_source(
    record: dict,
) -> str:

    for key in SOURCE_KEYS:

        value = record.get(
            key
        )

        if value not in (
            None,
            "",
        ):
            return str(
                value
            )

    return ""


def adapt_row(
    row: Any,
):

    if isinstance(
        row,
        list,
    ):

        original = deepcopy(
            row
        )

        def restore(
            values,
        ):

            output = deepcopy(
                original
            )

            if len(output) < len(values):

                output.extend(
                    [""] * (
                        len(values)
                        - len(output)
                    )
                )

            for index, value in enumerate(
                values
            ):

                output[
                    index
                ] = value

            return output

        return (
            list(row),
            restore,
        )

    if isinstance(
        row,
        dict,
    ):

        keys = list(
            row.keys()
        )

        original = deepcopy(
            row
        )

        def restore(
            values,
        ):

            output = deepcopy(
                original
            )

            for key, value in zip(
                keys,
                values,
            ):

                output[
                    key
                ] = value

            return output

        return (
            [
                row.get(key)
                for key in keys
            ],
            restore,
        )

    return (
        None,
        None,
    )


# ============================================================
# SPLIT SAMPLE NAMES
# ============================================================

def looks_like_name_continuation(
    previous: Any,
    current: Any,
) -> bool:

    previous_text = clean(
        previous
    )

    current_text = clean(
        current
    )

    if not previous_text:
        return False

    if not current_text:
        return False

    # --------------------------------------------------------
    # Never merge citation/reference continuations.
    #
    # Examples we observed:
    #
    # (Wannapeera et al., || 2011)
    #
    # (Morali and Sensoz, || 11 2015)
    # --------------------------------------------------------

    combined = (
        f"{previous_text} {current_text}"
    )

    if re.search(
        r"\b(?:18|19|20)\d{2}\b",
        combined,
    ):
        return False

    if re.search(
        r"\bet\s+al\.?\b",
        combined,
        re.I,
    ):
        return False

    # --------------------------------------------------------
    # Do not treat a numeric measurement as a continuation.
    # --------------------------------------------------------

    if looks_like_measurement_part(
        current_text
    ):
        return False

    # --------------------------------------------------------
    # Strong structural case:
    #
    # Mesocarp fibre (oil
    # palm)
    #
    # HHV(MJ/
    # kg)
    #
    # Previous cell has an unmatched opening parenthesis.
    # --------------------------------------------------------

    if (
        previous_text.count("(")
        >
        previous_text.count(")")
    ):

        # Fragment should still look like ordinary
        # text/unit material, not long prose.

        if len(
            current_text
        ) <= 60:

            return True

    # --------------------------------------------------------
    # Conservative lowercase continuation.
    # --------------------------------------------------------

    if (
        current_text[:1].islower()
        and
        re.search(
            r"[A-Za-z]",
            previous_text,
        )
        and
        len(current_text) <= 40
        and
        len(previous_text) <= 80
    ):
        return True

    return False

# STACKED MEASUREMENT PARSER
# ============================================================

def parse_measurement_groups(
    value: Any,
    n_samples: int,
) -> list[str] | None:

    """
    Examples:

    0.06<br>0.04

    becomes:

    [
        "0.06",
        "0.04",
    ]


    76.17±<br>
    0.19<br>
    84.60± 0.10

    becomes:

    [
        "76.17± 0.19",
        "84.60± 0.10",
    ]
    """

    parts = split_parts(
        value
    )

    if n_samples < 2:
        return None

    if not parts:
        return None

    groups = []

    index = 0

    while index < len(
        parts
    ):

        part = parts[
            index
        ]

        if (
            ends_with_numeric_pm(
                part
            )
            and
            index + 1
            < len(parts)
            and
            is_number_fragment(
                parts[
                    index + 1
                ]
            )
        ):

            groups.append(
                f"{part} "
                f"{parts[index + 1]}"
            )

            index += 2
            continue

        groups.append(
            part
        )

        index += 1

    if len(
        groups
    ) != n_samples:

        return None

    if not all(
        is_complete_measurement(
            group
        )
        for group in groups
    ):
        return None

    return groups


def find_multi_sample(
    values: list[Any],
):

    for sample_column, cell in enumerate(
        values
    ):

        names = split_parts(
            cell
        )

        if not (
            2
            <= len(names)
            <= 8
        ):
            continue

        # Critical protection:
        #
        # do NOT interpret stacked numerical
        # measurements as sample names.

        if not all(
            looks_like_sample_label(
                name
            )
            for name in names
        ):
            continue

        groups_by_column = {}

        for column, other_cell in enumerate(
            values
        ):

            if column == sample_column:
                continue

            groups = parse_measurement_groups(
                other_cell,
                len(names),
            )

            if groups is not None:

                groups_by_column[
                    column
                ] = groups

        # Need at least two independently
        # aligned measurement columns.

        if len(
            groups_by_column
        ) >= 2:

            return (
                sample_column,
                names,
                groups_by_column,
            )

    return None


# ============================================================
# FIGURE FRAGMENT DETECTION
# ============================================================

def figure_fragment_reason(
    values: list[Any],
) -> str | None:

    """
    Conservative detector.

    A table row such as:

        H/C atomic ratio | 0.99 ± ... | ...

    must NOT be removed.

    Axis extraction usually contains many
    standalone tick values and several
    numbers inside one malformed cell.
    """

    cleaned = [
        clean(value)
        for value in values
    ]

    text = " ".join(
        value
        for value in cleaned
        if value
    ).lower()

    if (
        "atomic ratio" not in text
        and
        "van krevelen" not in text
    ):
        return None

    pure_numeric_cells = sum(
        1
        for value in cleaned
        if is_number_fragment(
            value
        )
    )

    max_numeric_tokens_in_cell = max(
        (
            len(
                NUM_TOKEN_RE.findall(
                    value
                )
            )
            for value in cleaned
        ),
        default=0,
    )

    if (
        pure_numeric_cells >= 4
        and
        max_numeric_tokens_in_cell >= 2
    ):

        return (
            "axis_or_van_krevelen_fragment"
        )

    return None


# ============================================================
# JSONL
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: "
                    f"{error}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):

                raise TypeError(
                    f"Expected JSON object at "
                    f"{path}:{line_number}"
                )

            records.append(
                record
            )

    return records


def save_jsonl(
    path: Path,
    records: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in records:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# REPAIR ONE TABLE
# ============================================================

def repair_record(
    record: dict,
    record_number: int,
    report: list[dict],
):

    output = deepcopy(
        record
    )

    rows_key = find_rows_key(
        output
    )

    counts = {

        "uncertainty_moves": 0,

        "name_merges": 0,

        "multi_sample_explosions": 0,

        "figure_rows_removed": 0,
    }

    if rows_key is None:

        return (
            output,
            counts,
        )

    table_id = get_table_id(
        output,
        f"record_{record_number:05d}",
    )

    source = get_source(
        output
    )

    original_rows = output[
        rows_key
    ]

    work = []

    for row_index, row in enumerate(
        original_rows
    ):

        values, restore = adapt_row(
            row
        )

        work.append(
            {
                "row_index":
                    row_index,

                "values":
                    values,

                "restore":
                    restore,

                "original":
                    row,
            }
        )

    # ========================================================
    # PASS 1
    # UNCERTAINTY CONTINUATIONS
    #
    # Previous cell MUST contain a number
    # immediately before ±.
    #
    # Therefore:
    #
    # prev = ±
    #
    # will NOT be modified.
    # ========================================================

    for position in range(
        1,
        len(work),
    ):

        previous = work[
            position - 1
        ]["values"]

        current = work[
            position
        ]["values"]

        if previous is None:
            continue

        if current is None:
            continue

        width = min(
            len(previous),
            len(current),
        )

        for column in range(
            width
        ):

            if not ends_with_numeric_pm(
                previous[column]
            ):
                continue

            current_parts = split_parts(
                current[column]
            )

            if not current_parts:
                continue

            first = current_parts[
                0
            ]

            if not is_number_fragment(
                first
            ):
                continue

            before_previous = (
                previous[column]
            )

            before_current = (
                current[column]
            )

            previous[column] = (
                f"{str(previous[column]).rstrip()} "
                f"{first}"
            )

            current[column] = join_parts(
                current_parts[1:]
            )

            counts[
                "uncertainty_moves"
            ] += 1

            report.append(
                {
                    "table_id":
                        table_id,

                    "source":
                        source,

                    "action":
                        "move_uncertainty_fragment",

                    "row_index":
                        work[position][
                            "row_index"
                        ],

                    "column_index":
                        column,

                    "before":
                        (
                            f"prev={before_previous} | "
                            f"cur={before_current}"
                        ),

                    "after":
                        (
                            f"prev={previous[column]} | "
                            f"cur={current[column]}"
                        ),
                }
            )

    # ========================================================
    # PASS 2
    # SAMPLE-NAME CONTINUATION
    #
    # Require previous row to contain
    # at least TWO measurement cells.
    #
    # This prevents reference/citation prose
    # from being merged.
    # ========================================================

    rows_to_remove = set()

    for position in range(
        1,
        len(work),
    ):

        previous = work[
            position - 1
        ]["values"]

        current = work[
            position
        ]["values"]

        if previous is None:
            continue

        if current is None:
            continue

        nonempty_columns = [
            index
            for index, value
            in enumerate(current)
            if clean(value)
        ]

        if len(
            nonempty_columns
        ) != 1:
            continue

        column = (
            nonempty_columns[0]
        )

        if column >= len(
            previous
        ):
            continue

        measurement_cells = sum(
            1
            for index, value
            in enumerate(previous)
            if (
                index != column
                and
                cell_has_measurement(
                    value
                )
            )
        )

        if measurement_cells < 2:
            continue

        if not looks_like_name_continuation(
            previous[column],
            current[column],
        ):
            continue

        before = (
            f"{previous[column]} || "
            f"{current[column]}"
        )

        previous[column] = (
            f"{clean(previous[column])} "
            f"{clean(current[column])}"
        ).strip()

        rows_to_remove.add(
            position
        )

        counts[
            "name_merges"
        ] += 1

        report.append(
            {
                "table_id":
                    table_id,

                "source":
                    source,

                "action":
                    "merge_name_continuation",

                "row_index":
                    work[position][
                        "row_index"
                    ],

                "column_index":
                    column,

                "before":
                    before,

                "after":
                    previous[column],
            }
        )

    if rows_to_remove:

        work = [
            item
            for position, item
            in enumerate(work)
            if position
            not in rows_to_remove
        ]

    # ========================================================
    # PASS 3
    # EXPLODE REAL STACKED SAMPLE ROWS
    # ========================================================

    expanded_rows = []

    for item in work:

        values = item[
            "values"
        ]

        if values is None:

            expanded_rows.append(
                item
            )

            continue

        result = find_multi_sample(
            values
        )

        if result is None:

            expanded_rows.append(
                item
            )

            continue

        (
            sample_column,
            sample_names,
            groups_by_column,
        ) = result

        for sample_index, sample_name in enumerate(
            sample_names
        ):

            new_values = deepcopy(
                values
            )

            new_values[
                sample_column
            ] = sample_name

            for column, groups in groups_by_column.items():

                new_values[
                    column
                ] = groups[
                    sample_index
                ]

            expanded_rows.append(
                {
                    **item,
                    "values":
                        new_values,
                }
            )

        counts[
            "multi_sample_explosions"
        ] += 1

        report.append(
            {
                "table_id":
                    table_id,

                "source":
                    source,

                "action":
                    "explode_multi_sample_row",

                "row_index":
                    item[
                        "row_index"
                    ],

                "column_index":
                    sample_column,

                "before":
                    str(
                        values[
                            sample_column
                        ]
                    ),

                "after":
                    " || ".join(
                        sample_names
                    ),
            }
        )

    work = expanded_rows

    # ========================================================
    # PASS 4
    # REMOVE ONLY STRONG FIGURE AXIS FRAGMENTS
    # ========================================================

    kept_rows = []

    for item in work:

        values = item[
            "values"
        ]

        if values is None:

            kept_rows.append(
                item
            )

            continue

        reason = figure_fragment_reason(
            values
        )

        if reason is None:

            kept_rows.append(
                item
            )

            continue

        counts[
            "figure_rows_removed"
        ] += 1

        report.append(
            {
                "table_id":
                    table_id,

                "source":
                    source,

                "action":
                    "remove_figure_fragment",

                "row_index":
                    item[
                        "row_index"
                    ],

                "column_index":
                    "",

                "before":
                    " | ".join(
                        clean(value)
                        for value in values
                    ),

                "after":
                    reason,
            }
        )

    # ========================================================
    # RESTORE ORIGINAL ROW FORMAT
    # ========================================================

    repaired_rows = []

    for item in kept_rows:

        values = item[
            "values"
        ]

        restore = item[
            "restore"
        ]

        if (
            values is None
            or
            restore is None
        ):

            repaired_rows.append(
                item[
                    "original"
                ]
            )

        else:

            repaired_rows.append(
                restore(
                    values
                )
            )

    output[
        rows_key
    ] = repaired_rows

    output[
        "_step07f3_structural_repair"
    ] = {

        "version":
            2,

        "input_row_count":
            len(original_rows),

        "output_row_count":
            len(repaired_rows),

        **counts,
    }

    return (
        output,
        counts,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_text = os.environ.get(
        "INPUT_JSONL",
        "",
    ).strip()

    if input_text:

        input_path = Path(
            input_text
        ).expanduser()

        if not input_path.is_absolute():

            input_path = (
                ROOT
                / input_path
            )

    else:

        input_path = (
            DEFAULT_INPUT
        )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input JSONL not found: "
            f"{input_path}"
        )

    records = load_jsonl(
        input_path
    )

    print(
        f"Input:  {input_path}"
    )

    print(
        f"Tables: {len(records)}"
    )

    repaired_records = []

    report = []

    totals = {

        "tables_with_rows":
            0,

        "uncertainty_moves":
            0,

        "name_merges":
            0,

        "multi_sample_explosions":
            0,

        "figure_rows_removed":
            0,
    }

    for record_number, record in enumerate(
        records
    ):

        if find_rows_key(
            record
        ):

            totals[
                "tables_with_rows"
            ] += 1

        repaired, counts = repair_record(
            record,
            record_number,
            report,
        )

        repaired_records.append(
            repaired
        )

        for key, value in counts.items():

            totals[
                key
            ] += value

    save_jsonl(
        OUTPUT_JSONL,
        repaired_records,
    )

    fields = [

        "table_id",
        "source",
        "action",
        "row_index",
        "column_index",
        "before",
        "after",
    ]

    with REPORT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            report
        )

    print()

    print(
        "Step 07F3 v2 complete"
    )

    print(
        "-" * 60
    )

    print(
        "Tables with row arrays:      ",
        totals[
            "tables_with_rows"
        ],
    )

    print(
        "Uncertainty fragments moved: ",
        totals[
            "uncertainty_moves"
        ],
    )

    print(
        "Sample names merged:         ",
        totals[
            "name_merges"
        ],
    )

    print(
        "Multi-sample rows exploded:  ",
        totals[
            "multi_sample_explosions"
        ],
    )

    print(
        "Figure rows removed:         ",
        totals[
            "figure_rows_removed"
        ],
    )

    print()

    print(
        f"Output: {OUTPUT_JSONL}"
    )

    print(
        f"Report: {REPORT_CSV}"
    )

    print()

    print(
        "Original normalized input was NOT overwritten."
    )


if __name__ == "__main__":
    main()
