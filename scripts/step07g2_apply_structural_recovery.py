#!/usr/bin/env python3

"""
STEP 07G2
=========

High-confidence structural recovery after Step 07G.

This stage does NOT modify the frozen upstream files.

Inputs
------
processed_tables/deterministic_candidates.csv
processed_tables/semantic_recovery_queue.csv
processed_tables/rejected_table_rows.csv
processed_tables/repaired_candidate_tables.csv

Outputs
-------
processed_tables/structural_routed_rows.csv
processed_tables/deterministic_candidates_structural.csv
processed_tables/semantic_recovery_queue_structural.csv
processed_tables/rejected_table_rows_structural.csv
processed_tables/structural_recovery_audit.csv
processed_tables/structural_queue_summary.csv

Rules
-----
1. Shifted sample/material column:
   fill or correct sample_raw when the table header structurally shows
   a blank first column followed by an explicit sample/material column.

2. Supported sample-temperature code series:
   codes such as O300/O500/O700 or GH400/GH600 are accepted only when
   the same alphabetic stem occurs at >=2 distinct plausible
   temperatures within the same table.

3. Explicit °C*min condition:
   e.g. 450*20 under a °C*min header -> temperature 450 °C.
   Sample identity is not invented.

4. Parenthetical uncertainty companion rows:
   reject rows consisting only of uncertainty / missing-value
   companion cells.

5. Statistical summary rows:
   reject LSD and CV summary rows.

6. Separate uncertainty row following a ± mean row:
   reject only under strict structural evidence.

Principles
----------
- no paper IDs are hard-coded;
- no benchmark identities are used;
- source_key is preserved;
- raw source fields are never overwritten;
- ambiguous rows remain unresolved;
- recovery enriches data rather than fabricating missing context.
"""

from pathlib import Path
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

pt = (
    project_dir
    / "processed_tables"
)

det_path = (
    pt
    / "deterministic_candidates.csv"
)

rec_path = (
    pt
    / "semantic_recovery_queue.csv"
)

rej_path = (
    pt
    / "rejected_table_rows.csv"
)

meta_path = (
    pt
    / "repaired_candidate_tables.csv"
)


full_output_path = (
    pt
    / "structural_routed_rows.csv"
)

det_output_path = (
    pt
    / "deterministic_candidates_structural.csv"
)

rec_output_path = (
    pt
    / "semantic_recovery_queue_structural.csv"
)

rej_output_path = (
    pt
    / "rejected_table_rows_structural.csv"
)

audit_output_path = (
    pt
    / "structural_recovery_audit.csv"
)

summary_output_path = (
    pt
    / "structural_queue_summary.csv"
)


# ============================================================
# Helpers
# ============================================================

ELEMENTS = [
    "C",
    "H",
    "N",
    "O",
]


def clean(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def strip_markup(value):

    text = clean(
        value
    )

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = (
        text
        .replace(
            "**",
            "",
        )
        .replace(
            "_",
            "",
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def row_cells(raw):

    return [
        clean(cell)
        for cell
        in clean(raw).split("|")
    ]


def first_nonempty_cell(raw):

    for cell in row_cells(
        raw
    ):

        text = strip_markup(
            cell
        )

        if text:
            return text

    return ""


def numeric(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def valid_element_count(row):

    count = 0

    for element in ELEMENTS:

        value = numeric(
            row.get(
                f"{element}_value"
            )
        )

        if (
            value is not None
            and 0 <= value <= 100
        ):
            count += 1

    return count


# ============================================================
# Load current 07G queues
# ============================================================

det = pd.read_csv(
    det_path,
    low_memory=False,
)

rec = pd.read_csv(
    rec_path,
    low_memory=False,
)

rej = pd.read_csv(
    rej_path,
    low_memory=False,
)


det[
    "queue_before_07g2"
] = "deterministic"

rec[
    "queue_before_07g2"
] = "semantic_recovery"

rej[
    "queue_before_07g2"
] = "rejected"


df = pd.concat(
    [
        det,
        rec,
        rej,
    ],
    ignore_index=True,
    sort=False,
)


df[
    "_original_order_07g2"
] = range(
    len(df)
)


print()
print("=" * 82)
print(
    "STEP 07G2 — "
    "STRUCTURAL RECOVERY"
)
print("=" * 82)

print(
    "Rows loaded:",
    len(df),
)

print(
    "Source-key duplicates:",
    df[
        "source_key"
    ].duplicated().sum(),
)


if df[
    "source_key"
].duplicated().any():

    raise RuntimeError(
        "Duplicate source_key found "
        "before structural recovery."
    )


# ============================================================
# Preserve pre-stage state
# ============================================================

df[
    "semantic_class_before_07g2"
] = df[
    "semantic_class"
]

df[
    "sample_raw_before_07g2"
] = df[
    "sample_raw"
]

df[
    "temperature_C_before_07g2"
] = df[
    "temperature_C"
]


df[
    "structural_action_07g2"
] = None

df[
    "structural_recovery_applied_07g2"
] = False

df[
    "sample_recovered_07g2"
] = False

df[
    "sample_corrected_07g2"
] = False

df[
    "temperature_recovered_07g2"
] = False

df[
    "structural_rejected_07g2"
] = False


# Distinguish a genuinely missing treatment temperature from
# a temperature that is scientifically not applicable, e.g.
# raw biomass/feedstock composition.
df[
    "temperature_not_applicable_07g2"
] = False


# ============================================================
# Load repaired table metadata
# ============================================================

meta = pd.read_csv(
    meta_path,
    low_memory=False,
)


header_map = {}

for _, row in meta.iterrows():

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    if (
        table_id
        and table_id
        not in header_map
    ):

        header_map[
            table_id
        ] = clean(
            row.get(
                "combined_header"
            )
        )


# ============================================================
# RULE 1
#
# Detect structurally shifted sample/material tables.
# ============================================================

shifted_tables = set()

sample_header_terms = (
    "biomass type",
    "feedstock",
    "sample",
    "material",
)


for table_id, header in (
    header_map.items()
):

    header_cells = [
        strip_markup(cell)
        for cell
        in header.split("|")
    ]

    if len(
        header_cells
    ) < 2:
        continue

    # Structural pattern:
    #
    # blank first column | explicit material/sample column
    if header_cells[0]:
        continue

    second_header = (
        header_cells[1]
        .lower()
    )

    if any(
        term in second_header
        for term
        in sample_header_terms
    ):

        shifted_tables.add(
            table_id
        )


shifted_proposals = {}


for idx, row in df.iterrows():

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    if (
        table_id
        not in shifted_tables
    ):
        continue

    semantic_class = clean(
        row.get(
            "semantic_class"
        )
    )

    if semantic_class.startswith(
        "REJECT"
    ):
        continue

    cells = row_cells(
        row.get(
            "raw_source_row"
        )
    )

    if len(cells) < 2:
        continue

    proposed = strip_markup(
        cells[1]
    )

    if not proposed:
        continue

    if not re.search(
        r"[A-Za-z]",
        proposed,
    ):
        continue

    # Do not recover visibly broken text such as:
    # "Mesocarp fibre (oil"
    if (
        proposed.count("(")
        != proposed.count(")")
    ):
        continue

    current = strip_markup(
        row.get(
            "sample_raw"
        )
    )

    if current == proposed:
        continue

    shifted_proposals[
        idx
    ] = {
        "sample":
            proposed,

        "action":
            (
                "SHIFTED_SAMPLE_COLUMN_FILL"
                if not current
                else
                "SHIFTED_SAMPLE_COLUMN_CORRECT"
            ),
    }


# ============================================================
# Work on original SEMANTIC_RECOVERY_REQUIRED rows
# ============================================================

semantic_required_indices = set(
    df.index[
        df[
            "semantic_class_before_07g2"
        ].eq(
            "SEMANTIC_RECOVERY_REQUIRED"
        )
    ]
)


# All rows that were not already rejected upstream.
#
# Some structural artifacts can arrive with classes such as
# NEEDS_TEMPERATURE_INTERPRETATION rather than
# SEMANTIC_RECOVERY_REQUIRED, so rejection rules that are
# independent of semantic class may inspect this wider set.
active_candidate_indices = set(
    df.index[
        ~df[
            "semantic_class_before_07g2"
        ]
        .astype(str)
        .str.startswith("REJECT")
    ]
)


# ============================================================
# RULE 2
#
# Supported sample-temperature code series.
# ============================================================

code_candidates = {}


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    first = (
        first_nonempty_cell(
            row.get(
                "raw_source_row"
            )
        )
    )

    compact = re.sub(
        r"\s+",
        "",
        first,
    )

    match = re.fullmatch(
        r"([A-Za-z][A-Za-z-]*)(\d{3,4})",
        compact,
    )

    if not match:
        continue

    stem = (
        match.group(1)
        .lower()
    )

    temperature = float(
        match.group(2)
    )

    if not (
        100
        <= temperature
        <= 3000
    ):
        continue

    key = (
        clean(
            row.get(
                "table_id"
            )
        ),
        stem,
    )

    code_candidates[
        idx
    ] = {
        "key":
            key,

        "sample":
            compact,

        "temperature":
            temperature,
    }


series_members = {}


for idx, info in (
    code_candidates.items()
):

    series_members.setdefault(
        info[
            "key"
        ],
        [],
    ).append(
        (
            idx,
            info[
                "temperature"
            ],
        )
    )


supported_code_indices = set()


for _, members in (
    series_members.items()
):

    temperatures = {
        temperature
        for _, temperature
        in members
    }

    if (
        len(members) >= 2
        and len(
            temperatures
        ) >= 2
    ):

        supported_code_indices.update(
            idx
            for idx, _
            in members
        )


# ============================================================
# RULE 3
#
# Explicit number*number condition under °C*min header.
# ============================================================

star_temperature_proposals = {}


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    header = strip_markup(
        header_map.get(
            table_id,
            "",
        )
    )

    normalized_header = (
        header
        .replace(
            "◦",
            "°",
        )
        .replace(
            "º",
            "°",
        )
        .lower()
    )

    if not (
        "c*min"
        in normalized_header
        or
        "°c*min"
        in normalized_header
    ):
        continue

    first = (
        first_nonempty_cell(
            row.get(
                "raw_source_row"
            )
        )
    )

    match = re.fullmatch(
        r"\s*(\d{3,4})\s*\*\s*"
        r"(\d+(?:\.\d+)?)\s*",
        first,
    )

    if not match:
        continue

    temperature = float(
        match.group(1)
    )

    if (
        100
        <= temperature
        <= 3000
    ):

        star_temperature_proposals[
            idx
        ] = temperature


# ============================================================
# RULE 4
#
# Parenthetical / marked uncertainty companion rows.
# ============================================================

def parenthetical_companion(row):

    raw = clean(
        row.get(
            "raw_source_row"
        )
    )

    original_cells = (
        row_cells(
            raw
        )
    )

    if not original_cells:
        return False

    # Companion row begins with an empty sample cell.
    if strip_markup(
        original_cells[0]
    ):
        return False

    qualifying = 0
    nonempty = 0


    for original in (
        original_cells
    ):

        cell = strip_markup(
            original
        )

        if not cell:
            continue

        nonempty += 1

        low = cell.lower()

        if low in {
            "n.a.",
            "n.a",
            "na",
            "n/a",
            "-",
            "–",
            "—",
        }:

            qualifying += 1
            continue


        if re.fullmatch(
            r"\(\s*[-+]?"
            r"(?:\d+(?:\.\d+)?|\.\d+)"
            r"\s*\)",
            cell,
        ):

            qualifying += 1
            continue


        # Markdown can preserve:
        #
        # _0.0_
        #
        # rather than:
        #
        # _(0.0)_
        if re.fullmatch(
            r"_\s*[-+]?"
            r"(?:\d+(?:\.\d+)?|\.\d+)"
            r"\s*_",
            clean(
                original
            ),
        ):

            qualifying += 1
            continue


    return (
        nonempty >= 2
        and qualifying
        == nonempty
    )


parenthetical_rejects = set()


for idx in (
    semantic_required_indices
):

    if parenthetical_companion(
        df.loc[
            idx
        ]
    ):

        parenthetical_rejects.add(
            idx
        )


# ============================================================
# RULE 5
#
# LSD / CV statistical summary rows.
# ============================================================

statistical_rejects = set()


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    first = (
        first_nonempty_cell(
            row.get(
                "raw_source_row"
            )
        )
    )

    if re.match(
        r"^(?:"
        r"LSD(?:\s*\d|\b)"
        r"|"
        r"CV\s*\(%?\)"
        r")",
        first,
        flags=re.IGNORECASE,
    ):

        statistical_rejects.add(
            idx
        )


# ============================================================
# RULE 6
#
# Separate uncertainty row following a ± mean row.
# ============================================================

table_groups = {}


for table_id, group in (
    df.groupby(
        "table_id",
        sort=False,
    )
):

    table_groups[
        str(table_id)
    ] = (
        group
        .sort_values(
            "source_row_index"
        )
        .copy()
    )


following_pm_rejects = set()


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    source_row_index = (
        row.get(
            "source_row_index"
        )
    )

    table = table_groups.get(
        table_id
    )

    if table is None:
        continue

    positions = list(
        table.index
    )

    if idx not in positions:
        continue

    position = positions.index(
        idx
    )

    if position == 0:
        continue

    previous = table.iloc[
        position - 1
    ]

    if not clean(
        previous.get(
            "sample_raw"
        )
    ):
        continue

    pm_count = 0

    for element in ELEMENTS:

        raw = clean(
            previous.get(
                f"{element}_candidate_raw"
            )
        )

        if "±" in raw:
            pm_count += 1

    if pm_count < 2:
        continue

    values = []

    for element in ELEMENTS:

        value = numeric(
            row.get(
                f"{element}_value"
            )
        )

        if value is not None:

            values.append(
                value
            )

    if len(values) < 2:
        continue

    if not all(
        0 <= value <= 1
        for value in values
    ):
        continue

    following_pm_rejects.add(
        idx
    )




# ============================================================
# RULE 6A
#
# Table-level sample identity for explicit °C*min condition
# series.
#
# Some scientific tables encode only temperature*time in the
# first data column because the material identity applies to
# the entire table and appears in the caption/header.
#
# Recovery is restricted to tables already supported by the
# strict star-temperature rule.
# ============================================================

star_table_sample_map = {}


star_table_ids = {
    clean(
        df.at[
            idx,
            "table_id",
        ]
    )
    for idx in star_temperature_proposals
}


for table_id in star_table_ids:

    header = clean(
        header_map.get(
            table_id,
            ""
        )
    )

    header_cells = [
        strip_markup(cell).lower()
        for cell in header.split("|")
    ]

    candidates = set()

    # Normal intact wording:
    # "... analysis of cornstalk biochars ..."
    full_header = " ".join(
        header_cells
    )

    for match in re.finditer(
        r"\bof\s+"
        r"([a-z][a-z -]{2,40}?)"
        r"\s+biochars?\b",
        full_header,
        flags=re.IGNORECASE,
    ):

        candidate = re.sub(
            r"\s+",
            " ",
            match.group(1),
        ).strip()

        if (
            candidate
            and len(candidate) <= 40
        ):
            candidates.add(
                candidate.lower()
            )

    # PDF table extraction can split a word across adjacent
    # columns:
    #
    # "... of cornstal ..." | "k biochars ..."
    #
    # Reconstruct only this tightly supported pattern.
    for left, right in zip(
        header_cells,
        header_cells[1:],
    ):

        left_match = re.search(
            r"\bof\s+([a-z]{3,24})\b",
            left,
            flags=re.IGNORECASE,
        )

        right_match = re.match(
            r"\s*([a-z])\s+biochars?\b",
            right,
            flags=re.IGNORECASE,
        )

        if (
            left_match
            and right_match
        ):

            candidate = (
                left_match.group(1)
                + right_match.group(1)
            ).lower()

            candidates.add(
                candidate
            )

    if len(candidates) == 1:

        star_table_sample_map[
            table_id
        ] = next(
            iter(candidates)
        )


# ============================================================
# RULE 6B
#
# Shifted sample name continued onto the following rejected
# row.
#
# Example structural pattern:
#
# blank | Mesocarp fibre (oil | ...
# blank | palm)              | ...
#
# The continuation is accepted only when:
# - the table already has the shifted sample-column pattern;
# - the first fragment has unmatched opening parentheses;
# - the immediately following row supplies the balancing
#   continuation;
# - the following row is already classified as rejected.
# ============================================================

continued_sample_proposals = {}


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    if table_id not in shifted_tables:
        continue

    cells = row_cells(
        row.get(
            "raw_source_row"
        )
    )

    if len(cells) < 2:
        continue

    if strip_markup(
        cells[0]
    ):
        continue

    first_fragment = strip_markup(
        cells[1]
    )

    if not re.search(
        r"[A-Za-z]",
        first_fragment,
    ):
        continue

    if (
        first_fragment.count("(")
        <=
        first_fragment.count(")")
    ):
        continue

    table = table_groups.get(
        table_id
    )

    if table is None:
        continue

    positions = list(
        table.index
    )

    if idx not in positions:
        continue

    position = positions.index(
        idx
    )

    if position + 1 >= len(positions):
        continue

    next_idx = positions[
        position + 1
    ]

    next_row = df.loc[
        next_idx
    ]

    next_class = clean(
        next_row.get(
            "semantic_class_before_07g2"
        )
    )

    if not next_class.startswith(
        "REJECT"
    ):
        continue

    next_cells = row_cells(
        next_row.get(
            "raw_source_row"
        )
    )

    if len(next_cells) < 2:
        continue

    if strip_markup(
        next_cells[0]
    ):
        continue

    continuation = strip_markup(
        next_cells[1]
    )

    if not re.search(
        r"[A-Za-z]",
        continuation,
    ):
        continue

    combined = (
        first_fragment
        + " "
        + continuation
    )

    combined = re.sub(
        r"\s+",
        " ",
        combined,
    ).strip()

    if (
        combined.count("(")
        != combined.count(")")
    ):
        continue

    if len(combined) > 100:
        continue

    continued_sample_proposals[
        idx
    ] = combined


# ============================================================
# RULE 6C
#
# Explicit sample + temperature in first two columns when the
# upstream header detector failed.
#
# Restricted to unresolved rows where:
# - first cell is a compact material identifier;
# - second cell is a plausible temperature;
# - second header cell explicitly denotes °C / temperature;
# - at least one mapped elemental value exists.
# ============================================================

explicit_sample_temp_proposals = {}


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    if clean(
        row.get(
            "sample_raw"
        )
    ):
        continue

    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    cells = row_cells(
        row.get(
            "raw_source_row"
        )
    )

    if len(cells) < 2:
        continue

    sample = strip_markup(
        cells[0]
    )

    temperature_text = strip_markup(
        cells[1]
    )

    if not re.fullmatch(
        r"[A-Za-z]"
        r"[A-Za-z0-9+\-]{1,15}",
        sample,
    ):
        continue

    temperature = numeric(
        temperature_text
    )

    if (
        temperature is None
        or not (
            100
            <= temperature
            <= 3000
        )
    ):
        continue

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    header = clean(
        header_map.get(
            table_id,
            ""
        )
    )

    header_cells = header.split(
        "|"
    )

    if len(header_cells) < 2:
        continue

    second_header = (
        strip_markup(
            header_cells[1]
        )
        .lower()
        .replace(
            "◦",
            "°"
        )
        .replace(
            "º",
            "°"
        )
    )

    header_supports_temperature = (
        (
            "°"
            in second_header
            and "c"
            in second_header
        )
        or "temperature"
        in second_header
        or "temp"
        in second_header
        or "empera"
        in second_header
    )

    if not header_supports_temperature:
        continue

    if valid_element_count(
        row
    ) < 1:
        continue

    explicit_sample_temp_proposals[
        idx
    ] = {
        "sample":
            sample,

        "temperature":
            float(
                temperature
            ),
    }


# ============================================================
# RULE 6D
#
# Broken header / caption text incorrectly emitted as a data
# row.
#
# Conservative rejection:
# - unresolved row;
# - no sample identity;
# - at most two mapped element values;
# - first textual cell itself identifies a table/header or
#   descriptive pH heading rather than a sample.
# ============================================================

header_text_fragment_rejects = set()


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    if clean(
        row.get(
            "sample_raw"
        )
    ):
        continue

    if valid_element_count(
        row
    ) > 2:
        continue

    first = first_nonempty_cell(
        row.get(
            "raw_source_row"
        )
    )

    low = first.lower()

    starts_table_heading = bool(
        re.match(
            r"^table\s*\d+",
            low,
            flags=re.IGNORECASE,
        )
    )

    descriptive_ph_heading = (
        (
            "ph"
            in low
        )
        and (
            "studied"
            in low
            or "cacl"
            in low
        )
    )

    if (
        starts_table_heading
        or descriptive_ph_heading
    ):

        header_text_fragment_rejects.add(
            idx
        )


# ============================================================
# RULE 7
#
# Correlation / coefficient matrix rows.
#
# Pattern:
# - variable-like first cell containing an underscore;
# - many numeric cells;
# - all numeric values constrained to [-1, 1].
#
# This distinguishes correlation coefficients from CHNO
# composition rows without using paper-specific identifiers.
# ============================================================

correlation_matrix_rejects = set()


def correlation_matrix_like(row):

    # Preserve underscores here because names such as
    # T_WO and qm_SG identify correlation variables.
    # strip_markup() intentionally removes underscores,
    # so it must not be used for this identifier test.
    raw_cells = row_cells(
        row.get(
            "raw_source_row"
        )
    )

    first = ""

    for raw_cell in raw_cells:

        candidate = clean(
            raw_cell
        )

        candidate = re.sub(
            r"<br\\s*/?>",
            " ",
            candidate,
            flags=re.IGNORECASE,
        )

        candidate = re.sub(
            r"<[^>]+>",
            " ",
            candidate,
        )

        candidate = candidate.replace(
            "**",
            "",
        )

        candidate = re.sub(
            r"\\s+",
            " ",
            candidate,
        ).strip()

        if candidate:
            first = candidate
            break

    if not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9]*"
        r"(?:_[A-Za-z0-9%+\-]+)+",
        first,
    ):
        return False

    cells = [
        strip_markup(cell)
        for cell in row_cells(
            row.get(
                "raw_source_row"
            )
        )
    ]

    trailing = [
        cell
        for cell in cells[1:]
        if cell
    ]

    if len(trailing) < 8:
        return False

    values = []

    for cell in trailing:

        normalized = (
            cell
            .replace(
                "−",
                "-"
            )
            .replace(
                "–",
                "-"
            )
        )

        value = numeric(
            normalized
        )

        if value is None:
            return False

        values.append(
            value
        )

    if len(values) < 8:
        return False

    return all(
        -1.000001
        <= value
        <= 1.000001
        for value in values
    )


for idx in (
    active_candidate_indices
):

    if correlation_matrix_like(
        df.loc[
            idx
        ]
    ):

        correlation_matrix_rejects.add(
            idx
        )


# ============================================================
# RULE 8
#
# Figure-derived table fragments.
#
# A table is considered figure-like when its reconstructed
# text explicitly identifies a Van Krevelen diagram, or when
# both H/C and O/C atomic-ratio axis terminology are present.
#
# Only unresolved rows with no sample identity are rejected.
# ============================================================

figure_like_tables = set()


for table_id, table in (
    table_groups.items()
):

    combined = " ".join(
        strip_markup(
            value
        )
        for value in (
            table[
                "raw_source_row"
            ]
            .fillna("")
            .astype(str)
        )
    )

    low = (
        combined
        .lower()
        .replace(
            "−",
            "-"
        )
    )

    is_van_krevelen = (
        "van krevelen"
        in low
    )

    has_hc_axis = bool(
        re.search(
            r"\bh\s*/\s*c\b",
            low,
        )
    )

    has_oc_axis = bool(
        re.search(
            r"\bo\s*/\s*c\b",
            low,
        )
    )

    has_atomic_ratio = (
        "atomic ratio"
        in low
    )

    if (
        is_van_krevelen
        or (
            has_hc_axis
            and has_oc_axis
            and has_atomic_ratio
        )
    ):

        figure_like_tables.add(
            table_id
        )


figure_fragment_rejects = set()


for idx in (
    semantic_required_indices
):

    row = df.loc[
        idx
    ]

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    if (
        table_id
        not in figure_like_tables
    ):
        continue

    if clean(
        row.get(
            "sample_raw"
        )
    ):
        continue

    figure_fragment_rejects.add(
        idx
    )


# ============================================================
# RULE 9
#
# Orphan small-number CHNO companion rows.
#
# Conservative pattern:
# - unresolved semantic row;
# - no sample identity;
# - no alphabetic text in the source row;
# - at least three mapped elemental values;
# - every mapped elemental value <= 1 wt%.
#
# Such a row cannot plausibly represent a complete carbon
# material composition and is characteristic of uncertainty
# fragments separated from a preceding mean row.
# ============================================================

small_numeric_companion_rejects = set()


for idx in (
    semantic_required_indices
):

    # Rule 9 is deliberately a fallback.
    # If a stronger structural rule already explains the row,
    # do not assign a second action.
    if (
        idx in parenthetical_rejects
        or idx in statistical_rejects
        or idx in following_pm_rejects
        or idx in correlation_matrix_rejects
        or idx in figure_fragment_rejects
    ):
        continue

    row = df.loc[
        idx
    ]

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    # Figure fragments are handled by Rule 8.
    if (
        table_id
        in figure_like_tables
    ):
        continue

    if clean(
        row.get(
            "sample_raw"
        )
    ):
        continue

    raw_text = strip_markup(
        row.get(
            "raw_source_row"
        )
    )

    if re.search(
        r"[A-Za-z]",
        raw_text,
    ):
        continue

    element_values = []

    for element in ELEMENTS:

        value = numeric(
            row.get(
                f"{element}_value"
            )
        )

        if value is not None:
            element_values.append(
                value
            )

    if len(
        element_values
    ) < 3:
        continue

    if all(
        0 <= value <= 1
        for value
        in element_values
    ):

        small_numeric_companion_rejects.add(
            idx
        )



# ============================================================
# RULE 10
#
# Treatment temperature NOT APPLICABLE for raw feedstock
# composition tables.
#
# Important scientific distinction:
#
#   missing temperature
#       !=
#   temperature not applicable
#
# A raw biomass ultimate/proximate-analysis table should not
# be sent to semantic recovery merely because it has no
# pyrolysis/carbonization temperature.
#
# Conservative table-level evidence required:
# - header explicitly identifies BIOMASS;
# - header describes proximate/ultimate analysis OR contains
#   the classic VM / FC / Ash raw-analysis structure;
# - header does NOT identify biochar/char yield, pyrolysis,
#   carbonization, torrefaction, HTC, or treatment temperature.
#
# Tmax is deliberately NOT treated as a process temperature.
# ============================================================

raw_feedstock_table_ids = set()


for table_id, header in (
    header_map.items()
):

    raw_header = clean(
        header
    )

    low = (
        strip_markup(
            raw_header
        )
        .lower()
        .replace(
            "◦",
            "°"
        )
        .replace(
            "º",
            "°"
        )
    )

    header_cells = [
        strip_markup(cell)
        .lower()
        .strip()
        for cell in raw_header.split("|")
    ]

    has_biomass = bool(
        re.search(
            r"\bbiomass\b",
            low,
        )
    )

    has_analysis_words = (
        "proxim"
        in low
        or "ultimat"
        in low
    )

    compact_cells = {
        re.sub(
            r"\s+",
            " ",
            cell,
        ).strip()
        for cell in header_cells
    }

    has_classic_raw_analysis = (
        "vm"
        in compact_cells
        and "fc"
        in compact_cells
        and "ash"
        in compact_cells
    )

    process_indicators = (
        "biochar",
        "biochars",
        "bc yield",
        "char yield",
        "pyrolys",
        "carboniz",
        "carbonis",
        "torref",
        "hydrothermal",
        "treatment temperature",
        "carbonization temperature",
        "carbonisation temperature",
    )

    has_process_indicator = any(
        term in low
        for term in process_indicators
    )

    # Standalone HTC token.
    has_htc = bool(
        re.search(
            r"\bhtc\b",
            low,
        )
    )

    if (
        has_biomass
        and (
            has_analysis_words
            or has_classic_raw_analysis
        )
        and not has_process_indicator
        and not has_htc
    ):

        raw_feedstock_table_ids.add(
            table_id
        )


temperature_not_applicable_indices = set()


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    if (
        table_id
        not in raw_feedstock_table_ids
    ):
        continue

    # Do not override a real extracted treatment temperature.
    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    # Must contain actual elemental information.
    if valid_element_count(
        row
    ) < 1:
        continue

    temperature_not_applicable_indices.add(
        idx
    )




# ============================================================
# RULE 10B
#
# Explicit precursor / raw-feedstock rows.
#
# These rows contain valid elemental composition, but a
# treatment temperature is scientifically not applicable.
#
# Supported evidence:
#
# 1. sample identifier explicitly ends in "feedstock";
#
# 2. sample identifier has PREFIX-Raw and the same table
#    contains PREFIX-<temperature>C processed siblings;
#
# 3. a transposed raw-vs-biochar table contains an OCR-like
#    "Row <material>" precursor column followed by explicit
#    biochar pyrolysis-temperature columns. In this specific
#    structural situation "Row" is repaired to "Raw".
#
# No paper IDs or benchmark sample identities are used.
# ============================================================

explicit_precursor_temp_na_indices = set()

raw_feedstock_label_corrections = {}


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    # A real treatment temperature already present wins.
    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    if valid_element_count(
        row
    ) < 1:
        continue

    sample = strip_markup(
        row.get(
            "sample_raw"
        )
    ).strip()

    if not sample:
        continue

    sample_low = sample.lower()

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    # --------------------------------------------------------
    # Case 1:
    # Explicit feedstock label.
    #
    # Examples structurally include:
    #   X-feedstock
    #   Feedstock
    # --------------------------------------------------------

    explicit_feedstock = bool(
        re.search(
            r"(?:^|[-_\s])"
            r"feedstocks?"
            r"\s*$",
            sample_low,
        )
    )

    if explicit_feedstock:

        explicit_precursor_temp_na_indices.add(
            idx
        )

        continue


    # --------------------------------------------------------
    # Case 2:
    # PREFIX-Raw with processed siblings such as
    # PREFIX-200C, PREFIX-400C, ...
    #
    # Requiring sibling evidence prevents treating a phrase
    # such as "raw biochar" as an untreated precursor.
    # --------------------------------------------------------

    raw_match = re.fullmatch(
        r"(.+?)[-_]raw",
        sample,
        flags=re.IGNORECASE,
    )

    if raw_match:

        prefix = (
            raw_match
            .group(1)
            .strip()
        )

        sibling_count = 0

        table = table_groups.get(
            table_id
        )

        if table is not None:

            sibling_pattern = re.compile(
                r"^"
                + re.escape(prefix)
                + r"[-_]"
                + r"\d{3,4}"
                + r"\s*°?\s*C"
                + r"$",
                flags=re.IGNORECASE,
            )

            for _, sibling in (
                table.iterrows()
            ):

                sibling_sample = (
                    strip_markup(
                        sibling.get(
                            "sample_raw"
                        )
                    )
                    .strip()
                )

                if sibling_pattern.fullmatch(
                    sibling_sample
                ):

                    sibling_count += 1

        if sibling_count >= 1:

            explicit_precursor_temp_na_indices.add(
                idx
            )

            continue


    # --------------------------------------------------------
    # Case 3:
    # OCR-like "Row <material>" precursor heading in a
    # transposed table explicitly contrasting the precursor
    # against biochars at pyrolysis temperatures.
    #
    # We do not globally change "Row" -> "Raw". The repair is
    # allowed only when this strong table structure exists.
    # --------------------------------------------------------

    row_match = re.fullmatch(
        r"Row\s+(.+)",
        sample,
        flags=re.IGNORECASE,
    )

    if not row_match:
        continue

    header = (
        strip_markup(
            header_map.get(
                table_id,
                ""
            )
        )
        .lower()
    )

    has_raw_vs_biochar_structure = (
        "biochar"
        in header
        and "pyrolys"
        in header
        and "temperature"
        in header
        and sample_low
        in header
    )

    if not has_raw_vs_biochar_structure:
        continue

    corrected_sample = (
        "Raw "
        + row_match.group(1).strip()
    )

    raw_feedstock_label_corrections[
        idx
    ] = corrected_sample

    explicit_precursor_temp_na_indices.add(
        idx
    )


# Merge with the previously established table-level
# temperature-not-applicable set.
temperature_not_applicable_indices.update(
    explicit_precursor_temp_na_indices
)



# ============================================================
# RULE 10C
#
# Explicit biomass/feedstock composition tables.
#
# A row may be treated as temperature-not-applicable when:
#
# - the first table header explicitly identifies the table as
#   Biomass or Feedstock;
# - the row contains elemental-composition data;
# - no treatment temperature is already present;
# - the table does not describe products such as biochar,
#   hydrochar, or char yield;
# - the table does not contain an explicit Process +
#   Temperature schema.
#
# The final condition is important. A row such as:
#
#   Rice straw | Pyrolysis | -
#
# has a genuinely missing process temperature and must remain
# in semantic recovery rather than becoming TEMP_NA.
# ============================================================

explicit_raw_table_temp_na_indices = set()


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    if valid_element_count(
        row
    ) < 1:
        continue

    # Already classified through a stronger TEMP_NA rule.
    if (
        idx
        in temperature_not_applicable_indices
    ):
        continue

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    raw_header = clean(
        header_map.get(
            table_id,
            ""
        )
    )

    header = (
        strip_markup(
            raw_header
        )
        .lower()
    )

    first_header = (
        strip_markup(
            raw_header.split("|")[0]
        )
        .lower()
        .strip()
    )

    explicit_raw_table = (
        first_header.startswith(
            "biomass"
        )
        or first_header.startswith(
            "feedstoc"
        )
    )

    if not explicit_raw_table:
        continue

    product_indicators = (
        "biochar",
        "hydrochar",
        "bc yield",
        "char yield",
        "pyrolys",
        "carboniz",
        "carbonis",
    )

    has_product_indicator = any(
        term in header
        for term in product_indicators
    )

    if has_product_indicator:
        continue

    # Explicit process schema means the process is applicable,
    # even if this particular row has a missing temperature.
    has_process_temperature_schema = (
        bool(
            re.search(
                r"\bprocess\b",
                header,
            )
        )
        and bool(
            re.search(
                r"\btemperature\b",
                header,
            )
        )
    )

    if has_process_temperature_schema:
        continue

    sample = (
        strip_markup(
            row.get(
                "sample_raw"
            )
        )
        .lower()
        .strip()
    )

    product_row_labels = {
        "biochar",
        "bio-oil",
        "bio oil",
        "hydrochar",
        "char",
    }

    if sample in product_row_labels:
        continue

    explicit_raw_table_temp_na_indices.add(
        idx
    )


temperature_not_applicable_indices.update(
    explicit_raw_table_temp_na_indices
)



# ============================================================
# RULE 10D
#
# Raw precursor identified by a derived-biochar product family.
#
# This handles mixed precursor/product tables where:
#
# - a yield or biochar-yield column exists;
# - the candidate precursor has missing yield;
# - the precursor name implies an abbreviation formed from
#   word initials + "B";
# - at least two sibling rows in the same table belong to that
#   biochar family and have numeric yields.
#
# Examples of the structural relationship:
#
#   Corn stalk     -> CSB, CSB-...
#   Rice husk      -> RHB, RHB-...
#   Rice straw     -> RSB300, RSB500, ...
#   Sewage sludge  -> SSB300, SSB500, ...
#
# A comparison material such as lignite is not captured because
# it has no corresponding derived-biochar sibling family.
#
# No paper IDs or benchmark sample identities are used.
# ============================================================

product_family_precursor_temp_na_indices = set()


def missing_yield_value_07g2(
    value,
):

    value = (
        strip_markup(
            value
        )
        .lower()
        .strip()
    )

    return value in {
        "",
        "-",
        "–",
        "—",
        "na",
        "n.a.",
        "n.a",
        "nd",
        "n.d.",
        "n.d",
    }


def precursor_family_prefix_07g2(
    sample,
):

    sample = strip_markup(
        sample
    )

    words = re.findall(
        r"[A-Za-z]+",
        sample,
    )

    if not words:
        return None

    stop_words = {
        "of",
        "the",
        "and",
    }

    words = [
        word
        for word in words
        if word.lower()
        not in stop_words
    ]

    if not words:
        return None

    initials = "".join(
        word[0].upper()
        for word in words
    )

    return (
        initials
        + "B"
    )


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    # Existing numeric treatment temperature wins.
    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    # Already resolved as TEMP_NA by a stronger rule.
    if (
        idx
        in temperature_not_applicable_indices
    ):
        continue

    if valid_element_count(
        row
    ) < 1:
        continue

    sample = (
        strip_markup(
            row.get(
                "sample_raw"
            )
        )
        .strip()
    )

    if not sample:
        continue

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    raw_header = clean(
        header_map.get(
            table_id,
            ""
        )
    )

    header_cells = [
        (
            strip_markup(
                cell
            )
            .lower()
            .strip()
        )
        for cell in raw_header.split(
            "|"
        )
    ]

    yield_indices = [
        column_index
        for column_index, header_cell
        in enumerate(
            header_cells
        )
        if re.search(
            r"\byield\b",
            header_cell,
        )
    ]

    if not yield_indices:
        continue

    yield_index = (
        yield_indices[0]
    )

    source_cells = str(
        row.get(
            "raw_source_row",
            ""
        )
    ).split(
        "|"
    )

    if (
        yield_index
        >= len(source_cells)
    ):
        continue

    # The precursor itself must not have a numeric product
    # yield.
    if not missing_yield_value_07g2(
        source_cells[
            yield_index
        ]
    ):
        continue

    family_prefix = (
        precursor_family_prefix_07g2(
            sample
        )
    )

    if not family_prefix:
        continue

    family_pattern = re.compile(
        r"^"
        + re.escape(
            family_prefix
        )
        + r"(?:$|[-_]|\d)",
        flags=re.IGNORECASE,
    )

    table = table_groups.get(
        table_id
    )

    if table is None:
        continue

    supported_siblings = 0

    for sibling_idx, sibling in (
        table.iterrows()
    ):

        if sibling_idx == idx:
            continue

        sibling_sample = (
            strip_markup(
                sibling.get(
                    "sample_raw"
                )
            )
            .strip()
        )

        if not sibling_sample:
            continue

        if not family_pattern.search(
            sibling_sample
        ):
            continue

        sibling_cells = str(
            sibling.get(
                "raw_source_row",
                ""
            )
        ).split(
            "|"
        )

        if (
            yield_index
            >= len(sibling_cells)
        ):
            continue

        sibling_yield = numeric(
            sibling_cells[
                yield_index
            ]
        )

        if sibling_yield is None:
            continue

        supported_siblings += 1

    # Require repeated product-family evidence, not a single
    # accidental abbreviation match.
    if supported_siblings < 2:
        continue

    product_family_precursor_temp_na_indices.add(
        idx
    )


temperature_not_applicable_indices.update(
    product_family_precursor_temp_na_indices
)



# ============================================================
# RULE 10E
#
# Cross-table precursor / temperature-coded product family.
#
# A compact unresolved base sample is considered an untreated
# precursor when the same paper contains at least three
# distinct treatment temperatures represented by sibling
# samples that:
#
# - contain the base sample as a complete token;
# - end in a 3-4 digit temperature code;
# - have a recovered numeric treatment temperature equal to
#   that terminal code.
#
# Examples of structural families:
#
#   SS       -> SS-300 ... SS-800
#   SD       -> SD400 ... SD700
#   HW       -> HW300, HW450, HW600
#   VP       -> VP-250 ... VP-600
#   PHC      -> PHC500, PHC600, PHC700
#   GP       -> H-GP-225 ... B-GP-500
#
# Requiring at least three DISTINCT temperatures makes this
# substantially more conservative than treating any single
# related code as proof of precursor status.
#
# No paper IDs or benchmark sample identities are used.
# ============================================================

cross_table_precursor_temp_na_indices = set()


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    if (
        idx
        in temperature_not_applicable_indices
    ):
        continue

    if valid_element_count(
        row
    ) < 1:
        continue

    sample = (
        strip_markup(
            row.get(
                "sample_raw"
            )
        )
        .strip()
    )

    # Restrict this inference to compact sample identifiers.
    if not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9]{1,7}",
        sample,
    ):
        continue

    paper_id = clean(
        row.get(
            "paper_id"
        )
    )

    if not paper_id:
        continue

    token_pattern = re.compile(
        r"(?:^|[-_])"
        + re.escape(
            sample
        )
        + r"[-_]?"
        + r"(\d{3,4})"
        + r"$",
        flags=re.IGNORECASE,
    )

    supported_temperatures = set()

    same_paper = df[
        df["paper_id"]
        .astype(str)
        .eq(paper_id)
    ]

    for sibling_idx, sibling in (
        same_paper.iterrows()
    ):

        if sibling_idx == idx:
            continue

        sibling_sample = (
            strip_markup(
                sibling.get(
                    "sample_raw"
                )
            )
            .strip()
        )

        if not sibling_sample:
            continue

        match = token_pattern.search(
            sibling_sample
        )

        if not match:
            continue

        encoded_temperature = numeric(
            match.group(1)
        )

        recovered_temperature = numeric(
            sibling.get(
                "temperature_C"
            )
        )

        if (
            encoded_temperature is None
            or recovered_temperature is None
        ):
            continue

        if abs(
            encoded_temperature
            - recovered_temperature
        ) > 1e-9:
            continue

        supported_temperatures.add(
            recovered_temperature
        )

    # Strong repeated family evidence.
    if len(
        supported_temperatures
    ) < 3:
        continue

    cross_table_precursor_temp_na_indices.add(
        idx
    )


temperature_not_applicable_indices.update(
    cross_table_precursor_temp_na_indices
)


# ============================================================
# RULE 11
#
# Treatment temperature encoded in sample identifier.
#
# Examples of the general structure:
#
#   PREFIX-MATERIAL-500
#   SAMPLE600
#
# A numeric suffix is NOT sufficient by itself. Recovery
# requires additional table-level support:
#
# 1. explicit thermal-processing terminology in the table
#    header/context; OR
#
# 2. at least two rows in the same table using compatible
#    terminal-temperature codes with the same temperature.
#
# This avoids interpreting arbitrary sample numbers,
# percentages, replicate IDs, etc. as temperatures.
# ============================================================

sample_code_temperature_candidates = {}


def terminal_temperature_from_sample(
    sample,
):

    sample = strip_markup(
        sample
    ).strip()

    if not sample:
        return None

    # Must contain alphabetic material/process information.
    if not re.search(
        r"[A-Za-z]",
        sample,
    ):
        return None

    match = re.search(
        r"(?<!\d)"
        r"(\d{3,4})"
        r"\s*$",
        sample,
    )

    if not match:
        return None

    value = float(
        match.group(1)
    )

    # Broad but scientifically plausible thermal-processing
    # range. Context requirements below provide the main
    # protection against false positives.
    if not (
        100
        <= value
        <= 3000
    ):
        return None

    return value


# First collect structurally eligible suffix candidates.
suffix_candidates_by_table = {}


for idx in (
    active_candidate_indices
):

    row = df.loc[
        idx
    ]

    if numeric(
        row.get(
            "temperature_C"
        )
    ) is not None:
        continue

    sample = clean(
        row.get(
            "sample_raw"
        )
    )

    if not sample:
        continue

    temperature = (
        terminal_temperature_from_sample(
            sample
        )
    )

    if temperature is None:
        continue

    # Rows already established as raw feedstock / temperature
    # not applicable must never be converted here.
    if (
        idx
        in temperature_not_applicable_indices
    ):
        continue

    table_id = clean(
        row.get(
            "table_id"
        )
    )

    suffix_candidates_by_table.setdefault(
        table_id,
        [],
    ).append(
        (
            idx,
            sample,
            temperature,
        )
    )


for table_id, candidates in (
    suffix_candidates_by_table.items()
):

    raw_header = clean(
        header_map.get(
            table_id,
            ""
        )
    )

    header = (
        strip_markup(
            raw_header
        )
        .lower()
        .replace(
            "◦",
            "°"
        )
        .replace(
            "º",
            "°"
        )
    )

    thermal_terms = (
        "pyrolys",
        "carboniz",
        "carbonis",
        "temperature",
        "thermal treatment",
        "heat treatment",
        "dry carbonization",
        "hydrothermal",
        "torref",
    )

    explicit_thermal_context = (
        any(
            term in header
            for term in thermal_terms
        )
        or "°c"
        in header.replace(
            " ",
            ""
        )
    )

    # Family support:
    # at least two sibling rows encode the SAME plausible
    # terminal temperature.
    temperature_counts = {}

    for _, _, temperature in candidates:

        temperature_counts[
            temperature
        ] = (
            temperature_counts.get(
                temperature,
                0,
            )
            + 1
        )

    for idx, sample, temperature in candidates:

        family_support = (
            temperature_counts.get(
                temperature,
                0,
            )
            >= 2
        )

        # For family-only evidence, require an explicit
        # separator before the numeric suffix. This is much
        # stronger than accepting arbitrary labels such as
        # B500 or Sample600 with no contextual evidence.
        separator_supported = bool(
            re.search(
                r"[-_]"
                r"\d{3,4}"
                r"\s*$",
                sample,
            )
        )

        supported = (
            explicit_thermal_context
            or (
                family_support
                and separator_supported
            )
        )

        if not supported:
            continue

        sample_code_temperature_candidates[
            idx
        ] = temperature


# ============================================================
# Build semantic-rule actions
# ============================================================

semantic_actions = {}


def add_semantic_action(
    idx,
    action,
):
    """
    Enforce non-overlapping structural rules.
    """

    if idx in semantic_actions:

        raise RuntimeError(
            "Row matched multiple semantic "
            f"structural rules: index={idx}, "
            f"existing={semantic_actions[idx]}, "
            f"new={action}"
        )

    semantic_actions[
        idx
    ] = action


for idx in (
    raw_feedstock_label_corrections
):

    add_semantic_action(
        idx,
        "RECOVER_RAW_FEEDSTOCK_LABEL",
    )


for idx in (
    sample_code_temperature_candidates
):

    add_semantic_action(
        idx,
        "RECOVER_TEMPERATURE_FROM_SAMPLE_CODE",
    )


for idx in (
    continued_sample_proposals
):

    add_semantic_action(
        idx,
        "RECOVER_SHIFTED_SAMPLE_CONTINUATION",
    )


for idx in (
    explicit_sample_temp_proposals
):

    add_semantic_action(
        idx,
        "RECOVER_EXPLICIT_SAMPLE_TEMP_COLUMNS",
    )


for idx in (
    supported_code_indices
):

    add_semantic_action(
        idx,
        "RECOVER_SAMPLE_TEMP_CODE_SERIES",
    )


for idx in (
    star_temperature_proposals
):

    add_semantic_action(
        idx,
        "RECOVER_STAR_HEADER_TEMPERATURE",
    )


for idx in (
    parenthetical_rejects
):

    add_semantic_action(
        idx,
        "REJECT_PARENTHETICAL_COMPANION",
    )


for idx in (
    statistical_rejects
):

    add_semantic_action(
        idx,
        "REJECT_STATISTICAL_SUMMARY",
    )


for idx in (
    following_pm_rejects
):

    add_semantic_action(
        idx,
        "REJECT_FOLLOWING_PM_UNCERTAINTY",
    )


for idx in (
    header_text_fragment_rejects
):

    add_semantic_action(
        idx,
        "REJECT_HEADER_TEXT_FRAGMENT",
    )


for idx in (
    correlation_matrix_rejects
):

    add_semantic_action(
        idx,
        "REJECT_CORRELATION_MATRIX_ROW",
    )


for idx in (
    figure_fragment_rejects
):

    add_semantic_action(
        idx,
        "REJECT_FIGURE_FRAGMENT",
    )


for idx in (
    small_numeric_companion_rejects
):

    add_semantic_action(
        idx,
        "REJECT_SMALL_NUMERIC_COMPANION",
    )


# ============================================================
# Shifted-sample rule must not overlap another structural rule.
# ============================================================

overlap = (
    set(
        shifted_proposals
    )
    &
    set(
        semantic_actions
    )
)


if overlap:

    raise RuntimeError(
        "Shifted-sample rule overlaps "
        f"another structural rule: {overlap}"
    )


# ============================================================
# Apply rules
# ============================================================

audit_records = []


def audit_row(
    idx,
    action,
):

    row = df.loc[
        idx
    ]

    audit_records.append(
        {
            "source_key":
                row.get(
                    "source_key"
                ),

            "paper_id":
                row.get(
                    "paper_id"
                ),

            "table_id":
                row.get(
                    "table_id"
                ),

            "source_row_index":
                row.get(
                    "source_row_index"
                ),

            "queue_before_07g2":
                row.get(
                    "queue_before_07g2"
                ),

            "action":
                action,

            "sample_before":
                row.get(
                    "sample_raw_before_07g2"
                ),

            "sample_after":
                row.get(
                    "sample_raw"
                ),

            "temperature_before":
                row.get(
                    "temperature_C_before_07g2"
                ),

            "temperature_after":
                row.get(
                    "temperature_C"
                ),

            "semantic_class_before":
                row.get(
                    "semantic_class_before_07g2"
                ),

            "semantic_class_after":
                row.get(
                    "semantic_class"
                ),

            "raw_source_row":
                row.get(
                    "raw_source_row"
                ),
        }
    )


# ------------------------------------------------------------
# Shifted sample fills/corrections
# ------------------------------------------------------------

for idx, info in (
    shifted_proposals.items()
):

    old_sample = strip_markup(
        df.at[
            idx,
            "sample_raw",
        ]
    )

    new_sample = info[
        "sample"
    ]

    df.at[
        idx,
        "sample_raw",
    ] = new_sample

    df.at[
        idx,
        "structural_action_07g2",
    ] = info[
        "action"
    ]

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    if old_sample:

        df.at[
            idx,
            "sample_corrected_07g2",
        ] = True

    else:

        df.at[
            idx,
            "sample_recovered_07g2",
        ] = True


# ------------------------------------------------------------
# Shifted sample continuation recovery
# ------------------------------------------------------------

for idx, sample in (
    continued_sample_proposals.items()
):

    df.at[
        idx,
        "sample_raw",
    ] = sample

    df.at[
        idx,
        "structural_action_07g2",
    ] = (
        "RECOVER_SHIFTED_SAMPLE_CONTINUATION"
    )

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    df.at[
        idx,
        "sample_recovered_07g2",
    ] = True


# ------------------------------------------------------------
# Explicit first-column sample + second-column temperature
# ------------------------------------------------------------

for idx, info in (
    explicit_sample_temp_proposals.items()
):

    df.at[
        idx,
        "sample_raw",
    ] = info[
        "sample"
    ]

    df.at[
        idx,
        "temperature_C",
    ] = info[
        "temperature"
    ]

    df.at[
        idx,
        "temperature_exact_C",
    ] = info[
        "temperature"
    ]

    df.at[
        idx,
        "temperature_type",
    ] = "exact"

    df.at[
        idx,
        "structural_action_07g2",
    ] = (
        "RECOVER_EXPLICIT_SAMPLE_TEMP_COLUMNS"
    )

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    df.at[
        idx,
        "sample_recovered_07g2",
    ] = True

    df.at[
        idx,
        "temperature_recovered_07g2",
    ] = True


# ------------------------------------------------------------
# Supported sample-temperature code series
# ------------------------------------------------------------

for idx in (
    supported_code_indices
):

    info = code_candidates[
        idx
    ]

    df.at[
        idx,
        "sample_raw",
    ] = info[
        "sample"
    ]

    df.at[
        idx,
        "temperature_C",
    ] = info[
        "temperature"
    ]

    # Synchronize semantic temperature metadata with the
    # deterministic temperature recovered by 07G2.
    df.at[
        idx,
        "temperature_exact_C",
    ] = info[
        "temperature"
    ]

    df.at[
        idx,
        "temperature_type",
    ] = "exact"

    df.at[
        idx,
        "structural_action_07g2",
    ] = (
        "RECOVER_SAMPLE_TEMP_CODE_SERIES"
    )

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    df.at[
        idx,
        "sample_recovered_07g2",
    ] = True

    df.at[
        idx,
        "temperature_recovered_07g2",
    ] = True


# ------------------------------------------------------------
# °C*min temperature
# ------------------------------------------------------------

for idx, temperature in (
    star_temperature_proposals.items()
):

    df.at[
        idx,
        "temperature_C",
    ] = temperature

    # Synchronize semantic temperature metadata with the
    # deterministic temperature recovered by 07G2.
    df.at[
        idx,
        "temperature_exact_C",
    ] = temperature

    df.at[
        idx,
        "temperature_type",
    ] = "exact"

    table_id = clean(
        df.at[
            idx,
            "table_id",
        ]
    )

    table_sample = (
        star_table_sample_map.get(
            table_id
        )
    )

    if (
        table_sample
        and not clean(
            df.at[
                idx,
                "sample_raw",
            ]
        )
    ):

        df.at[
            idx,
            "sample_raw",
        ] = table_sample

        df.at[
            idx,
            "sample_recovered_07g2",
        ] = True

        action = (
            "RECOVER_STAR_TEMP_AND_TABLE_SAMPLE"
        )

    else:

        action = (
            "RECOVER_STAR_HEADER_TEMPERATURE"
        )

    df.at[
        idx,
        "structural_action_07g2",
    ] = action

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    df.at[
        idx,
        "temperature_recovered_07g2",
    ] = True


# ------------------------------------------------------------
# Structural rejects
# ------------------------------------------------------------

reject_action_sets = [
    (
        header_text_fragment_rejects,
        "REJECT_HEADER_TEXT_FRAGMENT",
    ),
    (
        parenthetical_rejects,
        "REJECT_PARENTHETICAL_COMPANION",
    ),
    (
        statistical_rejects,
        "REJECT_STATISTICAL_SUMMARY",
    ),
    (
        following_pm_rejects,
        "REJECT_FOLLOWING_PM_UNCERTAINTY",
    ),
    (
        correlation_matrix_rejects,
        "REJECT_CORRELATION_MATRIX_ROW",
    ),
    (
        figure_fragment_rejects,
        "REJECT_FIGURE_FRAGMENT",
    ),
    (
        small_numeric_companion_rejects,
        "REJECT_SMALL_NUMERIC_COMPANION",
    ),
]


for indices, action in (
    reject_action_sets
):

    for idx in indices:

        df.at[
            idx,
            "structural_action_07g2",
        ] = action

        df.at[
            idx,
            "structural_recovery_applied_07g2",
        ] = True

        df.at[
            idx,
            "structural_rejected_07g2",
        ] = True

        df.at[
            idx,
            "semantic_class",
        ] = (
            "REJECT_STRUCTURAL_ARTIFACT"
        )


# ------------------------------------------------------------
# Final TEMP_NA consistency gate
#
# TEMP_NA is a scientific state for a valid precursor row.
# It must never remain attached to a row that a stronger
# structural rule has classified as an artifact/rejection.
#
# Some table-level precursor rules intentionally run before
# structural rejection rules are applied. Therefore remove
# all rows scheduled for REJECT_* actions from the final
# TEMP_NA candidate set here, after semantic_actions has been
# fully assembled but before TEMP_NA is written to df.
# ------------------------------------------------------------

reject_action_indices = {
    idx
    for idx, action in semantic_actions.items()
    if clean(action).startswith(
        "REJECT_"
    )
}

temperature_not_applicable_indices.difference_update(
    reject_action_indices
)


# ------------------------------------------------------------
# Raw-feedstock sample-label correction
# ------------------------------------------------------------

for idx, corrected_sample in (
    raw_feedstock_label_corrections.items()
):

    old_sample = strip_markup(
        df.at[
            idx,
            "sample_raw",
        ]
    )

    df.at[
        idx,
        "sample_raw",
    ] = corrected_sample

    df.at[
        idx,
        "structural_action_07g2",
    ] = (
        "RECOVER_RAW_FEEDSTOCK_LABEL"
    )

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    df.at[
        idx,
        "sample_corrected_07g2",
    ] = True


# ------------------------------------------------------------
# Temperature encoded in sample identifier
# ------------------------------------------------------------

for idx, temperature in (
    sample_code_temperature_candidates.items()
):

    df.at[
        idx,
        "temperature_C",
    ] = temperature

    df.at[
        idx,
        "temperature_exact_C",
    ] = temperature

    df.at[
        idx,
        "temperature_type",
    ] = "exact"

    df.at[
        idx,
        "structural_action_07g2",
    ] = (
        "RECOVER_TEMPERATURE_FROM_SAMPLE_CODE"
    )

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    df.at[
        idx,
        "temperature_recovered_07g2",
    ] = True


# ------------------------------------------------------------
# Raw feedstock: treatment temperature not applicable
# ------------------------------------------------------------

for idx in (
    temperature_not_applicable_indices
):

    df.at[
        idx,
        "temperature_not_applicable_07g2",
    ] = True

    df.at[
        idx,
        "structural_recovery_applied_07g2",
    ] = True

    # Preserve a more specific structural action if this row
    # was already repaired, for example a shifted sample.
    current_action = clean(
        df.at[
            idx,
            "structural_action_07g2",
        ]
    )

    if not current_action:

        df.at[
            idx,
            "structural_action_07g2",
        ] = (
            "MARK_TEMPERATURE_NOT_APPLICABLE"
        )


# ============================================================
# Reclassify non-rejected touched rows
# ============================================================

touched = df[
    "structural_recovery_applied_07g2"
].fillna(
    False
).astype(bool)


for idx in df.index[
    touched
]:

    if bool(
        df.at[
            idx,
            "structural_rejected_07g2",
        ]
    ):
        continue

    row = df.loc[
        idx
    ]

    sample_present = bool(
        clean(
            row.get(
                "sample_raw"
            )
        )
    )

    temperature_present = (
        numeric(
            row.get(
                "temperature_C"
            )
        )
        is not None
    )

    element_count = (
        valid_element_count(
            row
        )
    )

    temperature_not_applicable = bool(
        row.get(
            "temperature_not_applicable_07g2",
            False,
        )
    )

    if (
        sample_present
        and temperature_not_applicable
        and element_count == 4
    ):

        new_class = (
            "DIRECT_COMPLETE_CHNO_TEMP_NA"
        )

    elif (
        sample_present
        and temperature_not_applicable
        and element_count >= 1
    ):

        new_class = (
            "DIRECT_PARTIAL_CHNO_TEMP_NA"
        )

    elif (
        sample_present
        and temperature_present
        and element_count == 4
    ):

        new_class = (
            "DIRECT_COMPLETE_CHNO"
        )

    elif (
        sample_present
        and temperature_present
        and element_count >= 1
    ):

        new_class = (
            "DIRECT_PARTIAL_CHNO"
        )

    elif (
        sample_present
        and not temperature_present
    ):

        new_class = (
            "NEEDS_TEMPERATURE_INTERPRETATION"
        )

    elif (
        not sample_present
        and temperature_present
    ):

        new_class = (
            "NEEDS_SAMPLE_INTERPRETATION"
        )

    else:

        new_class = (
            "SEMANTIC_RECOVERY_REQUIRED"
        )

    df.at[
        idx,
        "semantic_class",
    ] = new_class


df[
    "semantic_class_after_07g2"
] = df[
    "semantic_class"
]


df[
    "valid_element_count_07g2"
] = df.apply(
    valid_element_count,
    axis=1,
)


# ============================================================
# Build audit after final classes are known
# ============================================================

audit_records = []


for idx in df.index[
    touched
]:

    row = df.loc[
        idx
    ]

    audit_records.append(
        {
            "source_key":
                row.get(
                    "source_key"
                ),

            "paper_id":
                row.get(
                    "paper_id"
                ),

            "table_id":
                row.get(
                    "table_id"
                ),

            "source_row_index":
                row.get(
                    "source_row_index"
                ),

            "queue_before_07g2":
                row.get(
                    "queue_before_07g2"
                ),

            "action":
                row.get(
                    "structural_action_07g2"
                ),

            "sample_before":
                row.get(
                    "sample_raw_before_07g2"
                ),

            "sample_after":
                row.get(
                    "sample_raw"
                ),

            "temperature_before":
                row.get(
                    "temperature_C_before_07g2"
                ),

            "temperature_after":
                row.get(
                    "temperature_C"
                ),

            "semantic_class_before":
                row.get(
                    "semantic_class_before_07g2"
                ),

            "semantic_class_after":
                row.get(
                    "semantic_class_after_07g2"
                ),

            "valid_element_count":
                row.get(
                    "valid_element_count_07g2"
                ),

            "raw_source_row":
                row.get(
                    "raw_source_row"
                ),
        }
    )


audit = pd.DataFrame(
    audit_records
)


# ============================================================
# Final routing
# ============================================================

reject_mask = (
    df[
        "semantic_class"
    ]
    .fillna("")
    .astype(str)
    .str.startswith(
        "REJECT"
    )
)

det_mask = (
    df[
        "semantic_class"
    ]
    .isin(
        [
            "DIRECT_COMPLETE_CHNO",
            "DIRECT_PARTIAL_CHNO",
            "DIRECT_COMPLETE_CHNO_TEMP_NA",
            "DIRECT_PARTIAL_CHNO_TEMP_NA",
        ]
    )
)

rec_mask = (
    ~reject_mask
    &
    ~det_mask
)


df[
    "queue_after_07g2"
] = None

df.loc[
    det_mask,
    "queue_after_07g2",
] = "deterministic"

df.loc[
    rec_mask,
    "queue_after_07g2",
] = "semantic_recovery"

df.loc[
    reject_mask,
    "queue_after_07g2",
] = "rejected"


if df[
    "queue_after_07g2"
].isna().any():

    raise RuntimeError(
        "Some rows were not routed."
    )


if (
    int(
        det_mask.sum()
    )
    +
    int(
        rec_mask.sum()
    )
    +
    int(
        reject_mask.sum()
    )
    != len(df)
):

    raise RuntimeError(
        "Queue conservation failed."
    )


# ============================================================
# Output queues
# ============================================================

det_out = (
    df[
        det_mask
    ]
    .sort_values(
        "_original_order_07g2"
    )
    .copy()
)

rec_out = (
    df[
        rec_mask
    ]
    .sort_values(
        "_original_order_07g2"
    )
    .copy()
)

rej_out = (
    df[
        reject_mask
    ]
    .sort_values(
        "_original_order_07g2"
    )
    .copy()
)


# New run-local IDs.
# source_key remains the stable identity.
det_out[
    "candidate_id_07g2"
] = [
    f"DET2_{i:05d}"
    for i in range(
        1,
        len(det_out) + 1,
    )
]

rec_out[
    "candidate_id_07g2"
] = [
    f"REC2_{i:05d}"
    for i in range(
        1,
        len(rec_out) + 1,
    )
]

rej_out[
    "candidate_id_07g2"
] = [
    f"REJ2_{i:05d}"
    for i in range(
        1,
        len(rej_out) + 1,
    )
]


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "queue":
                "deterministic",
            "row_count":
                len(
                    det_out
                ),
        },
        {
            "queue":
                "semantic_recovery",
            "row_count":
                len(
                    rec_out
                ),
        },
        {
            "queue":
                "rejected",
            "row_count":
                len(
                    rej_out
                ),
        },
    ]
)


# ============================================================
# Remove internal-only ordering field
# ============================================================

for frame in [
    df,
    det_out,
    rec_out,
    rej_out,
]:

    if (
        "_original_order_07g2"
        in frame.columns
    ):

        frame.drop(
            columns=[
                "_original_order_07g2"
            ],
            inplace=True,
        )


# ============================================================
# Save
# ============================================================

df.to_csv(
    full_output_path,
    index=False,
)

det_out.to_csv(
    det_output_path,
    index=False,
)

rec_out.to_csv(
    rec_output_path,
    index=False,
)

rej_out.to_csv(
    rej_output_path,
    index=False,
)

audit.to_csv(
    audit_output_path,
    index=False,
)

summary.to_csv(
    summary_output_path,
    index=False,
)


# ============================================================
# Console report
# ============================================================

print()
print("=" * 82)
print("STRUCTURAL ACTIONS")
print("=" * 82)

print(
    audit[
        "action"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "Rows touched:",
    len(
        audit
    ),
)

print(
    "Rows structurally rejected:",
    int(
        df[
            "structural_rejected_07g2"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ),
)


print()
print("=" * 82)
print("QUEUE COUNTS")
print("=" * 82)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 82)
print("DETERMINISTIC CLASSES")
print("=" * 82)

print(
    det_out[
        "semantic_class"
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 82)
print("RECOVERY CLASSES")
print("=" * 82)

print(
    rec_out[
        "semantic_class"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "Output source-key duplicates:",
    df[
        "source_key"
    ].duplicated().sum(),
)

print(
    "Rows conserved:",
    len(df),
)

print()
print("Generated files:")

for path in [
    full_output_path,
    det_output_path,
    rec_output_path,
    rej_output_path,
    audit_output_path,
    summary_output_path,
]:

    print(
        "-",
        path,
    )
