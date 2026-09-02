#!/usr/bin/env python3

from pathlib import Path
from difflib import SequenceMatcher
import hashlib
import html
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SCOPE_DIR = (
    ROOT
    / "processed_tables"
    / "benchmark_final_table_v1"
)

STRICT_DIR = (
    SCOPE_DIR
    / "match_v1"
)

STRICT_MATCHED = (
    STRICT_DIR
    / "matched_rows.csv"
)

STRICT_UNMATCHED = (
    STRICT_DIR
    / "unmatched_benchmark.csv"
)

OUT_DIR = (
    SCOPE_DIR
    / "match_v1_range_compatible"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ELEMENT_TOLERANCE = 0.15

ELEMENTS = [
    "C_value",
    "H_value",
    "N_value",
    "O_value",
]

BIG_COST = 1.0e9


# ============================================================
# Helpers
# ============================================================

def file_hash(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def number(value):

    if pd.isna(value):
        return None

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(value):
        return None

    return value


def normalize_sample(value):

    if pd.isna(value):
        return ""

    text = html.unescape(
        str(value)
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = text.lower()

    text = (
        text
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def sample_similarity(a, b):

    a = normalize_sample(a)
    b = normalize_sample(b)

    if not a or not b:
        return 0.0

    seq = SequenceMatcher(
        None,
        a,
        b,
    ).ratio()

    at = set(a.split())
    bt = set(b.split())

    union = at | bt

    jaccard = (
        len(at & bt)
        / len(union)
        if union
        else 0.0
    )

    return max(
        seq,
        jaccard,
    )


def compare_elements(er, mr):

    diffs = {}

    for col in ELEMENTS:

        ev = number(
            er.get(col)
        )

        mv = number(
            mr.get(col)
        )

        if (
            ev is None
            or mv is None
        ):
            continue

        diffs[col] = abs(
            ev - mv
        )

    values = list(
        diffs.values()
    )

    if not values:

        return {
            "comparable_elements": 0,
            "element_mae": np.nan,
            "element_max_abs_diff":
                np.nan,
            "all_within_0_02": False,
            "all_within_0_15": False,
        }

    return {
        "comparable_elements":
            len(values),

        "element_mae":
            float(
                np.mean(values)
            ),

        "element_max_abs_diff":
            float(
                np.max(values)
            ),

        "all_within_0_02":
            all(
                x <= 0.02
                for x in values
            ),

        "all_within_0_15":
            all(
                x <= ELEMENT_TOLERANCE
                for x in values
            ),
    }


def find_final_with_doi():

    required = {
        "final_record_id",
        "source_key",
        "paper_id",
        "doi_normalized",
        "sample",
        "temperature_status",
        "temperature_low_C",
        "temperature_high_C",
        *ELEMENTS,
    }

    candidates = []

    for path in sorted(
        SCOPE_DIR.glob("*.csv")
    ):

        try:
            header = pd.read_csv(
                path,
                nrows=0,
            )
        except Exception:
            continue

        if not required.issubset(
            header.columns
        ):
            continue

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        range_count = (
            df[
                "temperature_status"
            ]
            .astype(str)
            .eq("range")
            .sum()
        )

        if range_count == 0:
            continue

        preferred = int(
            "final_with_doi"
            in path.name.lower()
        )

        candidates.append(
            (
                preferred,
                len(df),
                path,
                df,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Could not locate DOI-enriched "
            "final table. Rerun Step 09B0."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    return (
        candidates[0][2],
        candidates[0][3],
    )


# ============================================================
# Load validated strict result
# ============================================================

strict_hash_before = (
    file_hash(
        STRICT_MATCHED
    ),
    file_hash(
        STRICT_UNMATCHED
    ),
)


strict = pd.read_csv(
    STRICT_MATCHED,
    low_memory=False,
)

manual = pd.read_csv(
    STRICT_UNMATCHED,
    low_memory=False,
)


required_manual = {
    "benchmark_row_id",
    "doi_normalized",
    "Feedstock",
    "temperature_C",
    *ELEMENTS,
}

missing = (
    required_manual
    - set(manual.columns)
)

if missing:

    raise RuntimeError(
        "Unmatched benchmark missing: "
        + ", ".join(
            sorted(missing)
        )
    )


if (
    strict[
        "benchmark_row_id"
    ]
    .astype(str)
    .duplicated()
    .any()
):

    raise RuntimeError(
        "Strict benchmark matches "
        "are not one-to-one."
    )


if (
    strict[
        "final_record_id"
    ]
    .astype(str)
    .duplicated()
    .any()
):

    raise RuntimeError(
        "Strict extracted matches "
        "are not one-to-one."
    )


strict_benchmark_ids = set(
    strict[
        "benchmark_row_id"
    ].astype(str)
)

strict_extracted_ids = set(
    strict[
        "final_record_id"
    ].astype(str)
)


# ============================================================
# Load DOI-enriched final table
# ============================================================

(
    final_path,
    final,
) = find_final_with_doi()


ranges = final[
    final[
        "temperature_status"
    ]
    .astype(str)
    .eq("range")
].copy()


ranges[
    "temperature_low_C"
] = pd.to_numeric(
    ranges[
        "temperature_low_C"
    ],
    errors="coerce",
)

ranges[
    "temperature_high_C"
] = pd.to_numeric(
    ranges[
        "temperature_high_C"
    ],
    errors="coerce",
)


valid = (
    ranges[
        "temperature_low_C"
    ].notna()
    &
    ranges[
        "temperature_high_C"
    ].notna()
    &
    (
        ranges[
            "temperature_low_C"
        ]
        <=
        ranges[
            "temperature_high_C"
        ]
    )
)

if not valid.all():

    raise RuntimeError(
        "Invalid temperature range "
        "found in final table."
    )


ranges = ranges.reset_index(
    drop=True
)

manual = manual.reset_index(
    drop=True
)


# ============================================================
# Build range candidates
# ============================================================

records = []


common_dois = sorted(
    set(
        ranges[
            "doi_normalized"
        ]
        .dropna()
        .astype(str)
    )
    &
    set(
        manual[
            "doi_normalized"
        ]
        .dropna()
        .astype(str)
    )
)


for doi in common_dois:

    erows = ranges[
        ranges[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    mrows = manual[
        manual[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    for ei, er in erows.iterrows():

        low = number(
            er[
                "temperature_low_C"
            ]
        )

        high = number(
            er[
                "temperature_high_C"
            ]
        )

        for mi, mr in mrows.iterrows():

            mt = number(
                mr[
                    "temperature_C"
                ]
            )

            contains = (
                low is not None
                and high is not None
                and mt is not None
                and low <= mt <= high
            )

            element = compare_elements(
                er,
                mr,
            )

            sim = sample_similarity(
                er["sample"],
                mr["Feedstock"],
            )

            admissible = (
                contains
                and
                element[
                    "comparable_elements"
                ] >= 2
                and
                element[
                    "all_within_0_15"
                ]
            )


            # Do NOT use distance from
            # the range midpoint.
            #
            # A midpoint is not an exact
            # source-stated temperature.
            if admissible:

                cost = (
                    1000.0
                    * element[
                        "element_mae"
                    ]
                    +
                    25.0
                    * (
                        1.0
                        - sim
                    )
                    +
                    0.5
                    * (
                        4
                        - element[
                            "comparable_elements"
                        ]
                    )
                )

            else:

                mae = element[
                    "element_mae"
                ]

                if pd.isna(mae):
                    mae = 999.0

                cost = (
                    BIG_COST
                    +
                    1000.0
                    * mae
                    +
                    25.0
                    * (
                        1.0
                        - sim
                    )
                )


            rec = {
                "doi":
                    doi,

                "range_index":
                    int(ei),

                "manual_index":
                    int(mi),

                "final_record_id":
                    er[
                        "final_record_id"
                    ],

                "source_key":
                    er[
                        "source_key"
                    ],

                "paper_id":
                    er[
                        "paper_id"
                    ],

                "branch":
                    er.get(
                        "branch",
                        pd.NA,
                    ),

                "sample_extracted":
                    er["sample"],

                "sample_manual":
                    mr["Feedstock"],

                "sample_similarity":
                    sim,

                "temperature_low_C":
                    low,

                "temperature_high_C":
                    high,

                "temperature_manual":
                    mt,

                "range_contains_benchmark":
                    contains,

                "benchmark_row_id":
                    mr[
                        "benchmark_row_id"
                    ],

                "admissible":
                    admissible,

                "assignment_cost":
                    cost,
            }

            rec.update(
                element
            )

            for col in ELEMENTS:

                rec[
                    f"{col}_extracted"
                ] = er.get(col)

                rec[
                    f"{col}_manual"
                ] = mr.get(col)

            records.append(rec)


pairs = pd.DataFrame(
    records
)


# ============================================================
# Hungarian one-to-one assignment within DOI
# ============================================================

matches = []


for doi in common_dois:

    erows = ranges[
        ranges[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    mrows = manual[
        manual[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    eidx = list(
        erows.index
    )

    midx = list(
        mrows.index
    )

    if not eidx or not midx:
        continue


    epos = {
        idx: pos
        for pos, idx
        in enumerate(eidx)
    }

    mpos = {
        idx: pos
        for pos, idx
        in enumerate(midx)
    }


    matrix = np.full(
        (
            len(eidx),
            len(midx),
        ),
        BIG_COST * 10.0,
    )


    dp = pairs[
        pairs["doi"]
        .eq(doi)
    ]


    for _, row in dp.iterrows():

        matrix[
            epos[
                int(
                    row[
                        "range_index"
                    ]
                )
            ],
            mpos[
                int(
                    row[
                        "manual_index"
                    ]
                )
            ],
        ] = float(
            row[
                "assignment_cost"
            ]
        )


    rr, cc = (
        linear_sum_assignment(
            matrix
        )
    )


    for rp, cp in zip(
        rr,
        cc,
    ):

        ei = eidx[rp]
        mi = midx[cp]


        selected = dp[
            (
                dp[
                    "range_index"
                ].eq(ei)
            )
            &
            (
                dp[
                    "manual_index"
                ].eq(mi)
            )
        ]


        if len(selected) != 1:

            raise RuntimeError(
                "Expected exactly one "
                "candidate pair."
            )


        row = selected.iloc[0]


        if not bool(
            row["admissible"]
        ):
            continue


        rec = row.to_dict()

        rec[
            "match_mode"
        ] = (
            "RANGE_CONTAINS_BENCHMARK"
        )


        if bool(
            row[
                "all_within_0_02"
            ]
        ):

            rec[
                "match_class"
            ] = (
                "RANGE_COMPATIBLE_"
                "EXACT_NUMERIC_WITHIN_0_02"
            )

        else:

            rec[
                "match_class"
            ] = (
                "RANGE_COMPATIBLE_"
                "RELIABLE_WITHIN_0_15"
            )


        matches.append(rec)


range_matches = pd.DataFrame(
    matches
)


# ============================================================
# One-to-one invariants
# ============================================================

if len(range_matches):

    if (
        range_matches[
            "final_record_id"
        ]
        .astype(str)
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Range extracted IDs "
            "are not one-to-one."
        )


    if (
        range_matches[
            "benchmark_row_id"
        ]
        .astype(str)
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Range benchmark IDs "
            "are not one-to-one."
        )


range_benchmark_ids = (
    set(
        range_matches[
            "benchmark_row_id"
        ].astype(str)
    )
    if len(range_matches)
    else set()
)

range_extracted_ids = (
    set(
        range_matches[
            "final_record_id"
        ].astype(str)
    )
    if len(range_matches)
    else set()
)


if (
    range_benchmark_ids
    & strict_benchmark_ids
):

    raise RuntimeError(
        "Range match reused a "
        "strict benchmark row."
    )


if (
    range_extracted_ids
    & strict_extracted_ids
):

    raise RuntimeError(
        "Range match reused a "
        "strict extracted row."
    )


# ============================================================
# Remaining unmatched
# ============================================================

remaining = manual[
    ~manual[
        "benchmark_row_id"
    ]
    .astype(str)
    .isin(
        range_benchmark_ids
    )
].copy()


# ============================================================
# Summary
# ============================================================

strict_n = len(strict)

strict_unmatched_n = len(
    manual
)

benchmark_n = (
    strict_n
    + strict_unmatched_n
)

range_n = len(
    range_matches
)

combined_n = (
    strict_n
    + range_n
)


strict_recall = (
    100.0
    * strict_n
    / benchmark_n
)

compatible_recall = (
    100.0
    * combined_n
    / benchmark_n
)


summary = pd.DataFrame(
    [
        {
            "metric":
                "benchmark_rows",
            "value":
                benchmark_n,
        },
        {
            "metric":
                "strict_exact_matches_locked",
            "value":
                strict_n,
        },
        {
            "metric":
                "strict_exact_recall_pct",
            "value":
                strict_recall,
        },
        {
            "metric":
                "range_rows_considered",
            "value":
                len(ranges),
        },
        {
            "metric":
                "range_compatible_additions",
            "value":
                range_n,
        },
        {
            "metric":
                "combined_compatible_matches",
            "value":
                combined_n,
        },
        {
            "metric":
                "remaining_unmatched_benchmark",
            "value":
                len(remaining),
        },
        {
            "metric":
                "source_temperature_compatible_recall_pct",
            "value":
                compatible_recall,
        },
    ]
)


# ============================================================
# Save
# ============================================================

pairs_path = (
    OUT_DIR
    / "range_candidate_pairs.csv"
)

matches_path = (
    OUT_DIR
    / "range_compatible_matches.csv"
)

remaining_path = (
    OUT_DIR
    / "remaining_unmatched_benchmark.csv"
)

summary_path = (
    OUT_DIR
    / "range_match_summary.csv"
)


pairs.to_csv(
    pairs_path,
    index=False,
)

range_matches.to_csv(
    matches_path,
    index=False,
)

remaining.to_csv(
    remaining_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Prove strict outputs untouched
# ============================================================

strict_hash_after = (
    file_hash(
        STRICT_MATCHED
    ),
    file_hash(
        STRICT_UNMATCHED
    ),
)

strict_unchanged = (
    strict_hash_before
    ==
    strict_hash_after
)

if not strict_unchanged:

    raise RuntimeError(
        "Strict benchmark outputs "
        "changed unexpectedly."
    )


# ============================================================
# Report
# ============================================================

print("=" * 100)
print(
    "STEP 09B1R — RANGE-COMPATIBLE "
    "BENCHMARK MATCH"
)
print("=" * 100)

print()
print(
    "DOI-enriched final input:",
    final_path,
)

print()
print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("RANGE MATCHES")
print("=" * 100)


if len(range_matches):

    show = [
        "benchmark_row_id",
        "doi",
        "final_record_id",
        "source_key",
        "sample_manual",
        "sample_extracted",
        "temperature_manual",
        "temperature_low_C",
        "temperature_high_C",
        "comparable_elements",
        "element_mae",
        "element_max_abs_diff",
        "match_class",
    ]

    print(
        range_matches[
            show
        ]
        .sort_values(
            "benchmark_row_id"
        )
        .to_string(
            index=False
        )
    )

else:

    print("None")


print()
print("=" * 100)
print("INVARIANTS")
print("=" * 100)

print(
    "Strict matches locked:",
    strict_n,
)

print(
    "Strict/range benchmark overlap:",
    len(
        range_benchmark_ids
        & strict_benchmark_ids
    ),
)

print(
    "Strict/range extracted overlap:",
    len(
        range_extracted_ids
        & strict_extracted_ids
    ),
)

print(
    "Range benchmark IDs unique:",
    (
        not range_matches[
            "benchmark_row_id"
        ]
        .astype(str)
        .duplicated()
        .any()
        if len(range_matches)
        else True
    ),
)

print(
    "Range extracted IDs unique:",
    (
        not range_matches[
            "final_record_id"
        ]
        .astype(str)
        .duplicated()
        .any()
        if len(range_matches)
        else True
    ),
)

print(
    "Combined + remaining benchmark:",
    combined_n
    + len(remaining),
)

print(
    "Expected benchmark:",
    benchmark_n,
)

print(
    "Strict source files unchanged:",
    strict_unchanged,
)


print()
print("Saved:")
print("-", pairs_path)
print("-", matches_path)
print("-", remaining_path)
print("-", summary_path)
