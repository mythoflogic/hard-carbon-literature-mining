#!/usr/bin/env python3

from pathlib import Path
from difflib import SequenceMatcher
import html
import re

import numpy as np
import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

BASE = (
    ROOT
    / "processed_tables"
    / "benchmark_final_table_v1"
)

MATCH_DIR = (
    BASE
    / "match_v1"
)

FINAL_PATH = (
    BASE
    / "final_table_with_doi.csv"
)

UNMATCHED_PATH = (
    MATCH_DIR
    / "unmatched_benchmark.csv"
)

PAIRS_PATH = (
    MATCH_DIR
    / "pair_candidates.csv"
)

MATCHED_PATH = (
    MATCH_DIR
    / "matched_rows.csv"
)

OUT_PATH = (
    MATCH_DIR
    / "unmatched_benchmark_diagnosed.csv"
)

SUMMARY_PATH = (
    MATCH_DIR
    / "unmatched_diagnosis_summary.csv"
)


TEMP_TOL = 0.5
ELEMENT_TOL = 0.15

ELEMENTS = [
    "C_value",
    "H_value",
    "N_value",
    "O_value",
]


# ============================================================
# Helpers
# ============================================================

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

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def sample_similarity(a, b):

    a = normalize_sample(a)
    b = normalize_sample(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def compare_chno(extracted_row, manual_row):

    diffs = {}

    for col in ELEMENTS:

        ev = number(
            extracted_row.get(col)
        )

        mv = number(
            manual_row.get(col)
        )

        if (
            ev is None
            or mv is None
        ):
            continue

        diffs[col] = abs(
            ev - mv
        )

    if not diffs:

        return {
            "comparable":
                0,

            "mae":
                np.nan,

            "max_diff":
                np.nan,

            "all_015":
                False,
        }

    vals = list(
        diffs.values()
    )

    return {
        "comparable":
            len(vals),

        "mae":
            float(
                np.mean(vals)
            ),

        "max_diff":
            float(
                np.max(vals)
            ),

        "all_015":
            (
                len(vals) >= 2
                and
                all(
                    x <= ELEMENT_TOL
                    for x in vals
                )
            ),
    }


# ============================================================
# Load
# ============================================================

final = pd.read_csv(
    FINAL_PATH,
    low_memory=False,
)

unmatched = pd.read_csv(
    UNMATCHED_PATH,
    low_memory=False,
)

pairs = pd.read_csv(
    PAIRS_PATH,
    low_memory=False,
)

matched = pd.read_csv(
    MATCHED_PATH,
    low_memory=False,
)


print("=" * 105)
print("STEP 09B2 — DIAGNOSE UNMATCHED BENCHMARK")
print("=" * 105)

print(
    "\nUnmatched benchmark rows:",
    len(unmatched),
)


# ============================================================
# Diagnose each unmatched benchmark row
# ============================================================

records = []


for _, mr in unmatched.iterrows():

    bid = str(
        mr["benchmark_row_id"]
    )

    doi = str(
        mr["doi_normalized"]
    )

    mt = number(
        mr["temperature_C"]
    )

    manual_sample = (
        mr["Feedstock"]
    )


    # --------------------------------------------------------
    # Eligible candidates already calculated in 09B1
    # --------------------------------------------------------

    pc = pairs[
        pairs["benchmark_row_id"]
        .astype(str)
        .eq(bid)
    ].copy()


    doi_eligible_count = len(
        pc
    )


    admissible = pc[
        pc["admissible"]
        .fillna(False)
        .astype(bool)
    ].copy()


    temp_aligned = pc[
        pc["temperature_ok"]
        .fillna(False)
        .astype(bool)
    ].copy()


    numeric_good_eligible = pc[
        (
            pd.to_numeric(
                pc["comparable_elements"],
                errors="coerce",
            )
            >= 2
        )
        &
        pc["all_within_0_15"]
        .fillna(False)
        .astype(bool)
    ].copy()


    # --------------------------------------------------------
    # Search ALL 521 final rows in same DOI, including rows
    # excluded from direct benchmark eligibility.
    # --------------------------------------------------------

    all_same_doi = final[
        final["doi_normalized"]
        .astype(str)
        .eq(doi)
    ].copy()


    all_candidates = []

    for _, er in all_same_doi.iterrows():

        comp = compare_chno(
            er,
            mr,
        )

        et = number(
            er.get(
                "temperature_C"
            )
        )

        if (
            et is not None
            and mt is not None
        ):
            tdiff = abs(
                et - mt
            )
        else:
            tdiff = np.nan


        low = number(
            er.get(
                "temperature_low_C"
            )
        )

        high = number(
            er.get(
                "temperature_high_C"
            )
        )


        range_contains = False

        if (
            low is not None
            and high is not None
            and mt is not None
        ):
            range_contains = (
                low
                <= mt
                <= high
            )


        all_candidates.append(
            {
                "final_record_id":
                    er[
                        "final_record_id"
                    ],

                "source_key":
                    er[
                        "source_key"
                    ],

                "sample":
                    er[
                        "sample"
                    ],

                "temperature_status":
                    er[
                        "temperature_status"
                    ],

                "temperature_C":
                    et,

                "temperature_low_C":
                    low,

                "temperature_high_C":
                    high,

                "temperature_difference":
                    tdiff,

                "range_contains_manual_temp":
                    range_contains,

                "benchmark_eligible":
                    bool(
                        er.get(
                            "benchmark_eligible",
                            False,
                        )
                    ),

                "sample_similarity":
                    sample_similarity(
                        er["sample"],
                        manual_sample,
                    ),

                "comparable":
                    comp[
                        "comparable"
                    ],

                "mae":
                    comp[
                        "mae"
                    ],

                "max_diff":
                    comp[
                        "max_diff"
                    ],

                "all_015":
                    comp[
                        "all_015"
                    ],
            }
        )


    all_df = pd.DataFrame(
        all_candidates
    )


    numerical_matches = all_df[
        all_df["all_015"]
        .fillna(False)
        .astype(bool)
    ].copy()


    exact_numeric_wrong_temp = (
        numerical_matches[
            numerical_matches[
                "temperature_status"
            ].eq("exact")
            &
            (
                pd.to_numeric(
                    numerical_matches[
                        "temperature_difference"
                    ],
                    errors="coerce",
                )
                > TEMP_TOL
            )
        ]
        .copy()
    )


    unresolved_numeric = (
        numerical_matches[
            numerical_matches[
                "temperature_status"
            ].eq("not_resolved")
        ]
        .copy()
    )


    temp_na_numeric = (
        numerical_matches[
            numerical_matches[
                "temperature_status"
            ].eq("not_applicable")
        ]
        .copy()
    )


    range_numeric = (
        numerical_matches[
            numerical_matches[
                "temperature_status"
            ].eq("range")
        ]
        .copy()
    )


    range_contains = (
        range_numeric[
            range_numeric[
                "range_contains_manual_temp"
            ]
            .fillna(False)
            .astype(bool)
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if len(admissible):

        diagnosis = (
            "ONE_TO_ONE_COMPETITION"
        )

    elif len(
        exact_numeric_wrong_temp
    ):

        diagnosis = (
            "CHNO_MATCH_BUT_TEMPERATURE_MISMATCH"
        )

    elif len(
        range_contains
    ):

        diagnosis = (
            "CHNO_MATCH_IN_TEMPERATURE_RANGE_ROW"
        )

    elif len(
        unresolved_numeric
    ):

        diagnosis = (
            "CHNO_MATCH_BUT_TEMPERATURE_UNRESOLVED"
        )

    elif len(
        temp_na_numeric
    ):

        diagnosis = (
            "CHNO_MATCH_IN_TEMP_NA_ROW"
        )

    elif len(
        range_numeric
    ):

        diagnosis = (
            "CHNO_MATCH_IN_RANGE_BUT_TEMP_OUTSIDE_RANGE"
        )

    elif len(
        temp_aligned
    ):

        diagnosis = (
            "TEMPERATURE_MATCH_BUT_CHNO_DISAGREES"
        )

    elif doi_eligible_count == 0:

        diagnosis = (
            "NO_BENCHMARK_ELIGIBLE_ROWS_FOR_DOI"
        )

    elif len(
        numeric_good_eligible
    ):

        diagnosis = (
            "CHNO_MATCH_WITHOUT_TEMPERATURE_ALIGNMENT"
        )

    else:

        diagnosis = (
            "NO_ALIGNED_NUMERIC_MATCH"
        )


    # --------------------------------------------------------
    # Choose most informative candidate for report
    # --------------------------------------------------------

    best = None


    candidate_groups = [
        admissible,
        exact_numeric_wrong_temp,
        range_contains,
        unresolved_numeric,
        temp_na_numeric,
        range_numeric,
    ]


    for group in candidate_groups:

        if len(group) == 0:
            continue

        if (
            "element_mae"
            in group.columns
        ):

            g = group.sort_values(
                [
                    "element_mae",
                    "sample_similarity",
                ],
                ascending=[
                    True,
                    False,
                ],
            )

        else:

            g = group.sort_values(
                [
                    "mae",
                    "sample_similarity",
                ],
                ascending=[
                    True,
                    False,
                ],
            )

        best = g.iloc[0]
        break


    # If none of the special groups had a candidate,
    # use best temperature-aligned eligible pair.
    if (
        best is None
        and len(temp_aligned)
    ):

        g = temp_aligned.copy()

        g["_mae"] = pd.to_numeric(
            g["element_mae"],
            errors="coerce",
        ).fillna(
            999.0
        )

        g = g.sort_values(
            [
                "_mae",
                "sample_similarity",
            ],
            ascending=[
                True,
                False,
            ],
        )

        best = g.iloc[0]


    # Otherwise use best overall eligible candidate.
    if (
        best is None
        and len(pc)
    ):

        g = pc.copy()

        g["_tdiff"] = pd.to_numeric(
            g[
                "temperature_difference"
            ],
            errors="coerce",
        ).fillna(
            9999.0
        )

        g["_mae"] = pd.to_numeric(
            g["element_mae"],
            errors="coerce",
        ).fillna(
            999.0
        )

        g = g.sort_values(
            [
                "_tdiff",
                "_mae",
                "sample_similarity",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

        best = g.iloc[0]


    # --------------------------------------------------------
    # Normalize best-candidate fields across pair/all-final
    # tables.
    # --------------------------------------------------------

    if best is None:

        best_id = pd.NA
        best_sample = pd.NA
        best_status = pd.NA
        best_temp = np.nan
        best_tdiff = np.nan
        best_sim = np.nan
        best_comp = 0
        best_mae = np.nan
        best_max = np.nan

    else:

        best_id = best.get(
            "final_record_id",
            pd.NA,
        )

        best_sample = best.get(
            "sample",
            best.get(
                "sample_extracted",
                pd.NA,
            ),
        )

        best_status = best.get(
            "temperature_status",
            pd.NA,
        )

        best_temp = best.get(
            "temperature_C",
            best.get(
                "temperature_extracted",
                np.nan,
            ),
        )

        best_tdiff = best.get(
            "temperature_difference",
            np.nan,
        )

        best_sim = best.get(
            "sample_similarity",
            np.nan,
        )

        best_comp = best.get(
            "comparable",
            best.get(
                "comparable_elements",
                0,
            ),
        )

        best_mae = best.get(
            "mae",
            best.get(
                "element_mae",
                np.nan,
            ),
        )

        best_max = best.get(
            "max_diff",
            best.get(
                "element_max_abs_diff",
                np.nan,
            ),
        )


    records.append(
        {
            "benchmark_row_id":
                bid,

            "doi":
                doi,

            "feedstock_manual":
                manual_sample,

            "temperature_manual":
                mt,

            "C_manual":
                mr["C_value"],

            "H_manual":
                mr["H_value"],

            "N_manual":
                mr["N_value"],

            "O_manual":
                mr["O_value"],

            "diagnosis":
                diagnosis,

            "eligible_pair_candidates":
                doi_eligible_count,

            "admissible_candidates":
                len(admissible),

            "all_final_rows_same_doi":
                len(all_same_doi),

            "all_final_numeric_matches":
                len(numerical_matches),

            "best_final_record_id":
                best_id,

            "best_sample_extracted":
                best_sample,

            "best_temperature_status":
                best_status,

            "best_temperature_extracted":
                best_temp,

            "best_temperature_difference":
                best_tdiff,

            "best_sample_similarity":
                best_sim,

            "best_comparable_elements":
                best_comp,

            "best_element_mae":
                best_mae,

            "best_element_max_abs_diff":
                best_max,
        }
    )


diagnosed = pd.DataFrame(
    records
)


# ============================================================
# Summary
# ============================================================

summary = (
    diagnosed[
        "diagnosis"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "diagnosis"
    )
    .reset_index(
        name="count"
    )
)


summary[
    "pct_of_unmatched"
] = (
    100.0
    * summary["count"]
    / len(diagnosed)
)


# ============================================================
# Save
# ============================================================

diagnosed.to_csv(
    OUT_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Console
# ============================================================

print()
print("=" * 105)
print("DIAGNOSIS SUMMARY")
print("=" * 105)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 105)
print("DIAGNOSIS BY DOI")
print("=" * 105)

cross = pd.crosstab(
    diagnosed["doi"],
    diagnosed["diagnosis"],
)

print(
    cross.to_string()
)


print()
print("=" * 105)
print(f"{len(diagnosed)} UNMATCHED BENCHMARK ROWS")
print("=" * 105)

show_cols = [
    "benchmark_row_id",
    "doi",
    "feedstock_manual",
    "temperature_manual",
    "diagnosis",
    "best_final_record_id",
    "best_sample_extracted",
    "best_temperature_status",
    "best_temperature_extracted",
    "best_temperature_difference",
    "best_sample_similarity",
    "best_comparable_elements",
    "best_element_mae",
    "best_element_max_abs_diff",
]

print(
    diagnosed[
        show_cols
    ].to_string(
        index=False,
        max_colwidth=55,
    )
)


print()
print("=" * 105)
print("INVARIANTS")
print("=" * 105)

print(
    "Diagnosed rows:",
    len(diagnosed),
)

print(
    "Expected unmatched rows:",
    len(unmatched),
)

print(
    "Duplicate benchmark IDs:",
    diagnosed[
        "benchmark_row_id"
    ]
    .duplicated()
    .sum(),
)

print(
    "Matched rows unchanged:",
    len(matched),
)

print(
    "Source datasets modified: False"
)

print()
print("Saved:")
print("-", OUT_PATH)
print("-", SUMMARY_PATH)
