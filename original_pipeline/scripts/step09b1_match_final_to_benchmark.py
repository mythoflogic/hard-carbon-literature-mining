#!/usr/bin/env python3

from pathlib import Path
from difflib import SequenceMatcher
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

EXTRACTED_PATH = (
    SCOPE_DIR
    / "final_benchmark_eligible.csv"
)

MANUAL_PATH = (
    SCOPE_DIR
    / "benchmark_scope.csv"
)

OUT_DIR = (
    SCOPE_DIR
    / "match_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TEMP_TOLERANCE_C = 0.5
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

def safe_float(value):

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


def token_set(text):

    return set(
        normalize_sample(text)
        .split()
    )


def sample_similarity(a, b):

    a_norm = normalize_sample(a)
    b_norm = normalize_sample(b)

    if (
        not a_norm
        or not b_norm
    ):
        return 0.0

    seq = SequenceMatcher(
        None,
        a_norm,
        b_norm,
    ).ratio()

    a_tokens = token_set(a_norm)
    b_tokens = token_set(b_norm)

    union = (
        a_tokens
        | b_tokens
    )

    if union:

        jaccard = (
            len(
                a_tokens
                & b_tokens
            )
            / len(union)
        )

    else:
        jaccard = 0.0

    # Use the stronger of character-level and token overlap.
    return max(
        seq,
        jaccard,
    )


def compare_elements(er, mr):

    diffs = {}

    for col in ELEMENTS:

        ev = safe_float(
            er.get(col)
        )

        mv = safe_float(
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

    comparable = len(diffs)

    if comparable == 0:

        return {
            "comparable_elements": 0,
            "element_mae": np.nan,
            "element_max_abs_diff": np.nan,
            "elements_within_0_02": 0,
            "elements_within_0_15": 0,
            "all_within_0_02": False,
            "all_within_0_15": False,
            "C_abs_diff": np.nan,
            "H_abs_diff": np.nan,
            "N_abs_diff": np.nan,
            "O_abs_diff": np.nan,
        }

    values = list(
        diffs.values()
    )

    return {
        "comparable_elements":
            comparable,

        "element_mae":
            float(
                np.mean(values)
            ),

        "element_max_abs_diff":
            float(
                np.max(values)
            ),

        "elements_within_0_02":
            sum(
                x <= 0.02
                for x in values
            ),

        "elements_within_0_15":
            sum(
                x <= ELEMENT_TOLERANCE
                for x in values
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

        "C_abs_diff":
            diffs.get(
                "C_value",
                np.nan,
            ),

        "H_abs_diff":
            diffs.get(
                "H_value",
                np.nan,
            ),

        "N_abs_diff":
            diffs.get(
                "N_value",
                np.nan,
            ),

        "O_abs_diff":
            diffs.get(
                "O_value",
                np.nan,
            ),
    }


# ============================================================
# Load benchmark scope
# ============================================================

extracted = pd.read_csv(
    EXTRACTED_PATH,
    low_memory=False,
)

manual = pd.read_csv(
    MANUAL_PATH,
    low_memory=False,
)


# ============================================================
# Input invariants
# ============================================================

required_extracted = {
    "final_record_id",
    "source_key",
    "paper_id",
    "doi_normalized",
    "sample",
    "temperature_C",
    *ELEMENTS,
}

required_manual = {
    "benchmark_row_id",
    "doi_normalized",
    "Feedstock",
    "temperature_C",
    *ELEMENTS,
}


missing = (
    required_extracted
    - set(extracted.columns)
)

if missing:
    raise RuntimeError(
        "Extracted input missing columns: "
        + ", ".join(
            sorted(missing)
        )
    )


missing = (
    required_manual
    - set(manual.columns)
)

if missing:
    raise RuntimeError(
        "Benchmark input missing columns: "
        + ", ".join(
            sorted(missing)
        )
    )


if (
    extracted[
        "final_record_id"
    ]
    .astype(str)
    .duplicated()
    .any()
):
    raise RuntimeError(
        "Duplicate extracted final_record_id."
    )


if (
    manual[
        "benchmark_row_id"
    ]
    .astype(str)
    .duplicated()
    .any()
):
    raise RuntimeError(
        "Duplicate benchmark_row_id."
    )


if (
    extracted[
        "doi_normalized"
    ]
    .isna()
    .any()
):
    raise RuntimeError(
        "Eligible extracted row missing DOI."
    )


if (
    manual[
        "doi_normalized"
    ]
    .isna()
    .any()
):
    raise RuntimeError(
        "Benchmark-scope row missing DOI."
    )


# ============================================================
# Build all same-DOI pair candidates
# ============================================================

pair_records = []


common_dois = sorted(
    set(
        extracted[
            "doi_normalized"
        ].astype(str)
    )
    &
    set(
        manual[
            "doi_normalized"
        ].astype(str)
    )
)


for doi in common_dois:

    e = extracted[
        extracted[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    m = manual[
        manual[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    for ei, er in e.iterrows():

        et = safe_float(
            er["temperature_C"]
        )

        for mi, mr in m.iterrows():

            mt = safe_float(
                mr["temperature_C"]
            )

            if (
                et is None
                or mt is None
            ):
                tdiff = np.nan
                temperature_ok = False
            else:
                tdiff = abs(
                    et - mt
                )

                temperature_ok = (
                    tdiff
                    <= TEMP_TOLERANCE_C
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
                temperature_ok
                and
                element[
                    "comparable_elements"
                ] >= 2
                and
                element[
                    "all_within_0_15"
                ]
            )


            # Numerical agreement dominates.
            #
            # Sample similarity is supporting/tie-breaking
            # evidence only.
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
                    5.0
                    * tdiff
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

                # Diagnostic cost for ranking bad/unmatched
                # alternatives. This value is NOT used to
                # accept a pair.
                diagnostic_mae = (
                    element[
                        "element_mae"
                    ]
                )

                if pd.isna(
                    diagnostic_mae
                ):
                    diagnostic_mae = 999.0

                diagnostic_t = (
                    tdiff
                    if not pd.isna(
                        tdiff
                    )
                    else 999.0
                )

                cost = (
                    BIG_COST
                    +
                    1000.0
                    * diagnostic_mae
                    +
                    10.0
                    * diagnostic_t
                    +
                    25.0
                    * (
                        1.0
                        - sim
                    )
                )


            record = {
                "doi":
                    doi,

                "extracted_index":
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
                    er[
                        "sample"
                    ],

                "sample_manual":
                    mr[
                        "Feedstock"
                    ],

                "sample_similarity":
                    sim,

                "temperature_extracted":
                    et,

                "temperature_manual":
                    mt,

                "temperature_difference":
                    tdiff,

                "temperature_ok":
                    temperature_ok,

                "benchmark_row_id":
                    mr[
                        "benchmark_row_id"
                    ],

                "admissible":
                    admissible,

                "assignment_cost":
                    cost,
            }

            record.update(
                element
            )

            for col in ELEMENTS:

                record[
                    f"{col}_extracted"
                ] = er.get(col)

                record[
                    f"{col}_manual"
                ] = mr.get(col)

            pair_records.append(
                record
            )


pairs = pd.DataFrame(
    pair_records
)


# ============================================================
# Global one-to-one assignment within each DOI
# ============================================================

matched_records = []


for doi in common_dois:

    e = extracted[
        extracted[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    m = manual[
        manual[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ]

    e_indices = list(
        e.index
    )

    m_indices = list(
        m.index
    )

    if (
        not e_indices
        or not m_indices
    ):
        continue

    e_pos = {
        idx: pos
        for pos, idx
        in enumerate(e_indices)
    }

    m_pos = {
        idx: pos
        for pos, idx
        in enumerate(m_indices)
    }


    matrix = np.full(
        (
            len(e_indices),
            len(m_indices),
        ),
        BIG_COST * 10.0,
        dtype=float,
    )


    doi_pairs = pairs[
        pairs["doi"]
        .eq(doi)
    ]


    for _, row in doi_pairs.iterrows():

        matrix[
            e_pos[
                int(
                    row[
                        "extracted_index"
                    ]
                )
            ],
            m_pos[
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


    row_ind, col_ind = (
        linear_sum_assignment(
            matrix
        )
    )


    for rp, cp in zip(
        row_ind,
        col_ind,
    ):

        ei = e_indices[rp]
        mi = m_indices[cp]

        selected = doi_pairs[
            (
                doi_pairs[
                    "extracted_index"
                ]
                .eq(ei)
            )
            &
            (
                doi_pairs[
                    "manual_index"
                ]
                .eq(mi)
            )
        ]

        if len(selected) != 1:
            raise RuntimeError(
                "Expected exactly one pair "
                f"for DOI={doi}, "
                f"ei={ei}, mi={mi}"
            )

        row = selected.iloc[0]

        # Assignment is only accepted when the pair passed
        # the conservative scientific gate.
        if not bool(
            row["admissible"]
        ):
            continue

        rec = row.to_dict()

        if bool(
            row[
                "all_within_0_02"
            ]
        ):
            rec[
                "match_class"
            ] = (
                "EXACT_NUMERIC_WITHIN_0_02"
            )
        else:
            rec[
                "match_class"
            ] = (
                "RELIABLE_WITHIN_0_15"
            )

        matched_records.append(
            rec
        )


matched = pd.DataFrame(
    matched_records
)


# ============================================================
# One-to-one invariants
# ============================================================

if len(matched):

    if (
        matched[
            "final_record_id"
        ]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Matched extracted IDs are "
            "not one-to-one."
        )

    if (
        matched[
            "benchmark_row_id"
        ]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Matched benchmark IDs are "
            "not one-to-one."
        )


matched_extracted_ids = set(
    matched[
        "final_record_id"
    ].astype(str)
) if len(matched) else set()


matched_manual_ids = set(
    matched[
        "benchmark_row_id"
    ].astype(str)
) if len(matched) else set()


# ============================================================
# Unmatched rows
# ============================================================

unmatched_extracted = extracted[
    ~extracted[
        "final_record_id"
    ]
    .astype(str)
    .isin(
        matched_extracted_ids
    )
].copy()


unmatched_manual = manual[
    ~manual[
        "benchmark_row_id"
    ]
    .astype(str)
    .isin(
        matched_manual_ids
    )
].copy()


# ============================================================
# Attach best diagnostic candidate to unmatched manual
# ============================================================

diagnostic_rows = []


for _, mr in unmatched_manual.iterrows():

    bid = str(
        mr[
            "benchmark_row_id"
        ]
    )

    cand = pairs[
        pairs[
            "benchmark_row_id"
        ]
        .astype(str)
        .eq(bid)
    ].copy()

    if len(cand) == 0:

        diagnostic_rows.append(
            {
                "benchmark_row_id":
                    bid,

                "best_final_record_id":
                    pd.NA,

                "best_sample_extracted":
                    pd.NA,

                "best_sample_similarity":
                    np.nan,

                "best_temperature_difference":
                    np.nan,

                "best_comparable_elements":
                    0,

                "best_element_mae":
                    np.nan,

                "best_element_max_abs_diff":
                    np.nan,

                "best_all_within_0_15":
                    False,
            }
        )

        continue


    # Diagnostic ranking:
    #
    # 1. temperature-compatible first
    # 2. lower numerical MAE
    # 3. higher sample similarity
    #
    cand[
        "_temp_rank"
    ] = (
        ~cand[
            "temperature_ok"
        ].astype(bool)
    ).astype(int)


    cand[
        "_mae_rank"
    ] = pd.to_numeric(
        cand[
            "element_mae"
        ],
        errors="coerce",
    ).fillna(
        999.0
    )


    cand = cand.sort_values(
        [
            "_temp_rank",
            "_mae_rank",
            "sample_similarity",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )


    best = cand.iloc[0]


    diagnostic_rows.append(
        {
            "benchmark_row_id":
                bid,

            "best_final_record_id":
                best[
                    "final_record_id"
                ],

            "best_sample_extracted":
                best[
                    "sample_extracted"
                ],

            "best_sample_similarity":
                best[
                    "sample_similarity"
                ],

            "best_temperature_difference":
                best[
                    "temperature_difference"
                ],

            "best_comparable_elements":
                best[
                    "comparable_elements"
                ],

            "best_element_mae":
                best[
                    "element_mae"
                ],

            "best_element_max_abs_diff":
                best[
                    "element_max_abs_diff"
                ],

            "best_all_within_0_15":
                best[
                    "all_within_0_15"
                ],
        }
    )


manual_diag = pd.DataFrame(
    diagnostic_rows
)


unmatched_manual = (
    unmatched_manual
    .merge(
        manual_diag,
        on="benchmark_row_id",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# Per-DOI summary
# ============================================================

rows = []


all_dois = sorted(
    set(
        manual[
            "doi_normalized"
        ].astype(str)
    )
    |
    set(
        extracted[
            "doi_normalized"
        ].astype(str)
    )
)


for doi in all_dois:

    n_manual = (
        manual[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
        .sum()
    )

    n_extracted = (
        extracted[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
        .sum()
    )

    if len(matched):

        n_matched = (
            matched["doi"]
            .astype(str)
            .eq(doi)
            .sum()
        )

    else:
        n_matched = 0


    rows.append(
        {
            "doi":
                doi,

            "benchmark_rows":
                int(n_manual),

            "eligible_extracted_rows":
                int(n_extracted),

            "matched_rows":
                int(n_matched),

            "unmatched_benchmark_rows":
                int(
                    n_manual
                    - n_matched
                ),

            "unmatched_extracted_rows":
                int(
                    n_extracted
                    - n_matched
                ),

            "benchmark_recall_pct":
                (
                    100.0
                    * n_matched
                    / n_manual
                    if n_manual
                    else np.nan
                ),

            "eligible_extracted_matched_pct":
                (
                    100.0
                    * n_matched
                    / n_extracted
                    if n_extracted
                    else np.nan
                ),
        }
    )


per_doi = pd.DataFrame(
    rows
)


# ============================================================
# Summary
# ============================================================

match_class_counts = (
    matched[
        "match_class"
    ]
    .value_counts()
    if len(matched)
    else pd.Series(
        dtype=int
    )
)


summary_rows = [
    {
        "metric":
            "benchmark_rows",
        "value":
            len(manual),
    },
    {
        "metric":
            "eligible_extracted_rows",
        "value":
            len(extracted),
    },
    {
        "metric":
            "matched_rows",
        "value":
            len(matched),
    },
    {
        "metric":
            "unmatched_benchmark_rows",
        "value":
            len(
                unmatched_manual
            ),
    },
    {
        "metric":
            "unmatched_extracted_rows",
        "value":
            len(
                unmatched_extracted
            ),
    },
    {
        "metric":
            "benchmark_recall_pct",
        "value":
            (
                100.0
                * len(matched)
                / len(manual)
                if len(manual)
                else np.nan
            ),
    },
    {
        "metric":
            "eligible_extracted_matched_pct",
        "value":
            (
                100.0
                * len(matched)
                / len(extracted)
                if len(extracted)
                else np.nan
            ),
    },
    {
        "metric":
            "exact_numeric_within_0_02",
        "value":
            int(
                match_class_counts.get(
                    "EXACT_NUMERIC_WITHIN_0_02",
                    0,
                )
            ),
    },
    {
        "metric":
            "reliable_within_0_15",
        "value":
            int(
                match_class_counts.get(
                    "RELIABLE_WITHIN_0_15",
                    0,
                )
            ),
    },
]


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Save
# ============================================================

pairs.to_csv(
    OUT_DIR
    / "pair_candidates.csv",
    index=False,
)

matched.to_csv(
    OUT_DIR
    / "matched_rows.csv",
    index=False,
)

unmatched_manual.to_csv(
    OUT_DIR
    / "unmatched_benchmark.csv",
    index=False,
)

unmatched_extracted.to_csv(
    OUT_DIR
    / "unmatched_extracted.csv",
    index=False,
)

per_doi.to_csv(
    OUT_DIR
    / "per_doi_summary.csv",
    index=False,
)

summary.to_csv(
    OUT_DIR
    / "match_summary.csv",
    index=False,
)


# ============================================================
# Console report
# ============================================================

print("=" * 100)
print("STEP 09B1 — CONSERVATIVE ONE-TO-ONE BENCHMARK MATCH")
print("=" * 100)

print()
print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("MATCH CLASSES")
print("=" * 100)

if len(matched):

    print(
        matched[
            "match_class"
        ]
        .value_counts()
        .to_string()
    )

else:

    print("No matches.")


print()
print("=" * 100)
print("PER-DOI SUMMARY")
print("=" * 100)

print(
    per_doi
    .sort_values(
        [
            "benchmark_recall_pct",
            "doi",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("ONE-TO-ONE INVARIANTS")
print("=" * 100)

print(
    "Matched extracted IDs unique:",
    (
        matched[
            "final_record_id"
        ]
        .nunique()
        == len(matched)
        if len(matched)
        else True
    ),
)

print(
    "Matched benchmark IDs unique:",
    (
        matched[
            "benchmark_row_id"
        ]
        .nunique()
        == len(matched)
        if len(matched)
        else True
    ),
)

print(
    "Matched + unmatched benchmark:",
    (
        len(matched)
        + len(unmatched_manual)
    ),
)

print(
    "Expected benchmark:",
    len(manual),
)

print(
    "Matched + unmatched extracted:",
    (
        len(matched)
        + len(unmatched_extracted)
    ),
)

print(
    "Expected eligible extracted:",
    len(extracted),
)

print()
print(
    "Source datasets modified: False"
)

print(
    "Saved:",
    OUT_DIR,
)
