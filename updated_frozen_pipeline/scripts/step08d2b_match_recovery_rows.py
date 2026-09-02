from pathlib import Path
import math
import re
import pandas as pd


ORIGINAL_PATH = Path(
    "processed_tables/"
    "original_extractions_flattened_normalized.csv"
)

RECOVERY_PATH = Path(
    "processed_tables/"
    "semantic_recovery_final_with_manual_normalized.csv"
)

CROSSWALK_PATH = Path(
    "processed_tables/"
    "paper_id_final_crosswalk.csv"
)

OUT_ALL = Path(
    "processed_tables/"
    "recovery_row_match_diagnostic.csv"
)

OUT_SUMMARY = Path(
    "processed_tables/"
    "recovery_row_match_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

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


def numeric_equal(a, b, tolerance=0.02):

    if pd.isna(a) or pd.isna(b):
        return False

    try:
        return abs(
            float(a) - float(b)
        ) <= tolerance

    except Exception:
        return False


def count_element_matches(
    original_row,
    recovery_row,
):

    pairs = [
        ("C_value", "C_value"),
        ("H_value", "H_value"),
        ("N_value", "N_value"),
        ("O_value", "O_value"),
    ]

    comparable = 0
    matched = 0

    for o_col, r_col in pairs:

        o = original_row.get(o_col)
        r = recovery_row.get(r_col)

        if (
            pd.isna(o)
            or pd.isna(r)
        ):
            continue

        comparable += 1

        if numeric_equal(
            o,
            r,
        ):
            matched += 1

    return matched, comparable


def sample_compatible(
    original_sample,
    recovery_sample,
):

    a = clean_text(
        original_sample
    )

    b = clean_text(
        recovery_sample
    )

    if not a or not b:
        return False

    if a == b:
        return True

    # Allow abbreviation/name containment only when
    # one side is meaningfully informative.
    if len(a) >= 3 and len(b) >= 3:

        if (
            a in b
            or b in a
        ):
            return True

    return False


# ============================================================
# LOAD
# ============================================================

original = pd.read_csv(
    ORIGINAL_PATH
)

recovery = pd.read_csv(
    RECOVERY_PATH
)

crosswalk = pd.read_csv(
    CROSSWALK_PATH
)


# ============================================================
# MAP RECOVERY PAPER IDS TO ORIGINAL PAPER IDS
# ============================================================

mapping = dict(
    zip(
        crosswalk[
            "recovery_paper_id"
        ],
        crosswalk[
            "original_paper_id"
        ],
    )
)


recovery[
    "original_paper_id"
] = recovery[
    "paper_id"
].map(
    mapping
)


if recovery[
    "original_paper_id"
].isna().any():

    bad = recovery[
        recovery[
            "original_paper_id"
        ].isna()
    ]

    print(
        bad[
            [
                "recovery_id",
                "paper_id",
            ]
        ].to_string(
            index=False
        )
    )

    raise SystemExit(
        "STOP: recovery rows with unmapped paper IDs."
    )


# ============================================================
# MATCH EACH RECOVERY ROW AGAINST ORIGINAL ROWS IN SAME PAPER
# ============================================================

results = []


for _, rr in recovery.iterrows():

    candidates = original[
        original[
            "paper_id"
        ].eq(
            rr[
                "original_paper_id"
            ]
        )
    ]

    scored = []

    for _, oo in candidates.iterrows():

        element_matches, comparable = (
            count_element_matches(
                oo,
                rr,
            )
        )

        sample_match = sample_compatible(
            oo.get(
                "feedstock"
            ),
            rr.get(
                "resolved_sample"
            ),
        )

        temperature_match = numeric_equal(
            oo.get(
                "temperature_C"
            ),
            rr.get(
                "resolved_temperature_C"
            ),
            tolerance=0.5,
        )

        # ----------------------------------------------------
        # Score:
        #
        # Elemental agreement is strongest.
        # Sample and temperature are supporting evidence.
        # ----------------------------------------------------

        score = (
            element_matches * 10
            + int(sample_match) * 3
            + int(temperature_match) * 3
        )

        scored.append(
            {
                "original_record_id":
                    oo[
                        "original_record_id"
                    ],

                "score":
                    score,

                "element_matches":
                    element_matches,

                "element_comparable":
                    comparable,

                "sample_match":
                    sample_match,

                "temperature_match":
                    temperature_match,

                "original_feedstock":
                    oo.get(
                        "feedstock"
                    ),

                "original_temperature_C":
                    oo.get(
                        "temperature_C"
                    ),

                "original_C":
                    oo.get(
                        "C_value"
                    ),

                "original_H":
                    oo.get(
                        "H_value"
                    ),

                "original_N":
                    oo.get(
                        "N_value"
                    ),

                "original_O":
                    oo.get(
                        "O_value"
                    ),
            }
        )


    scored = sorted(
        scored,
        key=lambda x: (
            x["score"],
            x["element_matches"],
            x["sample_match"],
            x["temperature_match"],
        ),
        reverse=True,
    )


    if not scored:

        results.append(
            {
                "recovery_id":
                    rr["recovery_id"],

                "original_paper_id":
                    rr[
                        "original_paper_id"
                    ],

                "match_class":
                    "NO_ORIGINAL_CANDIDATES",

                "best_original_record_id":
                    None,

                "best_score":
                    None,

                "second_score":
                    None,

                "best_element_matches":
                    None,

                "best_element_comparable":
                    None,

                "best_sample_match":
                    None,

                "best_temperature_match":
                    None,
            }
        )

        continue


    best = scored[0]

    second_score = (
        scored[1]["score"]
        if len(scored) > 1
        else None
    )

    tied_best = [
        x
        for x in scored
        if x["score"] == best["score"]
    ]


    # --------------------------------------------------------
    # Conservative classification
    # --------------------------------------------------------

    if (
        best[
            "element_matches"
        ] >= 3
        and len(
            tied_best
        ) == 1
    ):

        match_class = (
            "UNIQUE_STRONG_ELEMENT_MATCH"
        )

    elif (
        best[
            "element_matches"
        ] >= 2
        and best[
            "sample_match"
        ]
        and best[
            "temperature_match"
        ]
        and len(
            tied_best
        ) == 1
    ):

        match_class = (
            "UNIQUE_SUPPORTED_MATCH"
        )

    elif (
        best[
            "element_matches"
        ] >= 2
        and len(
            tied_best
        ) > 1
    ):

        match_class = (
            "AMBIGUOUS_ELEMENT_MATCH"
        )

    elif (
        best[
            "sample_match"
        ]
        and best[
            "temperature_match"
        ]
        and len(
            tied_best
        ) == 1
    ):

        match_class = (
            "UNIQUE_SAMPLE_TEMP_MATCH"
        )

    else:

        match_class = (
            "NO_CONFIDENT_MATCH"
        )


    result = {
        "recovery_id":
            rr[
                "recovery_id"
            ],

        "original_paper_id":
            rr[
                "original_paper_id"
            ],

        "recovery_final_status":
            rr.get(
                "final_status"
            ),

        "recovery_final_bucket":
            rr.get(
                "final_bucket"
            ),

        "recovery_sample":
            rr.get(
                "resolved_sample"
            ),

        "recovery_temperature_C":
            rr.get(
                "resolved_temperature_C"
            ),

        "recovery_C":
            rr.get(
                "C_value"
            ),

        "recovery_H":
            rr.get(
                "H_value"
            ),

        "recovery_N":
            rr.get(
                "N_value"
            ),

        "recovery_O":
            rr.get(
                "O_value"
            ),

        "match_class":
            match_class,

        "best_original_record_id":
            best[
                "original_record_id"
            ],

        "best_score":
            best[
                "score"
            ],

        "second_score":
            second_score,

        "best_element_matches":
            best[
                "element_matches"
            ],

        "best_element_comparable":
            best[
                "element_comparable"
            ],

        "best_sample_match":
            best[
                "sample_match"
            ],

        "best_temperature_match":
            best[
                "temperature_match"
            ],

        "best_original_feedstock":
            best[
                "original_feedstock"
            ],

        "best_original_temperature_C":
            best[
                "original_temperature_C"
            ],

        "best_original_C":
            best[
                "original_C"
            ],

        "best_original_H":
            best[
                "original_H"
            ],

        "best_original_N":
            best[
                "original_N"
            ],

        "best_original_O":
            best[
                "original_O"
            ],
    }

    results.append(
        result
    )


result_df = pd.DataFrame(
    results
)


# ============================================================
# DUPLICATE ORIGINAL TARGET AUDIT
# ============================================================

confident_classes = {
    "UNIQUE_STRONG_ELEMENT_MATCH",
    "UNIQUE_SUPPORTED_MATCH",
    "UNIQUE_SAMPLE_TEMP_MATCH",
}


confident = result_df[
    result_df[
        "match_class"
    ].isin(
        confident_classes
    )
].copy()


target_counts = confident[
    "best_original_record_id"
].value_counts()

duplicate_targets = set(
    target_counts[
        target_counts > 1
    ].index
)


result_df[
    "duplicate_original_target"
] = result_df[
    "best_original_record_id"
].isin(
    duplicate_targets
)


# ============================================================
# SAVE + SUMMARY
# ============================================================

result_df.to_csv(
    OUT_ALL,
    index=False,
)


summary = (
    result_df[
        "match_class"
    ]
    .value_counts()
    .rename_axis(
        "match_class"
    )
    .reset_index(
        name="row_count"
    )
)


summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print("=" * 72)
print("STEP 08D2B — RECOVERY ROW MATCH DIAGNOSTIC")
print("=" * 72)

print(
    "\nRecovery rows:",
    len(result_df),
)

print(
    "\nMATCH CLASSES:"
)

print(
    result_df[
        "match_class"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nConfident matches:",
    len(confident),
)

print(
    "Confident matches targeting "
    "a duplicate original record:",
    result_df[
        "duplicate_original_target"
    ].sum(),
)

print(
    "\nBy recovery final bucket:"
)

print(
    pd.crosstab(
        result_df[
            "recovery_final_bucket"
        ],
        result_df[
            "match_class"
        ],
    ).to_string()
)


print("\nSaved:")
print("-", OUT_ALL)
print("-", OUT_SUMMARY)
