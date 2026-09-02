from pathlib import Path
import re
import unicodedata
import pandas as pd


MANUAL_PATH = Path(
    "data/feedstock.xlsx"
)

RECOVERY_PATH = Path(
    "processed_tables/"
    "semantic_recovery_final_with_manual.csv"
)

ORIGINAL_PATH = Path(
    "processed_tables/"
    "original_extractions_flattened.csv"
)

CROSSWALK_PATH = Path(
    "processed_tables/"
    "paper_id_final_crosswalk.csv"
)

ALIAS_PATH = Path(
    "processed_tables/"
    "step08e2a_paper_name_aliases.csv"
)

OLD_MATCH_PATH = Path(
    "processed_extraction/"
    "validation_v4/"
    "combined_matched_rows.csv"
)

OUT_MATCHES = Path(
    "processed_tables/"
    "step08e2b_alias_identity_matches.csv"
)

OUT_UNMATCHED_MANUAL = Path(
    "processed_tables/"
    "step08e2b_manual_unmatched.csv"
)

OUT_UNMATCHED_RECOVERY = Path(
    "processed_tables/"
    "step08e2b_recovery_unmatched.csv"
)

OUT_SUMMARY = Path(
    "processed_tables/"
    "step08e2b_alias_identity_summary.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_doi(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        text,
    )

    text = re.sub(
        r"^doi:\s*",
        "",
        text,
    )

    return text.strip()


def normalize_name(value):

    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = (
        text.encode(
            "ascii",
            "ignore",
        )
        .decode("ascii")
        .lower()
    )

    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    # Remove trailing temperature coding:
    # CF-225, SS-500, P 600, etc.
    text = re.sub(
        r"[\s_-]*"
        r"\d{2,4}"
        r"\s*(?:deg\s*)?"
        r"(?:c)?$",
        "",
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


def temperature_equal(a, b):

    if pd.isna(a) or pd.isna(b):
        return False

    try:
        return abs(
            float(a) - float(b)
        ) <= 0.5

    except Exception:
        return False


# ============================================================
# LOAD
# ============================================================

manual = pd.read_excel(
    MANUAL_PATH
)

recovery = pd.read_csv(
    RECOVERY_PATH
)

original = pd.read_csv(
    ORIGINAL_PATH
)

crosswalk = pd.read_csv(
    CROSSWALK_PATH
)

aliases = pd.read_csv(
    ALIAS_PATH
)

old_matches = pd.read_csv(
    OLD_MATCH_PATH
)


# ============================================================
# DOI MAPPING
# ============================================================

paper_doi = (
    original[
        [
            "paper_id",
            "doi",
        ]
    ]
    .drop_duplicates()
)

paper_to_doi = dict(
    zip(
        paper_doi["paper_id"],
        paper_doi["doi"].apply(
            normalize_doi
        ),
    )
)

recovery_to_original = dict(
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
    recovery_to_original
)

recovery[
    "doi_norm"
] = recovery[
    "original_paper_id"
].map(
    paper_to_doi
)

manual[
    "doi_norm"
] = manual[
    "DOI"
].apply(
    normalize_doi
)

manual[
    "manual_name_norm"
] = manual[
    "Feedstock"
].apply(
    normalize_name
)

manual[
    "manual_row_id"
] = [
    f"MANUAL_{i:04d}"
    for i in range(
        1,
        len(manual) + 1,
    )
]


# ============================================================
# SAFE ALIAS LOOKUP
# ============================================================

alias_lookup = {
    (
        row["doi_norm"],
        row["extracted_name_norm"],
    ):
        row["manual_name_norm"]

    for _, row in aliases.iterrows()
}


def resolve_recovery_name(
    doi,
    sample,
):

    sample_norm = normalize_name(
        sample
    )

    if not sample_norm:
        return "", "EMPTY"

    key = (
        doi,
        sample_norm,
    )

    if key in alias_lookup:

        return (
            alias_lookup[key],
            "LEARNED_SAFE_ALIAS",
        )

    # If no alias is needed, retain literal
    # normalized name.
    return (
        sample_norm,
        "DIRECT_NORMALIZED_NAME",
    )


accepted = recovery[
    recovery[
        "final_bucket"
    ].eq(
        "ACCEPTED"
    )
].copy()


resolved = accepted.apply(
    lambda row:
        resolve_recovery_name(
            row["doi_norm"],
            row["resolved_sample"],
        ),
    axis=1,
)


accepted[
    "recovery_manual_name_norm"
] = [
    x[0]
    for x in resolved
]

accepted[
    "name_resolution_method"
] = [
    x[1]
    for x in resolved
]


# ============================================================
# CANDIDATE PAIRS
# ============================================================

candidate_rows = []


for _, rr in accepted.iterrows():

    if (
        not rr["doi_norm"]
        or not rr[
            "recovery_manual_name_norm"
        ]
        or pd.isna(
            rr["resolved_temperature_C"]
        )
    ):
        continue

    candidates = manual[
        manual[
            "doi_norm"
        ].eq(
            rr["doi_norm"]
        )
        &
        manual[
            "manual_name_norm"
        ].eq(
            rr[
                "recovery_manual_name_norm"
            ]
        )
    ]

    candidates = candidates[
        pd.to_numeric(
            candidates["T (°C)"],
            errors="coerce",
        )
        .sub(
            float(
                rr[
                    "resolved_temperature_C"
                ]
            )
        )
        .abs()
        <= 0.5
    ]


    for _, mm in candidates.iterrows():

        candidate_rows.append(
            {
                "recovery_id":
                    rr["recovery_id"],

                "manual_row_id":
                    mm["manual_row_id"],

                "doi_norm":
                    rr["doi_norm"],

                "recovery_sample":
                    rr["resolved_sample"],

                "resolved_manual_name":
                    rr[
                        "recovery_manual_name_norm"
                    ],

                "manual_feedstock":
                    mm["Feedstock"],

                "recovery_temperature_C":
                    rr[
                        "resolved_temperature_C"
                    ],

                "manual_temperature_C":
                    mm["T (°C)"],

                "name_resolution_method":
                    rr[
                        "name_resolution_method"
                    ],

                "recovery_status":
                    rr["final_status"],
            }
        )


candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# UNIQUE ONE-TO-ONE
# ============================================================

if len(candidates):

    rc = (
        candidates[
            "recovery_id"
        ]
        .value_counts()
    )

    mc = (
        candidates[
            "manual_row_id"
        ]
        .value_counts()
    )

    candidates[
        "recovery_candidate_count"
    ] = candidates[
        "recovery_id"
    ].map(rc)

    candidates[
        "manual_candidate_count"
    ] = candidates[
        "manual_row_id"
    ].map(mc)


    unique = candidates[
        (
            candidates[
                "recovery_candidate_count"
            ] == 1
        )
        &
        (
            candidates[
                "manual_candidate_count"
            ] == 1
        )
    ].copy()

else:

    unique = pd.DataFrame()


# ============================================================
# WHICH MANUAL ROWS WERE ALREADY MATCHED IN V4?
# ============================================================

old_manual_rows = set(
    pd.to_numeric(
        old_matches[
            "manual_row"
        ],
        errors="coerce",
    )
    .dropna()
    .astype(int)
)


def manual_number(
    manual_row_id
):

    return int(
        str(manual_row_id)
        .replace(
            "MANUAL_",
            ""
        )
    )


if len(unique):

    unique[
        "manual_row_number"
    ] = unique[
        "manual_row_id"
    ].apply(
        manual_number
    )

    unique[
        "already_matched_in_v4"
    ] = unique[
        "manual_row_number"
    ].isin(
        old_manual_rows
    )

else:

    unique[
        "manual_row_number"
    ] = []

    unique[
        "already_matched_in_v4"
    ] = []


# ============================================================
# UNMATCHED
# ============================================================

matched_manual = set(
    unique[
        "manual_row_id"
    ]
    if len(unique)
    else []
)

matched_recovery = set(
    unique[
        "recovery_id"
    ]
    if len(unique)
    else []
)


manual_unmatched = manual[
    ~manual[
        "manual_row_id"
    ].isin(
        matched_manual
    )
].copy()


recovery_unmatched = accepted[
    ~accepted[
        "recovery_id"
    ].isin(
        matched_recovery
    )
].copy()


# ============================================================
# SAVE
# ============================================================

unique.to_csv(
    OUT_MATCHES,
    index=False,
)

manual_unmatched.to_csv(
    OUT_UNMATCHED_MANUAL,
    index=False,
)

recovery_unmatched.to_csv(
    OUT_UNMATCHED_RECOVERY,
    index=False,
)


new_manual_matches = (
    (~unique["already_matched_in_v4"])
    .sum()
    if len(unique)
    else 0
)


summary = pd.DataFrame(
    [
        {
            "metric":
                "manual_rows",
            "value":
                len(manual),
        },
        {
            "metric":
                "accepted_recovery_rows",
            "value":
                len(accepted),
        },
        {
            "metric":
                "candidate_pairs_after_aliases",
            "value":
                len(candidates),
        },
        {
            "metric":
                "unique_one_to_one_alias_matches",
            "value":
                len(unique),
        },
        {
            "metric":
                "already_matched_in_validation_v4",
            "value":
                (
                    unique[
                        "already_matched_in_v4"
                    ].sum()
                    if len(unique)
                    else 0
                ),
        },
        {
            "metric":
                "new_manual_rows_reached_by_recovery",
            "value":
                int(
                    new_manual_matches
                ),
        },
        {
            "metric":
                "manual_rows_not_alias_matched",
            "value":
                len(
                    manual_unmatched
                ),
        },
        {
            "metric":
                "recovery_rows_not_alias_matched",
            "value":
                len(
                    recovery_unmatched
                ),
        },
    ]
)


summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print("=" * 72)
print("STEP 08E2B — ALIAS-AWARE RECOVERY VS MANUAL")
print("=" * 72)

print("\nSUMMARY\n")

print(
    summary.to_string(
        index=False
    )
)


if len(unique):

    print(
        "\nMATCHES BY NAME RESOLUTION METHOD:\n"
    )

    print(
        unique[
            "name_resolution_method"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nNEW MANUAL ROWS NOT ALREADY MATCHED IN V4:\n"
    )

    cols = [
        "recovery_id",
        "manual_row_number",
        "doi_norm",
        "recovery_sample",
        "manual_feedstock",
        "recovery_temperature_C",
        "name_resolution_method",
        "recovery_status",
    ]

    new_rows = unique[
        ~unique[
            "already_matched_in_v4"
        ]
    ]

    print(
        new_rows[
            cols
        ]
        .head(100)
        .to_string(
            index=False
        )
    )


print("\nSaved:")
print("-", OUT_MATCHES)
print("-", OUT_UNMATCHED_MANUAL)
print("-", OUT_UNMATCHED_RECOVERY)
print("-", OUT_SUMMARY)


