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

OUT_MATCHES = Path(
    "processed_tables/"
    "step08e1_recovery_manual_identity_matches.csv"
)

OUT_RECOVERY_UNMATCHED = Path(
    "processed_tables/"
    "step08e1_recovery_identity_unmatched.csv"
)

OUT_MANUAL_UNMATCHED = Path(
    "processed_tables/"
    "step08e1_manual_identity_unmatched.csv"
)

OUT_SUMMARY = Path(
    "processed_tables/"
    "step08e1_identity_summary.csv"
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

    # Normalize common dash variants.
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove processing temperature suffixes such as:
    # SS-300, BC400, P 600, PW-500 C
    #
    # This is for identity comparison only. The temperature
    # remains separately represented in temperature_C.
    text = re.sub(
        r"[\s_-]*"
        r"\d{2,4}"
        r"\s*(?:deg\s*)?"
        r"(?:c)?$",
        "",
        text,
    )

    # Remove bracketed abbreviation punctuation but retain words.
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def temperature_equal(a, b):

    if pd.isna(a) or pd.isna(b):
        return False

    try:
        return abs(
            float(a) - float(b)
        ) <= 0.5

    except Exception:
        return False


def name_score(a, b):

    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return 0

    if a == b:
        return 3

    if (
        len(a) >= 3
        and len(b) >= 3
        and (
            a in b
            or b in a
        )
    ):
        return 2

    # Compare token overlap.
    ta = set(a.split())
    tb = set(b.split())

    if not ta or not tb:
        return 0

    overlap = len(
        ta & tb
    ) / min(
        len(ta),
        len(tb),
    )

    if overlap >= 0.8:
        return 1

    return 0


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


print("=" * 72)
print("STEP 08E1 — SEMANTIC RECOVERY VS MANUAL IDENTITY")
print("=" * 72)


# ============================================================
# MAP PAPER -> DOI
# ============================================================

paper_doi = (
    original[
        [
            "paper_id",
            "doi",
        ]
    ]
    .dropna(
        subset=["paper_id"]
    )
    .drop_duplicates()
)

paper_doi["doi_norm"] = (
    paper_doi["doi"]
    .apply(normalize_doi)
)


paper_to_doi = dict(
    zip(
        paper_doi["paper_id"],
        paper_doi["doi_norm"],
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
    "manual_row_id"
] = [
    f"MANUAL_{i:04d}"
    for i in range(
        1,
        len(manual) + 1,
    )
]


# ============================================================
# ONLY ACCEPTED RECOVERY ROWS
# ============================================================

accepted = recovery[
    recovery[
        "final_bucket"
    ].eq(
        "ACCEPTED"
    )
].copy()


print("\nManual rows:", len(manual))
print("Accepted recovery rows:", len(accepted))

print(
    "Accepted recovery rows with DOI:",
    accepted[
        "doi_norm"
    ].ne("").sum(),
)


# ============================================================
# GENERATE IDENTITY CANDIDATES
# ============================================================

candidate_rows = []


for _, rr in accepted.iterrows():

    doi = rr["doi_norm"]

    if not doi:
        continue

    candidates = manual[
        manual[
            "doi_norm"
        ].eq(
            doi
        )
    ]

    for _, mm in candidates.iterrows():

        ns = name_score(
            rr.get(
                "resolved_sample"
            ),
            mm.get(
                "Feedstock"
            ),
        )

        ts = temperature_equal(
            rr.get(
                "resolved_temperature_C"
            ),
            mm.get(
                "T (°C)"
            ),
        )

        # Require both sample evidence and exact temperature
        # for automatic identity matching.
        if (
            ns > 0
            and ts
        ):

            candidate_rows.append(
                {
                    "recovery_id":
                        rr["recovery_id"],

                    "manual_row_id":
                        mm["manual_row_id"],

                    "doi_norm":
                        doi,

                    "recovery_sample":
                        rr.get(
                            "resolved_sample"
                        ),

                    "manual_feedstock":
                        mm.get(
                            "Feedstock"
                        ),

                    "recovery_temperature_C":
                        rr.get(
                            "resolved_temperature_C"
                        ),

                    "manual_temperature_C":
                        mm.get(
                            "T (°C)"
                        ),

                    "name_score":
                        ns,

                    "recovery_status":
                        rr.get(
                            "final_status"
                        ),
                }
            )


candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# UNIQUE ONE-TO-ONE IDENTITY MATCHES
# ============================================================

if len(candidates):

    recovery_counts = (
        candidates[
            "recovery_id"
        ]
        .value_counts()
    )

    manual_counts = (
        candidates[
            "manual_row_id"
        ]
        .value_counts()
    )

    candidates[
        "recovery_candidate_count"
    ] = candidates[
        "recovery_id"
    ].map(
        recovery_counts
    )

    candidates[
        "manual_candidate_count"
    ] = candidates[
        "manual_row_id"
    ].map(
        manual_counts
    )


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
# UNMATCHED SETS
# ============================================================

matched_recovery = set(
    unique[
        "recovery_id"
    ]
    if len(unique)
    else []
)

matched_manual = set(
    unique[
        "manual_row_id"
    ]
    if len(unique)
    else []
)


recovery_unmatched = accepted[
    ~accepted[
        "recovery_id"
    ].isin(
        matched_recovery
    )
].copy()


manual_unmatched = manual[
    ~manual[
        "manual_row_id"
    ].isin(
        matched_manual
    )
].copy()


# ============================================================
# SAVE
# ============================================================

unique.to_csv(
    OUT_MATCHES,
    index=False,
)

recovery_unmatched.to_csv(
    OUT_RECOVERY_UNMATCHED,
    index=False,
)

manual_unmatched.to_csv(
    OUT_MANUAL_UNMATCHED,
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "metric": "manual_rows",
            "value": len(manual),
        },
        {
            "metric": "accepted_recovery_rows",
            "value": len(accepted),
        },
        {
            "metric": "identity_candidate_pairs",
            "value": len(candidates),
        },
        {
            "metric": "unique_one_to_one_identity_matches",
            "value": len(unique),
        },
        {
            "metric": "manual_rows_not_identity_matched",
            "value": len(manual_unmatched),
        },
        {
            "metric": "recovery_rows_not_identity_matched",
            "value": len(recovery_unmatched),
        },
    ]
)


summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print("\nSUMMARY")
print("-" * 72)

print(
    summary.to_string(
        index=False
    )
)


if len(candidates):

    print("\nNAME SCORE DISTRIBUTION:")

    print(
        candidates[
            "name_score"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )


print("\nSaved:")
print("-", OUT_MATCHES)
print("-", OUT_RECOVERY_UNMATCHED)
print("-", OUT_MANUAL_UNMATCHED)
print("-", OUT_SUMMARY)

