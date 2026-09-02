from pathlib import Path
import re
import unicodedata
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

FINAL_PATH = (
    ROOT / "processed_tables"
    / "final_source_only_extractions.csv"
)

MANUAL_PATH = (
    ROOT / "data"
    / "feedstock.xlsx"
)

ORIGINAL_PATH = (
    ROOT / "processed_tables"
    / "original_extractions_flattened_normalized.csv"
)

CROSSWALK_PATH = (
    ROOT / "processed_tables"
    / "paper_id_final_crosswalk.csv"
)

OUT_DIR = (
    ROOT / "processed_tables"
    / "fresh_benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def clean_doi(x):
    if pd.isna(x):
        return ""

    s = str(x).strip().lower()

    s = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        s,
    )

    s = re.sub(
        r"^doi:\s*",
        "",
        s,
    )

    return s.strip()


def norm_name(x):
    if pd.isna(x):
        return ""

    s = unicodedata.normalize(
        "NFKD",
        str(x),
    )

    s = (
        s.encode("ascii", "ignore")
        .decode()
        .lower()
    )

    # Remove trailing temperature coding only.
    s = re.sub(
        r"[\s_-]*\d{2,4}"
        r"\s*(?:deg\s*)?(?:c)?$",
        "",
        s,
    )

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s,
    )

    return re.sub(
        r"\s+",
        " ",
        s,
    ).strip()


def name_score(a, b):
    a = norm_name(a)
    b = norm_name(b)

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

    ta = set(a.split())
    tb = set(b.split())

    if not ta or not tb:
        return 0

    overlap = (
        len(ta & tb)
        / min(len(ta), len(tb))
    )

    if overlap >= 0.8:
        return 1

    return 0


def number(x):
    try:
        v = float(x)
    except Exception:
        return None

    if not math.isfinite(v):
        return None

    # Manual dataset uses negative values
    # such as -1 as missing sentinels.
    if v < 0:
        return None

    return v


# ============================================================
# LOAD
# ============================================================

final = pd.read_csv(FINAL_PATH)
manual = pd.read_excel(MANUAL_PATH)
original = pd.read_csv(ORIGINAL_PATH)
crosswalk = pd.read_csv(CROSSWALK_PATH)


# ============================================================
# DOI MAPPING FOR FINAL TABLE-DERIVED ROWS
# ============================================================

paper_to_doi = (
    original[
        ["paper_id", "doi"]
    ]
    .dropna()
    .drop_duplicates("paper_id")
    .set_index("paper_id")["doi"]
    .to_dict()
)

recovery_to_original = dict(
    zip(
        crosswalk["recovery_paper_id"],
        crosswalk["original_paper_id"],
    )
)

mapped_original = final[
    "paper_id"
].map(
    recovery_to_original
)

mapped_doi = mapped_original.map(
    paper_to_doi
)

final["doi_final"] = (
    final["doi"]
    .combine_first(mapped_doi)
    .map(clean_doi)
)

manual["doi_final"] = (
    manual["DOI"]
    .map(clean_doi)
)


if final["doi_final"].eq("").any():
    raise RuntimeError(
        "Some final rows still lack DOI."
    )


# ============================================================
# STANDARD COLUMN NAMES
# ============================================================

manual = manual.rename(
    columns={
        "Feedstock": "sample",
        "T (°C)": "temperature_C",
        "C_char(wt%)": "C_value",
        "H_char(wt%)": "H_value",
        "N_char(wt%)": "N_value",
        "O_char(wt%)": "O_value",
    }
)

manual = manual.reset_index(
    drop=True
)

manual["manual_id"] = [
    f"MANUAL_{i:04d}"
    for i in range(
        1,
        len(manual) + 1
    )
]


# ============================================================
# PAIR FEATURES
# ============================================================

pairs = []

common_dois = sorted(
    set(final["doi_final"])
    & set(manual["doi_final"])
)

for doi in common_dois:

    e = final[
        final["doi_final"] == doi
    ]

    m = manual[
        manual["doi_final"] == doi
    ]

    for ei, er in e.iterrows():

        for mi, mr in m.iterrows():

            ns = name_score(
                er["sample"],
                mr["sample"],
            )

            et = number(
                er["temperature_C"]
            )

            mt = number(
                mr["temperature_C"]
            )

            if (
                et is not None
                and mt is not None
            ):
                td = abs(et - mt)
            else:
                td = None

            errors = []

            for c in [
                "C_value",
                "H_value",
                "N_value",
                "O_value",
            ]:

                a = number(er[c])
                b = number(mr[c])

                if (
                    a is not None
                    and b is not None
                ):
                    errors.append(
                        abs(a - b)
                    )

            ec = len(errors)

            mae = (
                float(np.mean(errors))
                if errors
                else None
            )

            mx = (
                float(np.max(errors))
                if errors
                else None
            )

            # ------------------------------------------------
            # Conservative candidate gate.
            #
            # If both temperatures exist, they must agree.
            # Then require either compatible names or very
            # strong numerical evidence.
            # ------------------------------------------------

            temperature_ok = (
                td is None
                or td <= 0.5
            )

            numeric_strong = (
                ec >= 2
                and mae is not None
                and mae <= 0.15
                and mx <= 0.50
            )

            identity_candidate = (
                temperature_ok
                and (
                    ns >= 1
                    or numeric_strong
                )
            )

            if not identity_candidate:
                continue

            temperature_cost = (
                0.0
                if td is None
                else 100.0 * td
            )

            name_cost = (
                3.0 - ns
            )

            numeric_cost = (
                5.0
                if mae is None
                else mae
            )

            cost = (
                temperature_cost
                + name_cost
                + numeric_cost
            )

            pairs.append(
                {
                    "extracted_index": ei,
                    "manual_index": mi,
                    "assembly_id": (
                        er["assembly_id"]
                    ),
                    "manual_id": (
                        mr["manual_id"]
                    ),
                    "doi": doi,
                    "extracted_sample": (
                        er["sample"]
                    ),
                    "manual_sample": (
                        mr["sample"]
                    ),
                    "name_score": ns,
                    "temperature_extracted": et,
                    "temperature_manual": mt,
                    "temperature_difference": td,
                    "element_count": ec,
                    "element_mae": mae,
                    "maximum_element_error": mx,
                    "cost": cost,
                }
            )


pair_df = pd.DataFrame(pairs)


# ============================================================
# GLOBAL ONE-TO-ONE MATCHING WITHIN EACH DOI
# ============================================================

matches = []

for doi in common_dois:

    p = pair_df[
        pair_df["doi"] == doi
    ].copy()

    if p.empty:
        continue

    e_ids = sorted(
        p["extracted_index"].unique()
    )

    m_ids = sorted(
        p["manual_index"].unique()
    )

    ei_map = {
        v: i
        for i, v in enumerate(e_ids)
    }

    mi_map = {
        v: i
        for i, v in enumerate(m_ids)
    }

    matrix = np.full(
        (
            len(e_ids),
            len(m_ids),
        ),
        1e9,
        dtype=float,
    )

    pair_lookup = {}

    for _, row in p.iterrows():

        i = ei_map[
            row["extracted_index"]
        ]

        j = mi_map[
            row["manual_index"]
        ]

        if row["cost"] < matrix[i, j]:

            matrix[i, j] = row["cost"]

            pair_lookup[
                (i, j)
            ] = row


    rr, cc = linear_sum_assignment(
        matrix
    )

    for i, j in zip(rr, cc):

        if matrix[i, j] >= 1e8:
            continue

        matches.append(
            pair_lookup[(i, j)]
            .to_dict()
        )


matched = pd.DataFrame(matches)


# ============================================================
# NUMERICAL AGREEMENT CLASSIFICATION
# ============================================================

def agreement(row):

    n = int(
        row["element_count"]
    )

    mae = row["element_mae"]
    mx = row[
        "maximum_element_error"
    ]

    if n == 0:
        return "NO_COMPARABLE_CHNO"

    if (
        mae <= 0.02
        and mx <= 0.02
    ):
        return "EXACT_WITHIN_0.02"

    if (
        mae <= 0.15
        and mx <= 0.50
    ):
        return "HIGH_AGREEMENT"

    if (
        mae <= 0.50
        and mx <= 1.50
    ):
        return "MODERATE_AGREEMENT"

    return "LARGE_DISAGREEMENT"


if len(matched):

    matched[
        "agreement_class"
    ] = matched.apply(
        agreement,
        axis=1,
    )


# ============================================================
# UNMATCHED SETS
# ============================================================

matched_manual = set(
    matched["manual_index"]
) if len(matched) else set()

matched_extracted = set(
    matched["extracted_index"]
) if len(matched) else set()

manual_unmatched = manual[
    ~manual.index.isin(
        matched_manual
    )
].copy()

extracted_unmatched = final[
    ~final.index.isin(
        matched_extracted
    )
].copy()


# ============================================================
# FIELD-LEVEL CHNO ACCURACY
# ============================================================

field_rows = []

if len(matched):

    for _, match in matched.iterrows():

        er = final.loc[
            int(match["extracted_index"])
        ]

        mr = manual.loc[
            int(match["manual_index"])
        ]

        for element in [
            "C_value",
            "H_value",
            "N_value",
            "O_value",
        ]:

            a = number(er[element])
            b = number(mr[element])

            if (
                a is None
                or b is None
            ):
                continue

            error = abs(a - b)

            field_rows.append(
                {
                    "element": element,
                    "absolute_error": error,
                    "within_0.02": (
                        error <= 0.02
                    ),
                    "within_0.15": (
                        error <= 0.15
                    ),
                    "within_0.50": (
                        error <= 0.50
                    ),
                }
            )


fields = pd.DataFrame(
    field_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = [
    {
        "metric": "manual_rows",
        "value": len(manual),
    },
    {
        "metric": "final_extracted_rows",
        "value": len(final),
    },
    {
        "metric": "matched_manual_rows",
        "value": len(matched),
    },
    {
        "metric": "manual_identity_coverage_pct",
        "value": (
            100.0 * len(matched)
            / len(manual)
        ),
    },
    {
        "metric": "unmatched_manual_rows",
        "value": len(
            manual_unmatched
        ),
    },
    {
        "metric": "matched_extracted_rows",
        "value": len(
            matched_extracted
        ),
    },
    {
        "metric": "unmatched_extracted_rows",
        "value": len(
            extracted_unmatched
        ),
    },
]


if len(fields):

    summary_rows.extend(
        [
            {
                "metric": "CHNO_fields_compared",
                "value": len(fields),
            },
            {
                "metric": "CHNO_within_0.02_pct",
                "value": (
                    100.0
                    * fields[
                        "within_0.02"
                    ].mean()
                ),
            },
            {
                "metric": "CHNO_within_0.15_pct",
                "value": (
                    100.0
                    * fields[
                        "within_0.15"
                    ].mean()
                ),
            },
            {
                "metric": "CHNO_within_0.50_pct",
                "value": (
                    100.0
                    * fields[
                        "within_0.50"
                    ].mean()
                ),
            },
        ]
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

pair_df.to_csv(
    OUT_DIR / "candidate_pairs.csv",
    index=False,
)

matched.to_csv(
    OUT_DIR / "matched_rows.csv",
    index=False,
)

manual_unmatched.to_csv(
    OUT_DIR / "unmatched_manual.csv",
    index=False,
)

extracted_unmatched.to_csv(
    OUT_DIR / "unmatched_extracted.csv",
    index=False,
)

fields.to_csv(
    OUT_DIR / "field_accuracy.csv",
    index=False,
)

summary.to_csv(
    OUT_DIR / "summary.csv",
    index=False,
)


print("=" * 72)
print("FRESH SOURCE-ONLY BENCHMARK")
print("=" * 72)

print()
print(
    summary.to_string(
        index=False
    )
)

if len(matched):

    print("\nROW AGREEMENT:")
    print(
        matched[
            "agreement_class"
        ]
        .value_counts()
        .to_string()
    )

    print("\nNAME SCORE:")
    print(
        matched[
            "name_score"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

print("\nSaved to:")
print(OUT_DIR)
