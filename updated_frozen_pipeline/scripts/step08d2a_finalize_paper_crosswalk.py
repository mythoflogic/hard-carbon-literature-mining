from pathlib import Path
import pandas as pd


AUDIT_PATH = Path(
    "processed_tables/"
    "paper_id_normalization_audit.csv"
)

OUT_PATH = Path(
    "processed_tables/"
    "paper_id_final_crosswalk.csv"
)


audit = pd.read_csv(
    AUDIT_PATH
)


def prefix_compatible(a, b):

    if pd.isna(a) or pd.isna(b):
        return False

    a = str(a).strip()
    b = str(b).strip()

    shorter = min(
        (a, b),
        key=len,
    )

    longer = max(
        (a, b),
        key=len,
    )

    if len(shorter) < 20:
        return False

    return longer.startswith(shorter)


# ------------------------------------------------------------
# 1. Exact normalized matches
# ------------------------------------------------------------

exact = audit[
    audit["_merge"].eq("both")
][
    [
        "original_paper_id",
        "recovery_paper_id",
        "paper_id_normalized",
    ]
].copy()

exact["match_method"] = (
    "exact_normalized"
)


# ------------------------------------------------------------
# 2. Still-unmatched IDs
# ------------------------------------------------------------

left = audit[
    audit["_merge"].eq("left_only")
][
    [
        "original_paper_id",
        "paper_id_normalized",
    ]
].dropna()

right = audit[
    audit["_merge"].eq("right_only")
][
    [
        "recovery_paper_id",
        "paper_id_normalized",
    ]
].dropna()


candidate_rows = []

for _, r in right.iterrows():

    for _, o in left.iterrows():

        if prefix_compatible(
            o["paper_id_normalized"],
            r["paper_id_normalized"],
        ):

            candidate_rows.append(
                {
                    "original_paper_id":
                        o["original_paper_id"],

                    "recovery_paper_id":
                        r["recovery_paper_id"],

                    "paper_id_normalized":
                        o["paper_id_normalized"],
                }
            )


candidates = pd.DataFrame(
    candidate_rows
)


# ------------------------------------------------------------
# 3. Only accept unique one-to-one prefix matches
# ------------------------------------------------------------

prefix_matches = []

if len(candidates):

    recovery_counts = (
        candidates[
            "recovery_paper_id"
        ]
        .value_counts()
    )

    original_counts = (
        candidates[
            "original_paper_id"
        ]
        .value_counts()
    )

    for _, row in candidates.iterrows():

        if (
            recovery_counts[
                row["recovery_paper_id"]
            ] == 1
            and
            original_counts[
                row["original_paper_id"]
            ] == 1
        ):

            prefix_matches.append(
                row.to_dict()
            )


prefix = pd.DataFrame(
    prefix_matches
)

if len(prefix):

    prefix["match_method"] = (
        "unique_prefix"
    )


# ------------------------------------------------------------
# 4. Final crosswalk
# ------------------------------------------------------------

parts = [
    exact
]

if len(prefix):
    parts.append(
        prefix[
            exact.columns
        ]
    )


crosswalk = pd.concat(
    parts,
    ignore_index=True,
)


# ------------------------------------------------------------
# 5. Safety checks
# ------------------------------------------------------------

dup_recovery = crosswalk[
    "recovery_paper_id"
].duplicated().sum()

dup_original = crosswalk[
    "original_paper_id"
].duplicated().sum()


print("=" * 72)
print("STEP 08D2A.3 — FINAL PAPER-ID CROSSWALK")
print("=" * 72)

print(
    "\nMatched recovery papers:",
    len(crosswalk),
)

print(
    "Duplicate recovery IDs:",
    dup_recovery,
)

print(
    "Duplicate original IDs:",
    dup_original,
)

print(
    "\nMatch methods:"
)

print(
    crosswalk[
        "match_method"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 6. Verify all recovery papers are covered
# ------------------------------------------------------------

recovery_all = set(
    audit[
        "recovery_paper_id"
    ]
    .dropna()
    .astype(str)
)

recovery_matched = set(
    crosswalk[
        "recovery_paper_id"
    ]
    .dropna()
    .astype(str)
)

unmatched = sorted(
    recovery_all
    - recovery_matched
)


print(
    "\nRecovery papers still unmatched:",
    len(unmatched),
)

for value in unmatched:
    print(value)


if (
    dup_recovery > 0
    or dup_original > 0
):

    raise SystemExit(
        "\nSTOP: crosswalk is not one-to-one."
    )


if unmatched:

    raise SystemExit(
        "\nSTOP: some recovery papers remain unmatched."
    )


crosswalk.to_csv(
    OUT_PATH,
    index=False,
)


print(
    "\nSaved:",
    OUT_PATH,
)

