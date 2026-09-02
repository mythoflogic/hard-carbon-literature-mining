from pathlib import Path
from urllib.parse import unquote
import re
import pandas as pd


ORIGINAL_PATH = Path(
    "processed_tables/"
    "original_extractions_flattened.csv"
)

RECOVERY_PATH = Path(
    "processed_tables/"
    "semantic_recovery_final_with_manual.csv"
)

OUT_ORIGINAL = Path(
    "processed_tables/"
    "original_extractions_flattened_normalized.csv"
)

OUT_RECOVERY = Path(
    "processed_tables/"
    "semantic_recovery_final_with_manual_normalized.csv"
)

OUT_MAPPING = Path(
    "processed_tables/"
    "paper_id_normalization_audit.csv"
)


def normalize_paper_id(value):

    if pd.isna(value):
        return None

    text = str(value)

    # Decode URL-style filename characters:
    # %28biochar%29 -> (biochar)
    text = unquote(text)

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Remove evidence suffix accidentally embedded
    # in a paper identifier.
    text = re.sub(
        r"_E\d+$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Elsevier identifiers sometimes differ only
    # by a trailing "-main".
    text = re.sub(
        r"-main$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize punctuation/spacing for comparison
    # without destroying the original ID.
    text = text.lower()

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


original = pd.read_csv(
    ORIGINAL_PATH
)

recovery = pd.read_csv(
    RECOVERY_PATH
)


original[
    "paper_id_normalized"
] = original[
    "paper_id"
].apply(
    normalize_paper_id
)


recovery[
    "paper_id_normalized"
] = recovery[
    "paper_id"
].apply(
    normalize_paper_id
)


# ------------------------------------------------------------
# Paper-level audit
# ------------------------------------------------------------

o_ids = (
    original[
        [
            "paper_id",
            "paper_id_normalized",
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "paper_id":
                "original_paper_id"
        }
    )
)

r_ids = (
    recovery[
        [
            "paper_id",
            "paper_id_normalized",
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "paper_id":
                "recovery_paper_id"
        }
    )
)


audit = o_ids.merge(
    r_ids,
    on="paper_id_normalized",
    how="outer",
    indicator=True,
)


original.to_csv(
    OUT_ORIGINAL,
    index=False,
)

recovery.to_csv(
    OUT_RECOVERY,
    index=False,
)

audit.to_csv(
    OUT_MAPPING,
    index=False,
)


print("=" * 72)
print("STEP 08D2A — NORMALIZE PAPER IDENTITIES")
print("=" * 72)

print(
    "\nOriginal unique papers:",
    original[
        "paper_id"
    ].nunique(),
)

print(
    "Recovery unique papers:",
    recovery[
        "paper_id"
    ].nunique(),
)

print("\nNORMALIZED ID OVERLAP:")

print(
    audit["_merge"]
    .value_counts()
    .to_string()
)


print("\nMATCHED NAME VARIANTS:\n")

variants = audit[
    (audit["_merge"] == "both")
    & (
        audit["original_paper_id"]
        != audit["recovery_paper_id"]
    )
]

if len(variants):

    print(
        variants[
            [
                "original_paper_id",
                "recovery_paper_id",
            ]
        ].to_string(
            index=False
        )
    )

else:

    print("None")


print("\nORIGINAL PAPERS WITH NO RECOVERY PACKAGE:\n")

remaining = audit[
    audit["_merge"] == "left_only"
]

print(
    remaining[
        "original_paper_id"
    ].to_string(
        index=False
    )
)


print("\nRECOVERY PAPERS STILL UNMATCHED:\n")

remaining_r = audit[
    audit["_merge"] == "right_only"
]

print(
    remaining_r[
        "recovery_paper_id"
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print("-", OUT_ORIGINAL)
print("-", OUT_RECOVERY)
print("-", OUT_MAPPING)

