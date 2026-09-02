from pathlib import Path
import re
import unicodedata
import pandas as pd


MATCHED_PATH = Path(
    "processed_extraction/"
    "validation_v4/"
    "combined_matched_rows.csv"
)

OUT_ALIAS = Path(
    "processed_tables/"
    "step08e2a_paper_name_aliases.csv"
)

OUT_AMBIGUOUS = Path(
    "processed_tables/"
    "step08e2a_ambiguous_name_aliases.csv"
)


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

    # Remove trailing processing-temperature codes.
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


df = pd.read_csv(
    MATCHED_PATH
)


print("=" * 72)
print("STEP 08E2A — BUILD PAPER-SPECIFIC NAME ALIASES")
print("=" * 72)

print("\nMatched benchmark rows:", len(df))


df["doi_norm"] = (
    df["doi_norm"]
    .apply(normalize_doi)
)


df["extracted_name_norm"] = (
    df["feedstock_extracted"]
    .apply(normalize_name)
)


df["manual_name_norm"] = (
    df["feedstock_manual"]
    .apply(normalize_name)
)


# Drop empty identifiers.
x = df[
    df["doi_norm"].ne("")
    & df["extracted_name_norm"].ne("")
    & df["manual_name_norm"].ne("")
].copy()


# One alias pair may occur at multiple temperatures.
pairs = (
    x.groupby(
        [
            "doi_norm",
            "extracted_name_norm",
            "manual_name_norm",
        ]
    )
    .size()
    .reset_index(
        name="supporting_matched_rows"
    )
)


# ------------------------------------------------------------
# A safe alias must map one normalized extracted name
# to exactly one normalized manual name within that DOI.
# ------------------------------------------------------------

counts = (
    pairs.groupby(
        [
            "doi_norm",
            "extracted_name_norm",
        ]
    )[
        "manual_name_norm"
    ]
    .nunique()
    .reset_index(
        name="manual_name_count"
    )
)


pairs = pairs.merge(
    counts,
    on=[
        "doi_norm",
        "extracted_name_norm",
    ],
    how="left",
)


safe = pairs[
    pairs[
        "manual_name_count"
    ].eq(1)
].copy()


ambiguous = pairs[
    pairs[
        "manual_name_count"
    ].gt(1)
].copy()


safe.to_csv(
    OUT_ALIAS,
    index=False,
)

ambiguous.to_csv(
    OUT_AMBIGUOUS,
    index=False,
)


print(
    "\nUnique safe paper-specific aliases:",
    len(safe),
)

print(
    "Ambiguous aliases:",
    ambiguous[
        [
            "doi_norm",
            "extracted_name_norm",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)


print("\nSAMPLE SAFE ALIASES:\n")

print(
    safe[
        [
            "doi_norm",
            "extracted_name_norm",
            "manual_name_norm",
            "supporting_matched_rows",
        ]
    ]
    .head(60)
    .to_string(index=False)
)


if len(ambiguous):

    print("\nAMBIGUOUS ALIASES:\n")

    print(
        ambiguous[
            [
                "doi_norm",
                "extracted_name_norm",
                "manual_name_norm",
                "supporting_matched_rows",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )


print("\nSaved:")
print("-", OUT_ALIAS)
print("-", OUT_AMBIGUOUS)

