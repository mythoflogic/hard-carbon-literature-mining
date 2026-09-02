import json
import re
from pathlib import Path

import pandas as pd


PROJECT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ROWS_PATH = (
    PROJECT
    / "processed_tables"
    / "classified_extractable_rows.csv"
)

TABLES_PATH = (
    PROJECT
    / "processed_tables"
    / "classified_elemental_tables.jsonl"
)

OUT_PATH = (
    PROJECT
    / "processed_tables"
    / "classified_extractable_rows_enriched.csv"
)

AUDIT_PATH = (
    PROJECT
    / "processed_tables"
    / "caption_abbreviation_resolution_audit.csv"
)


def clean_text(x):
    if x is None or pd.isna(x):
        return ""

    x = re.sub(
        r"<[^>]+>",
        " ",
        str(x),
    )

    x = re.sub(
        r"\s+",
        " ",
        x,
    )

    return x.strip()


def extract_caption_definitions(caption):

    caption = clean_text(
        caption
    )

    pair_pattern = re.compile(
        r"""
        (?P<long>
            [A-Za-z][A-Za-z0-9'’\-/ ,]*?
        )
        \s*
        \(
        (?P<short>
            [A-Za-z][A-Za-z0-9\-]{1,12}
        )
        \)
        """,
        re.VERBOSE,
    )

    connector_words = {
        "and",
        "or",
        "to",
        "at",
        "of",
        "the",
        "a",
        "an",
    }

    def abbreviation_matches(
        long_form,
        short_form,
    ):

        long_compact = re.sub(
            r"[^a-z0-9]",
            "",
            long_form.lower(),
        )

        short_compact = re.sub(
            r"[^a-z0-9]",
            "",
            short_form.lower(),
        )

        pos = 0

        for char in short_compact:

            found = long_compact.find(
                char,
                pos,
            )

            if found < 0:
                return False

            pos = found + 1

        return True


    found = []

    for match in pair_pattern.finditer(
        caption
    ):

        candidate = match.group(
            "long"
        ).strip(
            " ,;:"
        )

        short_form = match.group(
            "short"
        ).strip()

        pieces = re.split(
            r"[.;,]",
            candidate,
        )

        candidate = (
            pieces[-1].strip()
            if pieces
            else candidate
        )

        words = candidate.split()

        while (
            words
            and words[0].lower()
            in connector_words
        ):
            words.pop(0)

        best = None

        for start in range(
            len(words)
        ):

            suffix = " ".join(
                words[start:]
            )

            if abbreviation_matches(
                suffix,
                short_form,
            ):
                best = suffix

        if best is None:
            best = " ".join(
                words
            )

        if (
            len(words) <= 3
            and words
            and words[0].lower()
            == short_form.lower()
        ):
            best = " ".join(
                words
            )

        best = best.strip(
            " ,;:"
        )

        if not best:
            continue

        found.append(
            {
                "short_form":
                    short_form,

                "long_form":
                    best,
            }
        )

    return found


# ============================================================
# LOAD TABLE CAPTIONS
# ============================================================

table_records = []

with TABLES_PATH.open() as f:

    for line in f:

        if line.strip():
            table_records.append(
                json.loads(line)
            )


caption_by_table = {
    record.get("table_id"):
        record.get("caption", "")
    for record in table_records
}


# ============================================================
# LOAD CLASSIFIED ROWS
# ============================================================

df = pd.read_csv(
    ROWS_PATH
)

df[
    "sample_component_codes"
] = None

df[
    "sample_component_expansions"
] = None

df[
    "caption_abbreviation_resolution"
] = False


audit = []


# ============================================================
# RESOLVE SOURCE-DEFINED COMPONENTS
# ============================================================

for idx, row in df.iterrows():

    if row.get(
        "classification"
    ) == "NOT_CHNO_ROW":
        continue

    sample_raw = clean_text(
        row.get(
            "sample_raw"
        )
    )

    if not sample_raw:
        continue

    caption = caption_by_table.get(
        row.get(
            "table_id"
        ),
        "",
    )

    definitions = (
        extract_caption_definitions(
            caption
        )
    )

    if not definitions:
        continue

    # Build an ambiguity-safe lookup.
    #
    # If the same abbreviation is explicitly associated with
    # more than one different long form in the caption, do not
    # resolve it automatically.
    definition_groups = {}

    for definition in definitions:

        short_key = (
            definition[
                "short_form"
            ]
            .strip()
            .lower()
        )

        long_form = clean_text(
            definition[
                "long_form"
            ]
        )

        if (
            not short_key
            or not long_form
        ):
            continue

        definition_groups.setdefault(
            short_key,
            set(),
        ).add(
            long_form
        )

    lookup = {
        short_key:
            next(
                iter(long_forms)
            )
        for short_key, long_forms
        in definition_groups.items()
        if len(long_forms) == 1
    }

    ambiguous_codes = {
        short_key
        for short_key, long_forms
        in definition_groups.items()
        if len(long_forms) > 1
    }

    parts = [
        p.strip()
        for p in re.split(
            r"\s*[-/]\s*",
            sample_raw,
        )
        if p.strip()
    ]

    resolved = []

    for part in parts:

        expansion = lookup.get(
            part.lower()
        )

        if expansion is not None:

            resolved.append(
                (
                    part,
                    expansion,
                )
            )

    if not resolved:
        continue

    codes = " | ".join(
        code
        for code, _ in resolved
    )

    expansions = " | ".join(
        expansion
        for _, expansion in resolved
    )

    df.at[
        idx,
        "sample_component_codes"
    ] = codes

    df.at[
        idx,
        "sample_component_expansions"
    ] = expansions

    df.at[
        idx,
        "caption_abbreviation_resolution"
    ] = True

    audit.append(
        {
            "paper_id":
                row.get(
                    "paper_id"
                ),

            "table_id":
                row.get(
                    "table_id"
                ),

            "source_row_index":
                row.get(
                    "source_row_index"
                ),

            "sample_raw":
                sample_raw,

            "sample_component_codes":
                codes,

            "sample_component_expansions":
                expansions,

            "ambiguous_caption_codes":
                " | ".join(
                    sorted(
                        ambiguous_codes
                    )
                ),

            "caption":
                clean_text(
                    caption
                ),
        }
    )


audit_df = pd.DataFrame(
    audit
)

if (
    audit_df.empty
    and len(
        audit_df.columns
    ) == 0
):
    audit_df = pd.DataFrame(
        columns=[
            "paper_id",
            "table_id",
            "source_row_index",
            "sample_raw",
            "sample_component_codes",
            "sample_component_expansions",
            "ambiguous_caption_codes",
            "caption",
        ]
    )


df.to_csv(
    OUT_PATH,
    index=False,
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)


print("=" * 72)
print("STEP 07F — CAPTION ABBREVIATION ENRICHMENT")
print("=" * 72)

print(
    "\nSource rows:",
    len(df),
)

print(
    "Rows resolved from caption definitions:",
    len(audit_df),
)

print(
    "Tables affected:",
    (
        audit_df["table_id"].nunique()
        if len(audit_df)
        else 0
    ),
)

print(
    "Papers affected:",
    (
        audit_df["paper_id"].nunique()
        if len(audit_df)
        else 0
    ),
)

if len(audit_df):

    print("\nFIRST 30 RESOLUTIONS:\n")

    print(
        audit_df[
            [
                "table_id",
                "source_row_index",
                "sample_raw",
                "sample_component_codes",
                "sample_component_expansions",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

print("\nSaved:")
print("-", OUT_PATH)
print("-", AUDIT_PATH)

