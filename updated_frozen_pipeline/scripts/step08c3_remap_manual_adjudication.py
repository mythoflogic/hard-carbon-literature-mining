import hashlib
from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

OLD_QUEUE = (
    ROOT
    / "checkpoints"
    / "before_global_upstream_refresh"
    / "semantic_recovery_queue.csv"
)

NEW_QUEUE = (
    ROOT
    / "processed_tables"
    / "semantic_recovery_queue_structural.csv"
)

OLD_MANUAL = (
    ROOT
    / "processed_tables"
    / "semantic_recovery_manual_adjudication.csv"
)

OUT = (
    ROOT
    / "processed_tables"
    / "semantic_recovery_manual_adjudication_remapped.csv"
)

AUDIT = (
    ROOT
    / "processed_tables"
    / "semantic_recovery_manual_adjudication_remap_audit.csv"
)


def norm(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return " ".join(
        str(value).split()
    )


def legacy_source_key(row):

    parts = [
        str(row.get("paper_id", "")),
        str(row.get("table_id", "")),
        str(row.get("source_row_index", "")),
        str(row.get("raw_source_row", "")),
    ]

    payload = "\x1f".join(parts)

    return (
        "SRC_"
        + hashlib.sha1(
            payload.encode("utf-8")
        ).hexdigest()[:16]
    )


def stable_key(row):

    return "\x1e".join(
        [
            norm(row.get("paper_id")),
            norm(row.get("table_id")),
            norm(row.get("raw_source_row")),
        ]
    )


old_q = pd.read_csv(
    OLD_QUEUE,
    low_memory=False,
)

new_q = pd.read_csv(
    NEW_QUEUE,
    low_memory=False,
)

old_manual = pd.read_csv(
    OLD_MANUAL,
    low_memory=False,
)


# ------------------------------------------------------------
# Load previously curated/remapped decisions BEFORE rewriting.
# ------------------------------------------------------------

if OUT.exists() and OUT.stat().st_size > 0:

    try:
        previous = pd.read_csv(
            OUT,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        previous = pd.DataFrame()

else:
    previous = pd.DataFrame()


# ------------------------------------------------------------
# Build stable identities
# ------------------------------------------------------------

old_q["_legacy_source_key"] = (
    old_q.apply(
        legacy_source_key,
        axis=1,
    )
)

old_q["_stable_key"] = (
    old_q.apply(
        stable_key,
        axis=1,
    )
)

new_q["_stable_key"] = (
    new_q.apply(
        stable_key,
        axis=1,
    )
)


if old_q["_stable_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate stable identities in old queue."
    )

if new_q["_stable_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate stable identities in new queue."
    )

if new_q["source_key"].astype(str).duplicated().any():
    raise RuntimeError(
        "Duplicate source keys in new queue."
    )


old_rec_to_stable = dict(
    zip(
        old_q["recovery_id"].astype(str),
        old_q["_stable_key"],
    )
)

old_source_to_stable = dict(
    zip(
        old_q["_legacy_source_key"],
        old_q["_stable_key"],
    )
)

new_source_to_stable = dict(
    zip(
        new_q["source_key"].astype(str),
        new_q["_stable_key"],
    )
)

new_stable_to_rec = dict(
    zip(
        new_q["_stable_key"],
        new_q["recovery_id"].astype(str),
    )
)

new_stable_to_source = dict(
    zip(
        new_q["_stable_key"],
        new_q["source_key"].astype(str),
    )
)


# ------------------------------------------------------------
# Collect candidate decisions.
#
# Previously curated/remapped decisions have priority over the
# original historical manual file because some were deliberately
# made more conservative after source review.
# ------------------------------------------------------------

candidates = []


if not previous.empty:

    for _, row in previous.iterrows():

        source = norm(
            row.get("source_key")
        )

        stable = None
        method = None

        if source in new_source_to_stable:

            stable = new_source_to_stable[
                source
            ]

            method = (
                "CURRENT_SOURCE_KEY"
            )

        elif source in old_source_to_stable:

            stable = old_source_to_stable[
                source
            ]

            method = (
                "OLD_SOURCE_TO_STABLE"
            )

        candidates.append(
            {
                "priority": 2,
                "origin":
                    "previous_remapped",

                "stable_key":
                    stable,

                "match_method":
                    method,

                "old_recovery_id":
                    norm(
                        row.get(
                            "recovery_id"
                        )
                    ),

                "manual_decision":
                    row.get(
                        "manual_decision"
                    ),

                "resolved_sample":
                    row.get(
                        "resolved_sample"
                    ),

                "resolved_temperature_C":
                    row.get(
                        "resolved_temperature_C"
                    ),

                "resolved_temperature_type":
                    row.get(
                        "resolved_temperature_type"
                    ),

                "manual_reason":
                    row.get(
                        "manual_reason"
                    ),
            }
        )


for _, row in old_manual.iterrows():

    old_rec = norm(
        row.get("recovery_id")
    )

    stable = old_rec_to_stable.get(
        old_rec
    )

    candidates.append(
        {
            "priority": 1,
            "origin":
                "original_manual",

            "stable_key":
                stable,

            "match_method":
                (
                    "OLD_RECOVERY_ID_TO_STABLE"
                    if stable
                    else None
                ),

            "old_recovery_id":
                old_rec,

            "manual_decision":
                row.get(
                    "manual_decision"
                ),

            "resolved_sample":
                row.get(
                    "resolved_sample"
                ),

            "resolved_temperature_C":
                row.get(
                    "resolved_temperature_C"
                ),

            "resolved_temperature_type":
                row.get(
                    "resolved_temperature_type"
                ),

            "manual_reason":
                row.get(
                    "manual_reason"
                ),
        }
    )


# ------------------------------------------------------------
# Map to fresh namespace and deduplicate.
# ------------------------------------------------------------

chosen = {}
audit_rows = []


for position, item in enumerate(
    candidates
):

    stable = item[
        "stable_key"
    ]

    new_rec = (
        new_stable_to_rec.get(
            stable
        )
        if stable
        else None
    )

    new_source = (
        new_stable_to_source.get(
            stable
        )
        if stable
        else None
    )

    status = (
        "REUSED_IN_NEW_QUEUE"
        if new_rec
        else
        "NO_LONGER_IN_NEW_QUEUE"
    )

    audit_rows.append(
        {
            "origin":
                item["origin"],

            "old_recovery_id":
                item[
                    "old_recovery_id"
                ],

            "new_recovery_id":
                new_rec,

            "new_source_key":
                new_source,

            "status":
                status,

            "match_method":
                item[
                    "match_method"
                ],

            "manual_decision":
                item[
                    "manual_decision"
                ],

            "resolved_sample":
                item[
                    "resolved_sample"
                ],

            "resolved_temperature_C":
                item[
                    "resolved_temperature_C"
                ],
        }
    )

    if not new_rec:
        continue

    output_record = {
        "recovery_id":
            new_rec,

        "source_key":
            new_source,

        "manual_decision":
            item[
                "manual_decision"
            ],

        "resolved_sample":
            item[
                "resolved_sample"
            ],

        "resolved_temperature_C":
            item[
                "resolved_temperature_C"
            ],

        "resolved_temperature_type":
            item[
                "resolved_temperature_type"
            ],

        "manual_reason":
            item[
                "manual_reason"
            ],
    }

    rank = (
        item["priority"],
        position,
    )

    previous_choice = (
        chosen.get(
            new_rec
        )
    )

    if (
        previous_choice is None
        or rank
        > previous_choice["rank"]
    ):
        chosen[new_rec] = {
            "rank": rank,
            "record": output_record,
        }


out = pd.DataFrame(
    [
        item["record"]
        for item in chosen.values()
    ]
)


if not out.empty:

    if out["recovery_id"].duplicated().any():

        raise RuntimeError(
            "Duplicate recovery IDs after manual remap."
        )

    out = out.sort_values(
        "recovery_id"
    ).reset_index(
        drop=True
    )


out.to_csv(
    OUT,
    index=False,
)

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT,
    index=False,
)


print("=" * 72)
print(
    "STEP 08C3 — STABLE MANUAL ADJUDICATION REMAP"
)
print("=" * 72)

print()
print(
    "Previously curated decisions:",
    len(previous),
)

print(
    "Original historical decisions:",
    len(old_manual),
)

print(
    "Final fresh decisions:",
    len(out),
)

print(
    "Unique fresh recovery IDs:",
    (
        out["recovery_id"].nunique()
        if len(out)
        else 0
    ),
)

print()
print("AUDIT STATUS:")

print(
    audit_df[
        ["origin", "status"]
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()
print("Final decisions:")

if len(out):

    print(
        out[
            [
                "recovery_id",
                "source_key",
                "manual_decision",
                "resolved_sample",
                "resolved_temperature_C",
                "resolved_temperature_type",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("Saved:")
print("-", OUT)
print("-", AUDIT)
