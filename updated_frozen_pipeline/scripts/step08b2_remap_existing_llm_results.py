import json
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

LLM_DIR = (
    ROOT
    / "processed_tables"
    / "semantic_recovery_llm"
)

OLD_RESULTS = (
    LLM_DIR
    / "semantic_recovery_results.jsonl"
)

OLD_FAILURES = (
    LLM_DIR
    / "semantic_recovery_failures.jsonl"
)

OUT_RESULTS = (
    LLM_DIR
    / "semantic_recovery_results_remapped.jsonl"
)

OUT_FAILURES = (
    LLM_DIR
    / "semantic_recovery_failures_remapped.jsonl"
)

RESULT_AUDIT = (
    LLM_DIR
    / "semantic_recovery_result_remap_audit.csv"
)

FAILURE_AUDIT = (
    LLM_DIR
    / "semantic_recovery_failure_remap_audit.csv"
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


def make_source_key(row):
    """
    Historical source-key definition.

    Kept only so old source keys can be translated
    to stable row identity.
    """

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


def make_stable_key(row):
    """
    Stable identity deliberately excludes source_row_index.

    Restoring rows above an existing row must not change
    the identity of that existing source row.
    """

    parts = [
        norm(row.get("paper_id")),
        norm(row.get("table_id")),
        norm(row.get("raw_source_row")),
    ]

    return "\x1e".join(parts)


def load_jsonl(path):

    rows = []

    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path, rows):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


old_q = pd.read_csv(
    OLD_QUEUE,
    low_memory=False,
)

new_q = pd.read_csv(
    NEW_QUEUE,
    low_memory=False,
)


# ------------------------------------------------------------
# Build old and new identities
# ------------------------------------------------------------

old_q["_legacy_source_key"] = (
    old_q.apply(
        make_source_key,
        axis=1,
    )
)

old_q["_stable_key"] = (
    old_q.apply(
        make_stable_key,
        axis=1,
    )
)

new_q["_stable_key"] = (
    new_q.apply(
        make_stable_key,
        axis=1,
    )
)


# ------------------------------------------------------------
# Safety gates
# ------------------------------------------------------------

if old_q["_stable_key"].duplicated().any():

    raise RuntimeError(
        "Old queue contains duplicate stable identities."
    )


if new_q["_stable_key"].duplicated().any():

    raise RuntimeError(
        "New queue contains duplicate stable identities."
    )


if new_q["source_key"].duplicated().any():

    raise RuntimeError(
        "New queue contains duplicate source keys."
    )


if new_q["recovery_id"].astype(str).duplicated().any():

    raise RuntimeError(
        "New queue contains duplicate recovery IDs."
    )


# ------------------------------------------------------------
# Identity maps
# ------------------------------------------------------------

old_rec_to_stable = dict(
    zip(
        old_q["recovery_id"].astype(str),
        old_q["_stable_key"],
    )
)

old_rec_to_paper = dict(
    zip(
        old_q["recovery_id"].astype(str),
        old_q["paper_id"].map(norm),
    )
)

old_rec_to_table = dict(
    zip(
        old_q["recovery_id"].astype(str),
        old_q["table_id"].map(norm),
    )
)

old_source_to_stable = dict(
    zip(
        old_q["_legacy_source_key"],
        old_q["_stable_key"],
    )
)

new_source_to_rec = dict(
    zip(
        new_q["source_key"].astype(str),
        new_q["recovery_id"].astype(str),
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

new_stable_to_row_index = dict(
    zip(
        new_q["_stable_key"],
        new_q["source_row_index"],
    )
)


# ------------------------------------------------------------
# Mapping helper
# ------------------------------------------------------------

def to_float_or_none(value):

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if pd.isna(value):
        return None

    return value


def map_record(
    record,
    allow_historical_rec_fallback,
):

    old_rec = norm(
        record.get("recovery_id")
    )

    record_source_key = norm(
        record.get("source_key")
    )

    # --------------------------------------------------------
    # 1. Best possible case:
    #    result already carries a source key in the current
    #    fresh queue.
    # --------------------------------------------------------

    if (
        record_source_key
        and record_source_key
        in new_source_to_rec
    ):

        new_rec = (
            new_source_to_rec[
                record_source_key
            ]
        )

        stable = (
            new_q.loc[
                new_q["source_key"]
                .astype(str)
                .eq(record_source_key),
                "_stable_key",
            ]
            .iloc[0]
        )

        return {
            "new_recovery_id": new_rec,
            "new_source_key": record_source_key,
            "stable_key": stable,
            "match_method":
                "EXACT_CURRENT_SOURCE_KEY",
        }


    # --------------------------------------------------------
    # 1B. Current REC identity without source_key.
    #
    # Some valid current-schema LLM results were generated
    # before source_key was written into result records.
    #
    # A REC ID by itself is NEVER sufficient because REC IDs
    # can shift after upstream reconstruction.
    #
    # Reuse the REC ID only when it independently agrees with
    # the fresh queue on:
    # - paper_id
    # - table_id
    # - source_row_index
    # - at least two comparable CHNO values
    # - every comparable CHNO value within numerical tolerance
    # --------------------------------------------------------

    if (
        not record_source_key
        and old_rec
    ):

        current_match = new_q.loc[
            new_q["recovery_id"]
            .astype(str)
            .eq(old_rec)
        ]

        if len(current_match) == 1:

            current_row = (
                current_match.iloc[0]
            )

            record_paper = norm(
                record.get("paper_id")
            )

            record_table = norm(
                record.get("table_id")
            )

            current_paper = norm(
                current_row.get("paper_id")
            )

            current_table = norm(
                current_row.get("table_id")
            )

            record_index = (
                to_float_or_none(
                    record.get(
                        "source_row_index"
                    )
                )
            )

            current_index = (
                to_float_or_none(
                    current_row.get(
                        "source_row_index"
                    )
                )
            )

            identity_ok = (
                bool(record_paper)
                and bool(record_table)
                and record_paper
                == current_paper
                and record_table
                == current_table
                and record_index
                is not None
                and current_index
                is not None
                and abs(
                    record_index
                    - current_index
                )
                <= 1e-9
            )

            comparable = 0
            elemental_ok = True

            for element in [
                "C_value",
                "H_value",
                "N_value",
                "O_value",
            ]:

                old_value = (
                    to_float_or_none(
                        record.get(element)
                    )
                )

                new_value = (
                    to_float_or_none(
                        current_row.get(
                            element
                        )
                    )
                )

                if (
                    old_value is None
                    or new_value is None
                ):
                    continue

                comparable += 1

                if abs(
                    old_value
                    - new_value
                ) > 1e-6:

                    elemental_ok = False
                    break

            if (
                identity_ok
                and elemental_ok
                and comparable >= 2
            ):

                stable = (
                    current_row[
                        "_stable_key"
                    ]
                )

                return {
                    "new_recovery_id":
                        old_rec,

                    "new_source_key":
                        str(
                            current_row[
                                "source_key"
                            ]
                        ),

                    "stable_key":
                        stable,

                    "match_method":
                        "CURRENT_REC_IDENTITY",
                }


    # --------------------------------------------------------
    # 2. Source key belongs to the historical queue.
    #
    #    Convert historical source key -> stable identity
    #    -> current queue.
    #
    #    This safely handles row-index shifts.
    # --------------------------------------------------------

    if (
        record_source_key
        and record_source_key
        in old_source_to_stable
    ):

        stable = (
            old_source_to_stable[
                record_source_key
            ]
        )

        if stable in new_stable_to_rec:

            return {
                "new_recovery_id":
                    new_stable_to_rec[
                        stable
                    ],

                "new_source_key":
                    new_stable_to_source[
                        stable
                    ],

                "stable_key":
                    stable,

                "match_method":
                    "HISTORICAL_SOURCE_KEY_TO_STABLE",
            }


    # --------------------------------------------------------
    # 3. Historical result without explicit source_key.
    #
    #    REC-ID fallback is allowed ONLY for historical-schema
    #    records and is verified against paper/table identity.
    #
    #    It is NEVER used for current-schema records because
    #    current REC IDs may now refer to different rows.
    # --------------------------------------------------------

    if (
        allow_historical_rec_fallback
        and old_rec
        and old_rec in old_rec_to_stable
    ):

        record_paper = norm(
            record.get("paper_id")
        )

        record_table = norm(
            record.get("table_id")
        )

        expected_paper = (
            old_rec_to_paper[
                old_rec
            ]
        )

        expected_table = (
            old_rec_to_table[
                old_rec
            ]
        )

        if (
            record_paper
            and record_paper
            != expected_paper
        ):
            return None

        if (
            record_table
            and record_table
            != expected_table
        ):
            return None

        stable = (
            old_rec_to_stable[
                old_rec
            ]
        )

        if stable in new_stable_to_rec:

            return {
                "new_recovery_id":
                    new_stable_to_rec[
                        stable
                    ],

                "new_source_key":
                    new_stable_to_source[
                        stable
                    ],

                "stable_key":
                    stable,

                "match_method":
                    "HISTORICAL_RECOVERY_ID_TO_STABLE",
            }

    return None


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

old_results = load_jsonl(
    OLD_RESULTS
)

result_candidates = {}
result_audit = []
result_collisions = 0


METHOD_PRIORITY = {
    "EXACT_CURRENT_SOURCE_KEY": 5,
    "CURRENT_REC_IDENTITY": 4,
    "HISTORICAL_SOURCE_KEY_TO_STABLE": 3,
    "HISTORICAL_RECOVERY_ID_TO_STABLE": 1,
}


for position, record in enumerate(
    old_results
):

    old_rec = norm(
        record.get("recovery_id")
    )

    is_current_schema_result = (
        "resolved_temperature_low_C"
        in record
        or
        "resolved_temperature_high_C"
        in record
    )

    mapped = map_record(
        record,
        allow_historical_rec_fallback=(
            not is_current_schema_result
        ),
    )

    if mapped is None:

        result_audit.append(
            {
                "old_recovery_id":
                    old_rec,

                "old_source_key":
                    norm(
                        record.get(
                            "source_key"
                        )
                    ),

                "new_recovery_id":
                    None,

                "new_source_key":
                    None,

                "status":
                    "NOT_REUSED",

                "match_method":
                    None,

                "paper_id":
                    record.get(
                        "paper_id"
                    ),

                "table_id":
                    record.get(
                        "table_id"
                    ),

                "source_row_index":
                    record.get(
                        "source_row_index"
                    ),

                "current_schema":
                    is_current_schema_result,
            }
        )

        continue


    new_rec = (
        mapped[
            "new_recovery_id"
        ]
    )

    priority = (
        METHOD_PRIORITY[
            mapped[
                "match_method"
            ]
        ],
        int(
            is_current_schema_result
        ),
        position,
    )


    new_record = dict(
        record
    )

    new_record[
        "original_recovery_id"
    ] = old_rec

    new_record[
        "original_source_key"
    ] = norm(
        record.get("source_key")
    )

    new_record[
        "recovery_id"
    ] = new_rec

    new_record[
        "source_key"
    ] = mapped[
        "new_source_key"
    ]

    new_record[
        "source_row_index"
    ] = (
        new_stable_to_row_index[
            mapped[
                "stable_key"
            ]
        ]
    )


    if new_rec in result_candidates:

        result_collisions += 1


    previous = (
        result_candidates.get(
            new_rec
        )
    )

    if (
        previous is None
        or priority
        >= previous[
            "priority"
        ]
    ):

        result_candidates[
            new_rec
        ] = {
            "priority":
                priority,

            "record":
                new_record,
        }


    result_audit.append(
        {
            "old_recovery_id":
                old_rec,

            "old_source_key":
                norm(
                    record.get(
                        "source_key"
                    )
                ),

            "new_recovery_id":
                new_rec,

            "new_source_key":
                mapped[
                    "new_source_key"
                ],

            "status":
                "REUSED_IN_NEW_QUEUE",

            "match_method":
                mapped[
                    "match_method"
                ],

            "paper_id":
                record.get(
                    "paper_id"
                ),

            "table_id":
                record.get(
                    "table_id"
                ),

            "source_row_index":
                record.get(
                    "source_row_index"
                ),

            "current_schema":
                is_current_schema_result,
        }
    )


remapped_results = [
    item["record"]
    for item in
    result_candidates.values()
]


write_jsonl(
    OUT_RESULTS,
    remapped_results,
)


pd.DataFrame(
    result_audit
).to_csv(
    RESULT_AUDIT,
    index=False,
)


# ------------------------------------------------------------
# Failures
#
# Failures are deliberately stricter:
# only explicit current/historical source keys are reused.
#
# We do NOT reuse a failure merely because its REC ID happens
# to equal a REC ID in the new namespace.
# ------------------------------------------------------------

old_failures = load_jsonl(
    OLD_FAILURES
)

remapped_failures_by_rec = {}
failure_audit = []


for record in old_failures:

    old_rec = norm(
        record.get("recovery_id")
    )

    mapped = map_record(
        record,
        allow_historical_rec_fallback=False,
    )

    if mapped is None:

        failure_audit.append(
            {
                "old_recovery_id":
                    old_rec,

                "old_source_key":
                    norm(
                        record.get(
                            "source_key"
                        )
                    ),

                "new_recovery_id":
                    None,

                "status":
                    "NOT_REUSED",

                "match_method":
                    None,

                "paper_id":
                    record.get(
                        "paper_id"
                    ),

                "table_id":
                    record.get(
                        "table_id"
                    ),
            }
        )

        continue


    new_rec = (
        mapped[
            "new_recovery_id"
        ]
    )


    new_record = dict(
        record
    )

    new_record[
        "original_recovery_id"
    ] = old_rec

    new_record[
        "original_source_key"
    ] = norm(
        record.get("source_key")
    )

    new_record[
        "recovery_id"
    ] = new_rec

    new_record[
        "source_key"
    ] = mapped[
        "new_source_key"
    ]

    new_record[
        "source_row_index"
    ] = (
        new_stable_to_row_index[
            mapped[
                "stable_key"
            ]
        ]
    )


    remapped_failures_by_rec[
        new_rec
    ] = new_record


    failure_audit.append(
        {
            "old_recovery_id":
                old_rec,

            "old_source_key":
                norm(
                    record.get(
                        "source_key"
                    )
                ),

            "new_recovery_id":
                new_rec,

            "status":
                "REUSED_IN_NEW_QUEUE",

            "match_method":
                mapped[
                    "match_method"
                ],

            "paper_id":
                record.get(
                    "paper_id"
                ),

            "table_id":
                record.get(
                    "table_id"
                ),
        }
    )


remapped_failures = list(
    remapped_failures_by_rec.values()
)


write_jsonl(
    OUT_FAILURES,
    remapped_failures,
)


pd.DataFrame(
    failure_audit
).to_csv(
    FAILURE_AUDIT,
    index=False,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("=" * 72)
print(
    "STEP 08B2 — STABLE-IDENTITY LLM RESULT REMAP"
)
print("=" * 72)

print()
print(
    "Old LLM result records:",
    len(old_results),
)

print(
    "Remapped result records:",
    len(remapped_results),
)

print(
    "Result mapping collisions:",
    result_collisions,
)

print(
    "Unique remapped result IDs:",
    len(
        {
            r["recovery_id"]
            for r in remapped_results
        }
    ),
)

print()
print(
    "Old failure records:",
    len(old_failures),
)

print(
    "Remapped failure records:",
    len(remapped_failures),
)

print()
print("RESULT MATCH METHODS:")

result_audit_df = pd.DataFrame(
    result_audit
)

if not result_audit_df.empty:

    print(
        result_audit_df[
            ["status", "match_method"]
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

print()
print("Saved:")
print("-", OUT_RESULTS)
print("-", OUT_FAILURES)
print("-", RESULT_AUDIT)
print("-", FAILURE_AUDIT)
