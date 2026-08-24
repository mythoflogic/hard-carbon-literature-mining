#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

BASE = (
    ROOT
    / "processed_tables"
    / "benchmark_final_table_v1"
)

MATCH_DIR = (
    BASE
    / "match_v1"
)

DIAG_PATH = (
    MATCH_DIR
    / "unmatched_benchmark_diagnosed.csv"
)

FINAL_PATH = (
    BASE
    / "final_table_with_doi.csv"
)

DET_PATH = (
    ROOT
    / "processed_tables"
    / "deterministic_candidates_structural.csv"
)

QUEUE_PATH = (
    ROOT
    / "processed_tables"
    / "semantic_recovery_queue_structural.csv"
)


OUT_TEMP_MISMATCH = (
    MATCH_DIR
    / "rootcause_temperature_mismatch.csv"
)

OUT_TEMP_UNRESOLVED = (
    MATCH_DIR
    / "rootcause_temperature_unresolved.csv"
)

OUT_NO_ELIGIBLE = (
    MATCH_DIR
    / "rootcause_no_eligible_doi.csv"
)

OUT_CHNO = (
    MATCH_DIR
    / "rootcause_same_temperature_chno_candidates.csv"
)

OUT_RULES = (
    MATCH_DIR
    / "rootcause_rule_summary.csv"
)


# ============================================================
# Load
# ============================================================

diag = pd.read_csv(
    DIAG_PATH,
    low_memory=False,
)

final = pd.read_csv(
    FINAL_PATH,
    low_memory=False,
)

det = pd.read_csv(
    DET_PATH,
    low_memory=False,
)

queue = pd.read_csv(
    QUEUE_PATH,
    low_memory=False,
)


# ============================================================
# Build one current upstream-record table
# ============================================================

wanted_upstream = [
    "source_key",
    "paper_id",
    "table_id",
    "source_row_index",

    "sample_candidate_raw",
    "sample_raw",

    "temperature_candidate_raw",
    "temperature_original",
    "temperature_C",
    "temperature_exact_C",
    "temperature_low_C",
    "temperature_high_C",
    "temperature_type",

    "C_candidate_raw",
    "H_candidate_raw",
    "N_candidate_raw",
    "O_candidate_raw",

    "C_value",
    "H_value",
    "N_value",
    "O_value",

    "classification",
    "original_classification",
    "semantic_class",

    "temperature_recovered_07h",
    "temperature_recovery_rule_07h",
    "temperature_structural_support_07h",

    "sample_recovered_07i",
    "sample_recovery_rule_07i",

    "queue_before_07g2",
    "semantic_class_before_07g2",
    "temperature_C_before_07g2",

    "structural_action_07g2",
    "structural_recovery_applied_07g2",
    "temperature_recovered_07g2",
    "temperature_not_applicable_07g2",
    "semantic_class_after_07g2",
    "queue_after_07g2",

    "provenance_status",
    "recovery_id",
    "candidate_id",
    "candidate_id_07g2",

    "raw_source_row",
]


def available(df, cols):

    return [
        c
        for c in cols
        if c in df.columns
    ]


det_u = det[
    available(
        det,
        wanted_upstream,
    )
].copy()

det_u[
    "upstream_branch"
] = "DETERMINISTIC"


queue_u = queue[
    available(
        queue,
        wanted_upstream,
    )
].copy()

queue_u[
    "upstream_branch"
] = "SEMANTIC_QUEUE"


upstream = pd.concat(
    [
        det_u,
        queue_u,
    ],
    ignore_index=True,
    sort=False,
)


if (
    upstream["source_key"]
    .astype(str)
    .duplicated()
    .any()
):

    bad = upstream[
        upstream["source_key"]
        .astype(str)
        .duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Duplicate source_key across "
        "deterministic + queue:\n"
        + bad[
            [
                "source_key",
                "upstream_branch",
                "paper_id",
                "table_id",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Attach final IDs and DOI
# ============================================================

final_core = final[
    [
        "final_record_id",
        "source_key",
        "doi_normalized",
        "sample",
        "temperature_status",
        "temperature_C",
        "temperature_low_C",
        "temperature_high_C",
        "C_value",
        "H_value",
        "N_value",
        "O_value",
        "benchmark_eligible",
        "final_status",
        "raw_source_row",
    ]
].copy()


final_core = final_core.rename(
    columns={
        "sample":
            "final_sample",

        "temperature_status":
            "final_temperature_status",

        "temperature_C":
            "final_temperature_C",

        "temperature_low_C":
            "final_temperature_low_C",

        "temperature_high_C":
            "final_temperature_high_C",

        "C_value":
            "final_C",

        "H_value":
            "final_H",

        "N_value":
            "final_N",

        "O_value":
            "final_O",

        "raw_source_row":
            "final_raw_source_row",
    }
)


audit = final_core.merge(
    upstream,
    on="source_key",
    how="left",
    validate="one_to_one",
    suffixes=(
        "_final",
        "_upstream",
    ),
)


# ============================================================
# 1. CHNO exact but temperature mismatch
# ============================================================

tm = diag[
    diag["diagnosis"]
    .eq(
        "CHNO_MATCH_BUT_TEMPERATURE_MISMATCH"
    )
].copy()


tm = tm.merge(
    audit,
    left_on="best_final_record_id",
    right_on="final_record_id",
    how="left",
    validate="many_to_one",
)


tm.to_csv(
    OUT_TEMP_MISMATCH,
    index=False,
)


print("=" * 120)
print("A. CHNO MATCHES BUT TEMPERATURE IS WRONG")
print("=" * 120)

cols = [
    c
    for c in [
        "benchmark_row_id",
        "doi",
        "feedstock_manual",
        "temperature_manual",

        "best_final_record_id",
        "final_sample",
        "final_temperature_status",
        "final_temperature_C",

        "sample_candidate_raw",
        "temperature_candidate_raw",
        "temperature_original",
        "temperature_C_before_07g2",

        "temperature_recovery_rule_07h",
        "sample_recovery_rule_07i",

        "structural_action_07g2",
        "temperature_recovered_07g2",

        "semantic_class_before_07g2",
        "semantic_class_after_07g2",

        "final_raw_source_row",
    ]
    if c in tm.columns
]

print(
    tm[cols].to_string(
        index=False,
        max_colwidth=100,
    )
)


# ============================================================
# 2. CHNO exact but temperature unresolved
# ============================================================

tu = diag[
    diag["diagnosis"]
    .eq(
        "CHNO_MATCH_BUT_TEMPERATURE_UNRESOLVED"
    )
].copy()


tu = tu.merge(
    audit,
    left_on="best_final_record_id",
    right_on="final_record_id",
    how="left",
    validate="many_to_one",
)


tu.to_csv(
    OUT_TEMP_UNRESOLVED,
    index=False,
)


print()
print("=" * 120)
print("B. CHNO MATCHES BUT TEMPERATURE IS UNRESOLVED")
print("=" * 120)

cols = [
    c
    for c in [
        "benchmark_row_id",
        "doi",
        "feedstock_manual",
        "temperature_manual",

        "best_final_record_id",
        "final_sample",
        "final_temperature_status",
        "final_temperature_C",

        "sample_candidate_raw",
        "temperature_candidate_raw",
        "temperature_original",

        "temperature_recovery_rule_07h",
        "sample_recovery_rule_07i",

        "structural_action_07g2",
        "semantic_class_before_07g2",
        "semantic_class_after_07g2",

        "final_status",
        "final_raw_source_row",
    ]
    if c in tu.columns
]

print(
    tu[cols].to_string(
        index=False,
        max_colwidth=100,
    )
)


# ============================================================
# 3. DOIs with no benchmark-eligible extracted row
# ============================================================

no_eligible_diag = diag[
    diag["diagnosis"]
    .eq(
        "NO_BENCHMARK_ELIGIBLE_ROWS_FOR_DOI"
    )
].copy()


no_eligible_dois = set(
    no_eligible_diag["doi"]
    .astype(str)
)


no_eligible_rows = audit[
    audit["doi_normalized"]
    .astype(str)
    .isin(
        no_eligible_dois
    )
].copy()


no_eligible_rows.to_csv(
    OUT_NO_ELIGIBLE,
    index=False,
)


print()
print("=" * 120)
print("C. ALL FINAL ROWS FOR DOI WITH NO ELIGIBLE RECORDS")
print("=" * 120)

cols = [
    c
    for c in [
        "doi_normalized",
        "final_record_id",
        "final_sample",
        "final_temperature_status",
        "final_temperature_C",
        "benchmark_eligible",

        "C_value",
        "H_value",
        "N_value",
        "O_value",

        "sample_candidate_raw",
        "temperature_candidate_raw",
        "temperature_original",

        "temperature_recovery_rule_07h",
        "structural_action_07g2",

        "semantic_class_before_07g2",
        "semantic_class_after_07g2",

        "final_status",
        "final_raw_source_row",
    ]
    if c in no_eligible_rows.columns
]

print(
    no_eligible_rows[
        cols
    ]
    .sort_values(
        [
            "doi_normalized",
            "final_record_id",
        ]
    )
    .to_string(
        index=False,
        max_colwidth=100,
    )
)


# ============================================================
# 4. Temperature matches but CHNO disagrees:
#    show ALL extracted rows at that DOI + temperature.
# ============================================================

cd = diag[
    diag["diagnosis"]
    .eq(
        "TEMPERATURE_MATCH_BUT_CHNO_DISAGREES"
    )
].copy()


candidate_rows = []


for _, mr in cd.iterrows():

    doi = str(
        mr["doi"]
    )

    temp = pd.to_numeric(
        pd.Series(
            [
                mr[
                    "temperature_manual"
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]


    same = audit[
        audit[
            "doi_normalized"
        ]
        .astype(str)
        .eq(doi)
    ].copy()


    same_temp = pd.to_numeric(
        same[
            "final_temperature_C"
        ],
        errors="coerce",
    )


    same = same[
        (
            same[
                "final_temperature_status"
            ]
            .eq("exact")
        )
        &
        (
            (
                same_temp
                - temp
            )
            .abs()
            <= 0.5
        )
    ].copy()


    if len(same) == 0:

        candidate_rows.append(
            {
                "benchmark_row_id":
                    mr[
                        "benchmark_row_id"
                    ],

                "doi":
                    doi,

                "feedstock_manual":
                    mr[
                        "feedstock_manual"
                    ],

                "temperature_manual":
                    temp,

                "candidate_found":
                    False,
            }
        )

        continue


    for _, er in same.iterrows():

        candidate_rows.append(
            {
                "benchmark_row_id":
                    mr[
                        "benchmark_row_id"
                    ],

                "doi":
                    doi,

                "feedstock_manual":
                    mr[
                        "feedstock_manual"
                    ],

                "temperature_manual":
                    temp,

                "C_manual":
                    mr[
                        "C_manual"
                    ],

                "H_manual":
                    mr[
                        "H_manual"
                    ],

                "N_manual":
                    mr[
                        "N_manual"
                    ],

                "O_manual":
                    mr[
                        "O_manual"
                    ],

                "candidate_found":
                    True,

                "final_record_id":
                    er[
                        "final_record_id"
                    ],

                "sample_extracted":
                    er[
                        "final_sample"
                    ],

                "temperature_extracted":
                    er[
                        "final_temperature_C"
                    ],

                "C_extracted":
                    er[
                        "final_C"
                    ],

                "H_extracted":
                    er[
                        "final_H"
                    ],

                "N_extracted":
                    er[
                        "final_N"
                    ],

                "O_extracted":
                    er[
                        "final_O"
                    ],

                "table_id":
                    er.get(
                        "table_id",
                        pd.NA,
                    ),

                "source_row_index":
                    er.get(
                        "source_row_index",
                        pd.NA,
                    ),

                "classification":
                    er.get(
                        "classification",
                        pd.NA,
                    ),

                "structural_action_07g2":
                    er.get(
                        "structural_action_07g2",
                        pd.NA,
                    ),

                "raw_source_row":
                    er.get(
                        "final_raw_source_row",
                        pd.NA,
                    ),
            }
        )


chno_candidates = pd.DataFrame(
    candidate_rows
)


chno_candidates.to_csv(
    OUT_CHNO,
    index=False,
)


print()
print("=" * 120)
print("D. SAME-TEMPERATURE CANDIDATES FOR CHNO-DISAGREEMENT ROWS")
print("=" * 120)

print(
    chno_candidates
    .to_string(
        index=False,
        max_colwidth=90,
    )
)


# ============================================================
# 5. Recovery-rule summary for temperature failures
# ============================================================

rule_rows = []


for label, df in [
    (
        "WRONG_TEMPERATURE",
        tm,
    ),
    (
        "UNRESOLVED_TEMPERATURE",
        tu,
    ),
]:

    if len(df) == 0:
        continue

    grouped = (
        df.groupby(
            [
                "temperature_recovery_rule_07h",
                "structural_action_07g2",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    grouped.insert(
        0,
        "problem",
        label,
    )

    rule_rows.append(
        grouped
    )


if rule_rows:

    rules = pd.concat(
        rule_rows,
        ignore_index=True,
    )

else:

    rules = pd.DataFrame()


rules.to_csv(
    OUT_RULES,
    index=False,
)


print()
print("=" * 120)
print("E. TEMPERATURE FAILURE × RECOVERY RULE")
print("=" * 120)

if len(rules):

    print(
        rules.to_string(
            index=False
        )
    )

else:

    print("No rows.")


print()
print("=" * 120)
print("AUDIT ACCOUNTING")
print("=" * 120)

print(
    "Wrong-temperature rows:",
    len(tm),
)

print(
    "Unresolved-temperature rows:",
    len(tu),
)

print(
    "No-eligible benchmark rows:",
    len(no_eligible_diag),
)

print(
    "CHNO-disagreement benchmark rows:",
    len(cd),
)

print(
    "One-to-one competition rows:",
    (
        diag["diagnosis"]
        .eq(
            "ONE_TO_ONE_COMPETITION"
        )
        .sum()
    ),
)

print(
    "Total:",
    len(diag),
)

print()
print(
    "Source datasets modified: False"
)
