from pathlib import Path
import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ORIGINAL = (
    ROOT / "processed_tables"
    / "original_extractions_flattened_normalized.csv"
)

DET = (
    ROOT / "processed_tables"
    / "deterministic_candidates.csv"
)

REC = (
    ROOT / "processed_tables"
    / "semantic_recovery_final_with_manual.csv"
)

OVERLAPS = (
    ROOT / "processed_tables"
    / "final_assembly_safe_overlaps.csv"
)

OUT = (
    ROOT / "processed_tables"
    / "final_source_only_extractions.csv"
)

SUMMARY = (
    ROOT / "processed_tables"
    / "final_source_only_extractions_summary.csv"
)


# ------------------------------------------------------------
# Original branch
# ------------------------------------------------------------

o = pd.read_csv(ORIGINAL)

original = pd.DataFrame({
    "final_record_id": o["original_record_id"],
    "branch": "ORIGINAL",
    "paper_id": o["paper_id"],
    "doi": o["doi"],
    "sample": o["feedstock"],
    "temperature_C": o["temperature_C"],
    "temperature_type": o["temperature_type"],
    "C_value": o["C_value"],
    "H_value": o["H_value"],
    "N_value": o["N_value"],
    "O_value": o["O_value"],
    "source_key": pd.NA,
    "source_table_id": pd.NA,
    "source_row_index": pd.NA,
    "provenance": "original_llm_extraction",
})


# ------------------------------------------------------------
# Deterministic table branch
# ------------------------------------------------------------

d = pd.read_csv(DET)

det = pd.DataFrame({
    "final_record_id": d["candidate_id"],
    "branch": "TABLE_DETERMINISTIC",
    "paper_id": d["paper_id"],
    "doi": pd.NA,
    "sample": d["sample_raw"],
    "temperature_C": d["temperature_exact_C"],
    "temperature_type": d["temperature_type"],
    "C_value": d["C_value"],
    "H_value": d["H_value"],
    "N_value": d["N_value"],
    "O_value": d["O_value"],
    "source_key": d["source_key"],
    "source_table_id": d["table_id"],
    "source_row_index": d["source_row_index"],
    "provenance": d["provenance_status"],
})


# ------------------------------------------------------------
# Accepted recovery branch
# ------------------------------------------------------------

r = pd.read_csv(REC)

r = r[
    r["final_bucket"] == "ACCEPTED"
].copy()

rec = pd.DataFrame({
    "final_record_id": r["recovery_id"],
    "branch": "TABLE_RECOVERY",
    "paper_id": r["paper_id"],
    "doi": pd.NA,
    "sample": r["resolved_sample"],
    "temperature_C": r["resolved_temperature_C"],
    "temperature_type": r["resolved_temperature_type"],
    "C_value": r["C_value"],
    "H_value": r["H_value"],
    "N_value": r["N_value"],
    "O_value": r["O_value"],
    "source_key": r["source_key"],
    "source_table_id": r["table_id"],
    "source_row_index": pd.NA,
    "provenance": r["final_status"],
})


table = pd.concat(
    [det, rec],
    ignore_index=True,
)


# ------------------------------------------------------------
# Remove table rows already represented by original branch
# ------------------------------------------------------------

overlap = pd.read_csv(OVERLAPS)

remove_table_ids = set(
    overlap["table_id"]
)

table_keep = table[
    ~table["final_record_id"].isin(
        remove_table_ids
    )
].copy()


# ------------------------------------------------------------
# Final assembly
# ------------------------------------------------------------

final = pd.concat(
    [original, table_keep],
    ignore_index=True,
)

final.insert(
    0,
    "assembly_id",
    [
        f"FINAL_{i:05d}"
        for i in range(1, len(final) + 1)
    ],
)


# ------------------------------------------------------------
# Audits
# ------------------------------------------------------------

if final["assembly_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate assembly_id found."
    )

if table["final_record_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate table record IDs found."
    )


summary = pd.DataFrame([
    {
        "metric": "original_rows",
        "count": len(original),
    },
    {
        "metric": "deterministic_table_rows",
        "count": len(det),
    },
    {
        "metric": "accepted_recovery_rows",
        "count": len(rec),
    },
    {
        "metric": "total_table_rows_before_dedup",
        "count": len(table),
    },
    {
        "metric": "safe_source_only_overlaps_removed",
        "count": len(remove_table_ids),
    },
    {
        "metric": "table_rows_after_dedup",
        "count": len(table_keep),
    },
    {
        "metric": "final_rows",
        "count": len(final),
    },
])


final.to_csv(
    OUT,
    index=False,
)

summary.to_csv(
    SUMMARY,
    index=False,
)


print("=" * 72)
print("FINAL SOURCE-ONLY DATASET")
print("=" * 72)

print()
print(
    summary.to_string(
        index=False
    )
)

print("\nBRANCH COUNTS:")
print(
    final["branch"]
    .value_counts()
    .to_string()
)

print("\nSaved:")
print("-", OUT)
print("-", SUMMARY)
