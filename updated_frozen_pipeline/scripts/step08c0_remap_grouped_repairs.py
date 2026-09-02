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
    ROOT / "checkpoints"
    / "before_global_upstream_refresh"
    / "semantic_recovery_queue.csv"
)

NEW_QUEUE = (
    ROOT / "processed_tables"
    / "semantic_recovery_queue.csv"
)

OLD_REPAIRS = (
    ROOT / "processed_tables"
    / "grouped_row_repairs.csv"
)

OUT = (
    ROOT / "processed_tables"
    / "grouped_row_repairs_remapped.csv"
)

AUDIT = (
    ROOT / "processed_tables"
    / "grouped_row_repairs_remap_audit.csv"
)


def make_key(row):
    parts = [
        str(row.get("paper_id", "")),
        str(row.get("table_id", "")),
        str(row.get("source_row_index", "")),
        str(row.get("raw_source_row", "")),
    ]

    payload = "\x1f".join(parts)

    return "SRC_" + hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()[:16]


old_q = pd.read_csv(OLD_QUEUE)
new_q = pd.read_csv(NEW_QUEUE)
repairs = pd.read_csv(OLD_REPAIRS)

old_q["source_key"] = old_q.apply(
    make_key,
    axis=1,
)

old_map = old_q[
    ["recovery_id", "source_key"]
].rename(
    columns={
        "recovery_id": "old_recovery_id"
    }
)

new_map = new_q[
    ["recovery_id", "source_key"]
].rename(
    columns={
        "recovery_id": "new_recovery_id"
    }
)

x = repairs.rename(
    columns={
        "recovery_id": "old_recovery_id"
    }
)

x = x.merge(
    old_map,
    on="old_recovery_id",
    how="left",
)

x = x.merge(
    new_map,
    on="source_key",
    how="left",
)

x["status"] = x["new_recovery_id"].apply(
    lambda v:
    "REUSED_IN_NEW_QUEUE"
    if pd.notna(v)
    else "NO_LONGER_IN_NEW_QUEUE"
)

x.to_csv(
    AUDIT,
    index=False,
)

keep = x[
    x["status"] == "REUSED_IN_NEW_QUEUE"
].copy()

keep["recovery_id"] = keep[
    "new_recovery_id"
]

drop_cols = [
    "old_recovery_id",
    "new_recovery_id",
    "status",
]

keep = keep.drop(
    columns=drop_cols
)

keep.to_csv(
    OUT,
    index=False,
)

print("=" * 72)
print("GROUPED REPAIR REMAP")
print("=" * 72)

print("Old grouped repairs:", len(repairs))
print("Remapped:", len(keep))
print(
    "No longer in recovery:",
    (x["status"] == "NO_LONGER_IN_NEW_QUEUE").sum(),
)
print(
    "Unique source keys:",
    keep["source_key"].nunique(),
)
print(
    "Unique new REC IDs:",
    keep["recovery_id"].nunique(),
)

print("\nSTATUS:")
print(
    x["status"]
    .value_counts()
    .to_string()
)
