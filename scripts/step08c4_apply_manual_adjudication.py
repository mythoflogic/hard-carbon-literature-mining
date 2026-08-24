from pathlib import Path
import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

AUTO = (
    ROOT / "processed_tables"
    / "semantic_recovery_final_accounting.csv"
)

MANUAL = (
    ROOT / "processed_tables"
    / "semantic_recovery_manual_adjudication_remapped.csv"
)

OUT = (
    ROOT / "processed_tables"
    / "semantic_recovery_final_with_manual.csv"
)

SUMMARY = (
    ROOT / "processed_tables"
    / "semantic_recovery_final_with_manual_summary.csv"
)


auto = pd.read_csv(AUTO)
manual = pd.read_csv(MANUAL)


auto["automatic_final_status"] = auto["final_status"]
auto["automatic_requires_manual_review"] = (
    auto["requires_manual_review"]
)

auto["manual_decision"] = pd.NA
auto["manual_reason"] = pd.NA


manual_by_key = manual.set_index(
    "source_key"
).to_dict("index")


# Current accounting may not yet contain source_key.
# Recover it from the current queue.
queue = pd.read_csv(
    ROOT / "processed_tables"
    / "semantic_recovery_queue_structural.csv"
)

key_map = dict(
    zip(
        queue["recovery_id"],
        queue["source_key"],
    )
)

auto["source_key"] = auto[
    "recovery_id"
].map(key_map)


for i, row in auto.iterrows():

    source_key = row["source_key"]

    if source_key not in manual_by_key:
        continue

    decision = manual_by_key[source_key]

    auto.at[i, "manual_decision"] = (
        decision.get("manual_decision")
    )

    auto.at[i, "manual_reason"] = (
        decision.get("manual_reason")
    )

    manual_decision = str(
        decision.get("manual_decision", "")
    )

    if manual_decision == "KEEP_PARTIAL":

        auto.at[i, "final_status"] = (
            "MANUAL_ACCEPTED_PARTIAL"
        )

        auto.at[i, "resolved_sample"] = (
            decision.get("resolved_sample")
        )

        auto.at[i, "resolved_temperature_C"] = pd.NA

        auto.at[i, "resolved_temperature_type"] = (
            decision.get(
                "resolved_temperature_type"
            )
        )

        auto.at[i, "requires_manual_review"] = False

    elif manual_decision == "RESOLVE":

        auto.at[i, "final_status"] = (
            "MANUAL_RESOLVED"
        )

        auto.at[i, "resolved_sample"] = (
            decision.get("resolved_sample")
        )

        auto.at[i, "resolved_temperature_C"] = (
            decision.get(
                "resolved_temperature_C"
            )
        )

        auto.at[i, "resolved_temperature_type"] = (
            decision.get(
                "resolved_temperature_type"
            )
        )

        auto.at[i, "requires_manual_review"] = False


accepted_statuses = {
    "LLM_RESOLVED",
    "LLM_RESOLVED_PARTIAL",
    "DETERMINISTIC_METADATA_RECOVERY",
    "DETERMINISTIC_GROUPED_ROW_RECOVERY",
    "DETERMINISTIC_SAMPLE_CODE_TEMPERATURE_RECOVERY",
    "DETERMINISTIC_COMPOSITE_CONDITION_RECOVERY",
    "DETERMINISTIC_CONTEXTUAL_TEMPERATURE_RECOVERY",
    "MANUAL_ACCEPTED_PARTIAL",
    "MANUAL_RESOLVED",
}

rejected_statuses = {
    "REJECT_FRAGMENTED_ROW",
    "REJECT_NO_ELEMENTAL_DATA",
    "REJECT_STRUCTURAL_ARTIFACT",
}


def bucket(status):

    if status in accepted_statuses:
        return "ACCEPTED"

    if status in rejected_statuses:
        return "REJECTED"

    return "MANUAL_REVIEW"


auto["final_bucket"] = auto[
    "final_status"
].apply(bucket)


auto.to_csv(
    OUT,
    index=False,
)


summary = (
    auto["final_status"]
    .value_counts()
    .rename_axis("final_status")
    .reset_index(name="row_count")
)

summary.to_csv(
    SUMMARY,
    index=False,
)


print("=" * 72)
print("REFRESHED FINAL WITH MANUAL")
print("=" * 72)

print("\nRows:", len(auto))

print("\nFINAL STATUS:")
print(
    auto["final_status"]
    .value_counts()
    .to_string()
)

print("\nFINAL BUCKET:")
print(
    auto["final_bucket"]
    .value_counts()
    .to_string()
)

print(
    "\nManual review remaining:",
    (auto["final_bucket"] == "MANUAL_REVIEW").sum(),
)

print("\nMANUAL ROWS:")
print(
    auto[
        auto["manual_decision"].notna()
    ][
        [
            "recovery_id",
            "source_key",
            "manual_decision",
            "resolved_sample",
            "resolved_temperature_C",
            "final_status",
        ]
    ].to_string(index=False)
)

