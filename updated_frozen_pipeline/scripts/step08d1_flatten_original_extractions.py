from pathlib import Path
import pandas as pd


INPUT = Path(
    "processed_extraction/"
    "final_merged_extractions_with_doi.jsonl"
)

OUTPUT = Path(
    "processed_tables/"
    "original_extractions_flattened.csv"
)


df = pd.read_json(
    INPUT,
    lines=True,
)


rows = []


for _, paper_row in df.iterrows():

    extraction = paper_row.get(
        "extraction"
    )

    if not isinstance(
        extraction,
        dict,
    ):
        continue

    records = extraction.get(
        "records",
        []
    )

    if not isinstance(
        records,
        list,
    ):
        continue

    doi = extraction.get(
        "doi"
    )

    source_filename = extraction.get(
        "source_filename"
    )

    paper_id = extraction.get(
        "paper_id"
    ) or paper_row.get(
        "paper_id"
    )


    for record_index, record in enumerate(
        records,
        start=1,
    ):

        if not isinstance(
            record,
            dict,
        ):
            continue

        evidence = record.get(
            "evidence"
        )

        if isinstance(
            evidence,
            list,
        ):
            evidence_text = " | ".join(
                str(x)
                for x in evidence
            )

        elif evidence is None:
            evidence_text = None

        else:
            evidence_text = str(
                evidence
            )


        rows.append(
            {
                "original_record_id": (
                    f"{paper_row['prompt_id']}"
                    f"_R{record_index:03d}"
                ),

                "prompt_id":
                    paper_row.get(
                        "prompt_id"
                    ),

                "paper_id":
                    paper_id,

                "doi":
                    doi,

                "source_filename":
                    source_filename,

                "feedstock":
                    record.get(
                        "feedstock"
                    ),

                "group":
                    record.get(
                        "group"
                    ),

                "temperature_C":
                    record.get(
                        "temperature_C"
                    ),

                "temperature_type":
                    record.get(
                        "temperature_type"
                    ),

                "C_value":
                    record.get(
                        "carbon_char_wt_percent"
                    ),

                "H_value":
                    record.get(
                        "hydrogen_char_wt_percent"
                    ),

                "N_value":
                    record.get(
                        "nitrogen_char_wt_percent"
                    ),

                "O_value":
                    record.get(
                        "oxygen_char_wt_percent"
                    ),

                "evidence":
                    evidence_text,
            }
        )


flat = pd.DataFrame(
    rows
)


print("=" * 72)
print("STEP 08D1 — FLATTEN ORIGINAL EXTRACTIONS")
print("=" * 72)

print(
    "\nPaper-level rows:",
    len(df),
)

print(
    "Flattened record rows:",
    len(flat),
)

print(
    "Unique papers represented:",
    flat["paper_id"].nunique(),
)

print(
    "Duplicate original_record_id:",
    flat[
        "original_record_id"
    ].duplicated().sum(),
)


print(
    "\nRows by paper:"
)

print(
    flat.groupby(
        "paper_id"
    )
    .size()
    .sort_values(
        ascending=False
    )
    .to_string()
)


flat.to_csv(
    OUTPUT,
    index=False,
)


print(
    "\nSaved:",
    OUTPUT,
)
