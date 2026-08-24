from pathlib import Path
import json
import re
import uuid

import pandas as pd

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# STEP 7
# Markdown papers -> table-aware traceable chunks

"""
This script:

1. Loads every Markdown paper.
2. Detects Markdown tables.
3. Keeps each table together as one standalone chunk.
4. Splits normal prose using LlamaIndex SentenceSplitter.
5. Preserves paper and chunk metadata.
6. Assigns global and per-paper chunk numbers.
7. Saves all chunks to JSONL.
8. Saves an Excel chunk inventory.
9. Saves a paper-level chunk summary.
10. Runs quality and terminology checks.

Important:
- Tables are not passed through SentenceSplitter.
- A table remains complete even when it is larger than chunk_size.
- Normal prose still uses overlap to preserve context.
"""

# Step 1: Define project paths

project_dir = Path.home() / "hardcarbon_project"

markdown_dir = project_dir / "papers_markdown"
outputs_dir = project_dir / "outputs"
chunks_dir = project_dir / "processed_chunks"

outputs_dir.mkdir(
    parents=True,
    exist_ok=True
)

chunks_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Step 2: Confirm the Markdown folder exists

if not markdown_dir.exists():
    raise FileNotFoundError(
        f"Markdown folder was not found:\n{markdown_dir}"
    )

markdown_files = sorted(
    markdown_dir.glob("*.md")
)

print("Markdown files found:", len(markdown_files))

if len(markdown_files) == 0:
    raise FileNotFoundError(
        "No Markdown files were found."
    )

# Step 3: Configure prose chunking

chunk_size = 1000
chunk_overlap = 150

splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    include_metadata=True,
    include_prev_next_rel=True
)

print("Prose chunk size:", chunk_size)
print("Prose chunk overlap:", chunk_overlap)


# Step 4: Define table-detection helper functions

def is_markdown_table_row(line):
    """
    Return True when a line appears to be part of a Markdown table.

    Examples:
        | Temperature | Carbon |
        |---|---|
        | 450 | 22.6 |
    """

    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith("```"):
        return False

    return stripped.count("|") >= 2


def is_table_caption(line):
    """
    Detect a likely table title or caption immediately before a table.

    Examples:
        Table 1
        Table 2. Ultimate analysis of the chars
        TABLE 3:
    """

    stripped = line.strip()

    return bool(
        re.match(
            r"^(table|tab\.)\s*\d*[\s.:–—-]*",
            stripped,
            flags=re.IGNORECASE
        )
    )


def split_markdown_into_blocks(markdown_text):
    """
    Split Markdown into ordered prose and table blocks.

    Output example:
        [
            {
                "block_type": "prose",
                "text": "Introduction..."
            },
            {
                "block_type": "table",
                "text": "Table 1...\\n| A | B |..."
            }
        ]

    Consecutive table rows remain in one block.

    When a line immediately before a table looks like a table caption,
    the caption is moved into the table block.
    """

    lines = markdown_text.splitlines()

    blocks = []
    prose_buffer = []
    table_buffer = []

    def flush_prose():
        nonlocal prose_buffer

        prose_text = "\n".join(prose_buffer).strip()

        if prose_text:
            blocks.append(
                {
                    "block_type": "prose",
                    "text": prose_text
                }
            )

        prose_buffer = []

    def flush_table():
        nonlocal table_buffer

        table_text = "\n".join(table_buffer).strip()

        if table_text:
            blocks.append(
                {
                    "block_type": "table",
                    "text": table_text
                }
            )

        table_buffer = []

    for line in lines:

        if is_markdown_table_row(line):

            # A new table is beginning.
            if not table_buffer:

                # Attach an immediately preceding table caption.
                if prose_buffer:

                    last_nonempty_index = None

                    for index in range(
                        len(prose_buffer) - 1,
                        -1,
                        -1
                    ):
                        if prose_buffer[index].strip():
                            last_nonempty_index = index
                            break

                    if last_nonempty_index is not None:

                        possible_caption = prose_buffer[
                            last_nonempty_index
                        ]

                        if is_table_caption(possible_caption):

                            caption = prose_buffer.pop(
                                last_nonempty_index
                            )

                            # Remove trailing empty lines after caption.
                            while (
                                prose_buffer
                                and not prose_buffer[-1].strip()
                            ):
                                prose_buffer.pop()

                            flush_prose()

                            table_buffer.append(caption)

                        else:
                            flush_prose()

                    else:
                        flush_prose()

                else:
                    flush_prose()

            table_buffer.append(line)

        else:

            # End the current table when a non-table line appears.
            if table_buffer:
                flush_table()

            prose_buffer.append(line)

    flush_table()
    flush_prose()

    return blocks

# Step 5: Process papers and create chunks

document_records = []
chunk_records = []

global_chunk_number = 0

for paper_number, markdown_path in enumerate(
    markdown_files,
    start=1
):

    markdown_text = markdown_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    paper_id = markdown_path.stem

    blocks = split_markdown_into_blocks(
        markdown_text
    )

    paper_chunk_number = 0
    prose_block_count = 0
    table_block_count = 0

    for block_number, block in enumerate(
        blocks,
        start=1
    ):

        block_type = block["block_type"]
        block_text = block["text"].strip()

        if not block_text:
            continue

        # Prose blocks: use SentenceSplitter
        if block_type == "prose":

            prose_block_count += 1

            document = Document(
                text=block_text,
                metadata={
                    "paper_id": paper_id,
                    "source_filename": markdown_path.name,
                    "source_path": str(markdown_path),
                    "document_number": paper_number,
                    "document_type":
                        "research_paper_markdown",
                    "block_number": block_number,
                    "block_type": "prose"
                }
            )

            prose_nodes = splitter.get_nodes_from_documents(
                [document]
            )

            for block_chunk_number, node in enumerate(
                prose_nodes,
                start=1
            ):

                chunk_text = node.text.strip()

                if not chunk_text:
                    continue

                paper_chunk_number += 1
                global_chunk_number += 1

                chunk_records.append(
                    {
                        "global_chunk_number":
                            global_chunk_number,

                        "paper_chunk_number":
                            paper_chunk_number,

                        "node_id":
                            node.node_id,

                        "paper_id":
                            paper_id,

                        "source_filename":
                            markdown_path.name,

                        "document_number":
                            paper_number,

                        "block_number":
                            block_number,

                        "block_chunk_number":
                            block_chunk_number,

                        "chunk_type":
                            "prose",

                        "chunk_text":
                            chunk_text,

                        "character_count":
                            len(chunk_text),

                        "word_count":
                            len(chunk_text.split())
                    }
                )

        # Table blocks: keep the whole table as one chunk

        elif block_type == "table":

            table_block_count += 1
            paper_chunk_number += 1
            global_chunk_number += 1

            table_node_id = (
                f"table_{paper_id}_"
                f"{block_number}_"
                f"{uuid.uuid4().hex[:12]}"
            )

            chunk_records.append(
                {
                    "global_chunk_number":
                        global_chunk_number,

                    "paper_chunk_number":
                        paper_chunk_number,

                    "node_id":
                        table_node_id,

                    "paper_id":
                        paper_id,

                    "source_filename":
                        markdown_path.name,

                    "document_number":
                        paper_number,

                    "block_number":
                        block_number,

                    "block_chunk_number":
                        1,

                    "chunk_type":
                        "table",

                    "chunk_text":
                        block_text,

                    "character_count":
                        len(block_text),

                    "word_count":
                        len(block_text.split())
                }
            )

        else:
            raise ValueError(
                f"Unknown block type: {block_type}"
            )

    document_records.append(
        {
            "Document_number":
                paper_number,

            "Paper_ID":
                paper_id,

            "Markdown_filename":
                markdown_path.name,

            "Markdown_characters":
                len(markdown_text),

            "Markdown_words":
                len(markdown_text.split()),

            "Number_of_blocks":
                len(blocks),

            "Number_of_prose_blocks":
                prose_block_count,

            "Number_of_table_blocks":
                table_block_count,

            "Number_of_chunks":
                paper_chunk_number
        }
    )

    print(
        f"Processed paper {paper_number}/{len(markdown_files)}: "
        f"{markdown_path.name} "
        f"({paper_chunk_number} chunks, "
        f"{table_block_count} tables)"
    )


print("\nPapers processed:", len(document_records))
print("Total chunks created:", len(chunk_records))

# Step 6: Save chunks as JSONL

chunks_jsonl_path = (
    chunks_dir
    / "paper_chunks.jsonl"
)

with chunks_jsonl_path.open(
    "w",
    encoding="utf-8"
) as jsonl_file:

    for record in chunk_records:

        jsonl_file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

print("\nChunks saved to:")
print(chunks_jsonl_path)

# Step 7: Create the chunk inventory

chunk_inventory_rows = []

for record in chunk_records:

    preview = record["chunk_text"][:500]

    chunk_inventory_rows.append(
        {
            "Global_chunk_number":
                record["global_chunk_number"],

            "Paper_chunk_number":
                record["paper_chunk_number"],

            "Node_ID":
                record["node_id"],

            "Paper_ID":
                record["paper_id"],

            "Source_filename":
                record["source_filename"],

            "Document_number":
                record["document_number"],

            "Block_number":
                record["block_number"],

            "Block_chunk_number":
                record["block_chunk_number"],

            "Chunk_type":
                record["chunk_type"],

            "Character_count":
                record["character_count"],

            "Word_count":
                record["word_count"],

            "Chunk_preview":
                preview
        }
    )

chunk_inventory_df = pd.DataFrame(
    chunk_inventory_rows
)

chunk_inventory_path = (
    outputs_dir
    / "chunk_inventory.xlsx"
)

chunk_inventory_df.to_excel(
    chunk_inventory_path,
    index=False
)

print("\nChunk inventory saved to:")
print(chunk_inventory_path)

# Step 8: Create the paper-level chunk summary

documents_df = pd.DataFrame(
    document_records
)

paper_chunk_summary_df = (
    chunk_inventory_df.groupby(
        [
            "Paper_ID",
            "Source_filename"
        ],
        as_index=False
    )
    .agg(
        Number_of_chunks=(
            "Global_chunk_number",
            "count"
        ),

        Number_of_prose_chunks=(
            "Chunk_type",
            lambda values: (
                values == "prose"
            ).sum()
        ),

        Number_of_table_chunks=(
            "Chunk_type",
            lambda values: (
                values == "table"
            ).sum()
        ),

        Minimum_chunk_characters=(
            "Character_count",
            "min"
        ),

        Maximum_chunk_characters=(
            "Character_count",
            "max"
        ),

        Average_chunk_characters=(
            "Character_count",
            "mean"
        )
    )
)

paper_chunk_summary_path = (
    outputs_dir
    / "paper_chunk_summary.xlsx"
)

paper_chunk_summary_df.to_excel(
    paper_chunk_summary_path,
    index=False
)

print("\nPaper chunk summary saved to:")
print(paper_chunk_summary_path)

# Step 9: Save a document-level summary

document_summary_path = (
    outputs_dir
    / "document_chunking_summary.xlsx"
)

documents_df.to_excel(
    document_summary_path,
    index=False
)

print("\nDocument summary saved to:")
print(document_summary_path)

# Step 10: Run chunk quality checks

empty_chunks = (
    chunk_inventory_df["Character_count"]
    .eq(0)
    .sum()
)

very_short_chunks = (
    chunk_inventory_df["Character_count"]
    .lt(100)
    .sum()
)

table_chunks = (
    chunk_inventory_df["Chunk_type"]
    .eq("table")
    .sum()
)

prose_chunks = (
    chunk_inventory_df["Chunk_type"]
    .eq("prose")
    .sum()
)

papers_with_no_chunks = (
    set(documents_df["Paper_ID"])
    - set(chunk_inventory_df["Paper_ID"])
)

duplicate_global_numbers = (
    chunk_inventory_df[
        "Global_chunk_number"
    ]
    .duplicated()
    .sum()
)

duplicate_paper_chunk_numbers = (
    chunk_inventory_df.duplicated(
        subset=[
            "Paper_ID",
            "Paper_chunk_number"
        ]
    )
    .sum()
)

print("\nChunk quality checks:")

print("Empty chunks:", empty_chunks)

print(
    "Very short chunks below 100 characters:",
    very_short_chunks
)

print("Prose chunks:", prose_chunks)
print("Table chunks:", table_chunks)

print(
    "Papers with no chunks:",
    len(papers_with_no_chunks)
)

print(
    "Duplicate global chunk numbers:",
    duplicate_global_numbers
)

print(
    "Duplicate per-paper chunk numbers:",
    duplicate_paper_chunk_numbers
)

if papers_with_no_chunks:

    print("\nPapers with no chunks:")

    for paper_id in sorted(
        papers_with_no_chunks
    ):
        print("-", paper_id)

# Step 11: Check whether important terms survived

test_terms = [
    "carbonization temperature",
    "pyrolysis temperature",
    "heat treatment temperature",
    "elemental analysis",
    "ultimate analysis",
    "carbon content",
    "hydrogen content",
    "nitrogen content",
    "oxygen content",
    "hard carbon",
    "soft carbon",
    "nanoporous carbon",
    "porous carbon",
    "feedstock",
    "biomass"
]

all_chunk_text = "\n".join(
    record["chunk_text"].lower()
    for record in chunk_records
)

print("\nScientific term check:")

for term in test_terms:

    found = term.lower() in all_chunk_text

    print(
        f"{term}:",
        "Found" if found else "Not found"
    )

# Step 12: Print sample prose chunks

sample_prose_records = [
    record
    for record in chunk_records
    if record["chunk_type"] == "prose"
][:2]

print("\n" + "=" * 70)
print("SAMPLE PROSE CHUNKS")
print("=" * 70)

for record in sample_prose_records:

    print("\nPaper:", record["source_filename"])
    print("Paper chunk:", record["paper_chunk_number"])
    print("Characters:", record["character_count"])
    print("-" * 70)
    print(record["chunk_text"][:1500])


# Step 13: Print sample table chunks

sample_table_records = [
    record
    for record in chunk_records
    if record["chunk_type"] == "table"
][:2]

print("\n" + "=" * 70)
print("SAMPLE TABLE CHUNKS")
print("=" * 70)

if not sample_table_records:
    print("No Markdown tables were detected.")

for record in sample_table_records:

    print("\nPaper:", record["source_filename"])
    print("Paper chunk:", record["paper_chunk_number"])
    print("Characters:", record["character_count"])
    print("-" * 70)
    print(record["chunk_text"][:2500])

# Step 14: Final completion check

documents_with_chunks = (
    chunk_inventory_df["Paper_ID"]
    .nunique()
)

expected_paper_count = len(markdown_files)

step7_complete = (
    len(document_records) == expected_paper_count
    and documents_with_chunks == expected_paper_count
    and len(chunk_records) > 0
    and empty_chunks == 0
    and len(papers_with_no_chunks) == 0
    and duplicate_global_numbers == 0
    and duplicate_paper_chunk_numbers == 0
)

print("\n" + "=" * 70)
print("STEP 7 FINAL SUMMARY")
print("=" * 70)

print(
    "Markdown papers:",
    len(markdown_files)
)

print(
    "Papers processed:",
    len(document_records)
)

print(
    "Papers represented in chunks:",
    documents_with_chunks
)

print(
    "Total chunks:",
    len(chunk_records)
)

print(
    "Prose chunks:",
    prose_chunks
)

print(
    "Table chunks:",
    table_chunks
)

print(
    "Empty chunks:",
    empty_chunks
)

print(
    "Very short chunks:",
    very_short_chunks
)

if step7_complete:

    print("\nSTEP 7 COMPLETE")

    print(
        "All Markdown papers were converted into "
        "traceable, table-aware chunks."
    )

else:

    print("\nSTEP 7 NEEDS REVIEW")

    print(
        "Check missing papers, empty chunks, duplicate "
        "chunk numbers, or the generated reports."
    )


print("\nGenerated files:")
print("-", chunks_jsonl_path)
print("-", chunk_inventory_path)
print("-", paper_chunk_summary_path)
print("-", document_summary_path)