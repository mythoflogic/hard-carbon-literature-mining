from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Step 1: Define project paths

project_dir = Path.home() / "Scratch"  / "hardcarbon_project"

chunks_path = ( project_dir / "processed_chunks" / "paper_chunks.jsonl")

embeddings_path =  ( project_dir / "processed_embeddings" / "chunk_embeddings.npy" )

metadata_path =  ( project_dir / "processed_embeddings" / "chunk_metadata.jsonl" )

retrieval_dir = project_dir / "processed_retrieval"

outputs_dir = project_dir / "outputs"

outputs_dir.mkdir(
    parents=True,
    exist_ok=True
)

retrieval_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Step 2: Confirm file exists

if not embeddings_path.exists():
    raise FileNotFoundError(
        f"Chunk file was not found:\n {chunks_path}"
    )
    
if not metadata_path.exists():
    raise FileNotFoundError(
        f"Metadata file was not found:\n {metadata_path}"
    )
    
# Step 3: Load embedding matrix

embeddings = np.load(
    embeddings_path
    )

embeddings = np.asarray(
    embeddings,
    dtype="float32"
    )

print("Embedding matrix shape:", embeddings.shape)

# Step 4: Load chunk metadata
    
chunk_metadata = []

with metadata_path.open(
        "r",
        encoding = "utf-8"
        ) as metadata_file:
    for line_number, line in enumerate(
            metadata_file,
            start = 1
            ):
        line = line.strip()
        
        if not line:
            continue
        
        try:
            record = json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError (
                f"Invalid JSON on metadata line {line_number}" 
                f"{ex}"
                ) from ex
            
        chunk_metadata.append(record)
        
print("Metadata records loaded: ", len(chunk_metadata))

if len(chunk_metadata) != embeddings.shape[0]:
    raise ValueError(
        "The number of metadata records does not match the number of embeddings"
        )
    
    
# Step 5: Group embedding positions by paper

paper_to_positions = {}

for  position, record in enumerate(
        chunk_metadata,
        ):
    
        paper_id = record.get(
            "paper_id",
            "unknown"
            )
        
        paper_to_positions.setdefault(
            paper_id,
            []
            ).append(position)

print("Papers represented:", len(paper_to_positions))


# Step 6: Define the scientific retrieval queries ( keywords relevant to research )

retrieval_queries = {
    "Feedstock_and_sample_definition": (
        "feedstock raw material biomass precursor sample labels abbreviations "
        "biochar names sample preparation pyrolysis carbonization"
    ),

    "Pyrolysis_conditions": (
        "pyrolysis temperature carbonization temperature heating rate residence time "
        "holding time nitrogen atmosphere furnace slow pyrolysis fast pyrolysis "
        "sample produced at degrees Celsius"
    ),

    "Ultimate_elemental_analysis": (
        "ultimate analysis elemental analysis biochar char carbon hydrogen nitrogen oxygen "
        "C H N O wt percent dry basis ash oxygen by difference"
    ),

    "Elemental_composition_table": (
        "table elemental composition ultimate analysis carbon content hydrogen content "
        "nitrogen content oxygen content C H N O biochar temperature"
    ),

    "Table_caption_and_headers": (
        "Table caption table headers sample temperature C H N O ash volatile matter "
        "fixed carbon ultimate proximate analysis"
    ),

    "Temperature_dependent_results": (
        "biochar properties at 200 300 400 450 500 600 700 800 degrees Celsius "
        "carbon increased hydrogen decreased oxygen decreased pyrolysis temperature"
    ),

    "Slow_and_fast_pyrolysis": (
        "slow pyrolysis fast pyrolysis char temperatures comparison "
        "batch reactor fluidized bed biochar elemental analysis"
    ),
}

# Step 7: Load the same embedding model used in step 8

model_name = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading the embedding model:")
print(model_name)

model = SentenceTransformer(
    model_name
    )

# Step 8: Encode all retrieved queries

query_names = list(
    retrieval_queries.keys()
    )

query_texts = [
    retrieval_queries[name]
    for name in query_names
    ]

if hasattr(model, "encode_query"):
    query_embeddings = model.encode_query(
        query_texts,
        convert_to_numpy = True,
        normalize_embeddings = True)
    
else:
    query_embeddings = model.encode( # Fallback for older versions
        query_texts,
        convert_to_numpy = True,
        normalize_embeddings = True)
    
query_embeddings = np.asarray(
    query_embeddings,
    dtype = "float32"
    )

print("Query embedding shape: ")
print(query_embeddings.shape)

# Step 9: Retrieve the top chunks from each paper

top_k_per_paper = 10
retrieval_records = []

for paper_id, positions in sorted(paper_to_positions.items()):

    paper_embeddings = embeddings[positions]

    for query_index, query_name in enumerate(query_names):
        query_embedding = query_embeddings[query_index]

        similarity_scores = paper_embeddings @ query_embedding

        adjusted_scores = []
        keyword_hits_by_position = []
        table_bonus_by_position = []

        for local_position, similarity_score in enumerate(
            similarity_scores
        ):
            global_position = positions[local_position]
            metadata = chunk_metadata[global_position]

            chunk_text = str(
                metadata.get("chunk_text", "")
            ).lower()

            target_terms = [
                "biochar",
                "pyrolysis",
                "carbonization",
                "pyrolysis temperature",
                "carbonization temperature",
                "ultimate analysis",
                "elemental analysis",
                "elemental composition",
                "proximate analysis",
                "carbon content",
                "hydrogen content",
                "nitrogen content",
                "oxygen content",
                "oxygen by difference",
                "wt%",
                "wt.%",
                "dry basis",
                "table",
                "feedstock",
                "slow pyrolysis",
                "fast pyrolysis",
            ]

            keyword_hits = sum(
                term in chunk_text
                for term in target_terms
            )

            table_bonus = 0.0

            if "table" in chunk_text:
                table_bonus += 0.08

            pipe_count = chunk_text.count("|")

            if pipe_count > 20:
                table_bonus += 0.20
            elif pipe_count > 8:
                table_bonus += 0.10

            if "ultimate analysis" in chunk_text:
                table_bonus += 0.10

            if "elemental analysis" in chunk_text:
                table_bonus += 0.10

            if any(
                term in chunk_text
                for term in [" c ", " h ", " n ", " o "]
            ):
                table_bonus += 0.04

            if (
                "%" in chunk_text
                or "wt%" in chunk_text
                or "wt.%" in chunk_text
            ):
                table_bonus += 0.05

            temperature_terms = [
                "200 °c",
                "300 °c",
                "400 °c",
                "450 °c",
                "500 °c",
                "600 °c",
                "700 °c",
                "800 °c",
                "200 c",
                "300 c",
                "400 c",
                "450 c",
                "500 c",
                "600 c",
                "700 c",
                "800 c",
            ]

            if any(
                temperature in chunk_text
                for temperature in temperature_terms
            ):
                table_bonus += 0.05

            adjusted_score = (
                float(similarity_score)
                + 0.03 * keyword_hits
                + table_bonus
            )

            adjusted_scores.append(
                adjusted_score
            )

            keyword_hits_by_position.append(
                keyword_hits
            )

            table_bonus_by_position.append(
                table_bonus
            )

        number_to_keep = min(
            top_k_per_paper,
            len(positions),
        )

        ranked_local_positions = np.argsort(
            np.asarray(
                adjusted_scores,
                dtype="float32",
            )
        )[::-1][:number_to_keep]

        selected_positions = set()

        for local_position in ranked_local_positions:
            local_position = int(local_position)

            selected_positions.add(
                local_position
            )

            if local_position > 0:
                selected_positions.add(
                    local_position - 1
                )

            if local_position < len(positions) - 1:
                selected_positions.add(
                    local_position + 1
                )

        selected_positions = sorted(
            selected_positions,
            key=lambda position: adjusted_scores[position],
            reverse=True,
        )

        for rank, local_position in enumerate(
            selected_positions,
            start=1,
        ):
            global_position = positions[local_position]
            metadata = chunk_metadata[global_position]

            score = float(
                similarity_scores[local_position]
            )

            keyword_hits = (
                keyword_hits_by_position[
                    local_position
                ]
            )

            table_bonus = (
                table_bonus_by_position[
                    local_position
                ]
            )

            adjusted_score = float(
                adjusted_scores[local_position]
            )

            retrieval_records.append(
                {
                    "faiss_index_position": global_position,
                    "global_chunk_number": metadata.get(
                        "global_chunk_number"
                    ),
                    "paper_chunk_number": metadata.get(
                        "paper_chunk_number"
                    ),
                    "node_id": metadata.get(
                        "node_id",
                        "",
                    ),
                    "paper_id": paper_id,
                    "source_filename": metadata.get(
                        "source_filename"
                    ),
                    "chunk_text": metadata.get(
                        "chunk_text",
                        "",
                    ),
                    "query_name": query_name,
                    "query_text": retrieval_queries[
                        query_name
                    ],
                    "rank_within_paper": rank,
                    "similarity_score": score,
                    "keyword_hits": keyword_hits,
                    "table_bonus": table_bonus,
                    "adjusted_score": adjusted_score,
                }
            )
                
print("Candidate passages retrieved", len(retrieval_records))

# Step 10: Save the complete results as JSONL

jsonl_output_path = ( retrieval_dir / "per_paper_retrieval_candidates.jsonl")

with jsonl_output_path.open(
        "w",
        encoding = "utf-8"
        ) as output_file:
    for record in retrieval_records:
        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False) + "\n")
        
print("FUll retrieval records saved to:", jsonl_output_path)

# Step 11: Save the human-readable excel report 

excel_rows = []

for record in retrieval_records:
    excel_rows.append(
        { 
            "Paper_ID":record["paper_id"],
            "Source_filename":record["source_filename"],
            "Query_name":record["query_name"],
            "Rank_within_paper":record["rank_within_paper"],
	    "Similarity_score": record["similarity_score"],
	    "Keyword_hits": record["keyword_hits"],
 	    "Table_bonus": record["table_bonus"],
	    "Adjusted_score": record["adjusted_score"],
            "Global_chunk_number":record["global_chunk_number"],
            "Paper_chunk_number":record["paper_chunk_number"],
            "Chunk_preview":record["chunk_text"][:1500],
            })
    
retrieval_df = pd.DataFrame(excel_rows)
retrieval_df = retrieval_df.sort_values(
    by=[
        "Paper_ID",
        "Query_name",
        "Adjusted_score",
    ],
    ascending=[
        True,
        True,
        False,
    ],
).reset_index(drop=True)

excel_output_path = outputs_dir / "per_paper_retrieval_candidates.xlsx"

retrieval_df.to_excel(excel_output_path,index = False)

print("Excel retrieval report saved to: ", excel_output_path)

# Step 12: Create the retrieval summary

paper_summary_df = (retrieval_df.groupby([
    "Paper_ID",
    "Source_filename"],as_index=False)
    .agg(
        Retrieved_rows = ("Paper_chunk_number", "count"),
        Retrieval_categories = ("Query_name", "nunique"),
        Mean_similarity_score = ("Similarity_score", "mean"),
        Maximum_similarity_score = ("Similarity_score", "max")
        )
    )

paper_summary_path = ( outputs_dir / "per_paper_retrieval_summary.xlsx")

paper_summary_df.to_excel(paper_summary_path, index = False)

print("Paper retrieval summary saved to: ", paper_summary_path)

# Step 13: Quality checks

expected_papers = len(paper_to_positions)
expected_categories = len(retrieval_queries)
paper_in_results = retrieval_df["Paper_ID"].nunique()
categories_in_results = retrieval_df["Query_name"].nunique()
empty_candidate_chunks = (retrieval_df["Chunk_preview"].fillna("").str.strip().eq("").sum())
papers_missing_from_results = ( set(paper_to_positions.keys()) - set(retrieval_df["Paper_ID"]))

print("Step 9 final summary: ")

print("input embedding vectors: ", embeddings.shape[0])
print("papers in embeddings: ", expected_papers)
print("retrieval categories: ", expected_categories)
print("Top chunks per paper/category: ", top_k_per_paper)
print("Papers represented in retrieval: ", paper_in_results)
print("Categories represented: ", categories_in_results)
print("Candidate passages retireved: ", len(retrieval_df))
print("Empty retrieved passages: ", empty_candidate_chunks)
print("Papers missing from retreival: ", len(papers_missing_from_results))

step_9_complete = (
    paper_in_results == expected_papers and
    categories_in_results == expected_categories and
    empty_candidate_chunks == 0 and
    len(papers_missing_from_results) == 0)

if step_9_complete:
    print("Step 9 Complete!")
    print("Every paper now has targeted evidence, passages for every extraction category.")
    
else:
    print("Step 9 not Completed!")
    print("Needs review")
    
print("Generated files: ")
print("-", jsonl_output_path)
print("-", excel_output_path)
print("-", paper_summary_path)
