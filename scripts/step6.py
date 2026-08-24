from pathlib import Path
import re

import pandas as pd
# Summary: 
"""
1. Load the corrected feedstock.xlsx database.
2. Normalize DOI values into a consistent format.
3. Count unique article numbers and unique DOI values.
4. Detect DOI candidates near the beginning of each Markdown paper.
5. Match expected database DOIs to the available documents.
6. Support confirmed DOI aliases, such as preprint and final-publication DOIs.
7. Report unmatched DOI values and papers requiring manual review.
8. Save paper-level and DOI-level verification reports.

"""

# STEP 6: VERIFY DATABASE REFERENCES AGAINST DOCUMENTS

# 1. Define project paths

project_dir = Path.home() / "hardcarbon_project"

excel_path = project_dir / "data" / "feedstock.xlsx"
markdown_dir = project_dir / "papers_markdown"
outputs_dir = project_dir / "outputs"

outputs_dir.mkdir(parents=True, exist_ok=True)

# 2. Check that required inputs exist

if not excel_path.exists():
    raise FileNotFoundError(
        f"Excel database was not found:\n{excel_path}"
    )

if not markdown_dir.exists():
    raise FileNotFoundError(
        f"Markdown folder was not found:\n{markdown_dir}"
    )

# 3. DOI cleaning function

def normalize_doi(value):
    """
    Convert a DOI into a consistent lowercase format.

    Examples
    --------
    https://doi.org/10.1234/example
    doi:10.1234/example

    Both become:
    10.1234/example
    """

    if pd.isna(value):
        return None

    doi = str(value).strip().lower()

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:"
    ]

    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    doi = doi.strip()

    # Remove punctuation or Markdown symbols attached to the end
    doi = doi.rstrip(".,;:)]}>\"'")

    if not doi:
        return None

    if not doi.startswith("10."):
        return None

    return doi

# 4. Optional DOI aliases

doi_aliases = {
    
    "10.5194/bg-11-6613-2014":
         "10.5194/bgd-11-11727-2014"
}


def apply_doi_alias(doi):
    """
    Convert an alternative DOI into the DOI used by
    the Excel benchmark.
    """

    if doi is None:
        return None

    return doi_aliases.get(doi, doi)

# 5. DOI extraction pattern

doi_pattern = re.compile(
    r"10\.\d{4,9}/[-._;()/:a-z0-9]+",
    re.IGNORECASE
)


def extract_dois(text):
    """
    Extract, normalize and deduplicate DOI candidates
    from a block of text.
    """

    detected_dois = set()

    for match in doi_pattern.findall(text):

        doi = normalize_doi(match)

        if doi is None:
            continue

        doi = apply_doi_alias(doi)

        detected_dois.add(doi)

    return sorted(detected_dois)

# 6. Load corrected Excel database

df = pd.read_excel(excel_path)

df.columns = [
    str(column).strip()
    for column in df.columns
]

required_columns = {
    "DOI",
    "Article number"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise KeyError(
        "Required columns are missing from feedstock.xlsx: "
        + ", ".join(sorted(missing_columns))
    )

print("=" * 60)
print("DATABASE INFORMATION")
print("=" * 60)

print("Database rows:", len(df))

print("\nDatabase columns:")
print(df.columns.tolist())

# 7. Normalize the database DOI column

doi_col = "DOI"
article_col = "Article number"

df["DOI_normalized"] = df[doi_col].apply(
    normalize_doi
)

expected_dois = sorted(
    df["DOI_normalized"]
    .dropna()
    .unique()
)

expected_doi_set = set(expected_dois)

unique_article_count = (
    df[article_col]
    .dropna()
    .nunique()
)

rows_without_valid_doi = (
    df["DOI_normalized"]
    .isna()
    .sum()
)

print("\nUnique article numbers:")
print(unique_article_count)

print("\nUnique valid DOI values:")
print(len(expected_dois))

print("\nRows without a valid DOI:")
print(rows_without_valid_doi)

# 8. Check article-number-to-DOI relationships

article_doi_df = (
    df[
        [
            article_col,
            "DOI_normalized"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        by=article_col
    )
)

article_doi_report_path = (
    outputs_dir
    / "article_number_to_doi_report.xlsx"
)

article_doi_df.to_excel(
    article_doi_report_path,
    index=False
)


# Count how many article numbers use each DOI

doi_article_counts = (
    article_doi_df
    .groupby("DOI_normalized")[article_col]
    .nunique()
)

shared_dois = (
    doi_article_counts[
        doi_article_counts > 1
    ]
)

if shared_dois.empty:

    print(
        "\nNo DOI is assigned to multiple "
        "article numbers."
    )

else:

    print(
        "\nDOIs assigned to multiple "
        "article numbers:"
    )

    print(shared_dois)

    shared_doi_rows = article_doi_df[
        article_doi_df[
            "DOI_normalized"
        ].isin(shared_dois.index)
    ]

    shared_doi_report_path = (
        outputs_dir
        / "shared_doi_article_numbers.xlsx"
    )

    shared_doi_rows.to_excel(
        shared_doi_report_path,
        index=False
    )

    print(
        "\nShared DOI report saved to:"
    )
    print(shared_doi_report_path)

# 9. Find converted Markdown papers

markdown_files = sorted(
    markdown_dir.glob("*.md")
)

print("\nMarkdown papers found:")
print(len(markdown_files))

if len(markdown_files) == 0:
    raise FileNotFoundError(
        "No Markdown files were found in "
        f"{markdown_dir}"
    )
    
# 10. Inspect DOI candidates in each paper

paper_records = []

for markdown_path in markdown_files:

    text = markdown_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Search approximately the beginning of the paper.
    # This reduces DOI matches from the references section.
    front_text = text[:20000]

    front_dois = extract_dois(
        front_text
    )

    full_document_dois = extract_dois(
        text
    )

    paper_records.append(
        {
            "Markdown_filename":
                markdown_path.name,

            "Front_DOIs":
                "; ".join(front_dois),

            "Front_DOI_count":
                len(front_dois),

            "Full_document_DOIs":
                "; ".join(full_document_dois),

            "Full_document_DOI_count":
                len(full_document_dois),

            "Markdown_characters":
                len(text)
        }
    )


papers_df = pd.DataFrame(
    paper_records
)

paper_doi_candidates_path = (
    outputs_dir
    / "paper_doi_candidates.xlsx"
)

papers_df.to_excel(
    paper_doi_candidates_path,
    index=False
)

print("\nDOIs detected near the beginning:")

print(
    papers_df["Front_DOI_count"]
    .value_counts()
    .sort_index()
)

print("\nPaper DOI candidate report saved to:")
print(paper_doi_candidates_path)

# 11. Match expected Excel DOIs to papers

doi_to_files = {}
file_to_expected_dois = {}

for markdown_path in markdown_files:

    text = markdown_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    front_text = text[:20000]

    front_dois = extract_dois(
        front_text
    )

    # Keep only DOI values expected by feedstock.xlsx
    expected_matches = sorted(
        {
            doi
            for doi in front_dois
            if doi in expected_doi_set
        }
    )

    file_to_expected_dois[
        markdown_path.name
    ] = expected_matches

    for doi in expected_matches:

        doi_to_files.setdefault(
            doi,
            []
        ).append(
            markdown_path.name
        )


matched_dois = set(
    doi_to_files.keys()
)

not_detected_dois = (
    expected_doi_set
    - matched_dois
)


print("\n" + "=" * 60)
print("STEP 6 DOI VERIFICATION")
print("=" * 60)

print(
    "Expected DOI values:",
    len(expected_doi_set)
)

print(
    "DOIs matched near paper beginning:",
    len(matched_dois)
)

print(
    "Expected DOIs not detected:",
    len(not_detected_dois)
)


if not_detected_dois:

    print("\nExpected DOI values not detected:")

    for doi in sorted(not_detected_dois):
        print("-", doi)

else:

    print(
        "\nAll expected DOI values were detected."
    )

# 12. Create DOI verification report

verification_rows = []

for doi in expected_dois:

    matching_files = doi_to_files.get(
        doi,
        []
    )

    if len(matching_files) == 1:

        status = "Confirmed"

    elif len(matching_files) > 1:

        status = (
            "Multiple matching files — review"
        )

    else:

        status = (
            "Not detected — review"
        )

    verification_rows.append(
        {
            "DOI":
                doi,

            "Status":
                status,

            "Number_of_matching_files":
                len(matching_files),

            "Matching_files":
                "; ".join(matching_files),

            "DOI_URL":
                f"https://doi.org/{doi}"
        }
    )


verification_df = pd.DataFrame(
    verification_rows
)

verification_path = (
    outputs_dir
    / "reference_verification_report.xlsx"
)

verification_df.to_excel(
    verification_path,
    index=False
)

print("\nVerification report saved to:")
print(verification_path)

# 13. Create paper-to-DOI matching report

paper_match_rows = []

for markdown_path in markdown_files:

    filename = markdown_path.name

    expected_matches = (
        file_to_expected_dois.get(
            filename,
            []
        )
    )

    paper_record = papers_df[
        papers_df["Markdown_filename"]
        == filename
    ].iloc[0]

    if len(expected_matches) == 1:

        status = "Matched"

    elif len(expected_matches) > 1:

        status = (
            "Multiple expected DOIs detected — review"
        )

    else:

        status = (
            "No expected DOI detected"
        )

    paper_match_rows.append(
        {
            "Markdown_filename":
                filename,

            "Expected_DOIs_detected":
                "; ".join(expected_matches),

            "All_front_DOI_candidates":
                paper_record["Front_DOIs"],

            "Front_DOI_count":
                paper_record["Front_DOI_count"],

            "Status":
                status
        }
    )


paper_match_df = pd.DataFrame(
    paper_match_rows
)

paper_match_path = (
    outputs_dir
    / "paper_to_doi_matching_report.xlsx"
)

paper_match_df.to_excel(
    paper_match_path,
    index=False
)

print("\nPaper matching report saved to:")
print(paper_match_path)

# 14. Print unmatched paper information

unmatched_papers = paper_match_df[
    paper_match_df["Status"]
    == "No expected DOI detected"
]

if not unmatched_papers.empty:

    print("\nUnmatched paper or papers:")

    print(
        unmatched_papers[
            [
                "Markdown_filename",
                "All_front_DOI_candidates",
                "Front_DOI_count"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nEvery Markdown paper has an "
        "expected DOI match."
    )
    
# 15. Save unmatched DOI values

missing_doi_path = (
    outputs_dir
    / "doi_values_not_detected.txt"
)

missing_doi_path.write_text(
    "\n".join(
        sorted(not_detected_dois)
    ),
    encoding="utf-8"
)

print("\nNot-detected DOI list saved to:")
print(missing_doi_path)

# 16. Calculate final counts

confirmed_count = (
    verification_df["Status"]
    .eq("Confirmed")
    .sum()
)

doi_review_count = (
    verification_df["Status"]
    .ne("Confirmed")
    .sum()
)

matched_paper_count = (
    paper_match_df["Status"]
    .eq("Matched")
    .sum()
)

paper_review_count = (
    paper_match_df["Status"]
    .ne("Matched")
    .sum()
)

papers_without_match = (
    paper_match_df["Status"]
    .eq("No expected DOI detected")
    .sum()
)

# 17. Final Step 6 summary

print("\n" + "=" * 60)
print("STEP 6 FINAL SUMMARY")
print("=" * 60)

print("Database rows:", len(df))
print("Unique article numbers:", unique_article_count)
print("Unique DOI values:", len(expected_doi_set))
print("Markdown papers:", len(markdown_files))

print(
    "Confirmed DOI matches:",
    confirmed_count
)

print(
    "DOI records needing review:",
    doi_review_count
)

print(
    "Matched papers:",
    matched_paper_count
)

print(
    "Papers needing review:",
    paper_review_count
)

print(
    "Papers without an expected DOI match:",
    papers_without_match
)

# 18. Determine completion status

step6_complete = (
    len(markdown_files) == len(expected_doi_set)
    and confirmed_count == len(expected_doi_set)
    and paper_review_count == 0
)

one_identity_check_needed = (
    len(not_detected_dois) == 1
    and papers_without_match == 1
)


if step6_complete:

    print("\nSTEP 6 COMPLETE")

    print(
        "Every expected DOI was matched to "
        "exactly one Markdown paper."
    )

elif one_identity_check_needed:

    print(
        "\nAUTOMATED STEP 6 FINISHED — "
        "ONE MANUAL IDENTITY CHECK REQUIRED"
    )

    print(
        "Compare the unmatched DOI with the "
        "title and DOI candidates of the "
        "unmatched paper."
    )

    print(
        "If it is a confirmed alternative or "
        "published-version DOI, add it to "
        "doi_aliases and rerun this script."
    )

else:

    print("\nSTEP 6 NEEDS REVIEW")

    print(
        "Open the generated reports and review "
        "all unmatched or multiple-match entries."
    )


print("\nGenerated files:")

print("-", article_doi_report_path)
print("-", paper_doi_candidates_path)
print("-", verification_path)
print("-", paper_match_path)
print("-", missing_doi_path)