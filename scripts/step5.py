from pathlib import Path
import pandas as pd
import pymupdf4llm
# Summary: 
"""
1. Locate every PDF in the raw-paper directory.
2. Convert each PDF into Markdown.
3. Save one Markdown file per research paper.
4. Continue processing even when an individual conversion fails.
5. Record PDF size, Markdown character count, conversion status,
   and any error messages.
6. Flag unusually short conversions for manual inspection.
7. Save a complete conversion-quality report to Excel.

"""

# Step 1: Define directories

project_dir = Path.home() / "hardcarbon_project"
papers_dir = project_dir / "papers_raw"
markdown_dir = project_dir / "papers_markdown"
outputs_dir = project_dir / "outputs"

papers_dir.mkdir(exist_ok = True)
markdown_dir.mkdir(exist_ok = True)
outputs_dir.mkdir(exist_ok = True)

excel_path = project_dir / "data" / "feedstock.xlsx"

print("Excel file exists:", excel_path.exists())

df = pd.read_excel(excel_path)

print("Database rows:", len(df))
print("Columns:", df.columns.tolist())

# Step 2: Find the PDFs

pdf_files = sorted(papers_dir.glob("*.pdf")) # Sort in alphabetical order

print("PDF files found:", len(pdf_files))

for number, pdf_path in enumerate(pdf_files, start = 1):
    print(number, pdf_path.name)

def convert_pdf_to_markdown(pdf_path, outputs_dir):
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    output_path = (outputs_dir / f"{pdf_path.stem}.md") #Removes file extension
    output_path.write_text(markdown_text, encoding = "utf-8")
    return output_path

conversion_records = []

for number, pdf_path in enumerate(pdf_files, start = 1):
    print(
        f"\n {number}/{len(pdf_files)}"
        f"Converting: {pdf_path.name}"
        )
    
    try: 
        markdown_path = convert_pdf_to_markdown(pdf_path, markdown_dir)
        
        conversion_records.append(
            {
                "PDF_filename" : pdf_path.name,
                "Markdown_filename" : markdown_path.name,#
                "PDF_size_MB": round(
            pdf_path.stat().st_size / (1024 ** 2),
            3
        ),
        "Markdown_characters": len(
            markdown_path.read_text(encoding="utf-8")
        ),
                "Status" : "Success",
                "Error" : ""
                })
        
        print("Success:", markdown_path.name)
        
    except Exception as ex:
        conversion_records.append( {
            "PDF_filename" : pdf_path.name,
            "Markdown_filename" : markdown_path.name,
            "PDF_size_MB": round(
            pdf_path.stat().st_size / (1024 ** 2),
            3
        ),
        "Markdown_characters": len(
            markdown_path.read_text(encoding="utf-8")
        ),
            "Status" : "Failed",
            "Error" : str(ex)
            })
        
        print("Failed!")
        
print(df["Article number"].dropna().unique())     
conversion_df = pd.DataFrame(conversion_records)

def classify_conversion(row):
    if row["Status"] == "Failed":
        return "Failed"

    if row["Markdown_characters"] < 1000:
        return "Very low text — inspect manually"

    if row["Markdown_characters"] < 5000:
        return "Low text — review"

    return "Likely usable"


conversion_df["Quality_check"] = conversion_df.apply(
    classify_conversion,
    axis=1
)

report_path = (
    outputs_dir / "pdf_conversion_report.xlsx"
)

conversion_df.to_excel(
    report_path,
    index = False
    )

print("\nConversion Summary:")
print(conversion_df["Status"].value_counts())

print("\nReport saved to:")
print(report_path)

print("\nQuality summary:")
print(conversion_df["Quality_check"].value_counts())

print(
    "\nTotal extracted characters:",
    conversion_df["Markdown_characters"].sum()
)


report_path = outputs_dir / "pdf_conversion_report.xlsx"

conversion_df.to_excel(
    report_path,
    index=False
)

papers_dir = project_dir / "papers_raw"

pdf_files = sorted(papers_dir.glob("*.pdf"))

print("PDF files found:", len(pdf_files))

markdown_dir = project_dir / "papers_markdown"

markdown_files = sorted(markdown_dir.glob("*.md"))

print("Markdown files created:", len(markdown_files))

print("\n" + "=" * 50)
print("STEP 5 FINAL CHECK")
print("=" * 50)

print("Corrected Excel exists:", excel_path.exists())
print("Database rows:", len(df))
print("PDF files:", len(pdf_files))
print("Markdown files:", len(markdown_files))

success_count = (
    conversion_df["Status"]
    .eq("Success")
    .sum()
)

failed_count = (
    conversion_df["Status"]
    .eq("Failed")
    .sum()
)

print("Successful conversions:", success_count)
print("Failed conversions:", failed_count)

step5_complete = (
    excel_path.exists()
    and len(df) == 582
    and len(pdf_files) == 60
    and len(markdown_files) == 60
    and failed_count == 0
)

print()

if step5_complete:
    print("STEP 5 COMPLETE")
else:
    print("STEP 5 NEEDS REVIEW")