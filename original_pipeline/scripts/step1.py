from pathlib import Path #To find paths easily
import pandas as pd #To work with tables
# Summary: 
"""
1. Define the project, data, output, and figure directories.
2. Load the corrected feedstock.xlsx database.
3. Remove accidental spaces from column names.
4. Convert temperature and elemental-composition columns to numeric data.
5. inspect missing values, feedstock groups, article numbers, and DOI values.
6. Save any cleaned or summary outputs needed by later steps.

"""


# Step 1: Locate the project

project_dir = Path.home() / "hardcarbon_project" #The main working file: /home/anderson/hardcarbon_project
data_dir = project_dir / "data" #The data is at: /home/anderson/hardcarbon_project/data
outputs_dir = project_dir / "outputs" #The outputs will be at: /home/anderson/hardcarbon_project/outputs

outputs_dir.mkdir(exist_ok=True) #Create the output folder if it does not exist already at said location

excel_path = data_dir / "feedstock.xlsx"

# Step 2: Read the Excel file

df = pd.read_excel(excel_path) #It loads the excel file into a pandas table called df (DataFrame)

# Clean column names: remove accidental spaces
df.columns = [str(col).strip() for col in df.columns]

# Step 3: Inspect the data

print("Dataset loaded successfully.")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
for col in df.columns:
    print("-", col)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())

# If there is a Group column, summarize it
if "Group" in df.columns:
    print("\nFeedstock group counts:")
    print(df["Group"].value_counts(dropna=False))

# Save a small summary file
summary_path = outputs_dir / "feedstock_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Dataset summary\n")
    f.write("================\n\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n\n")

    f.write("Column names:\n")
    for col in df.columns:
        f.write(f"- {col}\n")

    f.write("\nMissing values per column:\n")
    f.write(df.isna().sum().to_string())

    if "Group" in df.columns:
        f.write("\n\nFeedstock group counts:\n")
        f.write(df["Group"].value_counts(dropna=False).to_string())

print("\nSummary saved to:")
print(summary_path)