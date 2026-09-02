from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Summary: 
"""
1. Load or calculate the H/C and O/C atomic ratios.
2. Separate records by feedstock group where required.
3. Create Van Krevelen scatter plots.
4. Apply consistent labels, legends, axis limits, and figure formatting.
5. Save high-resolution figures for comparison with the manually
   reproduced reference plots.
6. Record the number of valid points used in each plot.

"""

# Step 1: Project folders

project_dir = Path.home() / "hardcarbon_project"
data_dir = project_dir / "data"
figures_dir = project_dir / "figures"
outputs_dir = project_dir / "outputs"

figures_dir.mkdir(exist_ok = True)
outputs_dir.mkdir(exist_ok = True)

excel_path = data_dir / "feedstock.xlsx"

# Step 2: Load data

df = pd.read_excel(excel_path)
df.columns = [str(col).strip() for col in df.columns]

temp_col = "T (°C)"
h_wt_col = "H_char(wt%)"
o_wt_col = "O_char(wt%)"
    
for col in [temp_col, h_wt_col, o_wt_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Rows: ", len(df))
print(df[["Group", temp_col, h_wt_col, o_wt_col]].head())
print(df["Group"].value_counts(dropna = False))
      
# Step 3: Define the plotting function

def plot_TvsProp(groupName, propertyCol, yLabel, outputName):
    group_df = df[df["Group"].astype(str).str.strip().str.lower() == groupName.lower()].copy()
    group_df = group_df.dropna(subset = [temp_col, propertyCol])
    print(groupName, propertyCol, "rows:", len(group_df))
    plt.figure(figsize=(5,4))
    plt.scatter(
       group_df[temp_col],
       group_df[propertyCol],
        s=12,
        alpha=0.75
    )
    plt.xlabel("Pyrolysis temperature / HTT (°C)")
    plt.ylabel(yLabel)
    plt.title(f"{groupName}: {yLabel} vs temperature")
    plt.grid(True, alpha=0.3)
    output_path = figures_dir / outputName
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
    plt.show()
    print("Woody figure saved to: ")
    print(output_path)

plot_TvsProp("Herbaceous", h_wt_col, "H_char(wt%)", "Herbaceous_H_vs_temprature.png")

plot_TvsProp("Herbaceous", o_wt_col, "O_char(wt%)", "Herbaceous_O_vs_temprature.png")

plot_TvsProp("Woody", h_wt_col, "H_char(wt%)", "Woody_H_vs_temprature.png")

plot_TvsProp("Woody", o_wt_col, "O_char(wt%)", "Woody_O_vs_temprature.png")
