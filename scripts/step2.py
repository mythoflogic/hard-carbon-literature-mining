from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Summary: 
"""
1. Load the cleaned feedstock database.
2. Remove rows missing the required C, H, or O measurements.
3. Convert weight percentages into molar quantities.
4. Calculate the atomic H/C and O/C ratios.
5. Check for invalid, missing, or infinite results.
6. Save the calculated ratios for plotting and comparison.

"""

# Step 1: Project folders

project_dir = Path.home() / "hardcarbon_project"
data_dir = project_dir / "data"
figures_dir = project_dir / "figures"
outputs_dir = project_dir / "outputs"

figures_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

excel_path = data_dir / "feedstock.xlsx"

# Step 2: Load data

df = pd.read_excel(excel_path)
df.columns = [str(col).strip() for col in df.columns]

print("Columns found:")
print(df.columns.tolist())

# Step 3: Identify useful columns

# We expect some version of these columns:
# Feedstock, Group, T (°C), H/C, O/C, N/C
# Sometimes the file may use slightly different spacing.

temperature_col = "T (°C)"
h_wt_col = "H_char(wt%)"
o_wt_col = "O_char(wt%)"
c_wt_col = "C_char(wt%)"
df[c_wt_col] = pd.to_numeric(df[c_wt_col], errors="coerce")
df[h_wt_col] = pd.to_numeric(df[h_wt_col], errors="coerce")
df[o_wt_col] = pd.to_numeric(df[o_wt_col], errors="coerce")

df["h_c_col"] =  (df[h_wt_col]/1.008) / (df[c_wt_col]/12.011)
df["o_c_col"] =  (df[o_wt_col]/15.999) / (df[c_wt_col]/12.011)
o_c_col = "o_c_col"
h_c_col = "h_c_col"
feedstock_col = "Feedstock"
required_cols = [temperature_col, h_c_col, o_c_col]

for col in df.columns:
    print(repr(col))

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")
print("Total rows before cleaning: ", len(df))
print("Non-numeric O values: ")
print(df[df[o_c_col].isna() == False][o_c_col].head())

# Convert to numeric in case Excel stored numbers as text
raw_df = df.copy()
df[temperature_col] = pd.to_numeric(df[temperature_col], errors="coerce")
df[h_c_col] = pd.to_numeric(df[h_c_col], errors="coerce")
df[o_c_col] = pd.to_numeric(df[o_c_col], errors="coerce")

bad_rows = df[
    df[temperature_col].isna() |
       df[h_c_col].isna() |
       df[o_c_col].isna()
]

print("\nMissing count inside excluded rows: ")
print(bad_rows[[temperature_col, h_c_col, o_c_col]].isna().sum())
print("\nExcluded rows: ")
print(bad_rows[[feedstock_col, "Group", temperature_col, h_c_col, o_c_col]])
print("Total rows: ", len(df))

print("Rows excluded: ", len(bad_rows))
                           
# Remove rows where plotting values are missing
plot_df = df.dropna(subset=[temperature_col, h_c_col, o_c_col]).copy()
print("Rows plotted: ", len(plot_df))
print("Rows available for Van Krevelen plot:", len(plot_df))

# Save cleaned data used for the plot
clean_path = outputs_dir / "van_krevelen_cleaned_data.xlsx"
plot_df.to_excel(clean_path, index=False)

print("Cleaned plotting data saved to:")
print(clean_path)

#To confirm how many entries are plotted:
print("Rows in full Excel:", len(df))
print("Rows used in PLot:", len(plot_df))
if "Group" in plot_df.columns:print(plot_df["Group"].value_counts(dropna=False))
# Step 4: Plot all data

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    plot_df[o_c_col],
    plot_df[h_c_col],
    c=plot_df[temperature_col],
    cmap="viridis",
    s=45,
    edgecolor="black",
    linewidth=0.4,
    alpha=0.9
)

plt.xlabel("Atomic O/C ratio")
plt.ylabel("Atomic H/C ratio")
plt.title("Van Krevelen Diagram - All Feedstocks")
plt.grid(True, alpha=0.3)

colorbar = plt.colorbar(scatter)
colorbar.set_label("Temperature (°C)")

output_path = figures_dir / "van_krevelen_all_feedstocks.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved to:")
print(output_path)

#Plot Herbaceous entries (273 data points)
herbaceous_df = plot_df[plot_df["Group"] == "Herbaceous"].copy()
print("Herbaceous rows:", len(herbaceous_df))
plt.figure(figsize=(5,4))
scatter = plt.scatter(
    herbaceous_df[o_c_col],
    herbaceous_df[h_c_col],
    c=herbaceous_df[temperature_col],
    cmap="viridis",
    s=12,
    alpha=0.75
)

plt.xlabel("Atomic O/C ratio")
plt.ylabel("Atomic H/C ratio")
plt.title("Van Krevelen Diagram - Herbaceous")
plt.grid(True, alpha=0.3)

colorbar = plt.colorbar(scatter)
colorbar.set_label("Temperature (°C)")

plt.text(
    0.95,
    0.95,
    f"{len(herbaceous_df)} data entries",
    transform = plt.gca().transAxes,
    ha = "right",
    va = "top",
    fontsize = 8
    )
output_path = figures_dir / "van_krevelen_herbaceous_273.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Herbaceous figure saved to: ")
print(output_path)

#Plot woody/Woody  entries (187 data points)
woody_df = plot_df[plot_df["Group"].astype(str).str.strip().str.lower() == "woody" ].copy()
print("Woody/woody rows:", len(woody_df))
plt.figure(figsize=(5,4))
scatter = plt.scatter(
    woody_df[o_c_col],
    woody_df[h_c_col],
    c=woody_df[temperature_col],
    cmap="viridis",
    s=12,
    alpha=0.75
)

plt.xlabel("Atomic O/C ratio")
plt.ylabel("Atomic H/C ratio")

plt.title("Van Krevelen Diagram - Woody")
plt.grid(True, alpha=0.3)

colorbar = plt.colorbar(scatter)
colorbar.set_label("Temperature (°C)")

plt.text(
    0.95,
    0.95,
    f"{len(woody_df)} data entries",
    transform = plt.gca().transAxes,
    ha = "right",
    va = "top",
    fontsize = 8
    )

output_path = figures_dir / "van_krevelen_woody_187.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Woody figure saved to: ")
print(output_path)
