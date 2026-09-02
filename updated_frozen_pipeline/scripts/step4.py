from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Summary: 
"""
1. Load the cleaned manual database.
2. Select a feedstock group and elemental property.
3. Remove records missing temperature or composition values.
4. Fit a four-parameter decreasing logistic curve.
5. Calculate fitted parameters and the coefficient of determination, R².
6. Plot the experimental points together with the fitted curve.
7. Save the figures and fitted parameters to Excel.
8. Compare the reproduced results with the reference plots.

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

woody_h_df = df[df["Group"].astype(str).str.strip().str.lower() == "woody"].copy()
woody_h_df = woody_h_df.dropna(subset = [temp_col, h_wt_col])

print("Woody H points:", len(woody_h_df))

x_data = woody_h_df[temp_col].to_numpy(dtype = float)
y_data = woody_h_df[h_wt_col].to_numpy(dtype = float)

print("Temperature range:", x_data.min(), "to", x_data.max())
print("Hydrogen range:", y_data.min(), "to", y_data.max())

def decreasing_logistics(x, bottom ,top , midpoint, scale):
    exponent = (x - midpoint) / scale
    return bottom + (top - bottom) / (1 + np.exp(exponent))

initial_guesses = [
    y_data.min(),
    y_data.max(),
    np.median(x_data),
    100
    ]

parameters, covariance = curve_fit(
    decreasing_logistics,
    x_data,
    y_data,
    p0 = initial_guesses,
    maxfev = 50000
    )

bottom, top, midpoint, scale = parameters

print("Bottom:", bottom)
print("Top:", top)
print("Midpoint:", midpoint)
print("Scale:", scale)

y_predicted = decreasing_logistics(x_data, *parameters)

ss_residual = np.sum((y_data - y_predicted)**2)
ss_total = np.sum((y_data - np.mean(y_data))**2)

r_squared = 1 - (ss_residual / ss_total)

print("Calculated R^2:", r_squared)

x_curve = np.linspace(x_data.min(), x_data.max(), 400)
y_curve = decreasing_logistics(x_curve, *parameters)

plt.figure(figsize=(6,4.5))
plt.scatter(
    x_data,
    y_data,
    s=18,
    alpha=0.7,
    label = "Experimental Data"
)

plt.plot(
    x_curve,
    y_curve,
    linewidth = 2,
    label = "Logistic Fit"
    )

plt.xlabel("Pyrolysis temperature / HTT (°C)")
plt.ylabel("H_char(wt%)")
plt.title("Woody feedstock: hydrogen vs temprature")

plt.text(
    0.68,
    0.90,
    f"$R^2$ = {r_squared: .4f}",
    transform = plt.gca().transAxes,
    )

plt.grid(True, alpha=0.3)
plt.legend()

figure_path = figures_dir / "Woody_H_Temperature.png"
plt.savefig(
    figure_path, 
    dpi = 300,
    bbox_inches = "tight"
    )
plt.show()
print("Figure saved to:", figure_path)
results = pd.DataFrame(
    [
     {"Group": "Woody",
      "Property": "H_char(wt%)",
      "Number_of_points": len(x_data),
      "Fit_type": "Four-parameter decreasing logistic",
      "Bottom": bottom,
      "Top": top,
      "Midpoint_C": midpoint,
      "Scale": scale,
      "R_squared": r_squared
      }])

results_path = outputs_dir / "woody_H_curve_fit_results.xlsx"

results.to_excel(
    results_path,
    index = False
    )

print("Fit results saved to:", results_path)