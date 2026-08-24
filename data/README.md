# Data

Raw research papers and the manually curated benchmark dataset are not
distributed with this repository.

## Expected benchmark schema

The validation scripts expect an Excel workbook at:

    data/feedstock.xlsx

with columns equivalent to:

- Feedstock
- Group
- T (°C)
- C_char(wt%)
- H_char(wt%)
- N_char(wt%)
- O_char(wt%)
- DOI
- Article number

The benchmark should be supplied locally by the user.

## Raw literature

Raw PDFs should be placed in the appropriate local input directory described
in the main README. PDFs are intentionally excluded from version control.

This repository contains code, aggregate validation results, and figures,
rather than redistributed source publications.
