#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

BASE = (
    ROOT
    / "processed_tables"
    / "benchmark_final_table_v1"
)

MATCH_DIR = (
    BASE
    / "match_v1"
)

OUT = (
    ROOT
    / "outputs"
    / "final_report_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


BENCHMARK_PATH = (
    BASE
    / "benchmark_scope.csv"
)

EXTRACTED_PATH = (
    BASE
    / "final_benchmark_eligible.csv"
)

MATCHED_PATH = (
    MATCH_DIR
    / "matched_rows.csv"
)

UNMATCHED_PATH = (
    MATCH_DIR
    / "unmatched_benchmark_diagnosed.csv"
)

PER_DOI_PATH = (
    MATCH_DIR
    / "per_doi_summary.csv"
)


RANGE_DIR = (
    BASE
    / "match_v1_range_compatible"
)


RANGE_MATCH_PATH = (
    RANGE_DIR
    / "range_compatible_matches.csv"
)


POST_RANGE_PATH = (
    MATCH_DIR
    / "post_range_unmatched_partition.csv"
)


POST_RANGE_SUMMARY_PATH = (
    MATCH_DIR
    / "post_range_unmatched_partition_summary.csv"
)


# ============================================================
# Constants
# ============================================================

ELEMENTS = {
    "C": "Carbon",
    "H": "Hydrogen",
    "N": "Nitrogen",
    "O": "Oxygen",
}


MANUAL_COLOR = "#1f77b4"
EXTRACTED_COLOR = "#d62728"

FIT_MANUAL_COLOR = "#0b3d91"
FIT_EXTRACTED_COLOR = "#8b0000"

DPI = 300


# ============================================================
# Load
# ============================================================

manual = pd.read_csv(
    BENCHMARK_PATH,
    low_memory=False,
)

extracted = pd.read_csv(
    EXTRACTED_PATH,
    low_memory=False,
)

matched = pd.read_csv(
    MATCHED_PATH,
    low_memory=False,
)

unmatched = pd.read_csv(
    UNMATCHED_PATH,
    low_memory=False,
)

per_doi = pd.read_csv(
    PER_DOI_PATH,
    low_memory=False,
)


range_matches = pd.read_csv(
    RANGE_MATCH_PATH,
    low_memory=False,
)


post_range = pd.read_csv(
    POST_RANGE_PATH,
    low_memory=False,
)


post_range_summary = pd.read_csv(
    POST_RANGE_SUMMARY_PATH,
    low_memory=False,
)


# ============================================================
# Numeric conversion
# ============================================================

for df in [
    manual,
    extracted,
]:

    for col in [
        "temperature_C",
        "C_value",
        "H_value",
        "N_value",
        "O_value",
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )


# ============================================================
# Final invariants
# ============================================================

if len(manual) != 351:
    raise RuntimeError(
        f"Expected 351 represented benchmark rows, "
        f"found {len(manual)}."
    )


if len(extracted) != 439:
    raise RuntimeError(
        f"Expected 439 benchmark-eligible extracted rows, "
        f"found {len(extracted)}."
    )


if len(matched) != 323:
    raise RuntimeError(
        f"Expected 323 final conservative matches, "
        f"found {len(matched)}."
    )


if len(unmatched) != 28:
    raise RuntimeError(
        f"Expected 28 unmatched benchmark rows, "
        f"found {len(unmatched)}."
    )


if len(range_matches) != 4:
    raise RuntimeError(
        f"Expected 4 validated range-compatible matches, "
        f"found {len(range_matches)}."
    )


if len(post_range) != 24:
    raise RuntimeError(
        f"Expected 24 post-range benchmark rows, "
        f"found {len(post_range)}."
    )


strict_match_ids = set(
    matched[
        "benchmark_row_id"
    ]
    .dropna()
    .astype(str)
)


strict_unmatched_ids = set(
    unmatched[
        "benchmark_row_id"
    ]
    .dropna()
    .astype(str)
)


range_match_ids = set(
    range_matches[
        "benchmark_row_id"
    ]
    .dropna()
    .astype(str)
)


post_range_ids = set(
    post_range[
        "benchmark_row_id"
    ]
    .dropna()
    .astype(str)
)


if len(strict_match_ids) != 323:
    raise RuntimeError(
        "Strict matched benchmark IDs are not unique."
    )


if len(strict_unmatched_ids) != 28:
    raise RuntimeError(
        "Strict unmatched benchmark IDs are not unique."
    )


if len(range_match_ids) != 4:
    raise RuntimeError(
        "Range-compatible benchmark IDs are not unique."
    )


if len(post_range_ids) != 24:
    raise RuntimeError(
        "Post-range benchmark IDs are not unique."
    )


if not strict_match_ids.isdisjoint(
    range_match_ids
):
    raise RuntimeError(
        "Strict matched and range-compatible benchmark "
        "IDs overlap."
    )


if not range_match_ids.issubset(
    strict_unmatched_ids
):
    raise RuntimeError(
        "Range-compatible IDs are not a subset of "
        "strict-unmatched IDs."
    )


if (
    strict_unmatched_ids
    - range_match_ids
    != post_range_ids
):
    raise RuntimeError(
        "Strict unmatched minus range-compatible does "
        "not equal the final post-range partition."
    )


unresolved_source_or_extraction = int(
    post_range[
        "post_range_partition"
    ]
    .eq(
        "UNRESOLVED_SOURCE_OR_EXTRACTION"
    )
    .sum()
)


if unresolved_source_or_extraction != 0:
    raise RuntimeError(
        "Expected zero unresolved source/extraction rows."
    )


if not (
    post_range[
        "post_range_partition"
    ]
    .astype(str)
    .str.startswith(
        "BENCHMARK_"
    )
    .all()
):
    raise RuntimeError(
        "Not all post-range rows are documented "
        "benchmark-side discrepancies."
    )


# ============================================================
# Helpers
# ============================================================

def savefig(name):

    path = (
        OUT
        / name
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        path,
    )


def logistic(
    x,
    low,
    high,
    midpoint,
    slope,
):

    z = np.clip(
        -slope
        * (
            x
            - midpoint
        ),
        -100,
        100,
    )

    return (
        low
        +
        (
            high
            - low
        )
        /
        (
            1.0
            + np.exp(z)
        )
    )


def fit_logistic(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        return None

    if (
        np.nanmax(x)
        ==
        np.nanmin(x)
    ):
        return None

    low0 = float(
        np.nanpercentile(
            y,
            10,
        )
    )

    high0 = float(
        np.nanpercentile(
            y,
            90,
        )
    )

    midpoint0 = float(
        np.nanmedian(x)
    )


    if (
        np.nanstd(x) > 0
        and
        np.nanstd(y) > 0
    ):

        corr = np.corrcoef(
            x,
            y,
        )[0, 1]

    else:

        corr = 0.0


    slope0 = (
        0.005
        if corr >= 0
        else -0.005
    )


    y_min = float(
        np.nanmin(y)
    )

    y_max = float(
        np.nanmax(y)
    )

    span = max(
        y_max - y_min,
        1.0,
    )


    lower = [
        y_min
        - 2 * span,
        y_min
        - 2 * span,
        np.nanmin(x)
        - 1000,
        -0.1,
    ]

    upper = [
        y_max
        + 2 * span,
        y_max
        + 2 * span,
        np.nanmax(x)
        + 1000,
        0.1,
    ]


    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            popt, pcov = curve_fit(
                logistic,
                x,
                y,
                p0=[
                    low0,
                    high0,
                    midpoint0,
                    slope0,
                ],
                bounds=(
                    lower,
                    upper,
                ),
                maxfev=100000,
            )

    except Exception:

        return None


    return {
        "params":
            popt,

        "covariance":
            pcov,

        "n":
            len(x),
    }


def regression_metrics(
    manual_values,
    extracted_values,
):

    m = pd.to_numeric(
        manual_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    e = pd.to_numeric(
        extracted_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    good = (
        np.isfinite(m)
        &
        np.isfinite(e)
    )

    m = m[good]
    e = e[good]


    if len(m) == 0:

        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "mean_bias_extracted_minus_manual":
                np.nan,
            "pearson_r": np.nan,
            "R_squared": np.nan,
            "within_0_02_pct": np.nan,
            "within_0_15_pct": np.nan,
        }


    diff = (
        e
        - m
    )


    mae = float(
        np.mean(
            np.abs(diff)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                diff ** 2
            )
        )
    )

    bias = float(
        np.mean(diff)
    )


    if (
        len(m) >= 2
        and
        np.std(m) > 0
        and
        np.std(e) > 0
    ):

        r = float(
            np.corrcoef(
                m,
                e,
            )[0, 1]
        )

    else:

        r = np.nan


    denominator = np.sum(
        (
            m
            - np.mean(m)
        )
        ** 2
    )


    if denominator > 0:

        r_squared = (
            1.0
            -
            np.sum(
                (
                    e
                    - m
                )
                ** 2
            )
            /
            denominator
        )

    else:

        r_squared = np.nan


    return {
        "n":
            len(m),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "mean_bias_extracted_minus_manual":
            bias,

        "pearson_r":
            r,

        "R_squared":
            float(
                r_squared
            ),

        "within_0_02_pct":
            float(
                100
                * np.mean(
                    np.abs(diff)
                    <= 0.02
                )
            ),

        "within_0_15_pct":
            float(
                100
                * np.mean(
                    np.abs(diff)
                    <= 0.15
                )
            ),
    }


def atomic_ratios(df):

    x = df.copy()

    C = pd.to_numeric(
        x["C_value"],
        errors="coerce",
    )

    H = pd.to_numeric(
        x["H_value"],
        errors="coerce",
    )

    O = pd.to_numeric(
        x["O_value"],
        errors="coerce",
    )


    carbon_moles = (
        C / 12.011
    )

    x["H_C_atomic"] = (
        (
            H / 1.008
        )
        /
        carbon_moles
    )

    x["O_C_atomic"] = (
        (
            O / 15.999
        )
        /
        carbon_moles
    )


    bad = (
        ~np.isfinite(
            x[
                "H_C_atomic"
            ]
        )
        |
        ~np.isfinite(
            x[
                "O_C_atomic"
            ]
        )
        |
        (
            C <= 0
        )
    )

    x.loc[
        bad,
        [
            "H_C_atomic",
            "O_C_atomic",
        ],
    ] = np.nan


    return x


# ============================================================
# Overall benchmark summary
# ============================================================

benchmark_recall = (
    100
    * len(matched)
    / len(manual)
)

eligible_match_fraction = (
    100
    * len(matched)
    / len(extracted)
)


source_temperature_compatible_matches = (
    len(matched)
    + len(range_matches)
)


source_temperature_compatible_recall = (
    100
    * source_temperature_compatible_matches
    / len(manual)
)


accounted_benchmark_rows = (
    source_temperature_compatible_matches
    + len(post_range)
)


accounted_benchmark_pct = (
    100
    * accounted_benchmark_rows
    / len(manual)
)


if source_temperature_compatible_matches != 327:
    raise RuntimeError(
        "Expected 327 source-temperature-compatible "
        "benchmark matches."
    )


if accounted_benchmark_rows != 351:
    raise RuntimeError(
        "Expected all 351 represented benchmark rows "
        "to be accounted for."
    )


exact_002 = (
    matched[
        "match_class"
    ]
    .eq(
        "EXACT_NUMERIC_WITHIN_0_02"
    )
    .sum()
)


reliable_015 = (
    matched[
        "match_class"
    ]
    .eq(
        "RELIABLE_WITHIN_0_15"
    )
    .sum()
)


overall = pd.DataFrame(
    [
        {
            "metric":
                "represented_benchmark_rows",
            "value":
                len(manual),
        },
        {
            "metric":
                "benchmark_eligible_extracted_rows",
            "value":
                len(extracted),
        },
        {
            "metric":
                "conservative_one_to_one_matches",
            "value":
                len(matched),
        },
        {
            "metric":
                "unmatched_benchmark_rows",
            "value":
                len(unmatched),
        },
        {
            "metric":
                "benchmark_recall_pct",
            "value":
                benchmark_recall,
        },
        {
            "metric":
                "eligible_extracted_matched_pct",
            "value":
                eligible_match_fraction,
        },
        {
            "metric":
                "matches_all_elements_within_0_02",
            "value":
                exact_002,
        },
        {
            "metric":
                "additional_matches_within_0_15",
            "value":
                reliable_015,
        },
        {
            "metric":
                "initial_matches_before_last_fix",
            "value":
                294,
        },
        {
            "metric":
                "additional_matches_from_final_fix",
            "value":
                len(matched)
                - 294,
        },
        {
            "metric":
                "initial_benchmark_recall_pct",
            "value":
                100
                * 294
                / 351,
        },
        {
            "metric":
                "recall_improvement_percentage_points",
            "value":
                benchmark_recall
                -
                (
                    100
                    * 294
                    / 351
                ),
        },
    ]
)


overall = pd.concat(
    [
        overall,
        pd.DataFrame(
            [
                {
                    "metric":
                        "range_compatible_additional_matches",
                    "value":
                        len(range_matches),
                },
                {
                    "metric":
                        "source_temperature_compatible_matches",
                    "value":
                        source_temperature_compatible_matches,
                },
                {
                    "metric":
                        "source_temperature_compatible_recall_pct",
                    "value":
                        source_temperature_compatible_recall,
                },
                {
                    "metric":
                        "post_range_benchmark_side_discrepancies",
                    "value":
                        len(post_range),
                },
                {
                    "metric":
                        "unresolved_source_or_extraction_rows",
                    "value":
                        unresolved_source_or_extraction,
                },
                {
                    "metric":
                        "accounted_benchmark_rows",
                    "value":
                        accounted_benchmark_rows,
                },
                {
                    "metric":
                        "accounted_benchmark_rows_pct",
                    "value":
                        accounted_benchmark_pct,
                },
            ]
        ),
    ],
    ignore_index=True,
)


overall.to_csv(
    OUT
    / "final_benchmark_summary.csv",
    index=False,
)


# ============================================================
# Element-level matched metrics
# ============================================================

metric_rows = []


for element, full_name in ELEMENTS.items():

    manual_col = (
        f"{element}_value_manual"
    )

    extracted_col = (
        f"{element}_value_extracted"
    )

    result = regression_metrics(
        matched[manual_col],
        matched[extracted_col],
    )

    result["element"] = element
    result["element_name"] = full_name

    metric_rows.append(
        result
    )


metrics = pd.DataFrame(
    metric_rows
)


metrics = metrics[
    [
        "element",
        "element_name",
        "n",
        "MAE",
        "RMSE",
        "mean_bias_extracted_minus_manual",
        "pearson_r",
        "R_squared",
        "within_0_02_pct",
        "within_0_15_pct",
    ]
]


metrics.to_csv(
    OUT
    / "matched_element_metrics.csv",
    index=False,
)


# ============================================================
# Plot 1-4:
# Manual vs final extracted elemental content vs temperature
# + curve fits
# ============================================================

fit_rows = []


for element, full_name in ELEMENTS.items():

    value_col = (
        f"{element}_value"
    )


    m = manual[
        [
            "temperature_C",
            value_col,
        ]
    ].dropna()


    e = extracted[
        [
            "temperature_C",
            value_col,
        ]
    ].dropna()


    plt.figure(
        figsize=(8.2, 5.8)
    )


    plt.scatter(
        m["temperature_C"],
        m[value_col],
        s=24,
        alpha=0.48,
        label=(
            f"Manual benchmark "
            f"(n={len(m)})"
        ),
        color=MANUAL_COLOR,
    )


    plt.scatter(
        e["temperature_C"],
        e[value_col],
        s=24,
        alpha=0.48,
        label=(
            f"Final extraction "
            f"(n={len(e)})"
        ),
        color=EXTRACTED_COLOR,
        marker="x",
    )


    for dataset_name, data, fit_color in [
        (
            "Manual benchmark",
            m,
            FIT_MANUAL_COLOR,
        ),
        (
            "Final extraction",
            e,
            FIT_EXTRACTED_COLOR,
        ),
    ]:

        fit = fit_logistic(
            data[
                "temperature_C"
            ],
            data[
                value_col
            ],
        )

        if fit is None:

            fit_rows.append(
                {
                    "element":
                        element,

                    "dataset":
                        dataset_name,

                    "fit_status":
                        "FAILED_OR_NOT_SUPPORTED",
                }
            )

            continue


        params = fit[
            "params"
        ]


        x_grid = np.linspace(
            min(
                m[
                    "temperature_C"
                ].min(),
                e[
                    "temperature_C"
                ].min(),
            ),
            max(
                m[
                    "temperature_C"
                ].max(),
                e[
                    "temperature_C"
                ].max(),
            ),
            400,
        )


        y_grid = logistic(
            x_grid,
            *params,
        )


        plt.plot(
            x_grid,
            y_grid,
            linewidth=2.0,
            color=fit_color,
            label=(
                dataset_name
                + " logistic fit"
            ),
        )


        fit_rows.append(
            {
                "element":
                    element,

                "dataset":
                    dataset_name,

                "fit_status":
                    "OK",

                "n":
                    fit["n"],

                "low":
                    params[0],

                "high":
                    params[1],

                "midpoint":
                    params[2],

                "slope":
                    params[3],
            }
        )


    plt.xlabel(
        "Processing temperature (°C)"
    )

    plt.ylabel(
        f"{full_name} (wt%)"
    )

    plt.title(
        f"{full_name} vs processing temperature\n"
        "Manual benchmark compared with final extraction"
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend(
        frameon=False,
        fontsize=8,
    )


    savefig(
        f"temperature_overlay_{element}.png"
    )


fit_results = pd.DataFrame(
    fit_rows
)


fit_results.to_csv(
    OUT
    / "temperature_curve_fit_parameters.csv",
    index=False,
)


# ============================================================
# Plot 5-8:
# Matched manual vs extracted parity plots
# ============================================================

for element, full_name in ELEMENTS.items():

    mcol = (
        f"{element}_value_manual"
    )

    ecol = (
        f"{element}_value_extracted"
    )


    p = matched[
        [
            mcol,
            ecol,
        ]
    ].copy()


    p[mcol] = pd.to_numeric(
        p[mcol],
        errors="coerce",
    )

    p[ecol] = pd.to_numeric(
        p[ecol],
        errors="coerce",
    )

    p = p.dropna()


    lo = float(
        min(
            p[mcol].min(),
            p[ecol].min(),
        )
    )

    hi = float(
        max(
            p[mcol].max(),
            p[ecol].max(),
        )
    )

    pad = max(
        (
            hi
            - lo
        )
        * 0.05,
        0.05,
    )


    plt.figure(
        figsize=(6.2, 6.2)
    )


    plt.scatter(
        p[mcol],
        p[ecol],
        s=28,
        alpha=0.58,
        color=EXTRACTED_COLOR,
    )


    plt.plot(
        [
            lo - pad,
            hi + pad,
        ],
        [
            lo - pad,
            hi + pad,
        ],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="1:1 agreement",
    )


    metric = metrics[
        metrics[
            "element"
        ].eq(element)
    ].iloc[0]


    annotation = (
        f"n = {int(metric['n'])}\n"
        f"MAE = {metric['MAE']:.4f}\n"
        f"RMSE = {metric['RMSE']:.4f}\n"
        f"r = {metric['pearson_r']:.5f}\n"
        f"R² = {metric['R_squared']:.5f}"
    )


    plt.text(
        0.04,
        0.96,
        annotation,
        transform=plt.gca().transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle":
                "round,pad=0.35",
            "facecolor":
                "white",
            "alpha":
                0.85,
        },
    )


    plt.xlim(
        lo - pad,
        hi + pad,
    )

    plt.ylim(
        lo - pad,
        hi + pad,
    )

    plt.xlabel(
        f"Manual {full_name} (wt%)"
    )

    plt.ylabel(
        f"Extracted {full_name} (wt%)"
    )

    plt.title(
        f"Matched {full_name}: "
        "manual vs final extraction"
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend(
        frameon=False
    )


    savefig(
        f"parity_{element}.png"
    )


# ============================================================
# Van Krevelen comparison
# ============================================================

manual_vk = atomic_ratios(
    manual
)

extracted_vk = atomic_ratios(
    extracted
)


manual_vk_valid = manual_vk[
    manual_vk[
        [
            "O_C_atomic",
            "H_C_atomic",
        ]
    ]
    .notna()
    .all(
        axis=1
    )
].copy()


extracted_vk_valid = extracted_vk[
    extracted_vk[
        [
            "O_C_atomic",
            "H_C_atomic",
        ]
    ]
    .notna()
    .all(
        axis=1
    )
].copy()


manual_vk_valid.to_csv(
    OUT
    / "manual_van_krevelen_points.csv",
    index=False,
)

extracted_vk_valid.to_csv(
    OUT
    / "extracted_van_krevelen_points.csv",
    index=False,
)


plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    manual_vk_valid[
        "O_C_atomic"
    ],
    manual_vk_valid[
        "H_C_atomic"
    ],
    s=28,
    alpha=0.50,
    color=MANUAL_COLOR,
    label=(
        "Manual benchmark "
        f"(n={len(manual_vk_valid)})"
    ),
)


plt.scatter(
    extracted_vk_valid[
        "O_C_atomic"
    ],
    extracted_vk_valid[
        "H_C_atomic"
    ],
    s=28,
    alpha=0.50,
    color=EXTRACTED_COLOR,
    marker="x",
    label=(
        "Final extraction "
        f"(n={len(extracted_vk_valid)})"
    ),
)


plt.xlabel(
    "Atomic O/C ratio"
)

plt.ylabel(
    "Atomic H/C ratio"
)

plt.title(
    "Van Krevelen comparison\n"
    "Manual benchmark vs final extraction"
)

plt.grid(
    alpha=0.20
)

plt.legend(
    frameon=False
)


savefig(
    "van_krevelen_overlay_full.png"
)


# Zoomed version corresponding to the dense scientific region.

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    manual_vk_valid[
        "O_C_atomic"
    ],
    manual_vk_valid[
        "H_C_atomic"
    ],
    s=30,
    alpha=0.55,
    color=MANUAL_COLOR,
    label="Manual benchmark",
)


plt.scatter(
    extracted_vk_valid[
        "O_C_atomic"
    ],
    extracted_vk_valid[
        "H_C_atomic"
    ],
    s=30,
    alpha=0.55,
    color=EXTRACTED_COLOR,
    marker="x",
    label="Final extraction",
)


plt.xlim(
    0,
    1.5,
)

plt.ylim(
    0,
    2.0,
)

plt.xlabel(
    "Atomic O/C ratio"
)

plt.ylabel(
    "Atomic H/C ratio"
)

plt.title(
    "Van Krevelen comparison — dense region"
)

plt.grid(
    alpha=0.20
)

plt.legend(
    frameon=False
)


savefig(
    "van_krevelen_overlay_zoom.png"
)


# ============================================================
# H/C and O/C vs temperature
# ============================================================

for ratio, label in [
    (
        "H_C_atomic",
        "Atomic H/C ratio",
    ),
    (
        "O_C_atomic",
        "Atomic O/C ratio",
    ),
]:

    m = manual_vk_valid[
        [
            "temperature_C",
            ratio,
        ]
    ].dropna()


    e = extracted_vk_valid[
        [
            "temperature_C",
            ratio,
        ]
    ].dropna()


    plt.figure(
        figsize=(8.2, 5.8)
    )


    plt.scatter(
        m[
            "temperature_C"
        ],
        m[ratio],
        s=25,
        alpha=0.50,
        color=MANUAL_COLOR,
        label="Manual benchmark",
    )


    plt.scatter(
        e[
            "temperature_C"
        ],
        e[ratio],
        s=25,
        alpha=0.50,
        marker="x",
        color=EXTRACTED_COLOR,
        label="Final extraction",
    )


    plt.xlabel(
        "Processing temperature (°C)"
    )

    plt.ylabel(
        label
    )

    plt.title(
        f"{label} vs processing temperature\n"
        "Manual benchmark vs final extraction"
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend(
        frameon=False
    )


    savefig(
        f"{ratio}_vs_temperature_overlay.png"
    )


# ============================================================
# Feedstock-group recovery comparison
# ============================================================

manual_group = manual.copy()

manual_group[
    "Group"
] = (
    manual_group[
        "Group"
    ]
    .fillna(
        "Unspecified"
    )
    .astype(str)
)


manual_group_counts = (
    manual_group
    .groupby(
        "Group"
    )
    .size()
    .rename(
        "benchmark_rows"
    )
)


matched_group = matched.merge(
    manual[
        [
            "benchmark_row_id",
            "Group",
        ]
    ],
    on="benchmark_row_id",
    how="left",
    validate="one_to_one",
)


matched_group[
    "Group"
] = (
    matched_group[
        "Group"
    ]
    .fillna(
        "Unspecified"
    )
    .astype(str)
)


matched_group_counts = (
    matched_group
    .groupby(
        "Group"
    )
    .size()
    .rename(
        "matched_rows"
    )
)


group_summary = (
    pd.concat(
        [
            manual_group_counts,
            matched_group_counts,
        ],
        axis=1,
    )
    .fillna(0)
    .reset_index()
)


group_summary[
    "benchmark_rows"
] = (
    group_summary[
        "benchmark_rows"
    ]
    .astype(int)
)

group_summary[
    "matched_rows"
] = (
    group_summary[
        "matched_rows"
    ]
    .astype(int)
)


group_summary[
    "recall_pct"
] = (
    100
    * group_summary[
        "matched_rows"
    ]
    /
    group_summary[
        "benchmark_rows"
    ]
)


group_summary = (
    group_summary
    .sort_values(
        "recall_pct",
        ascending=True,
    )
)


group_summary.to_csv(
    OUT
    / "benchmark_recovery_by_group.csv",
    index=False,
)


plt.figure(
    figsize=(
        8.5,
        max(
            4.8,
            0.48
            * len(
                group_summary
            ),
        ),
    )
)


plt.barh(
    group_summary[
        "Group"
    ],
    group_summary[
        "recall_pct"
    ],
)


plt.axvline(
    benchmark_recall,
    linestyle="--",
    linewidth=1.3,
    color="black",
    label=(
        f"Overall = "
        f"{benchmark_recall:.1f}%"
    ),
)


plt.xlim(
    0,
    105,
)

plt.xlabel(
    "Benchmark recovery (%)"
)

plt.ylabel(
    "Feedstock group"
)

plt.title(
    "Benchmark recovery by feedstock group"
)

plt.grid(
    axis="x",
    alpha=0.20,
)

plt.legend(
    frameon=False
)


savefig(
    "benchmark_recovery_by_group.png"
)


# ============================================================
# Per-paper recovery
# ============================================================

doi_plot = (
    per_doi
    .sort_values(
        "benchmark_recall_pct",
        ascending=True,
    )
    .copy()
)


doi_plot.to_csv(
    OUT
    / "benchmark_recovery_by_doi.csv",
    index=False,
)


plt.figure(
    figsize=(
        10,
        11,
    )
)


plt.barh(
    doi_plot[
        "doi"
    ],
    doi_plot[
        "benchmark_recall_pct"
    ],
)


plt.axvline(
    benchmark_recall,
    linestyle="--",
    color="black",
    linewidth=1.2,
    label=(
        f"Overall = "
        f"{benchmark_recall:.1f}%"
    ),
)


plt.xlim(
    0,
    105,
)

plt.xlabel(
    "Benchmark recovery (%)"
)

plt.ylabel(
    "DOI"
)

plt.title(
    "Benchmark recovery by paper"
)

plt.grid(
    axis="x",
    alpha=0.20,
)

plt.legend(
    frameon=False
)


savefig(
    "benchmark_recovery_by_paper.png"
)


# ============================================================
# Final outcome bar chart
# ============================================================

outcomes = pd.DataFrame(
    {
        "Outcome": [
            "Matched",
            "Unmatched",
        ],

        "Rows": [
            len(matched),
            len(unmatched),
        ],
    }
)


outcomes.to_csv(
    OUT
    / "benchmark_outcomes.csv",
    index=False,
)


plt.figure(
    figsize=(6.5, 5.2)
)


plt.bar(
    outcomes[
        "Outcome"
    ],
    outcomes[
        "Rows"
    ],
)


for i, row in outcomes.iterrows():

    plt.text(
        i,
        row["Rows"]
        + 3,
        str(
            int(
                row["Rows"]
            )
        ),
        ha="center",
    )


plt.ylabel(
    "Benchmark rows"
)

plt.title(
    "Final conservative benchmark outcome\n"
    f"Recovery = {benchmark_recall:.2f}%"
)

plt.grid(
    axis="y",
    alpha=0.20,
)


savefig(
    "final_benchmark_outcomes.png"
)


# ============================================================
# Source-temperature-compatible + audit accounting
# ============================================================

validation_accounting = pd.DataFrame(
    {
        "Outcome": [
            "Strict matches",
            "Range-compatible additions",
            "Benchmark-side discrepancies",
            "Unresolved source/extraction",
        ],

        "Rows": [
            len(matched),
            len(range_matches),
            len(post_range),
            unresolved_source_or_extraction,
        ],
    }
)


validation_accounting.to_csv(
    OUT
    / "final_validation_accounting.csv",
    index=False,
)


plt.figure(
    figsize=(8.5, 5.6)
)


plt.bar(
    validation_accounting[
        "Outcome"
    ],
    validation_accounting[
        "Rows"
    ],
)


for i, row in validation_accounting.iterrows():

    plt.text(
        i,
        row["Rows"]
        + 3,
        str(
            int(
                row["Rows"]
            )
        ),
        ha="center",
    )


plt.ylabel(
    "Represented benchmark rows"
)


plt.title(
    "Final benchmark validation accounting\n"
    f"Strict = {benchmark_recall:.2f}% | "
    f"Source-temperature-compatible = "
    f"{source_temperature_compatible_recall:.2f}% | "
    f"Accounted = {accounted_benchmark_pct:.2f}%"
)


plt.xticks(
    rotation=15,
    ha="right",
)


plt.grid(
    axis="y",
    alpha=0.20,
)


savefig(
    "final_validation_accounting.png"
)


# ============================================================
# Remaining-failure composition
# ============================================================

diagnosis_summary = (
    post_range[
        "post_range_partition"
    ]
    .value_counts()
    .rename_axis(
        "diagnosis"
    )
    .reset_index(
        name="count"
    )
)


diagnosis_summary[
    "pct_of_unmatched"
] = (
    100
    * diagnosis_summary[
        "count"
    ]
    / len(post_range)
)


diagnosis_summary.to_csv(
    OUT
    / "remaining_failure_breakdown.csv",
    index=False,
)


plot_diag = (
    diagnosis_summary
    .sort_values(
        "count",
        ascending=True,
    )
)


plt.figure(
    figsize=(9.5, 5.8)
)


plt.barh(
    plot_diag[
        "diagnosis"
    ],
    plot_diag[
        "count"
    ],
)


plt.xlabel(
    "Post-range benchmark rows"
)

plt.title(
    "Post-range benchmark-side discrepancies"
)

plt.grid(
    axis="x",
    alpha=0.20,
)


savefig(
    "remaining_failure_breakdown.png"
)


# ============================================================
# Matched point export for report / further plots
# ============================================================

matched.to_csv(
    OUT
    / "final_matched_comparison_points.csv",
    index=False,
)


# ============================================================
# Report-ready text summary
# ============================================================

lines = []

lines.append(
    "FINAL LITERATURE-MINING VALIDATION SUMMARY"
)

lines.append(
    "=" * 72
)

lines.append(
    ""
)

lines.append(
    f"Represented benchmark papers: "
    f"{manual['doi_normalized'].nunique()}"
)

lines.append(
    f"Represented benchmark rows: "
    f"{len(manual)}"
)

lines.append(
    f"Benchmark-eligible extracted rows: "
    f"{len(extracted)}"
)

lines.append(
    f"Strict one-to-one matches: "
    f"{len(matched)}"
)

lines.append(
    f"Strict benchmark recall: "
    f"{benchmark_recall:.2f}%"
)

lines.append(
    f"Strict unmatched benchmark rows: "
    f"{len(unmatched)}"
)

lines.append(
    ""
)

lines.append(
    f"Matches with all comparable CHNO "
    f"values within 0.02 wt%: "
    f"{exact_002}/{len(matched)} "
    f"({100 * exact_002 / len(matched):.2f}%)"
)

lines.append(
    f"Additional matches within 0.15 wt%: "
    f"{reliable_015}"
)

lines.append(
    ""
)

lines.append(
    "SOURCE-TEMPERATURE-COMPATIBLE VALIDATION"
)

lines.append(
    "-" * 72
)

lines.append(
    f"Strict exact-temperature matches: "
    f"{len(matched)}/{len(manual)} "
    f"({benchmark_recall:.2f}%)"
)

lines.append(
    f"Validated range-compatible additions: "
    f"{len(range_matches)}"
)

lines.append(
    f"Source-temperature-compatible matches: "
    f"{source_temperature_compatible_matches}/"
    f"{len(manual)} "
    f"({source_temperature_compatible_recall:.2f}%)"
)

lines.append(
    ""
)

lines.append(
    "POST-RANGE AUDIT"
)

lines.append(
    "-" * 72
)

for _, row in post_range_summary.iterrows():

    lines.append(
        f"{row['post_range_partition']}: "
        f"{int(row['row_count'])}"
    )

lines.append(
    f"Unresolved source/extraction failures: "
    f"{unresolved_source_or_extraction}"
)

lines.append(
    f"Accounted represented benchmark rows: "
    f"{accounted_benchmark_rows}/{len(manual)} "
    f"({accounted_benchmark_pct:.2f}%)"
)

lines.append(
    ""
)

lines.append(
    "Improvement after final contextual "
    "temperature correction:"
)

lines.append(
    "  Before: 294/351 "
    f"({100 * 294 / 351:.2f}%)"
)

lines.append(
    "  Final:  "
    f"{len(matched)}/351 "
    f"({benchmark_recall:.2f}%)"
)

lines.append(
    "  Gain:   "
    f"{len(matched) - 294} rows, "
    f"+{benchmark_recall - 100 * 294 / 351:.2f} "
    "percentage points"
)

lines.append(
    ""
)

lines.append(
    "ELEMENT-LEVEL AGREEMENT"
)

lines.append(
    "-" * 72
)


for _, row in metrics.iterrows():

    lines.append(
        (
            f"{row['element_name']}: "
            f"n={int(row['n'])}, "
            f"MAE={row['MAE']:.4f}, "
            f"RMSE={row['RMSE']:.4f}, "
            f"r={row['pearson_r']:.5f}, "
            f"R2={row['R_squared']:.5f}, "
            f"within 0.02="
            f"{row['within_0_02_pct']:.2f}%"
        )
    )


lines.append(
    ""
)

lines.append(
    "REMAINING DIFFERENCE CLASSES"
)

lines.append(
    "-" * 72
)


for _, row in diagnosis_summary.iterrows():

    lines.append(
        (
            f"{row['diagnosis']}: "
            f"{int(row['count'])} "
            f"({row['pct_of_unmatched']:.2f}%)"
        )
    )


lines.append(
    ""
)

lines.append(
    "Interpretation:"
)

lines.append(
    (
        "Where the pipeline identifies the corresponding "
        "benchmark record, numerical CHNO extraction is "
        "extremely accurate. Remaining benchmark differences "
        "are dominated by coverage/alignment and source-versus-"
        "benchmark discrepancies rather than general numerical "
        "extraction error."
    )
)

lines.append(
    ""
)

lines.append(
    (
        "The four 525 C benchmark records from "
        "10.1016/j.jaap.2022.105616 remain unresolved because "
        "the source evidence available to the pipeline reports "
        "a 500-550 C processing interval rather than a unique "
        "525 C temperature. No benchmark-derived value was "
        "inserted into the extraction."
    )
)


report_path = (
    OUT
    / "REPORT_READY_RESULTS.txt"
)


report_path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ============================================================
# Final console report
# ============================================================

print()
print("=" * 100)
print("FINAL REPORT COMPARISON COMPLETE")
print("=" * 100)

print()
print("OVERALL")
print(
    overall.to_string(
        index=False
    )
)

print()
print("ELEMENT METRICS")
print(
    metrics.to_string(
        index=False
    )
)

print()
print("REMAINING DIFFERENCES")
print(
    diagnosis_summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("FINAL NUMBERS")
print("=" * 100)

print(
    "Benchmark papers:",
    manual[
        "doi_normalized"
    ].nunique(),
)

print(
    "Benchmark rows:",
    len(manual),
)

print(
    "Final eligible extracted rows:",
    len(extracted),
)

print(
    "Strict matched:",
    len(matched),
)

print(
    "Strict unmatched:",
    len(unmatched),
)

print(
    "Strict benchmark recall:",
    f"{benchmark_recall:.2f}%",
)

print(
    "Matches within 0.02:",
    exact_002,
)

print(
    "Matches within 0.15 only:",
    reliable_015,
)

print(
    "Range-compatible additions:",
    len(range_matches),
)

print(
    "Source-temperature-compatible matches:",
    source_temperature_compatible_matches,
)

print(
    "Source-temperature-compatible recall:",
    f"{source_temperature_compatible_recall:.2f}%",
)

print(
    "Post-range benchmark-side discrepancies:",
    len(post_range),
)

print(
    "Unresolved source/extraction failures:",
    unresolved_source_or_extraction,
)

print(
    "Accounted benchmark rows:",
    f"{accounted_benchmark_rows}/{len(manual)}",
)

print(
    "Accounted benchmark rows (%):",
    f"{accounted_benchmark_pct:.2f}%",
)

print()
print(
    "All figures and tables saved to:"
)

print(
    OUT
)
