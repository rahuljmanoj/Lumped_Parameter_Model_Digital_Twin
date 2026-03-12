# -*- coding: utf-8 -*-
"""
Population level UQ + SA visualization (single aggregated run).

Assumes the following files exist in:
    UQSA_ROOT / <SUB_ID> /

    <SUB_ID>_uq_summary.json
    <SUB_ID>_uq_samples.csv
    <SUB_ID>_uq_params.csv
    <SUB_ID>_mc_invalid_params.csv

    <SUB_ID>_sobol_params.csv
    <SUB_ID>_sobol_lvedp.csv
    <SUB_ID>_sobol_validity.csv
    <SUB_ID>_sobol_invalid_params.csv
    <SUB_ID>_sobol_validity_invalid_params.csv

Outputs figures into:
    UQSA_ROOT / "_viz_population"
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# GLOBAL STYLE
# -----------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

PHYS_RANGE = (4.0, 35.0)
CM_TO_INCH = 1.0 / 2.54  # for figsize conversion

# pretty labels for Sobol parameter names
PARAM_LABELS = {
    "R_sys":  r"$R_{\mathrm{sys}}$",
    "Z_ao":   r"$Z_{\mathrm{ao}}$",
    "C_sa":   r"$C_{\mathrm{sa}}$",
    "R_mv":   r"$R_{\mathrm{mv}}$",
    "E_max":  r"$E_{\max}$",
    "E_min":  r"$E_{\min}$",
    "t_peak": r"$t_{\mathrm{peak}}$",
    "V_tot":  r"$V_{\mathrm{tot}}$",
    "C_sv":   r"$C_{\mathrm{sv}}$",
}

# -----------------------------
# CONFIG: edit these two lines
# -----------------------------
UQSA_ROOT = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Results_UQSA_Population_Level\UQ_SA"
SUB_ID = 999  # population level run id

OUTDIR = os.path.join(UQSA_ROOT, "_viz_population")
os.makedirs(OUTDIR, exist_ok=True)


def savefig(fig, name, dpi=300):
    path = os.path.abspath(os.path.join(OUTDIR, name))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved -> {path}")


# -----------------------------
# Loaders for population run
# -----------------------------
def load_population_artifacts(uqsa_root, sub_id):
    folder = os.path.join(uqsa_root, str(sub_id))

    def f(name):
        return os.path.join(folder, f"{sub_id}_{name}")

    # UQ summary
    summary_path = f("uq_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing UQ summary: {summary_path}")
    with open(summary_path, "r") as fp:
        s = json.load(fp)
    PI95 = s.get("PI95", [np.nan, np.nan])
    summary = {
        "sub_id": sub_id,
        "N": s.get("N", np.nan),
        "mean": s.get("mean", np.nan),
        "sd": s.get("sd", np.nan),
        "PI95_low": PI95[0],
        "PI95_high": PI95[1],
    }
    df_summary = pd.DataFrame([summary])

    # UQ samples (LVEDP)
    uq_samples_path = f("uq_samples.csv")
    if not os.path.isfile(uq_samples_path):
        raise FileNotFoundError(f"Missing UQ samples: {uq_samples_path}")
    df_samples = pd.read_csv(uq_samples_path)

    # invalid UQ params
    mc_inv_path = f("mc_invalid_params.csv")
    if os.path.isfile(mc_inv_path):
        n_bad_uq = max(sum(1 for _ in open(mc_inv_path, "r")) - 1, 0)
    else:
        n_bad_uq = 0
    n_all_uq = int(summary["N"]) + n_bad_uq if np.isfinite(summary["N"]) else n_bad_uq
    bad_pct_uq = (100.0 * n_bad_uq / n_all_uq) if n_all_uq > 0 else np.nan

    # Sobol LVEDP
    sobol_lvedp_path = f("sobol_lvedp.csv")
    df_sa_lvedp = pd.read_csv(sobol_lvedp_path) if os.path.isfile(sobol_lvedp_path) else pd.DataFrame()

    # Sobol VALIDITY
    sobol_valid_path = f("sobol_validity.csv")
    df_sa_valid = pd.read_csv(sobol_valid_path) if os.path.isfile(sobol_valid_path) else pd.DataFrame()

    # Sobol invalid logs
    sobol_inv_lvedp_path = f("sobol_invalid_params.csv")
    sobol_inv_valid_path = f("sobol_validity_invalid_params.csv")

    # For SA we know the number of Sobol samples from params file
    sobol_params_path = f("sobol_params.csv")
    if os.path.isfile(sobol_params_path):
        n_all_sa = max(sum(1 for _ in open(sobol_params_path, "r")) - 1, 0)
    else:
        n_all_sa = 0

    if os.path.isfile(sobol_inv_lvedp_path):
        n_bad_sa_l = max(sum(1 for _ in open(sobol_inv_lvedp_path, "r")) - 1, 0)
    else:
        n_bad_sa_l = 0
    bad_pct_sa_l = (100.0 * n_bad_sa_l / n_all_sa) if n_all_sa > 0 else np.nan

    if os.path.isfile(sobol_inv_valid_path):
        n_bad_sa_v = max(sum(1 for _ in open(sobol_inv_valid_path, "r")) - 1, 0)
    else:
        n_bad_sa_v = 0
    bad_pct_sa_v = (100.0 * n_bad_sa_v / n_all_sa) if n_all_sa > 0 else np.nan

    invalid_info = {
        "UQ": {"n_all": n_all_uq, "n_bad": n_bad_uq, "bad_pct": bad_pct_uq},
        "SA_LVEDP": {"n_all": n_all_sa, "n_bad": n_bad_sa_l, "bad_pct": bad_pct_sa_l},
        "SA_VALIDITY": {"n_all": n_all_sa, "n_bad": n_bad_sa_v, "bad_pct": bad_pct_sa_v},
    }

    return df_summary, df_samples, df_sa_lvedp, df_sa_valid, invalid_info


# -----------------------------
# Plots
# -----------------------------
def plot_uq_hist_and_summary(df_summary, df_samples, sub_id):
    """Histogram of LVEDP samples plus vertical lines for mean and PI."""
    mean = float(df_summary.loc[0, "mean"])
    low = float(df_summary.loc[0, "PI95_low"])
    high = float(df_summary.loc[0, "PI95_high"])

    y = df_samples["LVEDP"].values

    # 10 cm x 8 cm
    fig, ax = plt.subplots(figsize=(14 * CM_TO_INCH, 8 * CM_TO_INCH))
    ax.hist(y, bins=40, edgecolor="black", alpha=0.8)
    ax.axvspan(PHYS_RANGE[0], PHYS_RANGE[1], color="gray", alpha=0.15, label="Phys. LVEDP range")
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.1f} mmHg")
    ax.axvline(low, color="blue", linestyle=":", label=f"95% PI = {low:.1f} mmHg")
    ax.axvline(high, color="blue", linestyle=":", label=f"95% PI = {high:.1f} mmHg")

    ax.set_xlabel("LVEDP [mmHg]")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Population UQ: LVEDP distribution (sub_id={sub_id})")
    ax.legend()
    savefig(fig, f"{sub_id}_uq_hist_summary.png")


def _plot_s1_st_bars(df_sa, metric_name, filename_suffix):
    if df_sa.empty:
        print(f"[plot_s1_st_bars] no data for metric {metric_name}")
        return

    # if there are multiple rows per param (e.g. bootstrap CI), average them
    agg = df_sa.groupby("param", as_index=False)[["S1", "ST"]].mean()
    params = agg["param"].tolist()
    x = np.arange(len(params))
    width = 0.35

    # 15 cm x 8 cm
    fig, ax = plt.subplots(figsize=(15 * CM_TO_INCH, 8 * CM_TO_INCH))
    ax.bar(x - width / 2, agg["S1"].values, width, label="S1")
    ax.bar(x + width / 2, agg["ST"].values, width, label="ST")
    ax.set_xticks(x)

    # convert param names to mathtext with subscripts
    xticklabels = [PARAM_LABELS.get(p, p) for p in params]
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax.set_ylabel("Sobol index")
    ax.set_title(f"Population SA: S1 and ST per parameter ({metric_name})")
    ax.legend()
    savefig(fig, f"{SUB_ID}_sa_{filename_suffix}.png")


def plot_sa_lvedp(df_sa_lvedp):
    _plot_s1_st_bars(df_sa_lvedp, metric_name="LVEDP", filename_suffix="lvedp")


def plot_sa_validity(df_sa_valid):
    _plot_s1_st_bars(df_sa_valid, metric_name="VALIDITY", filename_suffix="validity")


def plot_invalid_rates_single(invalid_info, sub_id):
    phases = ["UQ", "SA_LVEDP", "SA_VALIDITY"]
    vals = []
    labels = []
    for ph in phases:
        v = invalid_info.get(ph, {})
        if v and np.isfinite(v["bad_pct"]):
            vals.append(v["bad_pct"])
            labels.append(ph.replace("SA_", "SA "))

    if not vals:
        print("[plot_invalid_rates_single] no invalid info")
        return

    fig, ax = plt.subplots(figsize=(13 * CM_TO_INCH, 8 * CM_TO_INCH))
    x = np.arange(len(vals))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Invalid fraction (%)")
    ax.set_title(f"Invalid sample rates (population run {sub_id})")
    savefig(fig, f"{sub_id}_invalid_rates.png")


def print_summary_to_console(df_summary, invalid_info):
    row = df_summary.iloc[0]
    print("\n=== Population UQ summary ===")
    print(f"sub_id     : {int(row['sub_id'])}")
    print(f"N valid    : {int(row['N'])}")
    print(f"mean LVEDP : {row['mean']:.2f} mmHg")
    print(f"sd LVEDP   : {row['sd']:.2f} mmHg")
    print(f"95% PI     : [{row['PI95_low']:.2f}, {row['PI95_high']:.2f}] mmHg")

    print("\n=== Invalid sample info ===")
    for ph, info in invalid_info.items():
        print(
            f"{ph:10s} -> n_all={info['n_all']}, "
            f"n_bad={info['n_bad']}, bad_pct={info['bad_pct']:.2f}%"
        )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    (
        df_summary,
        df_samples,
        df_sa_lvedp,
        df_sa_valid,
        invalid_info,
    ) = load_population_artifacts(UQSA_ROOT, SUB_ID)

    print_summary_to_console(df_summary, invalid_info)

    # UQ plot: histogram + mean and PI overlay
    plot_uq_hist_and_summary(df_summary, df_samples, SUB_ID)

    # SA LVEDP: S1 and ST per parameter
    plot_sa_lvedp(df_sa_lvedp)

    # SA VALIDITY: S1 and ST per parameter
    plot_sa_validity(df_sa_valid)

    # Invalid sample rates
    plot_invalid_rates_single(invalid_info, SUB_ID)

    plt.show()
