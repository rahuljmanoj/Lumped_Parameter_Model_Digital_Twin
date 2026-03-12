# -*- coding: utf-8 -*-
"""
Visualize UQ + SA artifacts written by your UQ/SA runner.
- Loads per-subject JSON/CSV outputs from UQ_SA/<sub_id> folders
- Builds clean DataFrames (summary, Sobol, invalid-rate)
- Makes figures and saves them to an output folder, and also shows them

Fixes / updates:
- boxplot(..., tick_labels=...) to silence Matplotlib 3.9 deprecation
- set_xticks(...) before set_xticklabels(...) to avoid warnings
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PHYS_RANGE = (4.0, 35.0)   # used for shading/lines
# --- CONFIG ---
UQSA_ROOT = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Results_UQSA_Subject_Specific\UQ_SA"
OUTDIR    = os.path.join(UQSA_ROOT, "_viz")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(fig, name, dpi=200):
    path = os.path.abspath(os.path.join(OUTDIR, name))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path}")

# -----------------------------
# Loaders
# -----------------------------
def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None

def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None

def load_subject_artifacts(uqsa_root):
    rows_summary = []
    rows_fail = []
    sa_frames = []

    for entry in os.scandir(uqsa_root):
        if not entry.is_dir():
            continue
        sub_id = _safe_int(entry.name)
        if sub_id is None:
            continue

        folder = entry.path

        # UQ files
        f_summary = os.path.join(folder, f"{sub_id}_uq_summary.json")
        f_uq_X    = os.path.join(folder, f"{sub_id}_uq_params.csv")
        f_uq_inv  = os.path.join(folder, f"{sub_id}_mc_invalid_params.csv")

        # Sobol common params file
        f_sa_X        = os.path.join(folder, f"{sub_id}_sobol_params.csv")

        # Sobol results
        f_sa_lvedp    = os.path.join(folder, f"{sub_id}_sobol_lvedp.csv")
        f_sa_valid    = os.path.join(folder, f"{sub_id}_sobol_validity.csv")

        # Sobol invalid param logs
        f_sa_inv_lvedp = os.path.join(folder, f"{sub_id}_sobol_invalid_params.csv")
        f_sa_inv_valid = os.path.join(folder, f"{sub_id}_sobol_validity_invalid_params.csv")

        # ----------------- UQ summary -----------------
        if os.path.isfile(f_summary):
            try:
                with open(f_summary, "r") as fp:
                    s = json.load(fp)
                PI95 = s.get("PI95", [np.nan, np.nan])
                rows_summary.append({
                    "sub_id": sub_id,
                    "N": s.get("N", np.nan),
                    "mean": s.get("mean", np.nan),
                    "sd": s.get("sd", np.nan),
                    "PI95_low": PI95[0],
                    "PI95_high": PI95[1],
                })
            except Exception as e:
                print(f"[warn] could not parse summary for {sub_id}: {e}")

        # ----------------- Sobol indices -----------------
        # LVEDP Sobol
        if os.path.isfile(f_sa_lvedp):
            try:
                df_l = pd.read_csv(f_sa_lvedp)
                if "sub_id" not in df_l.columns:
                    df_l["sub_id"] = sub_id
                if "metric" not in df_l.columns:
                    df_l["metric"] = "LVEDP"
                sa_frames.append(df_l)
            except Exception as e:
                print(f"[warn] could not read sobol LVEDP csv for {sub_id}: {e}")

        # VALIDITY Sobol
        if os.path.isfile(f_sa_valid):
            try:
                df_v = pd.read_csv(f_sa_valid)
                if "sub_id" not in df_v.columns:
                    df_v["sub_id"] = sub_id
                if "metric" not in df_v.columns:
                    df_v["metric"] = "VALIDITY"
                sa_frames.append(df_v)
            except Exception as e:
                print(f"[warn] could not read sobol VALIDITY csv for {sub_id}: {e}")

        # ----------------- invalid accounting: UQ -----------------
        if os.path.isfile(f_uq_X):
            n_all = sum(1 for _ in open(f_uq_X, "r")) - 1
            n_bad = 0
            if os.path.isfile(f_uq_inv):
                try:
                    n_bad = sum(1 for _ in open(f_uq_inv, "r")) - 1
                except Exception:
                    n_bad = 0
            rows_fail.append({
                "sub_id": sub_id,
                "phase": "UQ",
                "n_all": n_all,
                "n_bad": n_bad,
                "bad_pct": (100.0 * n_bad / n_all) if n_all > 0 else np.nan
            })

        # ----------------- invalid accounting: Sobol -----------------
        if os.path.isfile(f_sa_X):
            n_all_sa = sum(1 for _ in open(f_sa_X, "r")) - 1

            # LVEDP
            if os.path.isfile(f_sa_inv_lvedp):
                try:
                    n_bad_l = sum(1 for _ in open(f_sa_inv_lvedp, "r")) - 1
                except Exception:
                    n_bad_l = 0
                rows_fail.append({
                    "sub_id": sub_id,
                    "phase": "SA_LVEDP",
                    "n_all": n_all_sa,
                    "n_bad": n_bad_l,
                    "bad_pct": (100.0 * n_bad_l / n_all_sa) if n_all_sa > 0 else np.nan
                })

            # VALIDITY
            if os.path.isfile(f_sa_inv_valid):
                try:
                    n_bad_v = sum(1 for _ in open(f_sa_inv_valid, "r")) - 1
                except Exception:
                    n_bad_v = 0
                rows_fail.append({
                    "sub_id": sub_id,
                    "phase": "SA_VALIDITY",
                    "n_all": n_all_sa,
                    "n_bad": n_bad_v,
                    "bad_pct": (100.0 * n_bad_v / n_all_sa) if n_all_sa > 0 else np.nan
                })

    # summary df
    df_summary = (pd.DataFrame(
        rows_summary,
        columns=["sub_id", "N", "mean", "sd", "PI95_low", "PI95_high"]
    ).sort_values("sub_id"))

    # SA df (both metrics stacked, long format)
    if sa_frames:
        df_sa = pd.concat(sa_frames, ignore_index=True)
    else:
        df_sa = pd.DataFrame(
            columns=["sub_id", "metric", "param",
                     "S1", "S1_low", "S1_high",
                     "ST", "ST_low", "ST_high"]
        )
    if not df_sa.empty:
        sort_cols = [c for c in ["sub_id", "metric", "param"] if c in df_sa.columns]
        df_sa = df_sa.sort_values(sort_cols)

    # invalid accounting df
    df_fail = (pd.DataFrame(
        rows_fail,
        columns=["sub_id", "phase", "n_all", "n_bad", "bad_pct"]
    ).sort_values(["sub_id", "phase"]))

    return df_summary, df_sa, df_fail


# -----------------------------
# Plot helpers
# -----------------------------
def savefig(fig, name, dpi=200):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path}")

# 1) UQ: LVEDP mean + intervals per subject
def plot_uq_summary(df_summary):
    if df_summary.empty:
        print("[plot_uq_summary] no data")
        return
    df = df_summary.copy()
    df["sub_id"] = df["sub_id"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    y = df["mean"].values
    yerr = np.vstack([y - df["PI95_low"].values, df["PI95_high"].values - y])

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    ax.axhspan(4, 35, alpha=0.07, color="gray", label="Phys. LVEDP range")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sub_id"], rotation=0)
    ax.set_ylabel("LVEDP (mmHg)")
    ax.set_xlabel("Subject ID")
    ax.set_title("UQ: LVEDP mean and 95% prediction interval per subject")
    ax.legend(loc="upper left")
    savefig(fig, "uq_lvedp_summary.png")

# 2) SA: distribution of ST across subjects
def plot_st_box(df_sa, metric="LVEDP"):
    if df_sa.empty:
        print("[plot_st_box] no data")
        return
    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if df.empty:
        print(f"[plot_st_box] no data for metric={metric}")
        return

    order = (df.groupby("param")["ST"].median()
             .sort_values(ascending=False).index.tolist())
    data = [df.loc[df["param"] == p, "ST"].dropna().values for p in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("Total Sobol index (ST)")
    ax.set_title(f"Distribution of ST across subjects (metric={metric})")
    ax.set_xticklabels(order, rotation=45, ha="right")
    savefig(fig, f"sa_st_boxplot_{metric.lower()}.png")


# 3) Per-subject top-k ST bars
def plot_topk_st_per_subject(df_sa, k=3, metric="LVEDP"):
    if df_sa.empty:
        print("[plot_topk_st_per_subject] no data")
        return

    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if df.empty:
        print(f"[plot_topk_st_per_subject] no data for metric={metric}")
        return

    for sub_id, g in df.groupby("sub_id"):
        g = g.sort_values("ST", ascending=False).head(k)
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = np.arange(len(g))
        ax.bar(xs, g["ST"].values)
        ax.set_xticks(xs)
        ax.set_xticklabels(g["param"].values, rotation=45, ha="right")
        ax.set_ylim(0, max(1.0, g["ST"].max() * 1.1))
        ax.set_ylabel("ST")
        ax.set_title(f"Subject {int(sub_id)}: Top {k} ST (metric={metric})")
        savefig(fig, f"sa_top{k}_sub_{int(sub_id)}_{metric.lower()}.png")


# 4) Invalid-rate distributions
def plot_invalid_rates(df_fail):
    if df_fail.empty:
        print("[plot_invalid_rates] no data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    phases = ["UQ", "SA_LVEDP", "SA_VALIDITY"]
    boxdata = [df_fail.loc[df_fail["phase"] == ph, "bad_pct"].dropna().values for ph in phases]
    labels = ["UQ", "SA (LVEDP)", "SA (VALIDITY)"]

    # remove empty series to avoid empty boxes
    boxdata_clean = []
    labels_clean = []
    for bd, lab in zip(boxdata, labels):
        if len(bd) > 0:
            boxdata_clean.append(bd)
            labels_clean.append(lab)

    ax.boxplot(boxdata_clean, tick_labels=labels_clean, showfliers=False)
    ax.set_ylabel("Invalid fraction (%)")
    ax.set_title("Invalid LVEDP / validity rate across subjects")
    savefig(fig, "invalid_rates.png")


# 5) Correlate per-subject invalid rate with average ST per parameter
def plot_corr_invalid_vs_st(df_sa, df_fail, metric="LVEDP"):
    if df_sa.empty or df_fail.empty:
        print("[plot_corr_invalid_vs_st] no data")
        return

    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if df.empty:
        print(f"[plot_corr_invalid_vs_st] no SA data for metric={metric}")
        return

    mean_st = (df.groupby(["sub_id", "param"])["ST"]
               .mean().reset_index())
    bad = df_fail[df_fail["phase"] == "SA_LVEDP"][["sub_id", "bad_pct"]]

    merged = mean_st.merge(bad, on="sub_id", how="left").dropna()
    if merged.empty:
        print("[plot_corr_invalid_vs_st] merged empty")
        return

    for p, g in merged.groupby("param"):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(g["bad_pct"], g["ST"], alpha=0.7)
        ax.set_xlabel("Invalid fraction in SA (LVEDP) (%)")
        ax.set_ylabel("Mean ST")
        ax.set_title(f"Param {p}: ST vs invalid fraction (metric={metric})")
        savefig(fig, f"corr_invalid_ST_{p}_{metric.lower()}.png")


# 6) Table-like bar plot: subject-wise ST for a few key params
def plot_subjectwise_params(df_sa, params=("V_tot", "C_sv", "E_min"), metric="LVEDP"):
    if df_sa.empty:
        print("[plot_subjectwise_params] no data")
        return

    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    df = df[df["param"].isin(params)]
    if df.empty:
        print(f"[plot_subjectwise_params] selected params not found for metric={metric}")
        return

    pivot = df.pivot_table(index="sub_id", columns="param", values="ST", aggfunc="mean")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot))
    width = 0.8 / len(params)
    for i, p in enumerate(params):
        vals = pivot[p].values
        ax.bar(x + i * width, vals, width, label=p)
    ax.set_xticks(x + width * (len(params) - 1) / 2)
    ax.set_xticklabels(pivot.index.astype(int), rotation=0)
    ax.set_ylabel("ST")
    ax.set_title(f"Subject-wise ST for selected parameters (metric={metric})")
    ax.legend()
    savefig(fig, f"sa_subjectwise_key_params_{metric.lower()}.png")


def fig_uq_and_invalid(df_summary, df_fail):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # (A) UQ summary
    ax = axes[0]
    if not df_summary.empty:
        df = df_summary.copy()
        df["sub_id"] = df["sub_id"].astype(int)
        x = np.arange(len(df))
        y = df["mean"].values
        yerr = np.vstack([y - df["PI95_low"].values, df["PI95_high"].values - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
        ax.axhspan(4, 35, alpha=0.08, color="gray", label="Phys. LVEDP range")
        ax.set_xticks(x)
        ax.set_xticklabels(df["sub_id"], rotation=0)
        ax.set_ylabel("LVEDP (mmHg)")
        ax.set_xlabel("Subject ID")
        ax.set_title("UQ: LVEDP mean ± 95% PI")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "No UQ summary", ha="center")

    # (B) Invalid rate boxplot
    ax = axes[1]
    if not df_fail.empty:
        boxdata = [
            df_fail.loc[df_fail["phase"]=="UQ", "bad_pct"].dropna().values,
            df_fail.loc[df_fail["phase"]=="SA", "bad_pct"].dropna().values
        ]
        bp = ax.boxplot(boxdata, showfliers=False)
        ax.set_xticks([1,2])
        ax.set_xticklabels(["UQ", "SA"])
        ax.set_ylabel("Invalid fraction (%)")
        ax.set_title("Invalid LVEDP rates")
    else:
        ax.text(0.5, 0.5, "No invalid-rate data", ha="center")

    savefig(fig, "fig_01_uq_invalid.png")

def fig_sobol_overview(df_sa, key_params=("V_tot", "C_sv", "E_min"), metric="LVEDP"):
    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (A) ST distribution across subjects
    ax = axes[0]
    if not df.empty:
        order = (df.groupby("param")["ST"].median()
                 .sort_values(ascending=False).index.tolist())
        data = [df.loc[df["param"] == p, "ST"].dropna().values for p in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_ylabel("Total Sobol index (ST)")
        ax.set_title(f"ST across subjects (metric={metric})")
        ax.set_xticklabels(order, rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, f"No SA data (metric={metric})", ha="center")

    # (B) Subjectwise ST bars for key params
    ax = axes[1]
    sub = df[df["param"].isin(key_params)].copy()
    if not sub.empty:
        pivot = sub.pivot_table(index="sub_id", columns="param", values="ST", aggfunc="mean")
        pivot = pivot.sort_index()
        x = np.arange(len(pivot))
        width = 0.8 / len(key_params)
        for i, p in enumerate(key_params):
            vals = pivot[p].values
            ax.bar(x + i * width, vals, width, label=p)
        ax.set_xticks(x + width * (len(key_params) - 1) / 2)
        ax.set_xticklabels(pivot.index.astype(int))
        ax.set_ylabel("ST")
        ax.set_title(f"Subject-wise ST for key parameters (metric={metric})")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Selected params not found", ha="center")

    savefig(fig, f"fig_02_sobol_overview_{metric.lower()}.png")


def fig_topk_grid(df_sa, k=3, cols=4, metric="LVEDP"):
    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if df.empty:
        print(f"[fig_topk_grid] no SA data for metric={metric}")
        return

    subjects = sorted(df["sub_id"].unique())
    rows = int(np.ceil(len(subjects) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for i, sub_id in enumerate(subjects):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        g = (df[df["sub_id"] == sub_id]
             .sort_values("ST", ascending=False).head(k))
        xs = np.arange(len(g))
        ax.bar(xs, g["ST"].values)
        ax.set_xticks(xs)
        ax.set_xticklabels(g["param"].values, rotation=45, ha="right")
        ax.set_ylim(0, max(1.0, g["ST"].max() * 1.1))
        ax.set_title(f"Sub {int(sub_id)}", fontsize=10)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Top-{k} ST per subject (metric={metric})", y=0.995)
    savefig(fig, f"fig_03_topk_per_subject_{metric.lower()}.png")


def fig_s1_st_distributions(df_sa, metric="LVEDP"):
    if df_sa.empty:
        print("[fig_s1_st_distributions] no SA data")
        return

    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if df.empty:
        print(f"[fig_s1_st_distributions] no SA data for metric={metric}")
        return

    need = {"param", "S1", "ST"}
    if not need.issubset(df.columns):
        print("[fig_s1_st_distributions] missing columns:", need - set(df.columns))
        return

    df["int_mass"] = df["ST"] - df["S1"]

    order = (df.groupby("param")["ST"].median()
               .sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # (A) S1 boxplots
    ax = axes[0]
    data_s1 = [df.loc[df["param"] == p, "S1"].dropna().values for p in order]
    ax.boxplot(data_s1, tick_labels=order, showfliers=False)
    ax.set_title(f"Main effects (S1) across subjects (metric={metric})")
    ax.set_ylabel("S1")
    ax.set_xticklabels(order, rotation=45, ha="right")

    # (B) ST boxplots
    ax = axes[1]
    data_st = [df.loc[df["param"] == p, "ST"].dropna().values for p in order]
    ax.boxplot(data_st, tick_labels=order, showfliers=False)
    ax.set_title("Total effects (ST) across subjects")
    ax.set_ylabel("ST")
    ax.set_xticklabels(order, rotation=45, ha="right")

    # (C) Interaction mass
    ax = axes[2]
    data_int = [df.loc[df["param"] == p, "int_mass"].dropna().values for p in order]
    ax.boxplot(data_int, tick_labels=order, showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Interaction / nonlinearity (ST − S1)")
    ax.set_ylabel("ST − S1")
    ax.set_xticklabels(order, rotation=45, ha="right")

    fig.suptitle(f"Sobol indices across subjects (metric={metric})", y=0.98)
    savefig(fig, f"fig_04_s1_st_distributions_{metric.lower()}.png")


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def plot_pi_width_boxplot(df_summary: pd.DataFrame, out_dir=None, show=True):
    """
    Boxplot of 95% PI width across subjects.
    """
    _ensure_dir(out_dir)
    df = df_summary.copy()
    df["PI_width"] = df["PI95_high"] - df["PI95_low"]

    plt.figure(figsize=(6, 5))
    plt.boxplot(df["PI_width"].values, showfliers=False)
    plt.ylabel("95% PI width of LVEDP (mmHg)")
    plt.title("Uncertainty across subjects (PI width)")
    plt.grid(True, axis="y", alpha=0.3)

    if out_dir:
        plt.savefig(os.path.join(out_dir, "uq_pi_width_boxplot.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
def plot_population_histogram(df_summary: pd.DataFrame, out_dir=None, show=True, bins=15):
    """
    Histogram of subject mean LVEDP with physiological band.
    """
    _ensure_dir(out_dir)
    means = df_summary["mean"].values

    plt.figure(figsize=(7, 5))
    plt.hist(means, bins=bins, edgecolor="black", alpha=0.8)
    plt.axvspan(PHYS_RANGE[0], PHYS_RANGE[1], color="gray", alpha=0.15, label="Phys. LVEDP range")
    plt.axvline(12, linestyle="--", alpha=0.6, label="~Upper normal (≈12 mmHg)")
    plt.axvline(16, linestyle="--", alpha=0.6, label="Often used elevated cutoff (≈16 mmHg)")
    plt.xlabel("Mean LVEDP (mmHg)")
    plt.ylabel("Number of subjects")
    plt.title("Population distribution of mean LVEDP")
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)

    if out_dir:
        plt.savefig(os.path.join(out_dir, "uq_population_hist_mean.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_uq_sa_joint(df_summary: pd.DataFrame, df_sa: pd.DataFrame,
                     out_dir=None, show=True, top_k=4, metric="LVEDP"):
    _ensure_dir(out_dir)
    df = df_sa.copy()
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if df.empty:
        print(f"[plot_uq_sa_joint] no SA data for metric={metric}")
        return

    w = df_summary[["sub_id", "PI95_low", "PI95_high"]].copy()
    w["PI_width"] = w["PI95_high"] - w["PI95_low"]

    rank = (df.groupby("param", as_index=False)["ST"]
            .median()
            .sort_values("ST", ascending=False))
    keep = rank.head(top_k)["param"].tolist()

    n = len(keep)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for i, p in enumerate(keep):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        d = (df[df["param"] == p][["sub_id", "ST"]]
             .merge(w[["sub_id", "PI_width"]], on="sub_id", how="inner"))

        if len(d) == 0:
            ax.set_title(f"{p} (no data)")
            ax.axis("off")
            continue

        ax.scatter(d["ST"], d["PI_width"], alpha=0.75)
        if len(d) >= 2 and np.isfinite(d["ST"]).all() and np.isfinite(d["PI_width"]).all():
            m, b = np.polyfit(d["ST"].values, d["PI_width"].values, deg=1)
            xs = np.linspace(d["ST"].min(), d["ST"].max(), 100)
            ax.plot(xs, m * xs + b, linestyle="--", linewidth=1)

        ax.set_xlabel("Total-effect index ST")
        ax.set_ylabel("95% PI width (mmHg)")
        ax.set_title(f"UQ vs SA driver: {p} (metric={metric})")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle("Uncertainty width vs SA drivers (LVEDP)", y=1.02, fontsize=14)
    plt.tight_layout()

    if out_dir:
        plt.savefig(os.path.join(out_dir, f"uq_sa_joint_scatter_{metric.lower()}.png"),
                    dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df_summary, df_sa, df_fail = load_subject_artifacts(UQSA_ROOT)
    OUT_DIR = os.path.join(UQSA_ROOT, "_figs")

    print("summary shape:", df_summary.shape, "| columns:", list(df_summary.columns))
    print("sobol   shape:", df_sa.shape, "| columns:", list(df_sa.columns))
    print("fail    shape:", df_fail.shape, "| columns:", list(df_fail.columns))

    plot_uq_summary(df_summary)
    plot_invalid_rates(df_fail)

    fig_sobol_overview(df_sa, key_params=("V_tot", "C_sv", "E_min"), metric="LVEDP")
    fig_topk_grid(df_sa, k=3, cols=4, metric="LVEDP")
    fig_s1_st_distributions(df_sa, metric="LVEDP")
    plot_pi_width_boxplot(df_summary, out_dir=OUT_DIR, show=True)
    plot_population_histogram(df_summary, out_dir=OUT_DIR, show=True, bins=15)
    plot_uq_sa_joint(df_summary, df_sa, out_dir=OUT_DIR, show=True, top_k=4, metric="LVEDP")

    plt.show()

