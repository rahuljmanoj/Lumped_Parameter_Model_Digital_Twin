# -*- coding: utf-8 -*-
"""
LVEDP analysis: regressions, cross-validation, ROC & classification metrics,
and Bland–Altman plots (console + figures).

Edits requested:
- Console shows explicit 'Sensitivity' term (in addition to Recall/TPR).
- Console prints full Bland–Altman stats (bias, SD, LoA, CIs, proportional bias).
- Legends in plots are commented out/disabled.
- Figure sizes controlled in cm; Times New Roman styling.

Author: Rahul Manoj (organized)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    r2_score, roc_curve, auc, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score, average_precision_score,
    cohen_kappa_score
)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import pearsonr, ttest_rel, t, linregress


# ============================== CONFIG =======================================

# Paths
FIG_SAVE_DIR = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Figure_Results_Test"
EXCEL_PATH   = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx"
SHEET_NAME   = "Study_9_V4.1_T7"
HEADER_ROW   = 3
NROWS        = 68

# Columns
SIM_COL  = 'sim LVEDP (LPM) (mmHg)'     # Simulated LVEDP
PRED_COL = 'pred LVEDP (UVR) (mmHg)'    # UVR regression-predicted LVEDP
GT_COL   = 'GT LVEDP (mmHg)'            # Ground truth LVEDP
SWE_COL  = 'SWS (m/s)'                  # SWE velocity
X_FEATURES = ['SWS (m/s)', 'GT bMAP (mmHg)']  # features for linear model(s)

# Classification threshold(s)
LVEDP_THRESHOLD = 16.0  # mmHg
# SWE mapping to 16 mmHg: LVEDP ≈ a*SWE + b
A_SWE_TO_LVEDP = 2.4033
B_SWE_TO_LVEDP = 6.3966

# Cross-validation
N_SPLITS = 5
CV_RANDOM_STATE = 42

# Figure sizes (centimeters)
CM = 1.0 / 2.54
FIGSIZE_REG_CM = (8.0, 5.0)
FIGSIZE_BOX_CM = (8.0, 5.0)
FIGSIZE_ROC_CM = (8.0, 5.0)
FIGSIZE_BA_CM  = (8.0, 5.0)

# Matplotlib style (Times New Roman)
FONT_FAMILY = "Times New Roman"
FONT_SIZE   = 9

# Export classification table?
SAVE_CLASSIF_TABLE = True
CLASSIF_TABLE_NAME = "classification_metrics_summary.csv"

# ============================================================================


# ============================ UTILITIES =====================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fig_size_cm(w_cm: float, h_cm: float):
    return (w_cm * CM, h_cm * CM)


def style_axes(ax, xlabel: str, ylabel: str, add_legend: bool = False):
    """
    Times New Roman styling + clean axes.
    NOTE: Legends are disabled by default (add_legend=False).
    """
    ax.set_xlabel(xlabel, fontname=FONT_FAMILY, fontsize=FONT_SIZE, color="black")
    ax.set_ylabel(ylabel, fontname=FONT_FAMILY, fontsize=FONT_SIZE, color="black")
    ax.tick_params(axis='both', labelsize=FONT_SIZE, colors="black")
    for lbl in ax.get_xticklabels():
        lbl.set_fontname(FONT_FAMILY)
    for lbl in ax.get_yticklabels():
        lbl.set_fontname(FONT_FAMILY)
    # Legends intentionally suppressed unless explicitly enabled
    if add_legend:
        leg = ax.legend(fontsize=FONT_SIZE, frameon=False)
        for txt in leg.get_texts():
            txt.set_fontname(FONT_FAMILY)
            txt.set_color("black")
    ax.grid(False)
    ax.set_title("")


def correlation_print(df: pd.DataFrame, cols: list):
    """Print Pearson correlations of selected columns."""
    print("\n=== Pearson Correlations among predictors ===")
    print(df[cols].corr())


def vif_table(X: pd.DataFrame, name: str):
    """Compute VIFs for each column of X (no constant)."""
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    out = pd.DataFrame({'Variable': X.columns, 'VIF': vifs})
    print(f"\nVariance Inflation Factors ({name}):")
    print(out)
    return out


def cv_r2(X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int = 42):
    """Return mean CV R² for a linear regression."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='r2').mean()
    return score


def compute_swe_threshold(lvedp_cut: float, a: float, b: float) -> float:
    return (lvedp_cut - b) / a


# ====================== CLASSIFICATION METRICS ===============================

def classification_report_at_cutoff(y_true_bin: np.ndarray,
                                    scores_cont: np.ndarray,
                                    cutoff: float,
                                    label: str) -> dict:
    """
    Compute and print a comprehensive classification report at a fixed cutoff.
    Works with continuous 'scores_cont' (mmHg/m/s).
    Prints explicit 'Sensitivity' line in addition to Recall/TPR.
    """
    y_true_bin = np.asarray(y_true_bin, int)
    scores_cont = np.asarray(scores_cont, float)

    y_pred_bin = (scores_cont >= cutoff).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    n = tn + fp + fn + tp

    def _safe(num, den): return float(num) / den if den > 0 else np.nan

    sensitivity = _safe(tp, tp + fn)   # recall/TPR
    specificity = _safe(tn, tn + fp)   # TNR
    precision   = _safe(tp, tp + fp)   # PPV
    npv         = _safe(tn, tn + fn)   # NPV
    fpr         = _safe(fp, fp + tn)
    fnr         = 1.0 - sensitivity
    fdr         = _safe(fp, tp + fp)   # 1 - precision
    _for        = _safe(fn, fn + tn)   # "FOR"

    acc     = accuracy_score(y_true_bin, y_pred_bin)
    f1      = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    mcc     = matthews_corrcoef(y_true_bin, y_pred_bin) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else np.nan
    kappa   = cohen_kappa_score(y_true_bin, y_pred_bin)

    # Curves with continuous scores
    fpr_curve, tpr_curve, _ = roc_curve(y_true_bin, scores_cont)
    auroc = auc(fpr_curve, tpr_curve)
    ap    = average_precision_score(y_true_bin, scores_cont)  # PR-AUC

    # Print nicely (with an explicit Sensitivity line)
    print(f"\n{label} @ cutoff = {cutoff:.2f}")
    print(f"  Confusion matrix (tn fp / fn tp): [[{tn:3d} {fp:3d}] / [{fn:3d} {tp:3d}]]")
    print(f"  Accuracy          : {acc:.3f}")
    print(f"  Precision (PPV)   : {precision:.3f}    NPV  : {npv:.3f}")
    print(f"  Sensitivity       : {sensitivity:.3f}")  # <-- explicit term
    print(f"  Recall (TPR)      : {sensitivity:.3f}    Specificity (TNR): {specificity:.3f}")
    print(f"  F1-score          : {f1:.3f}    BalAcc: {bal_acc:.3f}    MCC: {mcc:.3f}    κ: {kappa:.3f}")
    print(f"  FPR               : {fpr:.3f}   FNR: {fnr:.3f}   FDR: {fdr:.3f}   FOR: {_for:.3f}")
    print(f"  AUROC             : {auroc:.3f}  PR-AUC (AP): {ap:.3f}")

    return {
        "model": label, "cutoff": float(cutoff),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(n),
        "accuracy": float(acc), "precision": float(precision), "recall": float(sensitivity),
        "specificity": float(specificity), "f1": float(f1), "balanced_acc": float(bal_acc),
        "npv": float(npv), "mcc": float(mcc), "kappa": float(kappa),
        "fpr": float(fpr), "fnr": float(fnr), "fdr": float(fdr), "for": float(_for),
        "auroc": float(auroc), "pr_auc": float(ap)
    }


# ======================= BLAND–ALTMAN TOOLS ==================================

def _coerce_numeric_pair(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def bland_altman_stats(a, b):
    """
    Bland–Altman stats with 95% LoA and their CIs (Bland & Altman, 1999).
    Returns dict containing arrays + scalars + CI bounds + proportional bias test.
    """
    a, b = _coerce_numeric_pair(a, b)
    mean = (a + b) / 2.0
    diff = a - b

    n = len(diff)
    if n < 3:
        raise ValueError("Not enough paired points for Bland–Altman (need n ≥ 3).")

    bias = np.mean(diff)
    sd   = np.std(diff, ddof=1)
    z    = 1.96
    loa_low  = bias - z * sd
    loa_high = bias + z * sd

    # 95% CIs (Bland & Altman, 1999)
    tcrit  = t.ppf(0.975, df=n-1)
    se_bias = sd / np.sqrt(n)
    ci_bias = (bias - tcrit * se_bias, bias + tcrit * se_bias)

    se_loa = sd * np.sqrt(1.0/n + (z**2)/(2*(n-1)))
    ci_loa_low  = (loa_low  - tcrit * se_loa, loa_low  + tcrit * se_loa)
    ci_loa_high = (loa_high - tcrit * se_loa, loa_high + tcrit * se_loa)

    # Proportional bias: diff ~ mean
    lr = linregress(mean, diff)
    prop_bias = dict(slope=lr.slope, intercept=lr.intercept, pval=lr.pvalue, r=lr.rvalue)

    return dict(
        n=n, mean=mean, diff=diff, bias=bias, sd=sd,
        loa_low=loa_low, loa_high=loa_high,
        ci_bias=ci_bias, ci_loa_low=ci_loa_low, ci_loa_high=ci_loa_high,
        prop_bias=prop_bias
    )


def print_ba_summary(stats_dict: dict, label: str):
    """Nicely print BA stats to console."""
    n       = stats_dict["n"]
    bias    = stats_dict["bias"]
    sd      = stats_dict["sd"]
    loa_low = stats_dict["loa_low"]
    loa_high= stats_dict["loa_high"]
    ci_bias = stats_dict["ci_bias"]
    ci_ll   = stats_dict["ci_loa_low"]
    ci_lh   = stats_dict["ci_loa_high"]
    prop    = stats_dict["prop_bias"]

    print(f"\n=== Bland–Altman Summary: {label} ===")
    print(f"  n           : {n}")
    print(f"  Bias        : {bias:.3f}   (95% CI {ci_bias[0]:.3f} to {ci_bias[1]:.3f})")
    print(f"  SD          : {sd:.3f}")
    print(f"  LoA         : {loa_low:.3f} to {loa_high:.3f}")
    print(f"  LoA 95% CI  : low {ci_ll[0]:.3f} to {ci_ll[1]:.3f} | high {ci_lh[0]:.3f} to {ci_lh[1]:.3f}")
    print(f"  Prop. bias  : slope={prop['slope']:.4f}, intercept={prop['intercept']:.4f}, "
          f"p={prop['pval']:.3g}, r={prop['r']:.3f}")


def plot_bland_altman(stats_dict, title="", x_label="Mean of methods",
                      y_label="Difference (A - B)", save_path=None):
    """BA plot without legend (as requested)."""
    mean = stats_dict["mean"]
    diff = stats_dict["diff"]
    bias = stats_dict["bias"]
    loa_low = stats_dict["loa_low"]
    loa_high = stats_dict["loa_high"]
    ci_bias = stats_dict["ci_bias"]
    ci_loa_low = stats_dict["ci_loa_low"]
    ci_loa_high = stats_dict["ci_loa_high"]

    fig, ax = plt.subplots(figsize=fig_size_cm(*FIGSIZE_BA_CM), dpi=300)
    ax.scatter(mean, diff, s=12, alpha=0.8, edgecolor='none')

    ax.axhline(bias, color='k', linestyle='-', linewidth=1.0)
    ax.axhline(loa_low, color='k', linestyle='--', linewidth=0.9)
    ax.axhline(loa_high, color='k', linestyle='--', linewidth=0.9)

    ax.fill_between([np.min(mean), np.max(mean)], ci_bias[0], ci_bias[1], alpha=0.12)
    ax.fill_between([np.min(mean), np.max(mean)], ci_loa_low[0], ci_loa_low[1], alpha=0.08)
    ax.fill_between([np.min(mean), np.max(mean)], ci_loa_high[0], ci_loa_high[1], alpha=0.08)

    style_axes(ax, x_label, y_label, add_legend=False)   # legend disabled
    ax.set_xlim(5, 40)
    ax.set_ylim(-20, 20)
    ax.set_xticks(np.arange(5, 41, 5))
    ax.set_yticks(np.arange(-20, 21, 5))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# =============================== MAIN ========================================

def main():
    ensure_dir(FIG_SAVE_DIR)

    # ----- Load data -----
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW, nrows=NROWS)

    # For modeling with X features (raw + standardized)
    X = df[X_FEATURES].copy()
    y = df[GT_COL].copy()

    # Drop NaNs for modeling subset
    model_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X.loc[model_mask].copy()
    y = y.loc[model_mask].copy()

    # Standardize X
    X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

    # ----- Basic linear regression & OLS -----
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print("\n=== Linear model (raw) ===")
    correlation_print(df, X_FEATURES)
    print(f"Raw-data R²: {r2_score(y, y_pred):.3f}")

    # OLS on raw X (with constant)
    Xs = sm.add_constant(X)
    ols = sm.OLS(y, Xs).fit()
    print("\nOLS results (raw features):")
    print(ols.summary())

    # ----- VIF -----
    _ = vif_table(X, name="raw X")
    _ = vif_table(X_std, name="standardized X")

    # ----- Cross-validation R² -----
    cv_r2_val = cv_r2(X, y, n_splits=N_SPLITS, random_state=CV_RANDOM_STATE)
    print(f"\n{len(X_FEATURES)}-var CV R² (LinearRegression, {N_SPLITS}-fold): {cv_r2_val:.3f}")

    # ----- Prepare arrays for downstream analyses -----
    required_cols = [SIM_COL, PRED_COL, GT_COL, SWE_COL]
    df2 = df.dropna(subset=required_cols).copy()

    sim_lvedp  = df2[SIM_COL].to_numpy(float)
    pred_lvedp = df2[PRED_COL].to_numpy(float)
    gt_lvedp   = df2[GT_COL].to_numpy(float)
    swe_vals   = df2[SWE_COL].to_numpy(float)

    # ----- Paired t-test: GT vs simulated LVEDP -----
    t_stat, p_val = ttest_rel(gt_lvedp, sim_lvedp)
    print("\n=== Paired t-test: GT LVEDP vs Sim LVEDP (LPM) ===")
    print(f"  n           = {len(gt_lvedp)}")
    print(f"  mean(GT)    = {np.mean(gt_lvedp):.2f} mmHg")
    print(f"  mean(Sim)   = {np.mean(sim_lvedp):.2f} mmHg")
    print(f"  t statistic = {t_stat:.3f}")
    print(f"  p value     = {p_val:.3g}")

    # ----- Regression plot: GT vs Sim -----
    x = gt_lvedp.reshape(-1, 1)
    y_sim = sim_lvedp
    reg = LinearRegression().fit(x, y_sim)
    y_fit = reg.predict(x)

    fig, ax = plt.subplots(figsize=fig_size_cm(*FIGSIZE_REG_CM), dpi=300)
    ax.scatter(gt_lvedp, sim_lvedp, s=10, marker='*', color='orange')
    x_line = np.linspace(gt_lvedp.min(), gt_lvedp.max(), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)
    ax.plot(x_line, y_line, linestyle='--', linewidth=0.75, color='black')
    ax.set_xlim(5, 40); ax.set_ylim(5, 40)
    ax.set_xticks(np.arange(5, 41, 5)); ax.set_yticks(np.arange(5, 41, 5))
    style_axes(ax, "GT LVEDP [mmHg]", "LPM-derived LVEDP [mmHg]", add_legend=False)  # legend disabled
    reg_path = os.path.join(FIG_SAVE_DIR, "Fig1_regression_GT_vs_Sim_LVEDP.png")
    fig.savefig(reg_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {reg_path}")
    print(f"Regression (GT→Sim): slope={reg.coef_[0]:.3f}, intercept={reg.intercept_:.3f}, R²={r2_score(y_sim, y_fit):.3f}")

    # ----- Boxplots -----
    # GT vs Sim
    fig, ax = plt.subplots(figsize=fig_size_cm(*FIGSIZE_BOX_CM), dpi=300)
    ax.boxplot([gt_lvedp, sim_lvedp], labels=["GT", "LPM"])
    style_axes(ax, "", "LVEDP [mmHg]", add_legend=False)  # legend disabled
    ax.set_ylim(5, 40); ax.set_yticks(np.arange(5, 41, 5))
    box1_path = os.path.join(FIG_SAVE_DIR, "Fig2_box_GT_vs_Sim_LVEDP.png")
    fig.savefig(box1_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {box1_path}")

    # GT vs Sim vs UVR
    fig, ax = plt.subplots(figsize=fig_size_cm(*FIGSIZE_BOX_CM), dpi=300)
    ax.boxplot([gt_lvedp, sim_lvedp, pred_lvedp], labels=["GT", "LPM", "UVR"])
    style_axes(ax, "", "LVEDP [mmHg]", add_legend=False)  # legend disabled
    ax.set_ylim(5, 40); ax.set_yticks(np.arange(5, 41, 5))
    box2_path = os.path.join(FIG_SAVE_DIR, "Fig3_box_GT_vs_Sim_vs_UVR_LVEDP.png")
    fig.savefig(box2_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {box2_path}")

    # ----- Correlations vs GT -----
    print("\n=== Correlations vs GT LVEDP ===")
    for name, arr in [(SIM_COL, sim_lvedp), (PRED_COL, pred_lvedp), ("SWE Velocity", swe_vals)]:
        r, p = pearsonr(arr, gt_lvedp)
        print(f"{name:>24}: r = {r:.2f}, p = {p:.3g}")

    # ----- ROC & AUC + Full Classification Metrics at fixed cutoffs -----
    gt_elevated = (gt_lvedp >= LVEDP_THRESHOLD).astype(int)
    swe_thresh = compute_swe_threshold(LVEDP_THRESHOLD, A_SWE_TO_LVEDP, B_SWE_TO_LVEDP)
    print(f"\nSWE threshold for LVEDP={LVEDP_THRESHOLD:.0f} mmHg: {swe_thresh:.2f} m/s")

    # Build ROC data for plots (AUROC) and print Youden info
    roc_data = []
    for name, scores in [(SIM_COL, sim_lvedp), (PRED_COL, pred_lvedp), ("SWE Velocity", swe_vals)]:
        fpr, tpr, thr = roc_curve(gt_elevated, scores)
        auc_score = auc(fpr, tpr)
        j_scores = tpr - fpr
        ix = np.argmax(j_scores)
        best_thr = thr[ix]
        roc_data.append((name, fpr, tpr, auc_score))

        print(f"\n{name} ROC:")
        print(f"  AUROC        : {auc_score:.3f}")
        print(f"  Youden J max : {j_scores[ix]:.3f} at threshold={best_thr:.3f}")

    # Classification metrics table at fixed cutoffs (16 mmHg / SWE-threshold)
    metrics_rows = []
    for name, scores in [(SIM_COL, sim_lvedp), (PRED_COL, pred_lvedp), ("SWE Velocity", swe_vals)]:
        cutoff = swe_thresh if name == "SWE Velocity" else LVEDP_THRESHOLD
        row = classification_report_at_cutoff(gt_elevated, scores, cutoff, name)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    print("\n===== Classification summary (fixed cutoffs) =====")
    print(metrics_df[[
        "model","cutoff","accuracy","precision","recall","specificity",
        "f1","balanced_acc","npv","mcc","kappa","auroc","pr_auc"
    ]].to_string(index=False))

    if SAVE_CLASSIF_TABLE:
        out_csv = os.path.join(FIG_SAVE_DIR, CLASSIF_TABLE_NAME)
        metrics_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    # ----- Plot ROC curves (no legends) -----
    # Combined (Sim + UVR)
    fig, ax = plt.subplots(figsize=fig_size_cm(*FIGSIZE_ROC_CM), dpi=300)
    for name, fpr, tpr, auc_score in roc_data:
        if name in (SIM_COL, PRED_COL):
            ax.plot(fpr, tpr, linewidth=1.0)  # label suppressed; legend disabled
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    style_axes(ax, "1 - Specificity", "Sensitivity", add_legend=False)  # legend disabled
    roc_comb_path = os.path.join(FIG_SAVE_DIR, "Fig4_ROC_Sim_plus_UVR_LVEDP.png")
    fig.savefig(roc_comb_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {roc_comb_path}")

    # Sim only
    fig, ax = plt.subplots(figsize=fig_size_cm(*FIGSIZE_ROC_CM), dpi=300)
    for name, fpr, tpr, auc_score in roc_data:
        if name == SIM_COL:
            ax.plot(fpr, tpr, linewidth=1.0)  # label suppressed; legend disabled
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    style_axes(ax, "1 - Specificity", "Sensitivity", add_legend=False)  # legend disabled
    roc_sim_path = os.path.join(FIG_SAVE_DIR, "Fig5_ROC_Sim_LVEDP_only.png")
    fig.savefig(roc_sim_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {roc_sim_path}")

    # ----- Bland–Altman plots + console summaries -----
    # (A) UVR vs GT
    ba_uvr = bland_altman_stats(pred_lvedp, gt_lvedp)
    print_ba_summary(ba_uvr, label="UVR vs GT")
    ba_uvr_path = os.path.join(FIG_SAVE_DIR, "Fig6_BA_UVR_vs_GT.png")
    plot_bland_altman(ba_uvr, title="UVR vs GT", x_label="", y_label="", save_path=ba_uvr_path)

    # (B) Sim vs GT
    ba_sim = bland_altman_stats(sim_lvedp, gt_lvedp)
    print_ba_summary(ba_sim, label="Sim vs GT")
    ba_sim_path = os.path.join(FIG_SAVE_DIR, "Fig7_BA_Sim_vs_GT.png")
    plot_bland_altman(ba_sim, title="Sim vs GT", x_label="", y_label="", save_path=ba_sim_path)


if __name__ == "__main__":
    main()