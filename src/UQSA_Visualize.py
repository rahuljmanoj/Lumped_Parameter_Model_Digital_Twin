# -*- coding: utf-8 -*-
"""
LPM_V4.1_UQSA_Visualize.py  (figure sizes in cm, combined S1/ST, fixed y-limits, TNR labels)

Creates:
  1) LVEDP histogram with mean & 95% PI (styled; adjustable bin width),
  2) Sobol bar charts with S1 and ST combined in a single axes (no CI; fixed y-cap; Times New Roman y-labels),
  3) Scatter LVEDP vs each parameter (linear regression, R², Pearson ρ).

Reads files written by your V4.1 UQSA script:
  - <SUB_ID>_uq_lvedp.csv | _uq_samples.csv
  - <SUB_ID>_uq_theta.csv | _uq_params.csv
  - <SUB_ID>_uq_summary.json
  - <SUB_ID>_sobol_lvedp.csv, <SUB_ID>_sobol_validity.csv
  - <SUB_ID>_uqsa_conditions.csv (for parameter order)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ====================== USER CONFIG ======================
UQSA_ROOT = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Results_999_UQSA_LPM_4.1_without_regression"   # <-- set to your SAVE_ROOT
SUB_ID    = 999
PHYS_RANGE = (4.0, 35.0)       # mmHg
MAX_SCATTER = 5000            # downsample cap for scatter

# ----------- FIGURE SIZES (in centimeters) -----------
CM = 1.0 / 2.54  # cm -> inch

# Histogram figure size (W_cm, H_cm)
FIGSIZE_HIST_CM   = (16, 7.0)       # e.g., 24×12 cm

# Sobol (combined S1+ST on one axes) figure size
FIGSIZE_SOBOL_CM  = (16, 7.0)      # e.g., 31×12 cm

# Scatter panel: either explicit figure size or per-subplot size × grid
# Option A: fixed figure size (comment out Option B below if you prefer this)
FIGSIZE_SCATTER_CM = (29.7, 15.0)      # A4 landscape-ish

# Option B (alternative): per-subplot size in cm (used only if FIGSIZE_SCATTER_CM=None)
SUBPLOT_W_CM, SUBPLOT_H_CM = 8.0, 6.5  # each small subplot size

# Histogram binning:
# - If HIST_BIN_WIDTH_MM is not None, it is used (larger => wider bars)
# - Else if HIST_NBINS is not None, that exact number of bins is used
# - Else Freedman–Diaconis
HIST_BIN_WIDTH_MM = 0.30    # e.g., 0.10–0.20 mmHg to spread bars; set None to disable
HIST_NBINS        = None    # or e.g., 100; keep None to use bin width or FD rule

# Sobol (combined chart) options
SOBOL_AS_PERCENT = False    # show S1/ST as 0–100%
SOBOL_YMAX       = 0.70     # axis cap in index units (0..1). If SOBOL_AS_PERCENT=True, this means 80%
BAR_ALPHA        = 0.88
COLOR_S1         = "#2ca02c"
COLOR_ST         = "#ff7f0e"
BAR_WIDTH        = 0.38     # width for each bar; S1 left, ST right

# ----- Fonts -----
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10, "axes.labelsize":10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

# Sobol x‑axis label styling
X_TICK_ROT_DEG = 0          # 0 = straight; use 45 if you ever want angled again
X_TICK_HA      = "center"   # horizontal alignment: "center" | "right" | "left"
X_TICK_PAD     = 4          # px padding between labels and axis

# ---------- OUTPUT DIR ----------
OUTDIR = os.path.join(UQSA_ROOT, str(SUB_ID), "_viz_subject")
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================

# Pretty labels
PLABEL = {
    "R_sys": r"$R_{\mathrm{sys}}$", "Z_ao": r"$Z_{\mathrm{ao}}$", "C_sa": r"$C_{\mathrm{sa}}$",
    "R_mv": r"$R_{\mathrm{mv}}$", "E_max": r"$E_{\max}$", "E_min": r"$E_{\min}$",
    "t_peak": r"$t_{\mathrm{peak}}$", "V_tot": r"$V_{\mathrm{tot}}$", "C_sv": r"$C_{\mathrm{sv}}$",
}
XLABELS = {
    "R_sys": r"$R_{\mathrm{sys}}$ (mmHg·s/ml)",
    "Z_ao":  r"$Z_{\mathrm{ao}}$ (mmHg·s/ml)",
    "C_sa":  r"$C_{\mathrm{sa}}$ (ml/mmHg)",
    "R_mv":  r"$R_{\mathrm{mv}}$ (mmHg·s/ml)",
    "E_max": r"$E_{\max}$ (mmHg/ml)",
    "E_min": r"$E_{\min}$ (mmHg/ml)",
    "t_peak":r"$t_{\mathrm{peak}}$ (s)",
    "V_tot": r"$V_{\mathrm{tot}}$ (ml)",
    "C_sv":  r"$C_{\mathrm{sv}}$ (ml/mmHg)",
}

# ---------- LOAD ----------
sub_dir = os.path.join(UQSA_ROOT, str(SUB_ID))
# UQ outputs (valid-only)
f_y_new = os.path.join(sub_dir, f"{SUB_ID}_uq_lvedp.csv")
f_y_old = os.path.join(sub_dir, f"{SUB_ID}_uq_samples.csv")
f_x_new = os.path.join(sub_dir, f"{SUB_ID}_uq_theta.csv")
f_x_old = os.path.join(sub_dir, f"{SUB_ID}_uq_params.csv")
f_sum   = os.path.join(sub_dir, f"{SUB_ID}_uq_summary.json")
# Sobol outputs (written by producer)
f_sa_l  = os.path.join(sub_dir, f"{SUB_ID}_sobol_lvedp.csv")
f_sa_v  = os.path.join(sub_dir, f"{SUB_ID}_sobol_validity.csv")
# Conditions for parameter order
f_cond  = os.path.join(sub_dir, f"{SUB_ID}_uqsa_conditions.csv")

# UQ samples
if   os.path.isfile(f_y_new): df_y = pd.read_csv(f_y_new)
elif os.path.isfile(f_y_old): df_y = pd.read_csv(f_y_old)
else: raise FileNotFoundError("No LVEDP UQ samples found in subject folder.")

if   os.path.isfile(f_x_new): df_x = pd.read_csv(f_x_new)
elif os.path.isfile(f_x_old): df_x = pd.read_csv(f_x_old)
else: raise FileNotFoundError("No UQ params/theta found in subject folder.")

# parameter order (optional)
param_order = list(df_x.columns)
if os.path.isfile(f_cond):
    try:
        dcond = pd.read_csv(f_cond)
        if "param" in dcond.columns:
            param_order = dcond["param"].tolist()
    except Exception:
        pass

# Sobol CSVs
df_sa_l = pd.read_csv(f_sa_l) if os.path.isfile(f_sa_l) else None
df_sa_v = pd.read_csv(f_sa_v) if os.path.isfile(f_sa_v) else None

# ---------- 1) LVEDP histogram ----------
y = df_y["LVEDP"].dropna().values
if os.path.isfile(f_sum):
    try:
        with open(f_sum, "r") as fp: s = json.load(fp)
        mean = float(s.get("mean", np.mean(y)))
        sd   = float(s.get("sd",   np.std(y, ddof=1)))
        pi   = s.get("PI95", [np.percentile(y,2.5), np.percentile(y,97.5)])
        lo, hi = float(pi[0]), float(pi[1])
    except Exception:
        mean, sd = float(np.mean(y)), float(np.std(y, ddof=1))
        lo, hi = np.percentile(y, [2.5,97.5])
else:
    mean, sd = float(np.mean(y)), float(np.std(y, ddof=1))
    lo, hi = np.percentile(y, [2.5,97.5])

fig_histsize = (FIGSIZE_HIST_CM[0]*CM, FIGSIZE_HIST_CM[1]*CM)
fig, ax = plt.subplots(figsize=fig_histsize)

# Compute bins
if HIST_BIN_WIDTH_MM is not None:
    pad = 0.02*(y.max()-y.min() if y.max()>y.min() else 1.0)
    bins = np.arange(y.min()-pad, y.max()+pad + HIST_BIN_WIDTH_MM, HIST_BIN_WIDTH_MM)
elif HIST_NBINS is not None:
    bins = int(HIST_NBINS)
else:
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    bin_w = 2*iqr*(len(y)**(-1/3)) if iqr>0 else (np.std(y)*3.49*(len(y)**(-1/3)))
    bins = max(30, int((y.max()-y.min())/bin_w)) if bin_w>0 else max(30, int(np.sqrt(len(y))))

ax.hist(y, bins=bins, color="#5B6770", edgecolor="black", alpha=0.88)
ax.axvspan(PHYS_RANGE[0], PHYS_RANGE[1], color="lightgrey", alpha=0.45, label="LVEDP Physiological Bound")
ax.axvline(mean, color="#d62728", ls="--", lw=1.6, label=f"Mean = {mean:.1f} mmHg")
ax.axvline(lo,   color="#1f77b4", ls=":",  lw=1.6, label=f"95% PI = {lo:.1f} mmHg")
ax.axvline(hi,   color="#1f77b4", ls=":",  lw=1.6, label=f"95% PI = {hi:.1f} mmHg")
ax.set_xlabel("LVEDP [mmHg]"); ax.set_ylabel("Number of samples")
#ax.set_title(f"UQ: LVEDP distribution (sub {SUB_ID})")
ax.legend(ncol=1, frameon=True, framealpha=0.95)
ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
fig.tight_layout()
out_hist = os.path.join(OUTDIR, f"{SUB_ID}_uq_lvedp_hist.png")
fig.savefig(out_hist, dpi=300); print(f"Saved -> {out_hist}")

# ---------- helpers ----------
def _to_num(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=s.index)

def _plot_sobol_combined(df_sa, title, fname):
    """Single axes with side-by-side S1 & ST bars per parameter; no CI bars; fixed y-limit; straight x labels."""
    if df_sa is None or df_sa.empty:
        print(f"[info] No SA data to plot for {title}")
        return

    # Enforce parameter order if present
    if "param" in df_sa.columns:
        df_sa = df_sa.set_index("param")
        keep = [p for p in param_order if p in df_sa.index]
        df_sa = df_sa.loc[keep].reset_index()
    else:
        df_sa = df_sa.copy()

    scale = 100.0 if SOBOL_AS_PERCENT else 1.0
    ylab  = "Sobol index (%)" if SOBOL_AS_PERCENT else "Sobol index"

    params = df_sa["param"].tolist() if "param" in df_sa.columns else [f"p{i}" for i in range(len(df_sa))]
    S1 = _to_num(df_sa.get("S1")).values * scale
    ST = _to_num(df_sa.get("ST")).values * scale

    # Combined bars
    x = np.arange(len(params))
    fig_sobolsize = (FIGSIZE_SOBOL_CM[0]*CM, FIGSIZE_SOBOL_CM[1]*CM)
    fig, ax = plt.subplots(figsize=fig_sobolsize)
    ax.bar(x - BAR_WIDTH/2, S1, width=BAR_WIDTH, color=COLOR_S1, alpha=BAR_ALPHA, label="S1")
    ax.bar(x + BAR_WIDTH/2, ST, width=BAR_WIDTH, color=COLOR_ST, alpha=BAR_ALPHA, label="ST")

    # Straight, centered x‑tick labels in Times New Roman
    ax.set_xticks(x)
    tick_texts = [PLABEL.get(p, p) for p in params]
    ax.set_xticklabels(tick_texts)  # set first, then style
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(X_TICK_ROT_DEG)          # <-- 0° = straight
        lbl.set_horizontalalignment(X_TICK_HA)    # <-- center
        lbl.set_fontfamily("Times New Roman")     # explicit TNR
    ax.tick_params(axis="x", pad=X_TICK_PAD)

    # Y‑axis label + ticks in Times New Roman
    ax.set_ylabel(ylab, fontfamily="Times New Roman")
    for lab in ax.get_yticklabels():
        lab.set_fontfamily("Times New Roman")

    # Fixed y-limit with gentle headroom if exceeded
    cap = SOBOL_YMAX * (100.0 if SOBOL_AS_PERCENT else 1.0)
    max_val = np.nanmax([np.nanmax(S1), np.nanmax(ST)])
    if np.isfinite(max_val) and max_val > cap:
        cap = min((100.0 if SOBOL_AS_PERCENT else 1.0), max_val * 1.05)
    ax.set_ylim(0, cap)

    #ax.grid(axis="y", alpha=0.25)
    #ax.set_title(title)
    ax.legend(ncol=2, frameon=False)

    # Slightly increase bottom margin since labels are now horizontal but centered
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    outp = os.path.join(OUTDIR, fname)
    fig.savefig(outp, dpi=300)
    print(f"Saved -> {outp}")

# ---------- 2) Sobol bar charts (combined S1+ST) ----------
_plot_sobol_combined(df_sa_l, f"Sobol – LVEDP (sub {SUB_ID})", f"{SUB_ID}_sa_lvedp_combined.png")
_plot_sobol_combined(df_sa_v, f"Sobol – VALIDITY (sub {SUB_ID})", f"{SUB_ID}_sa_validity_combined.png")

# ---------- 3) Scatter: LVEDP vs each parameter ----------
# Downsample for very large N
if len(df_x) > MAX_SCATTER:
    idx = np.random.default_rng(0).choice(len(df_x), size=MAX_SCATTER, replace=False)
    X = df_x.iloc[idx].reset_index(drop=True)
    Y = df_y.iloc[idx].reset_index(drop=True)
else:
    X, Y = df_x.copy(), df_y.copy()

params = [p for p in param_order if p in X.columns]

# Figure size for scatter
if FIGSIZE_SCATTER_CM is not None:
    fig_sc_size = (FIGSIZE_SCATTER_CM[0]*CM, FIGSIZE_SCATTER_CM[1]*CM)
else:
    # Compute from per-subplot size & grid
    n_cols = 3
    n_rows = int(np.ceil(len(params)/n_cols))
    W = n_cols * SUBPLOT_W_CM
    H = n_rows * SUBPLOT_H_CM
    fig_sc_size = (W*CM, H*CM)

# If explicit size is set, choose grid to fit
n_cols = 3
n_rows = int(np.ceil(len(params)/n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_sc_size)
axes = axes.flatten()

for i, p in enumerate(params):
    ax = axes[i]
    xv = X[p].values.astype(float)
    yv = Y["LVEDP"].values.astype(float)
    # linear regression
    mlin, blin = np.polyfit(xv, yv, 1)
    r = np.corrcoef(xv, yv)[0,1]; r2 = r*r
    ax.scatter(xv, yv, s=6, alpha=0.55, edgecolors="none", color="dimgray")
    xp = np.linspace(np.min(xv), np.max(xv), 120)
    ax.plot(xp, mlin*xp + blin, color="#d62728", lw=1.5)
    ax.set_xlabel(XLABELS.get(p, p)); ax.set_ylabel("LVEDP [mmHg]")
    ax.set_title(f"{PLABEL.get(p,p)}  (R²={r2:.2f}, ρ={r:.2f})")
    ax.grid(alpha=0.2)

# remove extra axes if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f"LVEDP vs Parameters (sub {SUB_ID})", y=0.98)
fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
out_sc = os.path.join(OUTDIR, f"{SUB_ID}_scatter_LVEDP_vs_params.png")
fig.savefig(out_sc, dpi=300); print(f"Saved -> {out_sc}")
plt.show()