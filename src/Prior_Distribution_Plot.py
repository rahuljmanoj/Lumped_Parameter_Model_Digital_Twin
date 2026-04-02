# -*- coding: utf-8 -*-
"""
LPM_V4.1_Prior_Distribution_Plot.py
Priors + Selected UQ Window + UQ samples (rug) per parameter.

Legend now includes:
  - Orange/beige area  : prior ~95% band (μ ± 2σ)
  - Blue area          : selected UQ window (±10% ∩ hard bounds)
  - Black curve        : prior PDF
  - Red dashed line    : prior mean
  - Black '|'          : UQ samples (valid-only)

Robust file reading for:
  - <SUB_ID>_uqsa_window_theta.csv  (preferred)
  - <SUB_ID>_uqsa_conditions.csv    (fallback with theta_low/high or hard bounds)
  - <SUB_ID>_uq_theta.csv | _uq_params.csv (UQ samples)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

# ------------------ USER CONFIG ------------------
UQSA_ROOT = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Results_999_UQSA_LPM_4.1_without_regression"   # <-- set to your SAVE_ROOT
SUB_ID    = 999

# Figure size in centimeters (cm)
CM = 1.0 / 2.54
FIGSIZE_PRIORS_CM = (20, 16)   # e.g., A4 landscape-ish

# Show shaded UQ window (blue) in each panel
SHOW_WINDOW = True

# Normal prior CVs (edit if needed)
PRIORS_CV = {
    "R_sys": 0.12, "Z_ao": 0.12, "C_sa": 0.15, "R_mv": 0.15,
    "E_max": 0.15, "E_min": 0.15, "t_peak": 0.10, "V_tot": 0.12, "C_sv": 0.15
}

# Labels
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

# ------------------ LOAD ARTIFACTS ------------------
sub_dir  = os.path.join(UQSA_ROOT, str(SUB_ID))
f_cond   = os.path.join(sub_dir, f"{SUB_ID}_uqsa_conditions.csv")
f_win_th = os.path.join(sub_dir, f"{SUB_ID}_uqsa_window_theta.csv")
f_uq_par = os.path.join(sub_dir, f"{SUB_ID}_uq_params.csv")
f_uq_th  = os.path.join(sub_dir, f"{SUB_ID}_uq_theta.csv")

if not os.path.isfile(f_cond):
    raise FileNotFoundError(f"Missing {f_cond} (written by the UQSA script).")

dfc = pd.read_csv(f_cond)  # has param, best_fit, hard_lower/upper, theta_low/high, ...
params = dfc["param"].tolist()
best   = dict(zip(dfc["param"], dfc["best_fit"]))

# Support both 'hard_lower/upper' and 'lower/upper'
lower_col = "hard_lower" if "hard_lower" in dfc.columns else ("lower" if "lower" in dfc.columns else None)
upper_col = "hard_upper" if "hard_upper" in dfc.columns else ("upper" if "upper" in dfc.columns else None)
hard_l = dict(zip(dfc["param"], dfc[lower_col])) if lower_col else {}
hard_u = dict(zip(dfc["param"], dfc[upper_col])) if upper_col else {}

# Window (prefer dedicated CSV; else use columns in conditions; else compute ±10% ∩ hard bounds)
if os.path.isfile(f_win_th):
    dww = pd.read_csv(f_win_th)  # param, theta_low, theta_high
    theta_low  = dict(zip(dww["param"], dww["theta_low"]))
    theta_high = dict(zip(dww["param"], dww["theta_high"]))
    print("[info] Using window from uqsa_window_theta.csv")
elif "theta_low" in dfc.columns and "theta_high" in dfc.columns:
    theta_low  = dict(zip(dfc["param"], dfc["theta_low"]))
    theta_high = dict(zip(dfc["param"], dfc["theta_high"]))
    print("[info] Using window from uqsa_conditions.csv")
else:
    print("[info] No window file/columns found; computing ±10% from best_fit ∩ hard bounds.")
    theta_low, theta_high = {}, {}
    for p in params:
        b = float(best[p])
        lo, hi = 0.9*b, 1.1*b
        if p in hard_l: lo = max(lo, float(hard_l[p]))
        if p in hard_u: hi = min(hi, float(hard_u[p]))
        theta_low[p], theta_high[p] = lo, hi

# UQ samples (valid-only)
df_uq = None
if   os.path.isfile(f_uq_th):  df_uq = pd.read_csv(f_uq_th)
elif os.path.isfile(f_uq_par): df_uq = pd.read_csv(f_uq_par)

# ------------------ STYLE ------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

# Layout
n_cols = 3
n_rows = int(np.ceil(len(params) / n_cols))
figsize = (FIGSIZE_PRIORS_CM[0]*CM, FIGSIZE_PRIORS_CM[1]*CM)
fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
axes = axes.flatten()

# Colors for legend handles (consistent across panels)
COLOR_BAND  = "#E6CBA8"       # beige/orange ~95% band
ALPHA_BAND  = 0.40
COLOR_WIN   = "#4C78A8"       # blue-ish (or use "tab:blue")
ALPHA_WIN   = 0.15
COLOR_PDF   = "tab:blue"
COLOR_MEAN  = "#d62728"

# Collect legend handles just once (consistent across panels)
band_handle   = Patch(facecolor=COLOR_BAND, alpha=ALPHA_BAND, label=r"≈95% prior band ($\mu\pm2\sigma$)")
window_handle = Patch(facecolor=COLOR_WIN,  alpha=ALPHA_WIN,  label="Selected UQ window (±10% ∩ bounds)")
pdf_handle    = Line2D([0], [0], color=COLOR_PDF, lw=1.8, label="Prior PDF")
mean_handle   = Line2D([0], [0], color=COLOR_MEAN, lw=1.2, ls="--", label="prior mean")
rug_handle    = Line2D([0], [0], color='black', marker='|', linestyle='None', markersize=8, label="UQ samples")

legend_handles = [band_handle, pdf_handle, mean_handle, rug_handle]
if SHOW_WINDOW:
    legend_handles.insert(1, window_handle)  # put window right after 95% band

# ------------------ DRAW PANELS ------------------
X_POINTS = 600

for i, p in enumerate(params):
    ax = axes[i]
    mu = float(best[p])
    cv = PRIORS_CV.get(p, 0.15)
    sigma = cv*abs(mu) if mu != 0 else cv

    # x-range; clamp to positive where appropriate
    x_min = mu - 4*sigma
    x_max = mu + 4*sigma
    if p in ["R_sys","Z_ao","C_sa","R_mv","E_max","E_min","V_tot","C_sv","t_peak"]:
        x_min = max(x_min, 0.0)

    x = np.linspace(x_min, x_max, X_POINTS)
    y = norm.pdf(x, loc=mu, scale=sigma)

    # --- shaded prior 95% band (orange/beige) ---
    lo95, hi95 = mu - 2*sigma, mu + 2*sigma
    mask = (x >= lo95) & (x <= hi95)
    ax.fill_between(x[mask], 0, y[mask], color=COLOR_BAND, alpha=ALPHA_BAND)

    # --- optional selected UQ window (blue) ---
    if SHOW_WINDOW and p in theta_low and p in theta_high:
        lo_w, hi_w = float(theta_low[p]), float(theta_high[p])
        # draw a rectangle via fill_between using top of current PDF for appearance
        mask_w = (x >= lo_w) & (x <= hi_w)
        if mask_w.any():
            ax.fill_between(x[mask_w], 0, np.interp(x[mask_w], x, y), color=COLOR_WIN, alpha=ALPHA_WIN)

    # prior PDF + mean
    ax.plot(x, y, color=COLOR_PDF, lw=1.8)
    ax.axvline(mu, color=COLOR_MEAN, ls="--", lw=1.2)

    # UQ samples as black rug
    if df_uq is not None and p in df_uq.columns:
        s = df_uq[p].dropna().values.astype(float)
        if s.size > 0:
            y_max = y.max()
            rug_y = -0.06*y_max*np.ones_like(s)
            ax.plot(s, rug_y, '|', color='black', alpha=0.6, markersize=7)
            ax.set_ylim(bottom=-0.10*y_max)

    ax.set_xlabel(XLABELS.get(p, p)); ax.set_ylabel("PDF")
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))

# remove extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Common legend at bottom, one line
fig.legend(legend_handles, [h.get_label() for h in legend_handles],
           loc="lower center", ncol=len(legend_handles), bbox_to_anchor=(0.5, 0.02))

fig.suptitle(f"Priors – Selected UQ Window – UQ Samples (Subject {SUB_ID})", y=0.98)
fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.95])
out_path = os.path.join(sub_dir, "_viz_subject", f"{SUB_ID}_priors_panel.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300)
print(f"Saved -> {out_path}")
plt.show()