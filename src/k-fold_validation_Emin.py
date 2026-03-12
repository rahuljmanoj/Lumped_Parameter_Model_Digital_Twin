
# Cross-validated monotone calibration of mechanistic Emin to pig-derived prior.
# Plots: 8 cm x 5 cm, Times New Roman size 9. Affine removed from plots (kept for metrics & deployment).

import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FormatStrFormatter

# --------- DEFAULT PATH TO YOUR EXCEL (Windows absolute path) ----------
DEFAULT_INPUT = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Emin_Calculation.xlsx"
# ----------------------------------------------------------------------

# --------------------------- PLOTTING STYLE ---------------------------
CM_TO_IN = 1 / 2.54
FIG_W = 8 * CM_TO_IN   # 8 cm
FIG_H = 6 * CM_TO_IN   # 5 cm

def set_plot_style(font_family="Times New Roman", font_size=9):
    """
    Apply global Matplotlib style: Times New Roman, size 9, small lines, tight layout.
    If Times New Roman is not available on the system, matplotlib will fall back.
    """
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.grid": False
    })

# --------------------------- UTILITIES --------------------------------
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the columns found: {candidates}")

def pos_slope_affine_fit(x, y):
    """
    Fit y ≈ a + b*x with b >= 0. If unconstrained slope < 0, set b=0 and a=mean(y).
    """
    x = x.reshape(-1,1)
    lr = LinearRegression().fit(x, y)
    b = lr.coef_[0]; a = lr.intercept_
    if b < 0:
        b = 0.0
        a = float(np.mean(y))
    return a, b

def apply_affine(a, b, x):
    return a + b*x

def reliability_bins(y_true, y_pred, n_bins=10):
    """
    Bin-wise calibration summary:
      - Sort by y_pred
      - Split into n_bins equal-sized contiguous bins
      - Return (x_bin_mean, y_bin_mean)
    If n_bins <= 1, returns None, None to indicate 'skip' (use all-points scatter instead).
    """
    if n_bins is None or n_bins <= 1:
        return None, None
    order = np.argsort(y_pred)
    y_true = y_true[order]; y_pred = y_pred[order]
    bins = np.array_split(np.arange(len(y_pred)), n_bins)
    xc = []; yc = []
    for idx in bins:
        xc.append(np.mean(y_pred[idx]))
        yc.append(np.mean(y_true[idx]))
    return np.array(xc), np.array(yc)

# --------------------------- MAIN PIPELINE ----------------------------
def main(args):
    set_plot_style(font_family="Times New Roman", font_size=9)

    # Resolve input and check it exists
    input_path = args.input if args.input else DEFAULT_INPUT
    if not os.path.isfile(input_path):
        sys.stderr.write(
            f"\n[ERROR] Could not find Excel at:\n  {input_path}\n"
            "       Check the path or pass a different file with --input \"C:\\path\\to\\file.xlsx\"\n"
        )
        sys.exit(1)

    # Load Excel
    df = pd.read_excel(input_path, engine="openpyxl")

    # Identify columns
    target_col = find_column(df, ["Emin_pig_exp_prior_mmHg_ml"])
    mech_col   = find_column(df, [
        "Emin_thick_corr_mmHg_ml",                   # preferred
        "Emin_mech_thick_wall_raw_mmHg_ml",          # fallback names
        "Emin_mech_thick_mmHg_ml"
    ])
    id_col = find_column(df, ["sub_id"])

    y = df[target_col].values.astype(float)   # target (pig prior)
    x = df[mech_col].values.astype(float)    # mechanistic predictor

    # Physiologic clip window for deployment
    phys_min, phys_max = 0.01, 2.50

    # Splitter (group-aware if requested)
    if args.group_col and args.group_col in df.columns:
        groups = df[args.group_col].values
        splitter = GroupKFold(n_splits=args.folds)
        split_iter = splitter.split(x, y, groups=groups)
        split_name = f"GroupKFold({args.group_col})"
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(x, y)
        split_name = f"KFold({args.folds}, shuffle=True, random_state={args.seed})"

    # Storage
    oof = pd.DataFrame({id_col: df[id_col].values, "y_true": y, "x_mech": x})
    oof["yhat_affine"]   = np.nan
    oof["yhat_isotonic"] = np.nan
    metrics = []

    # Cross-validated predictions
    for fold, (tr, te) in enumerate(split_iter, 1):
        xtr, ytr = x[tr], y[tr]
        xte, yte = x[te], y[te]

        # Affine (β>=0)
        a_aff, b_aff = pos_slope_affine_fit(xtr, ytr)
        yhat_aff = apply_affine(a_aff, b_aff, xte)

        # Isotonic (monotone), clipped OOB
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(xtr, ytr)
        yhat_iso = iso.predict(xte)

        # Store OOF
        oof.loc[te, "yhat_affine"]   = yhat_aff
        oof.loc[te, "yhat_isotonic"] = yhat_iso

        # Fold metrics (observed on predicted)
        for name, yhat in [("affine", yhat_aff), ("isotonic", yhat_iso)]:
            mae  = mean_absolute_error(yte, yhat)
            rmse = np.sqrt(np.mean((yte - yhat)**2))
            r2   = r2_score(yte, yhat)
            reg  = LinearRegression().fit(yhat.reshape(-1,1), yte)
            cal_slope, cal_intercept = reg.coef_[0], reg.intercept_
            metrics.append({
                "fold": fold, "model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
                "CalSlope": cal_slope, "CalIntercept": cal_intercept
            })

    met = pd.DataFrame(metrics)
    met_summary = met.groupby("model").agg(["mean","std"]).reset_index()

    # Output dir
    outdir = args.outdir if args.outdir else os.path.join(os.path.dirname(input_path), "calibration_outputs")
    os.makedirs(outdir, exist_ok=True)

    # Save CV artifacts
    oof.to_csv(os.path.join(outdir, "cv_oof_predictions.csv"), index=False)
    met.to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)
    met_summary.to_csv(os.path.join(outdir, "cv_metrics_summary.csv"), index=False)

    # Final fit on ALL data
    a_aff, b_aff = pos_slope_affine_fit(x, y)
    iso_final = IsotonicRegression(out_of_bounds="clip").fit(x, y)

    # Model card
    card = {
        "input_excel": input_path,
        "id_col": id_col,
        "target_col": target_col,
        "mech_col": mech_col,
        "cv": split_name,
        "affine": {"intercept": float(a_aff), "slope": float(b_aff)},
        "isotonic": {
            "X_thresholds": iso_final.X_thresholds_.tolist(),
            "y_thresholds": iso_final.y_thresholds_.tolist()
        },
        "phys_range": [phys_min, phys_max]
    }
    with open(os.path.join(outdir, "calibration_model.json"), "w") as f:
        json.dump(card, f, indent=2)

    # ---------------------------- PLOTTING ----------------------------
    dpi = args.dpi

    # -- Calibration fit (plot isotonic only; affine commented out) --

    plt.figure(figsize=(FIG_W, FIG_H))
    plt.scatter(x, y, s=16, facecolors='none', edgecolors='black',
                linewidth=0.8, marker='o')

    # Get the current Axes object
    ax = plt.gca()

    # Decimal formatting (one decimal)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    xx = np.linspace(x.min(), x.max(), 400)

    # Isotonic line
    plt.plot(xx, iso_final.predict(xx), "g-", lw=1, label="Isotonic Fit")

    # Limits
    plt.xlim(0, 25)
    plt.ylim(0, 2.5)

    # Ticks
    plt.xticks(np.arange(0, 25 + 1e-12, 5))
    plt.yticks(np.arange(0, 2.5 + 1e-12, 0.5))

    plt.xlabel("Mechanistic $E_{min}$ (mmHg·mL$^{-1}$)")
    plt.ylabel("Pig-derived $E_{min}$ prior (mmHg·mL$^{-1}$)")
    plt.title("Calibration fit (Isotonic only)")
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, "fit_scatter_isotonic.png"), dpi=dpi)

    # -- Reliability curve (bin-based). If n_bins<=1, skip this figure. --
    n_bins = args.n_bins
    yhat_iso_all = iso_final.predict(x)

    if n_bins is not None and n_bins > 1:
        xc, yc = reliability_bins(y, yhat_iso_all, n_bins=n_bins)
        plt.figure(figsize=(FIG_W, FIG_H))
        lo = min(y.min(), yhat_iso_all.min())
        hi = max(y.max(), yhat_iso_all.max())
        plt.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.7, label="Ideal")
        plt.scatter(xc, yc, s=25, color="tab:blue", label=f"Bin means (n_bins={n_bins})")
        plt.xlabel("Predicted $E_{min}$ (bin mean)")
        plt.ylabel("Observed $E_{min}$ (bin mean)")
        plt.title("Reliability (Isotonic)")
        plt.grid(alpha=.3)
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"reliability_isotonic_bins{n_bins}.png"), dpi=dpi)

    # -- Observed vs Predicted (all points) --
    plt.figure(figsize=(FIG_W, FIG_H))
    plt.scatter(yhat_iso_all, y, s=12, alpha=0.6, color="tab:purple")
    lo = min(y.min(), yhat_iso_all.min())
    hi = max(y.max(), yhat_iso_all.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.7, label="Ideal")
    plt.xlabel("Predicted $E_{min}$ (isotonic)")
    plt.ylabel("Observed $E_{min}$")
    plt.title("Observed vs Predicted (all points)")
    plt.grid(alpha=.3)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "obs_vs_pred_allpoints.png"), dpi=dpi)

    # -- Isotonic output (Y) vs Pig data (X) --
    plt.figure(figsize=(FIG_W, FIG_H))
    plt.scatter(y, yhat_iso_all, s=12, alpha=0.6, color="tab:green")
    lo = min(y.min(), yhat_iso_all.min())
    hi = max(y.max(), yhat_iso_all.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.7, label="Ideal")
    plt.xlabel("Pig-derived $E_{min}$")
    plt.ylabel("Isotonic calibrated $E_{min}$")
    plt.title("Isotonic output vs Pig data")
    plt.grid(alpha=.3)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "iso_vs_pig.png"), dpi=dpi)

    # -- Deployment: calibrated predictions clipped to physiologic band --
    yhat_aff_all  = np.clip(apply_affine(a_aff, b_aff, x), phys_min, phys_max)
    yhat_iso_clip = np.clip(yhat_iso_all,                     phys_min, phys_max)

    deploy = pd.DataFrame({
        id_col: df[id_col].values,
        "Emin_mech": x,
        "Emin_prior": y,
        "Emin_affine_cal": yhat_aff_all,
        "Emin_isotonic_cal": yhat_iso_clip
    })
    deploy.to_csv(os.path.join(outdir, "calibrated_predictions.csv"), index=False)

    print(f"\nDone.\n  Input : {input_path}\n  Output: {outdir}\n"
          f"  CV    : {split_name}\n  Plots : 8cm x 5cm, font=Times New Roman, size=9, dpi={dpi}\n"
          f"  Reliability bins: {'skipped (all-points only)' if (n_bins is None or n_bins<=1) else n_bins}\n")

# --------------------------- CLI --------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cross-validated monotone calibration of mechanistic Emin.")
    p.add_argument("--input", default=None, help="Path to Excel. If omitted, uses DEFAULT_INPUT in the script.")
    p.add_argument("--outdir", default=None, help="Output directory. If omitted, creates 'calibration_outputs' next to the Excel.")
    p.add_argument("--folds", type=int, default=10, help="Number of CV folds (default 10).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffling.")
    p.add_argument("--group_col", default=None, help="Optional column for grouping (e.g., site/scanner) for GroupKFold.")
    p.add_argument("--dpi", type=int, default=300, help="DPI for saved figures (default 300).")
    p.add_argument("--n_bins", type=int, default=10,
                   help="Number of bins for reliability curve (default 10). "
                        "Set to 0 or 1 to skip bin-based reliability and keep only all-points plot.")
    args = p.parse_args()
    main(args)