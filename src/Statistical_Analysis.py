import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import pearsonr, ttest_rel
import os
import matplotlib.pyplot as plt
import os

FIG_SAVE_DIR = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Figure_Results_V4.1"
os.makedirs(FIG_SAVE_DIR, exist_ok=True)

# ----- Load Excel file -----
excel_path = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx"
sheet_name = "Study_3_V4.1_T6"
df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, nrows=68)

# ----- Select columns -----
sim_col = 'sim LVEDP (LPM) (mmHg)'    # Simulated LVEDP
pred_col = 'pred LVEDP (UVR) (mmHg)'  # Regression-predicted LVEDP from SWE
gt_col  = 'GT LVEDP (mmHg)'     # Ground-truth LVEDP
swe_col = 'SWS (m/s)'       # SWE velocity

#X = df[['SWS (m/s)']]  # adjust column names
X = df[['SWS (m/s)', 'GT bMAP (mmHg)']]  # adjust column names
y = df['GT LVEDP (mmHg)']
# 2. Standardize X
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# 3. Fit a linear model on the raw data (for R²)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print (df[['SWS (m/s)', 'GT bMAP (mmHg)']].corr())
print(f"Raw-data R²: {r2_score(y, y_pred):.3f}")

# 4. Fit OLS on the standardized data (to get standardized betas)
Xs = sm.add_constant(X)
ols_std = sm.OLS(y, Xs).fit()

print("\nStandardized-beta OLS results:")
print(ols_std.summary())

# 5. Compute VIFs on the original X to check collinearity
print("Columns used for VIF:", X.columns)
vif_df = pd.DataFrame({
    'Variable': X.columns,
    'VIF': [variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])]
})
vif_std = pd.DataFrame({
    'Variable': X_std.columns,
    'VIF': [variance_inflation_factor(X_std.values, i)
            for i in range(X_std.shape[1])]
})

print("\nVariance Inflation Factors:")
print(vif_df)
print("\nVariance Inflation Factors after scaling:")
print(vif_std)
print(np.corrcoef(X.values.T))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
y = df['GT LVEDP (mmHg)']

def style_axes(ax, xlabel, ylabel, add_legend=False):
    """Apply common formatting: Times New Roman, 9 pt, black, no grid, no title."""
    ax.set_xlabel(xlabel, fontname="Times New Roman", fontsize=9, color="black")
    ax.set_ylabel(ylabel, fontname="Times New Roman", fontsize=9, color="black")

    ax.tick_params(axis='both', labelsize=9, colors="black")
    for lbl in ax.get_xticklabels():
        lbl.set_fontname("Times New Roman")
    for lbl in ax.get_yticklabels():
        lbl.set_fontname("Times New Roman")

    if add_legend:
        leg = ax.legend(fontsize=9, frameon=False)
        for txt in leg.get_texts():
            txt.set_fontname("Times New Roman")
            txt.set_color("black")

    ax.grid(False)        # no grid
    ax.set_title("")      # no title


def cv_score(X):
    return cross_val_score(LinearRegression(), X, y,
                           cv=cv, scoring='r2').mean()

X2 = df[['SWS (m/s)', 'GT bMAP (mmHg)']]
#X3 = df[['SWS (m/s)', 'GT bMAP (mmHg)', 'LAVI (ml/m²)']]

print("2-var CV R²:", cv_score(X2))
#print("3-var CV R²:", cv_score(X3))




df = df.dropna(subset=[sim_col, pred_col, gt_col, swe_col])

sim_lvedp  = df[sim_col].values
pred_lvedp = df[pred_col].values
gt_lvedp   = df[gt_col].values
swe_vals   = df[swe_col].values

# ----- Paired, two tailed t test: GT vs simulated LVEDP -----
t_stat, p_val = ttest_rel(gt_lvedp, sim_lvedp)

print("\nPaired t test: GT LVEDP vs sim LVEDP (LPM)")
print(f"  n           = {len(gt_lvedp)}")
print(f"  mean(GT)    = {np.mean(gt_lvedp):.2f} mmHg")
print(f"  mean(sim)   = {np.mean(sim_lvedp):.2f} mmHg")
print(f"  t statistic = {t_stat:.3f}")
print(f"  p value     = {p_val:.3g}")  # two sided by default
# -------------------------------------------------------------

# ----- Regression plot: GT LVEDP vs sim LVEDP -----
x = gt_lvedp.reshape(-1, 1)
y = sim_lvedp

reg = LinearRegression()
reg.fit(x, y)
y_fit = reg.predict(x)
slope = reg.coef_[0]
intercept = reg.intercept_
r2 = r2_score(y, y_fit)

fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54), dpi=300)

# scatter as black "*"
ax.scatter(gt_lvedp, sim_lvedp, s=10, marker='*', color='orange')

# regression line as dashed black
x_line = np.linspace(gt_lvedp.min(), gt_lvedp.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
ax.plot(x_line, y_line, linestyle='--', linewidth=0.75, color='black')
ax.set_xlim(5, 40)
ax.set_ylim(5, 40)
ax.set_xticks(np.arange(5, 41, 5))
ax.set_yticks(np.arange(5, 41, 5))
style_axes(ax, "GT LVEDP [mmHg]", "LPM-deived LVEDP [mmHg]", add_legend=False)
#fig.tight_layout()

#plt.tight_layout()
reg_path = os.path.join(FIG_SAVE_DIR, "Fig1_regression_GT_vs_Sim_LVEDP.png")
fig.savefig(reg_path, dpi=300, bbox_inches='tight')
print(f"Saved: {reg_path}")
plt.close(fig)

# ----- Boxplot: GT vs Sim LVEDP -----
fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54), dpi=300)
ax.boxplot([gt_lvedp, sim_lvedp], labels=["GT", "LPM"])
style_axes(ax, "", "LVEDP [mmHg]", add_legend=False)
ax.set_ylim(5, 40)
ax.set_yticks(np.arange(5, 41, 5))
#plt.tight_layout()
box1_path = os.path.join(FIG_SAVE_DIR, "Fig2_box_GT_vs_Sim_LVEDP.png")
fig.savefig(box1_path, dpi=300, bbox_inches='tight')
print(f"Saved: {box1_path}")
plt.close(fig)

# ----- Boxplot: GT vs Sim vs UVR LVEDP -----
fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54), dpi=300)
ax.boxplot([gt_lvedp, sim_lvedp, pred_lvedp], labels=["GT", "LPM", "UVR"])
style_axes(ax, "", "LVEDP [mmHg]", add_legend=False)
ax.set_ylim(5, 40)
ax.set_yticks(np.arange(5, 41, 5))
#plt.tight_layout()

box2_path = os.path.join(FIG_SAVE_DIR, "Fig3_box_GT_vs_Sim_vs_UVR_LVEDP.png")
fig.savefig(box2_path, dpi=300, bbox_inches='tight')
print(f"Saved: {box2_path}")
plt.close(fig)


threshold = 16
gt_elevated = (gt_lvedp >= threshold).astype(int)

# ---- Calculate SWE velocity threshold corresponding to 16 mmHg ----
a = 2.4033
b = 6.3966
swe_thresh = (threshold - b) / a
print(f"SWE threshold for LVEDP=16 mmHg: {swe_thresh:.2f} m/s")

# ---- CORRELATION ----
for name, arr in [("sim LVEDP (LPM) (mmHg)", sim_lvedp), ("pred LVEDP (mmHg)", pred_lvedp), ("SWE Velocity", swe_vals)]:
    r, p = pearsonr(arr, gt_lvedp)
    print(f"{name} vs GT LVEDP: Correlation r = {r:.2f}, p = {p:.3g}")

# ---- ROC & AUC Analysis ----
roc_data = []
for name, pred in [
        ("sim LVEDP (LPM) (mmHg)", sim_lvedp),
        ("pred LVEDP (mmHg)", pred_lvedp),
        ("SWE Velocity", swe_vals)]:
    fpr, tpr, thresholds = roc_curve(gt_elevated, pred)
    auc_score = auc(fpr, tpr)
    # Best threshold by Youden's J (not for SWE plot in paper, but for info)
    j_scores = tpr - fpr
    ix = np.argmax(j_scores)
    best_thresh = thresholds[ix]
    # For paper, use 16 mmHg (for LVEDP), SWE threshold for SWE
    if name == "SWE Velocity":
        cutoff = swe_thresh
    else:
        cutoff = threshold
    pred_class = (pred >= cutoff).astype(int)
    cm = confusion_matrix(gt_elevated, pred_class)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_data.append((name, fpr, tpr, auc_score, cutoff, sensitivity, specificity))
    print(f"\n{name} ROC:")
    print(f"  AUC: {auc_score:.2f}")
    print(f"  Sensitivity @ cutoff {cutoff:.2f}: {sensitivity:.2f}")
    print(f"  Specificity @ cutoff {cutoff:.2f}: {specificity:.2f}")

# ---- Plot all three ROC curves together ----
# ---- ROC plots with standard formatting ----

# 5) ROC: combined Sim LVEDP + UVR LVEDP
fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54), dpi=300)
for name, fpr, tpr, auc_score, cutoff, sensitivity, specificity in roc_data:
    if name in ("sim LVEDP (LPM) (mmHg)", "pred LVEDP (mmHg)"):
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})', linewidth=1)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
style_axes(ax, "1 - Specificity", "Sensitivity", add_legend=False)
plt.tight_layout()
roc_comb_path = os.path.join(FIG_SAVE_DIR, "Fig4_ROC_Sim_plus_UVR_LVEDP.png")
fig.savefig(roc_comb_path, dpi=300, bbox_inches='tight')
print(f"Saved: {roc_comb_path}")
plt.close(fig)

# 6) ROC: Sim LVEDP alone
fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54), dpi=300)
for name, fpr, tpr, auc_score, cutoff, sensitivity, specificity in roc_data:
    if name == "sim LVEDP (LPM) (mmHg)":
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})', linewidth=1)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
style_axes(ax, "1 - Specificity", "Sensitivity", add_legend=False)
plt.tight_layout()
roc_sim_path = os.path.join(FIG_SAVE_DIR, "Fig5_ROC_Sim_LVEDP_only.png")
fig.savefig(roc_sim_path, dpi=300, bbox_inches='tight')
print(f"Saved: {roc_sim_path}")
plt.close(fig)



# ====== Bland–Altman Utils ======
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, linregress

def _coerce_numeric_pair(a: np.ndarray, b: np.ndarray):
    """Return aligned numeric arrays with NaNs removed (pairwise)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]

def bland_altman_stats(a, b):
    """
    Compute Bland–Altman stats:
      bias ± 1.96*SD (LoA) and 95% CIs for bias and LoA (Bland & Altman 1999).
    Returns dict with keys:
      n, mean, diff, bias, sd, loa_low, loa_high, ci_bias, ci_loa_low, ci_loa_high,
      prop_bias: {slope, intercept, pval, r}
    """
    a, b = _coerce_numeric_pair(a, b)
    mean = (a + b) / 2.0
    diff = a - b

    n = len(diff)
    if n < 3:
        raise ValueError("Not enough paired points for Bland–Altman (need n ≥ 3).")

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)

    # 95% Limits of Agreement
    z = 1.96
    loa_low = bias - z * sd
    loa_high = bias + z * sd

    # CIs (Bland & Altman, 1999):
    #   CI for bias: bias ± t_{n-1,0.975} * SD/sqrt(n)
    #   CI for LoA: LoA ± t_{n-1,0.975} * SD * sqrt(1/n + z^2/(2*(n-1)))
    tcrit = t.ppf(0.975, df=n-1)
    se_bias = sd / np.sqrt(n)
    ci_bias = (bias - tcrit * se_bias, bias + tcrit * se_bias)

    se_loa = sd * np.sqrt(1.0/n + (z**2)/(2*(n-1)))
    ci_loa_low  = (loa_low  - tcrit * se_loa, loa_low  + tcrit * se_loa)
    ci_loa_high = (loa_high - tcrit * se_loa, loa_high + tcrit * se_loa)

    # Proportional bias: diff ~ mean (simple linear regression)
    lr = linregress(mean, diff)
    prop_bias = dict(slope=lr.slope, intercept=lr.intercept, pval=lr.pvalue, r=lr.rvalue)

    return dict(
        n=n, mean=mean, diff=diff, bias=bias, sd=sd,
        loa_low=loa_low, loa_high=loa_high,
        ci_bias=ci_bias, ci_loa_low=ci_loa_low, ci_loa_high=ci_loa_high,
        prop_bias=prop_bias
    )

def plot_bland_altman(
    stats_dict, title="", x_label="Mean of methods", y_label="Difference (A - B)",
    point_alpha=0.8, annotate=True, save_name=None
):
    """
    Render a single Bland–Altman plot using stats from bland_altman_stats().
    If save_name is given, saves PNG into FIG_SAVE_DIR.
    """
    mean = stats_dict["mean"]
    diff = stats_dict["diff"]
    bias = stats_dict["bias"]
    loa_low = stats_dict["loa_low"]
    loa_high = stats_dict["loa_high"]
    ci_bias = stats_dict["ci_bias"]
    ci_loa_low = stats_dict["ci_loa_low"]
    ci_loa_high = stats_dict["ci_loa_high"]
    n = stats_dict["n"]
    prop = stats_dict["prop_bias"]

    fig, ax = plt.subplots(figsize=(7/2.54, 5/2.54), dpi=300)

    ax.scatter(mean, diff, s=12, alpha=point_alpha, edgecolor='none')

    ax.axhline(bias, color='k', linestyle='-', linewidth=1.0)
    ax.axhline(loa_low, color='k', linestyle='--', linewidth=0.9)
    ax.axhline(loa_high, color='k', linestyle='--', linewidth=0.9)

    ax.fill_between([np.min(mean), np.max(mean)], ci_bias[0], ci_bias[1], alpha=0.12)
    ax.fill_between([np.min(mean), np.max(mean)], ci_loa_low[0], ci_loa_low[1], alpha=0.08)
    ax.fill_between([np.min(mean), np.max(mean)], ci_loa_high[0], ci_loa_high[1], alpha=0.08)

    if annotate:
        txt = (
            f"n = {n}\n"
            f"Bias = {bias:.2f} (95% CI {ci_bias[0]:.2f} to {ci_bias[1]:.2f})\n"
            f"LoA = {loa_low:.2f} to {loa_high:.2f}\n"
            f"Prop. bias: slope={prop['slope']:.3f}, p={prop['pval']:.3g}"
        )
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, linewidth=0.5)
        )

    style_axes(ax, x_label, y_label, add_legend=False)
    ax.set_xlim(5, 35)
    ax.set_ylim(-15, 15)
    ax.set_xticks(np.arange(5, 36, 5))
    ax.set_yticks(np.arange(-15, 16, 5))
    fig.tight_layout()

    if save_name is not None:
        ba_path = os.path.join(FIG_SAVE_DIR, save_name)
        fig.savefig(ba_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {ba_path}")

    plt.close(fig)



def run_bland_altman(df, col_A, col_B, label_A=None, label_B=None, save_name=None):
    """
    Convenience wrapper to compute + plot BA between two dataframe columns.
    """
    if label_A is None: label_A = col_A
    if label_B is None: label_B = col_B
    a = df[col_A].values
    b = df[col_B].values
    stats_d = bland_altman_stats(a, b)
    title = f"{label_A} vs {label_B}"
    plot_bland_altman(
        stats_d, title=title,
        x_label=f" ",
        y_label=f" ",
        save_name=save_name
    )
    return stats_d


# ====== USAGE: choose your BA pairs ======
# Examples below assume the variables sim_col, pred_col, gt_col, swe_col already exist
# (as in your current script). Typically BA should compare the *same units/methods*,
# e.g., predicted LVEDP vs ground-truth LVEDP, or simulated vs ground-truth.
#
# Define the pairs you want to analyze and plot:
ba_pairs = [
    (pred_col, gt_col, "Fig6_BA_UVR_vs_GT.png"),
    (sim_col,  gt_col, "Fig7_BA_Sim_vs_GT.png"),
]

for A, B, fname in ba_pairs:
    print(f"\n=== Bland–Altman: {A} vs {B} ===")
    ba_stats = run_bland_altman(df, A, B, label_A=A, label_B=B, save_name=fname)




