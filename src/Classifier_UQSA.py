"""
Train a classifier that predicts whether a parameter set produces physiologically VALID (1)
or INVALID (0) outputs.  Hard-coded for Rahul’s 500k synthetic dataset.

Root folder:
  C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\
      3_Codes\Python\Dataset_Python\Results_Dataset_Creation_500k_samples\SYNTHETIC
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score, brier_score_loss,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from joblib import dump

# --------------------------------------------------------------------
# Hard-coded paths
# --------------------------------------------------------------------
CSV_PATH = (
    r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
    r"\3_Codes\Python\Dataset_Python\Results_Dataset_Creation_500k_samples\SYNTHETIC"
    r"\synthetic_dataset_merged.csv"
)
OUTDIR = (
    r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
    r"\3_Codes\Python\Dataset_Python\Results_Dataset_Creation_500k_samples\SYNTHETIC\classifier_artifacts"
)
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
FEATURE_COLS = [
    'R_sys','Z_ao','C_sa','R_mv','E_max','E_min','t_peak','V_tot','C_sv','T'
]
TARGET_COL = 'is_valid'
DESIRED_RECALL_VALID = 0.98   # target recall for valid class
RANDOM_SEED = 42

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def threshold_from_recall(y_true, y_prob, desired_recall):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.r_[0.0, thr, 1.0]
    rec = np.r_[rec[0], rec, rec[-1]]
    idx = np.where(rec >= desired_recall)[0]
    return 0.5 if len(idx) == 0 else float(thr[idx[-1]])

def evaluate_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": thr,
        "accuracy": (tp+tn)/cm.sum(),
        "precision_valid": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_valid": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_valid": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }, y_pred, cm

# --------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------
print(f"\n[INFO] Loading dataset:\n{CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Select feature columns that exist
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"[WARN] Missing expected columns (ignored): {missing}")
print(f"[INFO] Using features: {feature_cols}")

X = df[feature_cols].values.astype(float)
y = df[TARGET_COL].values.astype(int)

# Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"[INFO] Training samples: {len(y_train)}, Test samples: {len(y_test)}")

# Train Random Forest (handles imbalance automatically)
clf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1
)
print("[INFO] Training Random Forest...")
clf.fit(X_train, y_train)

# Predict probabilities
prob_test = clf.predict_proba(X_test)[:,1]

# Metrics at 0.5 threshold
base_metrics, y_pred_05, cm_05 = evaluate_at_threshold(y_test, prob_test, 0.5)
auc_roc = roc_auc_score(y_test, prob_test)
auc_pr = average_precision_score(y_test, prob_test)
brier = brier_score_loss(y_test, prob_test)

print("\n[BASELINE @ 0.5]")
for k,v in base_metrics.items():
    if isinstance(v,float):
        print(f"{k:>20}: {v:.4f}")
print(f"{'AUC-ROC':>20}: {auc_roc:.4f}")
print(f"{'AUC-PR':>20}: {auc_pr:.4f}")
print(f"{'Brier':>20}: {brier:.4f}")

# Tune threshold for desired recall
thr = threshold_from_recall(y_test, prob_test, DESIRED_RECALL_VALID)
tuned_metrics, y_pred_tuned, cm_tuned = evaluate_at_threshold(y_test, prob_test, thr)
print("\n[TUNED THRESHOLD]")
for k,v in tuned_metrics.items():
    if isinstance(v,float):
        print(f"{k:>20}: {v:.4f}")

# Save model bundle
from joblib import dump
bundle_path = os.path.join(OUTDIR, "model.joblib")
dump({"model": clf, "feature_cols": feature_cols, "threshold": thr}, bundle_path)
print(f"\n[OK] Saved model to {bundle_path}")

# Save metrics
metrics = {
    "feature_cols": feature_cols,
    "auc_roc": auc_roc,
    "auc_pr": auc_pr,
    "brier": brier,
    "baseline": base_metrics,
    "tuned": tuned_metrics,
}
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Permutation importance
imp = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=RANDOM_SEED)
fi_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": imp.importances_mean,
    "importance_std": imp.importances_std
}).sort_values("importance_mean", ascending=False)
fi_df.to_csv(os.path.join(OUTDIR, "feature_importance.csv"), index=False)

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(4,4))
disp = ConfusionMatrixDisplay(cm_tuned, display_labels=["Invalid","Valid"])
disp.plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix (thr={thr:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "confusion_matrix.png"), dpi=200)

# ROC + PR curves
fpr, tpr, _ = roc_curve(y_test, prob_test)
prec, rec, _ = precision_recall_curve(y_test, prob_test)
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].plot(fpr, tpr); axes[0].plot([0,1],[0,1],'--',lw=1)
axes[0].set_title(f"ROC AUC={auc_roc:.3f}"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[1].plot(rec, prec)
axes[1].set_title(f"PR AUC={auc_pr:.3f}"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "roc_pr_curves.png"), dpi=200)

# Calibration plot
frac_pos, mean_pred = calibration_curve(y_test, prob_test, n_bins=20, strategy='quantile')
fig, ax = plt.subplots(figsize=(4,4))
ax.plot([0,1],[0,1],'--',lw=1)
ax.plot(mean_pred, frac_pos, marker='o')
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "calibration_curve.png"), dpi=200)

print(f"\n[OK] All artifacts saved to:\n{OUTDIR}")
