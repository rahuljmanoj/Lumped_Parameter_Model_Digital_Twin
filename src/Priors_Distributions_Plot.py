import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator


# ------------------ GLOBAL STYLE ------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ------------------ CONFIG ------------------
excel_path = (r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
              r"\3_Codes\Python\Dataset_Python\Results_Validation_Paper_all_subjects.xlsx")
sheet_name = "Study_13"
subject_id = 999   # subject / population ID in the Excel file

UQSA_ROOT = (r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
             r"\3_Codes\Python\Dataset_Python\Results_UQSA_Population_Level\UQ_SA")
uqsa_sub_folder = os.path.join(UQSA_ROOT, str(subject_id))
uq_params_path = os.path.join(uqsa_sub_folder, f"{subject_id}_uq_params.csv")

n_points = 500

# ------------------ LOAD SUBJECT MEANS ------------------
combined_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, nrows=69)

data_row = combined_df.loc[combined_df['Sub ID'] == subject_id]
if data_row.shape[0] != 1:
    raise ValueError(f"Expected one row for subject {subject_id}, got {data_row.shape[0]}")
data_row = data_row.squeeze()


def safe_val(series_val):
    if isinstance(series_val, pd.Series):
        return float(series_val.iloc[0])
    return float(series_val)

means = {
    "R_sys": safe_val(data_row['R_sys (mmHg s/ml)']),
    "Z_ao":  safe_val(data_row['Z_ao (mmHg s/ml)']),
    "C_sa":  safe_val(data_row['C_sa (ml/mmHg)']),
    "R_mv":  safe_val(data_row['R_mv (mmHg s/ml)']),
    "E_max": safe_val(data_row['E_max (mmHg/ml)']),
    "E_min": safe_val(data_row['E_min (mmHg/ml)']),
    "t_peak": safe_val(data_row['t_peak (s)']),
    "V_tot": safe_val(data_row['V_tot (ml)']),
    "C_sv":  safe_val(data_row['C_sv (ml/mmHg)']),
}

# ------------------ PRIORS (NORMAL) ------------------
priors = {
    "R_sys":  ("normal", {"cv": 0.12}),
    "Z_ao":   ("normal", {"cv": 0.12}),
    "C_sa":   ("normal", {"cv": 0.15}),
    "R_mv":   ("normal", {"cv": 0.15}),
    "E_max":  ("normal", {"cv": 0.15}),
    "E_min":  ("normal", {"cv": 0.15}),
    "t_peak": ("normal", {"cv": 0.10}),
    "V_tot":  ("normal", {"cv": 0.12}),
    "C_sv":   ("normal", {"cv": 0.15}),
}

# x-axis labels with subscripts + units
xlabels = {
    "R_sys":  r"$R_{\mathrm{sys}}$ (mmHg·s/ml)",
    "Z_ao":   r"$Z_{\mathrm{ao}}$ (mmHg·s/ml)",
    "C_sa":   r"$C_{\mathrm{sa}}$ (ml/mmHg)",
    "R_mv":   r"$R_{\mathrm{mv}}$ (mmHg·s/ml)",
    "E_max":  r"$E_{\max}$ (mmHg/ml)",
    "E_min":  r"$E_{\min}$ (mmHg/ml)",
    "t_peak": r"$t_{\mathrm{peak}}$ (s)",
    "V_tot":  r"$V_{\mathrm{tot}}$ (ml)",
    "C_sv":   r"$C_{\mathrm{sv}}$ (ml/mmHg)",
}

# ------------------ LOAD UQ SAMPLES ------------------
df_samples = None
if os.path.isfile(uq_params_path):
    df_samples = pd.read_csv(uq_params_path)
else:
    print(f"[warn] UQ params not found at {uq_params_path}; "
          f"will plot priors without sample overlay.")

# ------------------ CREATE FIGURE ------------------
n_cols = 3
n_rows = int(np.ceil(len(priors) / n_cols))

cm_to_inch = 1.0 / 2.54
sub_w = 5.0 * cm_to_inch  # 5 cm
sub_h = 5.0 * cm_to_inch  # 5 cm
fig_w = n_cols * sub_w
fig_h = n_rows * sub_h + 1.0  # extra height for legend

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
axes = axes.flatten()

# for common legend
common_handles = None
common_labels = None

for i, (param, (dist_type, args)) in enumerate(priors.items()):
    mean_val = means[param]
    ax = axes[i]

    if dist_type != "normal":
        raise ValueError("This script assumes all priors are normal.")

    cv = args["cv"]
    sigma = cv * abs(mean_val) if mean_val != 0 else cv

    lo_95 = mean_val - 2 * sigma
    hi_95 = mean_val + 2 * sigma

    x_min = mean_val - 4 * sigma
    x_max = mean_val + 4 * sigma
    if param in ["R_sys", "Z_ao", "C_sa", "R_mv", "E_max", "E_min", "V_tot", "C_sv", "t_peak"]:
        x_min = max(x_min, 0.0)

    x = np.linspace(x_min, x_max, n_points)
    y = norm.pdf(x, loc=mean_val, scale=sigma)

    # 95% band
    mask_95 = (x >= lo_95) & (x <= hi_95)
    band = ax.fill_between(x[mask_95], 0, y[mask_95],
                           alpha=0.2, color="peru",
                           label=r"≈95% ($\mu\pm2\sigma$)")

    # PDF
    pdf_line, = ax.plot(x, y, lw=1.5, label="Prior PDF")

    # mean line (no numeric values)
    mean_line = ax.axvline(mean_val, color='red', linestyle='--', label="mean")

    # rug plot
    uq_handle = None
    y_max = y.max()
    if df_samples is not None and param in df_samples.columns:
        s = df_samples[param].dropna().values
        if len(s) > 0:
            rug_y = -0.04 * y_max * np.ones_like(s)
            ax.plot(s, rug_y, '|', color='black', alpha=0.4, markersize=5)
            uq_handle = Line2D([0], [0], color='black', marker='|',
                               linestyle='None', markersize=7,
                               label="UQ samples")
            ax.set_ylim(bottom=-0.08 * y_max)

    # x/y labels
    ax.set_xlabel(xlabels.get(param, param))
    ax.set_ylabel("PDF")
    ax.grid(False)

    # ---- ENFORCE AT LEAST 3 x-ticks AND 4 y-ticks ----
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))

    # Collect legend handles only once (from first subplot that has everything)
    if common_handles is None:
        handles = [band, pdf_line, mean_line]
        labels = [h.get_label() for h in handles]
        if uq_handle is not None:
            handles.append(uq_handle)
            labels.append("UQ samples")
        common_handles, common_labels = handles, labels


# remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

#fig.suptitle(f"Normal Priors + UQ Samples for Subject {subject_id}", y=0.98)

# common legend at bottom, all in one line
if common_handles is not None:
    fig.legend(common_handles,
               common_labels,
               loc="lower center",
               ncol=len(common_handles),
               bbox_to_anchor=(0.5, 0.02))

fig.tight_layout(rect=[0.03, 0.10, 0.97, 0.95])  # leave space bottom for legend
plt.show()
