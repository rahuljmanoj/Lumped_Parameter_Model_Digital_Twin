
# -*- coding: utf-8 -*-
"""
Patient↔Virtual hemodynamics summary (Python 3.9; no type hints)

- Inputs:
  * GT CSV (patients): sub_id, age, bSBP, bDBP, bMAP, SV, HR
  * Matches CSV: patient_sub_id, virtual_ref_sub_id  (use rank==1 if present)
  * PWDB CSV: ref_sub_id, PWV_a  (m/s)
  * Waveform CSVs (matrix form; 1st column = virtual subject ID; cols 2..N = samples @ 500 Hz):
      - PWs_Carotid_A.csv       (carotid area)
      - PWs_AorticRoot_A.csv    (aortic root area)
      - PWs_AorticRoot_U.csv    (aortic root velocity)

- Outputs:
  * CSV with: τ, R_sys (mmHg·s/mL), C_sa (mL/mmHg), Z_ao (mmHg·s/mL),
              mean aortic area, peak aortic velocity, ED (s from start),
              and diagnostics.
  * Plots: carotid diastolic fit per subject; optional aortic velocity debug plots.

Author: M365 Copilot (for Rahul)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER PATHS ----------------
GT_CSV = Path(r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Invasive_Study_Leuven_GT_matrices_all_subjects.csv")
MATCHES_CSV = Path(r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\patient_virtual_1NN_matches.csv")
PWDB_CSV = Path(r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\pwdb_haemod_params.csv")

# Matrix waveform CSVs (first column is subject id; remaining columns are samples @ 500 Hz)
WAVEFORMS_ROOT = Path(r"C:\Workspace\PhD Works HTIC IITM\Cardiovascular Research\Data Set\Virtual Subjects Database\Virtual Subjects Database CSV")
CAROTID_A_CSV   = WAVEFORMS_ROOT / "PWs_Carotid_A.csv"
AORTIC_A_CSV    = WAVEFORMS_ROOT / "PWs_AorticRoot_A.csv"
AORTIC_U_CSV    = WAVEFORMS_ROOT / "PWs_AorticRoot_U.csv"

# ---------------- OUTPUTS ----------------
OUT_DIR = Path("./Data_Results\Results_Pulse_Wave_Analysis")
PLOTS_DIR = OUT_DIR / "carotid_decay_plots"
DEBUG_U_DIR = OUT_DIR / "u_debug_plots"
RESULTS_CSV = OUT_DIR / "patient_virtual_hemodynamics_summary.csv"
for d in (OUT_DIR, PLOTS_DIR, DEBUG_U_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- CONSTANTS ----------------
FS_HZ = 500.0               # sampling rate
T_DIA_START = 0.35          # s (carotid diastolic fit start)
RHO = 1060.0                # kg/m^3 for water-hammer
MMHG_TO_PA = 133.322        # Pa per mmHg
PA_S_PER_M3_TO_MMHG_S_PER_ML = 1.0 / (MMHG_TO_PA * 1e6)

# --- ED from RAW within a fixed window ---
ED_WINDOW = (0.20, 0.40)    # seconds
ED_DOWN_ONLY = True         # True: require positive->negative crossing
ED_EPS_FRAC = 0.002         # relative tolerance for zero (0.2% amplitude)

# Debug toggles
DEBUG_ED = True             # save aortic velocity debug plots
DEBUG_MAX_PLOTS = 40        # cap number of debug plots

# Optional SciPy for curve_fit
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------- HELPERS ----------------
def read_csv_smart(path):
    return pd.read_csv(path, skipinitialspace=True)

def load_waveform_matrix(path):
    """
    Load matrix-style waveform CSV:
      - First column = virtual subject id (any header name)
      - Remaining columns = samples pt1..ptN
    Returns DataFrame indexed by subject id (int), numeric sample columns.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    if df.shape[1] < 2:
        raise ValueError(f"{path.name}: need >= 2 columns (id + samples)")
    id_col = df.columns[0]
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col]).copy()
    df[id_col] = df[id_col].astype(int)
    df = df.set_index(id_col)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_row_waveform(df_matrix, subj_id):
    """
    Extract waveform row for 'subj_id' as float array, trimming trailing NaNs.
    Returns (waveform_array_or_None, original_length_before_trim)
    """
    if subj_id not in df_matrix.index:
        return None, 0
    row = df_matrix.loc[subj_id]
    w = row.to_numpy(dtype=float)
    n_all = w.size
    if np.isnan(w).any():
        idx = np.where(~np.isnan(w))[0]
        if idx.size == 0:
            return None, n_all
        w = w[: idx[-1] + 1]
    return w, n_all

def exp_model(t, c, a, tau, t0):
    return c + a * np.exp(-(t - t0) / max(tau, 1e-6))

def fit_exponential_decay(t, y, t0=T_DIA_START):
    """
    Fit y(t) = c + a * exp(-(t - t0)/tau) over t >= t0. Returns (tau, fit_info).
    """
    mask = t >= t0
    t_fit = t[mask]; y_fit = y[mask]
    if t_fit.size < 5:
        return np.nan, {"ok": False, "reason": "insufficient_points"}

    c0 = float(np.nanmin(y_fit))
    a0 = max(float(y_fit[0] - c0), 1e-6)
    tau0 = 0.15

    if SCIPY_AVAILABLE:
        try:
            def f(tt, c, a, tau): return exp_model(tt, c, a, tau, t0)
            popt, _ = curve_fit(f, t_fit, y_fit, p0=[c0, a0, tau0],
                                bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, 10.0]),
                                maxfev=20000)
            c_hat, a_hat, tau_hat = popt
            y_hat = f(t_fit, *popt)
            rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2)))
            return float(tau_hat), {"ok": True, "c": float(c_hat), "a": float(a_hat),
                                    "tau": float(tau_hat), "rmse": rmse, "n": int(t_fit.size)}
        except Exception:
            pass

    # Fallback: small grid on c, then log-linear
    best = {"rmse": np.inf}
    grid = np.linspace(c0 - 0.05 * abs(y_fit[0]), c0 + 0.2 * abs(y_fit[0]), 25)
    for c_try in grid:
        y_shift = y_fit - c_try
        if np.any(y_shift <= 0):
            continue
        X = (t_fit - t0); Y = np.log(y_shift)
        A = np.vstack([np.ones_like(X), -X]).T  # Y = log(a) - (1/tau) X
        try:
            beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        except Exception:
            continue
        loga, inv_tau = beta
        a_hat = float(np.exp(loga))
        tau_hat = 1.0 / max(float(inv_tau), 1e-6)
        y_hat = c_try + a_hat * np.exp(-(t_fit - t0) / tau_hat)
        rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2)))
        if rmse < best["rmse"]:
            best = {"rmse": rmse, "c": float(c_try), "a": a_hat, "tau": float(tau_hat)}
    if best["rmse"] < np.inf:
        return float(best["tau"]), {"ok": True, **best, "n": int(t_fit.size)}
    return np.nan, {"ok": False, "reason": "fit_failed"}

def find_ed_from_raw_zero_cross(t, u, window=ED_WINDOW, down_only=ED_DOWN_ONLY,
                                eps_frac=ED_EPS_FRAC, save_plot_path=None):
    """
    ED from RAW aortic velocity:
      - Search the first zero-crossing inside [window[0], window[1]] seconds.
      - If down_only=True: require positive->negative (end-of-ejection).
      - Returns (ED_s, note), where ED_s is seconds from start; NaN if none found.
    """
    if t.size != u.size or t.size < 2:
        return np.nan, "bad_vectors"

    amp = float(np.nanmax(np.abs(u))) if np.isfinite(u).any() else 0.0
    eps = max(1e-12, eps_frac * amp)
    w0, w1 = float(window[0]), float(window[1])
    if w1 <= w0:
        return np.nan, "bad_window"

    def interp_zero(i):
        u0i, u1i = u[i], u[i+1]
        if (u1i - u0i) == 0:
            return (t[i] + t[i+1]) / 2.0
        return t[i] - u0i * (t[i+1] - t[i]) / (u1i - u0i)

    ED_s = np.nan
    # scan segments that intersect the window
    for i in range(len(u) - 1):
        t0, t1 = t[i], t[i+1]
        if t1 < w0 or t0 > w1:
            continue
        u0i, u1i = u[i], u[i+1]

        # skip pairs both tiny around zero
        if (-eps <= u0i <= eps) and (-eps <= u1i <= eps):
            continue

        crossed = False
        if down_only:
            crossed = (u0i >= -eps) and (u1i < -eps)      # +/0 -> -
        else:
            crossed = ((u0i <= -eps and u1i > eps) or     # - -> +
                        (u0i >= eps  and u1i < -eps))     # + -> -

        # also consider exact-zero endpoints inside window
        if not crossed and (abs(u0i) <= eps or abs(u1i) <= eps):
            crossed = (u0i * u1i) <= 0

        if crossed:
            tz = float(interp_zero(i))
            if w0 <= tz <= w1:
                ED_s = tz
                break

    note = "" if np.isfinite(ED_s) else "no_cross_in_window"

    # Debug plot
    if save_plot_path is not None:
        plt.figure(figsize=(7.8, 4.2))
        plt.plot(t, u, color="tab:gray", label="U raw")
        plt.axhline(0.0, color="k", lw=1)
        plt.axvspan(w0, w1, color="gold", alpha=0.20, label=f"ED window [{w0:.2f}, {w1:.2f}] s")
        if np.isfinite(ED_s):
            plt.plot(ED_s, 0.0, "o", color="tab:red", label=f"ED = {ED_s:.3f} s")
        plt.title("Aortic velocity (debug: raw zero-cross in fixed window)")
        plt.xlabel("Time (s)"); plt.ylabel("Velocity (raw)")
        plt.legend(); plt.tight_layout(); plt.savefig(save_plot_path, dpi=160); plt.close()

    return ED_s, note

def co_ml_per_s(sv_ml, hr_bpm):
    return (sv_ml * hr_bpm) / 60.0


# ---------------- LOAD TABLES ----------------
gt = read_csv_smart(GT_CSV)
matches = read_csv_smart(MATCHES_CSV)
pwdb = read_csv_smart(PWDB_CSV)

# GT normalize + compute MAP if missing
if "sub_id" not in gt.columns and "patient_sub_id" in gt.columns:
    gt = gt.rename(columns={"patient_sub_id": "sub_id"})
if "bMAP" not in gt.columns and {"bSBP", "bDBP"}.issubset(gt.columns):
    gt["bMAP"] = gt["bDBP"] + (gt["bSBP"] - gt["bDBP"]) / 3.0
for c in ["sub_id", "age", "bSBP", "bDBP", "bMAP", "SV", "HR"]:
    if c not in gt.columns:
        raise ValueError(f"GT missing column: {c}")

# Matches: keep rank 1 if present; normalize column names
if "rank" in matches.columns:
    matches = matches.loc[matches["rank"] == 1].copy()
if "patient_sub_id" not in matches.columns:
    for alt in ["patient_id", "sub_id"]:
        if alt in matches.columns:
            matches = matches.rename(columns={alt: "patient_sub_id"})
            break
for col in ["patient_sub_id", "virtual_ref_sub_id"]:
    if col not in matches.columns:
        raise ValueError(f"Matches missing column: {col}")

# PWDB: ensure 'ref_sub_id' + 'PWV_a'
pw_col = None
for cand in ["PWV_a", "PWV_a [m/s]"]:
    if cand in pwdb.columns:
        pw_col = cand; break
if pw_col is None:
    raise ValueError("PWDB missing 'PWV_a' column")
if "ref_sub_id" not in pwdb.columns:
    for c in ["ref_sub_id", "ref_subject_id", "refid", "ref_subid"]:
        if c in pwdb.columns:
            pwdb = pwdb.rename(columns={c: "ref_sub_id"}); break
if "ref_sub_id" not in pwdb.columns:
    raise ValueError("PWDB missing 'ref_sub_id'")

pwdb_slim = pwdb[["ref_sub_id", pw_col]].copy()
pwdb_slim["ref_sub_id"] = pwdb_slim["ref_sub_id"].astype(int)

# Merge master frame
df = matches.merge(gt[["sub_id", "age", "bSBP", "bDBP", "bMAP", "SV", "HR"]],
                   left_on="patient_sub_id", right_on="sub_id", how="inner")
df = df.merge(pwdb_slim.rename(columns={pw_col: "PWV_a"}),
              left_on="virtual_ref_sub_id", right_on="ref_sub_id", how="left")
df.drop(columns=["sub_id", "ref_sub_id"], inplace=True, errors="ignore")

# ---------------- LOAD MATRIX WAVEFORMS ----------------
carotid_mat = load_waveform_matrix(CAROTID_A_CSV)
aorticA_mat = load_waveform_matrix(AORTIC_A_CSV)
aorticU_mat = load_waveform_matrix(AORTIC_U_CSV)

# ---------------- MAIN ----------------
records = []
n_debug = 0

for _, row in df.iterrows():
    patient_id = int(row["patient_sub_id"])
    virt_id = int(row["virtual_ref_sub_id"])

    # ---- Carotid area -> tau ----
    A_c, n_all_c = get_row_waveform(carotid_mat, virt_id)
    loaded_carotid = A_c is not None and A_c.size >= 10
    N_carotid = int(A_c.size) if loaded_carotid else 0
    tau = np.nan; fit_info = {"ok": False}
    carotid_plot_path = ""
    if loaded_carotid:
        t_c = np.arange(N_carotid, dtype=float) / FS_HZ
        tau, fit_info = fit_exponential_decay(t_c, A_c, t0=T_DIA_START)
        # plot carotid diastolic fit
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(t_c, A_c, label="Carotid area (m²)", color="tab:blue")
        plt.axvline(T_DIA_START, color="k", linestyle="--", lw=1.25, label=f"t = {T_DIA_START:.2f} s")
        if fit_info.get("ok", False) and np.isfinite(tau):
            t_fit = t_c[t_c >= T_DIA_START]
            A_fit = exp_model(t_fit, fit_info["c"], fit_info["a"], fit_info.get("tau", tau), T_DIA_START)
            plt.plot(t_fit, A_fit, "r-", lw=2.0, label=f"Exp fit (τ={tau:.3f} s)")
        plt.title(f"Patient {patient_id} → Virtual {virt_id}: Carotid diastolic decay")
        plt.xlabel("Time (s)"); plt.ylabel("Area (m²)"); plt.legend()
        carotid_plot_path = str((PLOTS_DIR / f"patient_{patient_id}_virtual_{virt_id}_carotid_decay.png").resolve())
        plt.tight_layout(); plt.savefig(carotid_plot_path, dpi=160); plt.close()

    # ---- Hemodynamics (clinical units only) ----
    MAP_mmHg = float(row["bMAP"])
    SV_mL = float(row["SV"])
    HR_bpm = float(row["HR"])

    CO_mL_per_s = co_ml_per_s(SV_mL, HR_bpm)               # mL/s
    R_sys_mmHg_s_per_mL = (MAP_mmHg / CO_mL_per_s) if CO_mL_per_s > 0 else np.nan
    C_sa_mL_per_mmHg = (tau / R_sys_mmHg_s_per_mL) if (np.isfinite(tau) and np.isfinite(R_sys_mmHg_s_per_mL) and R_sys_mmHg_s_per_mL > 0) else np.nan

    # ---- Aortic area -> mean area + Z_ao (area-only) ----
    A_a, n_all_a = get_row_waveform(aorticA_mat, virt_id)
    loaded_aorticA = A_a is not None and A_a.size >= 5
    N_aorticA = int(A_a.size) if loaded_aorticA else 0
    mean_aortic_area_m2 = float(np.nanmean(A_a)) if loaded_aorticA else np.nan

    PWV_a = float(row.get("PWV_a", np.nan))
    Z_ao_mmHg_s_per_mL = np.nan
    if loaded_aorticA and np.isfinite(PWV_a) and mean_aortic_area_m2 > 0:
        Zao_SI = RHO * PWV_a / mean_aortic_area_m2              # Pa·s/m^3
        Z_ao_mmHg_s_per_mL = Zao_SI * PA_S_PER_M3_TO_MMHG_S_PER_ML

    # ---- Aortic velocity -> RAW zero-cross ED in fixed window ----
    U, n_all_u = get_row_waveform(aorticU_mat, virt_id)
    loaded_aorticU = U is not None and U.size >= 5
    N_aorticU = int(U.size) if loaded_aorticU else 0

    peak_u = np.nan
    ejection_duration_s = np.nan
    ed_note = ""
    if loaded_aorticU:
        t_u = np.arange(N_aorticU, dtype=float) / FS_HZ
        peak_u = float(np.nanmax(U))
        save_path = DEBUG_U_DIR / f"Udebug_patient_{patient_id}_virtual_{virt_id}.png" if DEBUG_ED and n_debug < DEBUG_MAX_PLOTS else None
        if save_path is not None:
            n_debug += 1
        ejection_duration_s, ed_note = find_ed_from_raw_zero_cross(
            t_u, U, window=ED_WINDOW, down_only=ED_DOWN_ONLY, eps_frac=ED_EPS_FRAC, save_plot_path=save_path
        )

    # ---- Collect ----
    records.append({
        "patient_sub_id": patient_id,
        "age": int(row["age"]),
        "bSBP_mmHg": float(row["bSBP"]),
        "bDBP_mmHg": float(row["bDBP"]),
        "bMAP_mmHg": MAP_mmHg,
        "SV_mL": SV_mL,
        "HR_bpm": HR_bpm,
        "virtual_ref_sub_id": virt_id,
        "PWV_a_m_per_s": PWV_a,
        "tau_s": float(tau) if np.isfinite(tau) else np.nan,
        "R_sys_mmHg_s_per_mL": float(R_sys_mmHg_s_per_mL) if np.isfinite(R_sys_mmHg_s_per_mL) else np.nan,
        "C_sa_mL_per_mmHg":    float(C_sa_mL_per_mmHg)    if np.isfinite(C_sa_mL_per_mmHg)    else np.nan,
        "Z_ao_mmHg_s_per_mL":  float(Z_ao_mmHg_s_per_mL)  if np.isfinite(Z_ao_mmHg_s_per_mL)  else np.nan,
        "mean_aortic_area_m2": mean_aortic_area_m2,
        "peak_aortic_velocity_m_per_s": peak_u,
        "ejection_duration_s": ejection_duration_s,   # <-- ED = time (s) from start to zero-cross in the window
        # Diagnostics
        "loaded_carotid": bool(loaded_carotid),
        "loaded_aortic_area": bool(loaded_aorticA),
        "loaded_aortic_u": bool(loaded_aorticU),
        "N_carotid": N_carotid, "N_aorticA": N_aorticA, "N_aorticU": N_aorticU,
        "ed_note": ed_note,
        "carotid_decay_plot": carotid_plot_path,
        "carotid_fit_ok": bool(fit_info.get("ok", False)),
        "carotid_fit_rmse": float(fit_info.get("rmse", np.nan)) if fit_info.get("ok", False) else np.nan,
        "carotid_fit_points": int(fit_info.get("n", 0)) if fit_info.get("ok", False) else 0,
    })

# ---------------- SAVE ----------------
result = pd.DataFrame(records).sort_values("patient_sub_id", na_position="last")
result.to_csv(RESULTS_CSV, index=False)

print(f"\nSaved results CSV: {RESULTS_CSV}")
print(f"Carotid plots: {PLOTS_DIR}")
print(f"Aortic velocity debug plots (first {DEBUG_MAX_PLOTS}): {DEBUG_U_DIR}")

if "carotid_fit_ok" in result:
    print(f"Exp fits OK: {int(result['carotid_fit_ok'].sum())}/{len(result)} subjects.")
if "ejection_duration_s" in result:
    n_ed = int(np.isfinite(result['ejection_duration_s']).sum())
    print(f"ED (zero-cross in {ED_WINDOW[0]:.2f}-{ED_WINDOW[1]:.2f} s) found for {n_ed}/{len(result)} subjects.")
