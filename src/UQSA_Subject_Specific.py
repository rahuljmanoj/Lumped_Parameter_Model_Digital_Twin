# -*- coding: utf-8 -*-
"""
LPM_V4.1_UQSA_AllInOne_10pct_UQdrop_SAkeep.py

Single-file UQ + SA for your 0D cardiovascular model (V4.1):
 - ED at dp/dt upstroke; ES at max(P/V) within AVO→AVC
 - Subject-specific ±10% bounds (θ-window) intersected with hard bounds
 - Sampling in z-space using the window mapped to [z_lo,z_hi] (log/lin safe)
 - UQ (MC): STRICT -> drop invalid (no replacement)
 - Sobol (LVEDP): KEEP ALL -> replace invalid with the median of valid Y (to preserve Saltelli structure)
 - Sobol (VALIDITY): sensitivity of probability(valid), unchanged
 - Exports V4.1 and legacy filenames

Author: Rahul Manoj (with Copilot assistance)
"""

# ======== USER SETTINGS ======================================================
EXCEL_PATH   = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx"
SHEET_NAME   = "Study_3_V4.1_T6"     # adjust if needed
#SUBJECT_LIST = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
                #344, 346, 347, 349, 351, 352]
#SUBJECT_LIST =  [354, 356, 357, 359, 360]
#SUBJECT_LIST =  [361, 362, 363, 364, 365]
SUBJECT_LIST =  [366, 367, 368, 369, 370, 371, 372, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 409, 410, 411, 412, 413, 414, 415]                # 999 = mean row; or [316, 319, ...]
SAVE_ROOT    = r".\Results_all_subjects_UQSA_LPM_4.1"

# Monte-Carlo (UQ)
N_MC     = 75000          # your requested 75k
MC_MODE  = "uniform"      # "uniform" in z-window, or "normal" around z_best
PRIOR_CV = 0.15           # only used if MC_MODE=="normal" (σ in z)

# Sobol (SA)
SOBOL_BASE_N =2048       # your requested base N
SOBOL_BOOT   = 2000       # your requested bootstrap resamples for CI
SECOND_ORDER = False      # can set True if you want S2 (costly)

# Subject-specific ±% windows (θ-space) intersected with hard bounds
PCT_WINDOW = {
    "R_sys":0.10, "Z_ao":0.10, "C_sa":0.10, "R_mv":0.10,
    "E_max":0.10, "E_min":0.10, "t_peak":0.10, "V_tot":0.10, "C_sv":0.10
}
# ============================================================================

import os
import time
import numpy as np
import pandas as pd

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc, norm

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

PHYS_LVEDP_RANGE = (4.0, 35.0)
EPS_NUM = 1e-12

# --------------------- helpers: formatting & logging -------------------------
def _sec_to_str(dt):
    if dt < 60: return f"{dt:.1f}s"
    m = int(dt // 60); s = dt - 60*m
    return f"{m:d}m {s:.1f}s"

def _hdr(title):
    line = "=" * (len(title) + 4)
    return f"\n{line}\n= {title} =\n{line}"

def _get(row, keys, default=np.nan):
    """Return first present, non-NaN field from candidate keys in a pandas Series."""
    if isinstance(keys, str): keys = [keys]
    for k in keys:
        if k in row and pd.notna(row[k]): return row[k]
    return default

# ========================= V4.1 MODEL (SubjectSimulator) =====================
class SubjectSimulator:
    def __init__(self, data_row, sub_id, save_root,
                 soft_weight_LVEDP=10.0, lambda_phys=100.0):
        self.sub_id = sub_id
        self.data_row = data_row
        self.save_root = save_root

        # weights/penalties
        self.soft_weight_LVEDP = soft_weight_LVEDP
        self.lambda_phys = lambda_phys

        # fixed hemodynamics & elastance
        self.cycles = 10
        self.P_ao0 = 70.0
        self.V_LV0 = 100.0
        self.P_th  = -4.0
        self.a = 1.55
        self.alpha1, self.alpha2 = 0.7, 1.17
        self.n1, self.n2 = 1.9, 21.9

        # subject fields
        self.SWE_velocity      = float(_get(data_row, ['SWE_vel_MVC', 'SWS (m/s)', 'SWE (m/s)'], 2.5))
        self.bMAP              = float(_get(data_row, ['bMAP', 'GT bMAP (mmHg)'], 90.0))
        self.Emin_mech_prior   = float(_get(data_row, ['Emin_isotonic_cal', 'E_min (mmHg/ml)'], 0.05))
        self.bpm               = float(_get(data_row, ['HR', 'HR (bpm)'], 60.0))

        self.T = 60.0 / self.bpm
        self.total = self.cycles * self.T
        self.dt = self.T / 500.0

        # paper-aligned ED/ES
        self.ED_DEF  = "dpdt_upstroke"
        self.ESP_DEF = "max_P_over_V"

        # parameters (nominal)
        self.param_names = ['R_sys','Z_ao','C_sa','R_mv','E_max','E_min','t_peak','V_tot','C_sv']
        self.params = {
            'R_sys': 2.09, 'Z_ao': 0.06, 'C_sa': 0.68, 'R_mv': 0.05,
            'E_max': 5.0,  'E_min': 0.05,'t_peak': 0.35,'V_tot': 300.0,'C_sv': 15.0,
            'T': self.T
        }

        # hard bounds
        self.bounds = [
            (0.5, 4.0),   (0.01, 1.0), (0.1, 10.0),
            (0.01, 0.1),  (0.9, 10.0), (0.01, 2.5),
            (0.1, 0.7),   (50.0, 2000),(0.5, 30.0)
        ]

        # IC
        V_sa0 = self.P_ao0 * self.params['C_sa']
        V_sv0 = self.params['V_tot'] - self.V_LV0 - V_sa0
        self.y0 = [self.V_LV0, V_sa0, V_sv0]

        # loss scaling
        self.weights = {
            'EDV':1, 'ESV':1, 'SV':0, 'EF':0,
            'bSBP':1, 'bDBP':1, 'bMAP':0, 'bPP':0,
            'LVOT_Flow_Peak':0, 'time_LVOT_Flow_Peak':0,
            'ET':1, 'IVRT':1, 'LVEDP':0.0
        }
        self.loss_scaling_mode = 'cohort_minmax'

        # prior for LVEDP
        self.sigma_prior     = 4.235  # mmHg RMSE
        self.prior_deadzone  = 3.0    # mmHg
        self.use_prior_deadzone = True
        self.kappa_prior     = 1.0

        self.subject_dir = os.path.join(self.save_root, str(self.sub_id))
        os.makedirs(self.subject_dir, exist_ok=True)

        self._precompute_bounds_arrays()

    # ---------- V4.1 z<->θ ----------
    def _precompute_bounds_arrays(self):
        self.lb = np.array([b[0] for b in self.bounds], float)
        self.ub = np.array([b[1] for b in self.bounds], float)
        self.scale_type = ['log','log','log','log','log','log','lin','log','log']

    def z_from_theta(self, theta):
        theta = np.asarray(theta, float)
        z = np.empty_like(theta)
        for i, st in enumerate(self.scale_type):
            lb, ub = self.lb[i], self.ub[i]
            if st == 'log':
                lb = max(lb, EPS_NUM); ub = max(ub, lb*(1+1e-12))
                z[i] = (np.log(theta[i]) - np.log(lb)) / (np.log(ub) - np.log(lb))
            else:
                z[i] = (theta[i] - lb) / (ub - lb)
        return np.clip(z, 0, 1)

    def theta_from_z(self, z):
        z = np.asarray(z, float); z = np.clip(z, 0, 1)
        theta = np.empty_like(z)
        for i, st in enumerate(self.scale_type):
            lb, ub = self.lb[i], self.ub[i]
            if st == 'log':
                lb = max(lb, EPS_NUM); ub = max(ub, lb*(1+1e-12))
                logθ = np.log(lb) + z[i]*(np.log(ub)-np.log(lb))
                theta[i] = np.exp(logθ)
            else:
                theta[i] = lb + z[i]*(ub-lb)
        return theta

    # ---------- model dynamics ----------
    def initial_state(self, p):
        V_sa0 = self.P_ao0 * p['C_sa']
        V_sv0 = p['V_tot'] - self.V_LV0 - V_sa0
        return [self.V_LV0, V_sa0, V_sv0]

    def elastance(self, t, p):
        tn = np.mod(t, p['T']) / p['t_peak']
        t1, t2 = tn / self.alpha1, tn / self.alpha2
        En = (t1**self.n1) / (1 + t1**self.n1)
        En *= 1.0 / (1 + t2**self.n2)
        return p['E_max'] * En * self.a + p['E_min']

    def circulation_odes(self, t, y, p):
        V_lv, V_sa, V_sv = y
        E_t = self.elastance(t, p)
        P_lv = E_t * V_lv + self.P_th
        P_sa = V_sa / p['C_sa']
        P_sv = V_sv / p['C_sv']
        Q_sv_lv = (P_sv - P_lv)/p['R_mv'] if P_sv > P_lv else 0.0
        Q_lv_ao = (P_lv - P_sa)/p['Z_ao'] if P_lv > P_sa else 0.0
        Q_sys   = (P_sa - P_sv)/p['R_sys'] if P_sa > P_sv else 0.0
        return [Q_sv_lv - Q_lv_ao, Q_lv_ao - Q_sys, Q_sys - Q_sv_lv]

    def run_simulation(self, params=None):
        if params is None: params = self.params
        y0 = self.initial_state(params)
        num_steps = int(np.round(self.total / self.dt))
        t_eval = np.linspace(0, self.total, num_steps + 1)
        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, params),
                        [0, self.total], y0, t_eval=t_eval, max_step=self.dt)
        t = sol.t
        V_lv, V_sa, V_sv = sol.y
        P_lv = self.elastance(t, params) * V_lv + self.P_th
        P_sa = V_sa / params['C_sa']
        P_ao = np.where(P_lv > P_sa, P_lv, P_sa)
        P_sv = V_sv / params['C_sv']
        Q_sv_lv = np.where(P_sv > P_lv, (P_sv - P_lv)/params['R_mv'], 0.0)
        Q_lv_ao = np.where(P_lv > P_sa, (P_lv - P_sa)/params['Z_ao'], 0.0)
        Q_sys   = np.where(P_sa > P_sv, (P_sa - P_sv)/params['R_sys'], 0.0)
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys

    def cycle_cutting_algo(self, t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao):
        dt = t[1] - t[0]; T = 60.0 / self.bpm
        spc = int(round(float(T/dt)))
        n = len(t)//spc
        if n < 2: raise RuntimeError("Not enough full cycles to slice out")
        start = (n-2)*spc; end = (n-1)*spc
        return {
            't': t[start:end]-t[start], 'P_ao':P_ao[start:end], 'P_lv':P_lv[start:end],
            'V_lv':V_lv[start:end], 'Q_sv':Q_sv_lv[start:end], 'Q_ao':Q_lv_ao[start:end]
        }

    # ---- event detection ----
    def _cycle_period(self, t):
        t = np.asarray(t, float); dt = float(t[1]-t[0])
        return float(t[-1]-t[0]+dt)

    def _dt_cyclic(self, t1, t2, T):
        return float(t2 - t1) if t2 >= t1 else float(t2 + T - t1)

    def _flow_segments(self, Q, thr_frac=0.01, abs_thr=1e-3, min_len=3):
        Q = np.asarray(Q, float)
        if np.max(Q) <= 0: return []
        thr = max(thr_frac*np.max(Q), abs_thr)
        mask = Q > thr
        if not np.any(mask): return []
        d = np.diff(mask.astype(int))
        starts = list(np.where(d==1)[0] + 1)
        ends   = list(np.where(d==-1)[0])
        if mask[0]:  starts = [0] + starts
        if mask[-1]: ends   = ends + [len(Q)-1]
        segs = []
        for s, e in zip(starts, ends):
            if e - s + 1 >= min_len: segs.append((int(s), int(e)))
        return segs

    def _choose_main_segment(self, Q, segs):
        if not segs: return None, None
        Q = np.asarray(Q, float)
        peaks = [np.max(Q[s:e+1]) for s, e in segs]
        k = int(np.argmax(peaks))
        return segs[k]

    def _valve_events_from_flows(self, t, Qao, Qmv, thr_ao=0.01, thr_mv=0.005):
        N = len(t)
        ao_segs = self._flow_segments(Qao, thr_frac=thr_ao, min_len=3)
        avo, avc = self._choose_main_segment(Qao, ao_segs)
        mv_segs = self._flow_segments(Qmv, thr_frac=thr_mv, min_len=3)
        if not mv_segs: return avo, avc, None, None
        if avo is None or avc is None:
            mvo = mv_segs[0][0]; mvc = mv_segs[-1][1]
            return avo, avc, int(mvo), int(mvc)
        avo_i, avc_i = int(avo), int(avc)
        starts = np.array([s for s, e in mv_segs], int)
        ends   = np.array([e for s, e in mv_segs], int)
        starts_shift = np.where(starts >= avc_i, starts, starts + N)
        mvo = int(starts[np.argmin(starts_shift)])
        ends_shift = np.where(ends <= avo_i, ends, ends - N)
        mvc = int(ends[np.argmax(ends_shift)])
        return avo_i, avc_i, mvo, mvc

    def _idx_esp_max_P_over_V(self, P, V, avo=None, avc=None):
        eps = 1e-6
        ratio = P/np.maximum(V, eps)
        if avo is not None and avc is not None:
            if avc >= avo: idx_range = np.arange(avo, avc+1)
            else:          idx_range = np.concatenate([np.arange(avo, len(P)), np.arange(0, avc+1)])
            return int(idx_range[np.argmax(ratio[idx_range])])
        return int(np.argmax(ratio))

    def extract_cycle_metrics(self, cyc):
        t   = np.asarray(cyc['t'], float)
        V   = np.asarray(cyc['V_lv'], float)
        P   = np.asarray(cyc['P_lv'], float)
        Qao = np.asarray(cyc['Q_ao'], float)
        Qmv = np.asarray(cyc['Q_sv'], float)
        Tcyc = self._cycle_period(t)

        avo_i, avc_i, mvo_i, mvc_i = self._valve_events_from_flows(t, Qao, Qmv, 0.01, 0.005)

        dPdt = np.gradient(P, t)
        if self.ED_DEF == "dpdt_upstroke":
            if mvc_i is not None:
                back_s = 0.25; dt = float(t[1]-t[0]); back_n = int(round(back_s/dt))
                i0 = max(0, int(mvc_i)-back_n); i1 = int(mvc_i)
                d = dPdt[i0:i1+1]
                crossings = np.where((d[:-1] <= 0) & (d[1:] > 0))[0] + 1
                ED_idx = int(i0 + crossings[-1]) if len(crossings) else int(np.argmax(V))
            else:
                i_peak = int(np.argmax(dPdt))
                crossings = np.where((dPdt[:-1] <= 0) & (dPdt[1:] > 0))[0] + 1
                crossings = crossings[crossings < i_peak]
                ED_idx = int(crossings[-1]) if len(crossings) else int(np.argmax(V))
        else:
            ED_idx = int(np.argmax(V))

        EDV   = float(V[ED_idx])
        LVEDP = float(P[ED_idx])

        if self.ESP_DEF == "max_P_over_V":
            es_idx = self._idx_esp_max_P_over_V(P, V, avo_i, avc_i)
        else:
            es_idx = int(np.argmin(V))
        ESP = float(P[es_idx])
        ESV = float(V[es_idx])

        ESV_minV      = float(np.min(V))
        ESV_idx_minV  = int(np.argmin(V))
        EDV_maxV      = float(np.max(V))

        peak_idx = int(np.argmax(Qao))
        Q_peak   = float(Qao[peak_idx])
        t_peak   = float(t[peak_idx])

        thresh = 0.01 * Q_peak
        mask   = Qao > thresh
        edur_start_idx = int(np.where(mask)[0][0]) if mask.any() else ED_idx
        edur_end_idx   = int(np.where(mask)[0][-1]) if mask.any() else ED_idx
        EDur = float(t[edur_end_idx] - t[edur_start_idx])

        if avc_i is not None and mvo_i is not None:
            IVRT = self._dt_cyclic(float(t[avc_i]), float(t[mvo_i]), Tcyc)
        else:
            IVRT = np.nan

        return (
            {
                'EDV': EDV, 'ESV': ESV, 'ESV_minV': ESV_minV, 'LVEDP': LVEDP,
                'ESP': ESP, 'EDV_maxV': EDV_maxV,
                'Q_peak': Q_peak, 't_peak': t_peak, 'EDur': EDur, 'ET': EDur, 'IVRT': IVRT
            },
            {
                'EDV_idx': int(ED_idx), 'ESV_idx': int(es_idx),
                'ESV_minV_idx': int(ESV_idx_minV), 'peak_idx': int(peak_idx),
                'edur_start': int(edur_start_idx), 'edur_end': int(edur_end_idx),
                'avo_idx': avo_i, 'avc_idx': avc_i, 'mvo_idx': mvo_i, 'mvc_idx': mvc_i
            }
        )

# ------------------------ optional Excel param override -----------------------
def try_override_params_from_row(sim, data_row):
    col_map = {
        "R_sys (mmHg s/ml)": "R_sys", "Z_ao (mmHg s/ml)": "Z_ao", "C_sa (ml/mmHg)": "C_sa",
        "R_mv (mmHg s/ml)": "R_mv", "E_max (mmHg/ml)": "E_max", "E_min (mmHg/ml)": "E_min",
        "t_peak (s)": "t_peak", "V_tot (ml)": "V_tot", "C_sv (ml/mmHg)": "C_sv",
        "Emin_isotonic_cal": "E_min",
    }
    updates = 0
    for src, dst in col_map.items():
        if src in data_row and pd.notna(data_row[src]):
            try: sim.params[dst] = float(data_row[src]); updates += 1
            except: pass
    if updates>0 and hasattr(sim, "_precompute_bounds_arrays"):
        sim._precompute_bounds_arrays()
    return updates

# ==================== subject-specific ±% window (θ) → z-window ===============
def make_subject_bounds(sim, pct=None):
    """Intersect hard bounds with a ±pct window around current sim.params."""
    names = list(sim.param_names)
    lb_hard = np.asarray(sim.lb, float); ub_hard = np.asarray(sim.ub, float)
    if pct is None: pct = {n:0.10 for n in names}
    lb_win = np.empty_like(lb_hard); ub_win = np.empty_like(ub_hard)
    for i, n in enumerate(names):
        base = float(sim.params[n]); r = float(pct.get(n, 0.10))
        lo_w, hi_w = (1-r)*base, (1+r)*base
        lb_win[i] = max(lb_hard[i], min(lo_w, hi_w))
        ub_win[i] = min(ub_hard[i], max(lo_w, hi_w))
    assert np.all(np.isfinite(lb_win)) and np.all(np.isfinite(ub_win)), "Non-finite bounds"
    assert np.all(lb_win < ub_win), "Lower bound not < upper bound"
    return names, lb_win, ub_win  # θ-window

def window_to_z_interval(sim, lb_win, ub_win):
    """Map θ-window [lb_win, ub_win] into per-dimension z-window [z_lo, z_hi]."""
    z_lo = sim.z_from_theta(lb_win); z_hi = sim.z_from_theta(ub_win)
    return np.minimum(z_lo, z_hi), np.maximum(z_lo, z_hi)

def z_best_from_sim(sim):
    theta_best = np.array([float(sim.params[n]) for n in sim.param_names], float)
    return sim.z_from_theta(theta_best)

def z_to_theta(sim, Z):
    return np.vstack([sim.theta_from_z(z) for z in Z])

def sample_Z_in_window(N, z_lo, z_hi, seed=0, mode="uniform", z_best=None, cv=0.15):
    """Sample z inside [z_lo, z_hi]."""
    d = len(z_lo)
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        U = qmc.LatinHypercube(d=d, seed=seed).random(N)
        return z_lo + U*(z_hi - z_lo)
    elif mode == "normal":
        if z_best is None:
            raise ValueError("z_best required for mode='normal'")
        Z = rng.normal(loc=z_best, scale=cv, size=(N, d))
        return np.clip(Z, z_lo, z_hi)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")

# ===================== evaluation utilities ==================================
def _lvedp_once(sim, theta):
    """Return LVEDP or np.nan (STRICT: invalid -> NaN)."""
    try:
        params = sim.params.copy()
        for name, val in zip(sim.param_names, theta):
            params[name] = float(val)
        t, Pao, Plv, Vlv, Qmv, Qao, Qsys = sim.run_simulation(params)
        cyc = sim.cycle_cutting_algo(t, Pao, Plv, Vlv, Qmv, Qao)
        mets, _ = sim.extract_cycle_metrics(cyc)
        y = float(mets["LVEDP"])
        if not (np.isfinite(y) and PHYS_LVEDP_RANGE[0] <= y <= PHYS_LVEDP_RANGE[1]):
            return np.nan
        return y
    except Exception:
        return np.nan

def eval_matrix_parallel_theta(sim, Theta, save_dir=None, label="eval"):
    """
    Evaluate LVEDP for all rows of Θ in parallel.
    Returns Y (float array) and mask_bad (True where invalid).
    Does NOT replace invalids (decision is made at caller).
    """
    def one_row(theta): return _lvedp_once(sim, theta)
    Y = Parallel(n_jobs=-1, verbose=10)(delayed(one_row)(theta) for theta in Theta)
    Y = np.asarray(Y, float)
    mask_bad = ~np.isfinite(Y)
    n_bad = int(mask_bad.sum()); n_all = len(Y)
    print(f"[info] {label}: invalid fraction = {100.0*n_bad/n_all:.2f}% ({n_bad}/{n_all})")
    if n_bad and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(Theta[mask_bad], columns=sim.param_names).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_{label}_invalid_params.csv"), index=False
        )
    return Y, mask_bad

# =========================== MC-UQ (drop invalid) ============================
def mc_uq_lvedp(sim, N=2000, seed=0, mode="uniform", prior_cv=0.15, save_dir=None, pct_window=None):
    # window
    if pct_window is not None:
        names, lb_win, ub_win = make_subject_bounds(sim, pct=pct_window)
        z_lo, z_hi = window_to_z_interval(sim, lb_win, ub_win)
    else:
        names = list(sim.param_names); z_lo = np.zeros(len(names)); z_hi = np.ones(len(names))

    # sample & evaluate
    z_best = z_best_from_sim(sim)
    Z = sample_Z_in_window(N=N, z_lo=z_lo, z_hi=z_hi, seed=seed, mode=mode, z_best=z_best, cv=prior_cv)
    Theta = z_to_theta(sim, Z)
    t0 = time.perf_counter()
    Y_raw, mask_bad = eval_matrix_parallel_theta(sim, Theta, save_dir=save_dir, label="mc")
    t1 = time.perf_counter()
    valid = ~mask_bad
    if valid.sum() == 0:
        raise RuntimeError(f"No valid LVEDP samples in MC-UQ (sub {sim.sub_id})")

    Theta_v, Z_v, Y_v = Theta[valid], Z[valid], Y_raw[valid]
    mean = float(np.mean(Y_v)); sd = float(np.std(Y_v, ddof=1))
    p025, p975 = np.percentile(Y_v, [2.5, 97.5])

    # Console summary
    print(_hdr(f"UQ Summary (sub {sim.sub_id})"))
    print(f"N_total: {N:,} | N_valid: {valid.sum():,} | Invalid: {mask_bad.mean()*100:.2f}%")
    print(f"LVEDP mean={mean:.2f}  sd={sd:.2f}  95%PI=[{p025:.2f}, {p975:.2f}]")
    print(f"UQ compute time: {_sec_to_str(t1-t0)}")

    # Save outputs
    pd.DataFrame(Theta_v, columns=names).to_csv(os.path.join(save_dir, f"{sim.sub_id}_uq_theta.csv"), index=False)
    pd.DataFrame(Z_v,     columns=names).to_csv(os.path.join(save_dir, f"{sim.sub_id}_uq_z.csv"), index=False)
    pd.DataFrame({"LVEDP": Y_v}).to_csv(os.path.join(save_dir, f"{sim.sub_id}_uq_lvedp.csv"), index=False)
    pd.Series({"mean":mean, "sd":sd, "PI95":(float(p025), float(p975)), "N":int(valid.sum()), "N_total":int(N)}).to_json(
        os.path.join(save_dir, f"{sim.sub_id}_uq_summary.json")
    )
    # legacy aliases for your readers/plots
    pd.DataFrame(Theta_v, columns=names).to_csv(os.path.join(save_dir, f"{sim.sub_id}_uq_params.csv"), index=False)
    pd.DataFrame({"LVEDP": Y_v}).to_csv(os.path.join(save_dir, f"{sim.sub_id}_uq_samples.csv"), index=False)

    # Save the window for traceability
    if pct_window is not None:
        pd.DataFrame({"param":names, "theta_low":lb_win, "theta_high":ub_win}).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_uqsa_window_theta.csv"), index=False
        )
        pd.DataFrame({"param":names, "z_low":z_lo, "z_high":z_hi}).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_uqsa_window_z.csv"), index=False
        )

    # Return bundle
    return {"names": names, "Z": Z_v, "Theta": Theta_v, "Y": Y_v, "mask_bad": mask_bad,
            "summary": {"mean":mean, "sd":sd, "PI95":(float(p025), float(p975)), "N":int(valid.sum()), "N_total":int(N)}}

# ========== Sobol SA (LVEDP keep-all via median replacement; VALIDITY std) ===
def sobol_with_ci(problem, Y, calc_second_order=False, n_resamples=500, conf_level=0.95, seed=0):
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
    rng = np.random.default_rng(seed)
    n = len(Y); S1_s, ST_s = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        Si_r = sobol_analyze.analyze(problem, Y[idx], calc_second_order=calc_second_order, print_to_console=False)
        S1_s.append(Si_r["S1"]); ST_s.append(Si_r["ST"])
    S1_s, ST_s = np.asarray(S1_s), np.asarray(ST_s)
    alpha = (1 - conf_level)/2.0
    Si["S1_conf"] = np.nanpercentile(S1_s,[100*alpha,100*(1-alpha)],axis=0).T
    Si["ST_conf"] = np.nanpercentile(ST_s,[100*alpha,100*(1-alpha)],axis=0).T
    return Si

def sobol_sa_validity(sim, N=1024, seed=0, n_resamples=500, save_dir=None, pct_window=None, second_order=False):
    names = list(sim.param_names); d = len(names)
    if pct_window is not None:
        _, lb_win, ub_win = make_subject_bounds(sim, pct=pct_window)
        z_lo, z_hi = window_to_z_interval(sim, lb_win, ub_win)
    else:
        z_lo = np.zeros(d); z_hi = np.ones(d)

    problem = {"num_vars": d, "names": names, "bounds": [[0.0, 1.0]]*d}
    U = sobol_sample.sample(problem, N, calc_second_order=second_order)
    Z = z_lo + U*(z_hi - z_lo)
    Theta = z_to_theta(sim, Z)

    Y_lvedp, mask_bad = eval_matrix_parallel_theta(sim, Theta, save_dir=save_dir, label="sobol_validity")
    Y_valid = (~mask_bad).astype(float)  # 0/1 validity

    # Analyze and save
    t0 = time.perf_counter()
    Si = sobol_with_ci(problem, Y_valid, calc_second_order=False, n_resamples=n_resamples)
    t1 = time.perf_counter()
    print(_hdr(f"Sobol VALIDITY (sub {sim.sub_id})"))
    print(f"Base N={N:,} | d={d} | Total evals={len(Y_valid):,} | Invalid (for LVEDP)={mask_bad.mean()*100:.2f}%")
    print(f"P(valid) = {Y_valid.mean():.3f} | SA time: {_sec_to_str(t1-t0)}")

    # Save results
    S1c, STc = np.asarray(Si["S1_conf"]), np.asarray(Si["ST_conf"])
    df_sa = pd.DataFrame({
        "sub_id": sim.sub_id, "metric": "VALIDITY", "param": names,
        "S1": Si["S1"].tolist(), "S1_low": S1c[:,0], "S1_high": S1c[:,1],
        "ST": Si["ST"].tolist(), "ST_low": STc[:,0], "ST_high": STc[:,1]
    })
    df_sa.to_csv(os.path.join(save_dir, f"{sim.sub_id}_sobol_validity.csv"), index=False)

    # Also write a sobol_params.csv (legacy) with the Theta used for VALIDITY SA
    pd.DataFrame(Theta, columns=names).to_csv(os.path.join(save_dir, f"{sim.sub_id}_sobol_params.csv"), index=False)

    return {"names": names, "Theta": Theta, "U": U, "Si": Si, "mask_bad": mask_bad}

def sobol_sa_lvedp_keep_all(sim, N=1024, seed=0, n_resamples=500, save_dir=None, pct_window=None, second_order=False):
    """
    Sobol SA on LVEDP where we KEEP the full Saltelli design:
      - Evaluate Y
      - If invalids exist, REPLACE invalid Y with median(valid Y)
      - Then run SALib analyze() on the full vector
    This avoids retries and preserves the estimator structure.
    """
    names = list(sim.param_names); d = len(names)
    if pct_window is not None:
        _, lb_win, ub_win = make_subject_bounds(sim, pct=pct_window)
        z_lo, z_hi = window_to_z_interval(sim, lb_win, ub_win)
    else:
        z_lo = np.zeros(d); z_hi = np.ones(d)

    problem = {"num_vars": d, "names": names, "bounds": [[0.0, 1.0]]*d}
    U = sobol_sample.sample(problem, N, calc_second_order=second_order)
    Z = z_lo + U*(z_hi - z_lo)
    Theta = z_to_theta(sim, Z)

    Y, mask_bad = eval_matrix_parallel_theta(sim, Theta, save_dir=save_dir, label="sobol_lvedp")
    n_bad = int(mask_bad.sum())
    if n_bad > 0:
        med = float(np.nanmedian(Y))
        Y[mask_bad] = med
        print(f"[note] Sobol LVEDP: replaced {n_bad} invalid outputs with median={med:.2f} (SA only).")

    # Analyze and save
    t0 = time.perf_counter()
    Si = sobol_with_ci(problem, Y, calc_second_order=second_order, n_resamples=n_resamples)
    t1 = time.perf_counter()
    print(_hdr(f"Sobol LVEDP (sub {sim.sub_id})"))
    print(f"Base N={N:,} | d={d} | Total evals={len(Y):,} | Invalid replaced={n_bad}")
    print(f"SA time: {_sec_to_str(t1-t0)}")

    # Save results
    S1c, STc = np.asarray(Si["S1_conf"]), np.asarray(Si["ST_conf"])
    df_sa = pd.DataFrame({
        "sub_id": sim.sub_id, "metric": "LVEDP", "param": names,
        "S1": Si["S1"].tolist(), "S1_low": S1c[:,0], "S1_high": S1c[:,1],
        "ST": Si["ST"].tolist(), "ST_low": STc[:,0], "ST_high": STc[:,1]
    })
    df_sa.to_csv(os.path.join(save_dir, f"{sim.sub_id}_sobol_lvedp.csv"), index=False)
    return {"names": names, "Theta": Theta, "U": U, "Si": Si, "mask_bad": mask_bad}

# ============================ Orchestration & Saving ==========================
def run_uq_sa_for_subject(sim, save_dir, N_mc, sobol_baseN, sobol_boot,
                          mc_mode, prior_cv, pct_window, second_order):
    os.makedirs(save_dir, exist_ok=True)

    # Show hard bounds and center θ*
    print(_hdr(f"Start (sub {sim.sub_id})"))
    for k in sim.param_names:
        print(f"θ* {k:7s} = {float(sim.params[k]):.6g}")
    for n, lo, hi in zip(sim.param_names, sim.lb, sim.ub):
        print(f"hard {n:7s}: {lo:.6g} -> {hi:.6g}")

    # Compute & print the active subject-specific window
    names, lb_win, ub_win = make_subject_bounds(sim, pct=pct_window)
    z_lo, z_hi = window_to_z_interval(sim, lb_win, ub_win)

    print(_hdr(f"Active Window (sub {sim.sub_id})"))
    print("θ-window:")
    for n, lo, hi in zip(names, lb_win, ub_win):
        cen = float(sim.params[n])
        print(f"  {n:7s}: {lo:.6g} -> {hi:.6g} (center={cen:.6g})")
    print("z-window:")
    for n, lo, hi in zip(names, z_lo, z_hi):
        print(f"  {n:7s}: {lo:.4f} -> {hi:.4f}")

    # Save window CSVs
    pd.DataFrame({"param":names, "theta_low":lb_win, "theta_high":ub_win}).to_csv(
        os.path.join(save_dir, f"{sim.sub_id}_uqsa_window_theta.csv"), index=False
    )
    pd.DataFrame({"param":names, "z_low":z_lo, "z_high":z_hi}).to_csv(
        os.path.join(save_dir, f"{sim.sub_id}_uqsa_window_z.csv"), index=False
    )

    # Save conditions for reproducibility
    df_cond = pd.DataFrame({
        "param": names, "hard_lower": sim.lb, "hard_upper": sim.ub,
        "theta_low": lb_win, "theta_high": ub_win,
        "best_fit": [float(sim.params[n]) for n in names],
        "scale_type": getattr(sim, "scale_type", ["?"]*len(names))
    })
    df_cond.to_csv(os.path.join(save_dir, f"{sim.sub_id}_uqsa_conditions.csv"), index=False)

    # ---- UQ (drop invalid) ----
    mc = mc_uq_lvedp(sim, N=N_mc, seed=0, mode=mc_mode, prior_cv=prior_cv, save_dir=save_dir, pct_window=pct_window)

    # ---- Sobol VALIDITY (0/1) ----
    sa_valid = sobol_sa_validity(sim, N=sobol_baseN, seed=0, n_resamples=sobol_boot,
                                 save_dir=save_dir, pct_window=pct_window, second_order=False)

    # ---- Sobol LVEDP (keep-all with median replacement) ----
    sa_lvedp = sobol_sa_lvedp_keep_all(sim, N=sobol_baseN, seed=0, n_resamples=sobol_boot,
                                       save_dir=save_dir, pct_window=pct_window, second_order=second_order)
    # Top contributors (reporting convenience)
    if sa_lvedp is not None:
        names = sa_lvedp["names"]
        ST = np.asarray(sa_lvedp["Si"]["ST"])
        S1 = np.asarray(sa_lvedp["Si"]["S1"])
        top_st = sorted(zip(names, ST), key=lambda x: -x[1])[:3]
        top_s1 = sorted(zip(names, S1), key=lambda x: -x[1])[:3]
        print(f"Top ST (LVEDP): {[(n, round(v,3)) for n,v in top_st]}")
        print(f"Top S1 (LVEDP): {[(n, round(v,3)) for n,v in top_s1]}")

    # VALIDITY summary
    names_v = sa_valid["names"]
    ST_v = np.asarray(sa_valid["Si"]["ST"])
    top_st_v = sorted(zip(names_v, ST_v), key=lambda x: -x[1])[:3]
    print(f"Top ST (VALIDITY): {[(n, round(v,3)) for n,v in top_st_v]}")

    return mc, sa_lvedp, sa_valid

# ================================== MAIN =====================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Load Excel (optional)
    df = None
    if os.path.exists(EXCEL_PATH):
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=3)
        except Exception as e:
            print(f"[warn] Excel read with header=3 failed: {e}")
            try:
                df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
                print("[info] Retried without header=3")
            except Exception as ee:
                print(f"[warn] Excel read still failed: {ee} -> continue with defaults")
                df = None
    else:
        print(f"[warn] Excel not found at: {EXCEL_PATH} -> continue with defaults")

    for sub_id in SUBJECT_LIST:
        # Prepare subject row
        if df is not None and 'Sub ID' in df.columns:
            row = df.loc[df['Sub ID'] == sub_id]
            data_row = row.squeeze() if len(row) > 0 else pd.Series(dtype=float)
            if len(row) == 0:
                print(f"[warn] Sub ID {sub_id} not found in Excel; using defaults.")
        else:
            data_row = pd.Series(dtype=float)

        sim = SubjectSimulator(data_row, sub_id, SAVE_ROOT)
        updates = try_override_params_from_row(sim, data_row)
        if updates: print(f"[info] Overrode {updates} param(s) from Excel for sub {sub_id}")

        save_dir = os.path.join(SAVE_ROOT, str(sub_id))
        t0 = time.perf_counter()
        mc, sa_l, sa_v = run_uq_sa_for_subject(
            sim, save_dir,
            N_mc=N_MC, sobol_baseN=SOBOL_BASE_N, sobol_boot=SOBOL_BOOT,
            mc_mode=MC_MODE, prior_cv=PRIOR_CV, pct_window=PCT_WINDOW,
            second_order=SECOND_ORDER
        )
        t1 = time.perf_counter()

        s = mc["summary"]
        print(_hdr(f"DONE (sub {sub_id})"))
        print(f"UQ LVEDP (valid only): mean={s['mean']:.2f}, sd={s['sd']:.2f}, 95%PI={tuple(round(x,2) for x in s['PI95'])} | N_valid={s['N']:,}/{s['N_total']:,}")
        print(f"Total wall time: {_sec_to_str(t1-t0)}")