#1. Save csv file after optimisation of each subject, and append to the file.
#2. Save additional information - DE loss and time taken, L-BFGS-B loss and time taken
#3. Save a csv file with optimised values and model output after DE alone for each subject.
#4. Integrated the new params for R_sys, Z_ao and C_sa as per the KNN Virtual to Patient Matching Algo.
#5. Modified the non_physiological range penalty to a hinge to bound method.
#6. Fixed the SWS-LVEDP soft prior weight based on the maximum‑a‑posteriori formulation in absolute units, prior based on RMSE of the regression model.
#7. Fixed the scaling of loss function error by physiological constants K as cohort mean as in Nikolai's paper (implimented as cohort_minmax) and Hybrid WLS
#8. Emin is fixed before PE to the isotonic calibration self.Emin_mech_prior = data_row['Emin_isotonic_cal']
#9. Redundant variables are removed in the loss function, added IVRT to loss function and changed ED to ET.
#10. LVOT Flow magnitude and time of peak flow are removed from loss function.
#11. Model params are normalized to 0,1 before optimization (z_from_thetaa, theta_from_z)
#12. ESP and EDP are identified based on dp/dt max and maximal P/V values, as per literature - Bezy et al, Caenen et al

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import os
import time
from multiprocessing import cpu_count

class SubjectSimulator:
    def __init__(self, data_row, sub_id, save_root, soft_weight_LVEDP = 10, lambda_phys = 100.0):
        """Initialize all subject-specific and fixed simulation parameters."""
        self.sub_id = sub_id
        self.data_row = data_row
        self.save_root = save_root
        # Weights for loss function
        self.soft_weight_LVEDP = soft_weight_LVEDP
        self.lambda_phys = lambda_phys

        # Fixed simulation and elastance parameters
        self.cycles = 10
        self.P_ao0 = 70
        self.V_LV0 = 100.0
        self.P_th = -4.0

        self.a = 1.55
        self.alpha1 = 0.7
        self.alpha2 = 1.17
        self.n1 = 1.9
        self.n2 = 21.9

        self.SWE_velocity = data_row['SWE_vel_MVC']
        self.bMAP = data_row['bMAP']

        #change here
        #self.Emin_mech_prior = data_row['Emin_mech_prior_thin_wall_k_2.5']
        #self.Emin_mech_prior = data_row['Emin_mech_thick_wall_scaled_0_2.5']
        #self.Emin_mech_prior = data_row['Emin_exp_prior']
        self.Emin_mech_prior = data_row['Emin_isotonic_cal']


        self.bpm = data_row['HR']
        self.T = 60.0 / self.bpm
        self.total = self.cycles * self.T
        self.dt = self.T / 500.0

        # === Paper-aligned ED/ES definitions used in optimization ===
        self.ED_DEF = "dpdt_upstroke"  # ED at onset of +dP/dt (ED corner)
        self.ESP_DEF = "max_P_over_V"  # ESP at max(P/V) during ejection

        # Model parameter names and initial guess (for bounds)
        self.param_names = ['R_sys', 'Z_ao', 'C_sa', 'R_mv', 'E_max', 'E_min', 't_peak', 'V_tot', 'C_sv']
        self.params = {
            'R_sys': 2.09, #'R_sys': data_row['R_sys'],#change here
            'Z_ao': 0.06, #'Z_ao': data_row['Z_ao'],   #change here
            'C_sa':0.68, #'C_sa': data_row['C_sa'],    #change here
            'R_mv': 0.05,
            'E_max': 5,
            'E_min': data_row['Emin_isotonic_cal'],   #'E_min': 0.05, #change here
            't_peak': 0.35,
            'V_tot': 300,
            'C_sv': 15,
            'T': self.T
        }
        # Bounds for optimization (subject specific)
        self.bounds = [
            (0.5, 4.5),#(self.params['R_sys'] * (1 - 0.20), self.params['R_sys'] * (1 + 0.20)),  #change here
            (0.01, 1.0),#(self.params['Z_ao'] * (1 - 0.20), self.params['Z_ao'] * (1 + 0.20)), #change here
            (0.1, 10.0),#(self.params['C_sa'] * (1 - 0.20), self.params['C_sa'] * (1 + 0.20)), #change here
            (0.01, 0.1),     # R_mv
            (0.9, 10.0),     # E_max
            (self.params['E_min'] * (1 - 0.70), self.params['E_min'] * (1 + 0.70)), #(0.01, 2.5),    #change here
            (0.1, 0.7),      # t_peak
            (50.0, 2000),    # V_tot
            (0.5, 30.0)      # C_sv
        ]
        # Initial conditions
        V_sa0 = self.P_ao0 * self.params['C_sa']
        V_sv0 = self.params['V_tot'] - self.V_LV0 - V_sa0
        self.y0 = [self.V_LV0, V_sa0, V_sv0]
        # Metric weights for loss calculation

        self.weights = {
            'EDV': 1, 'ESV': 1, 'SV': 0, 'EF': 0,
            'bSBP': 1, 'bDBP': 1, 'bMAP': 0, 'bPP': 0,
            'LVOT_Flow_Peak': 0, 'time_LVOT_Flow_Peak': 0,
            'ET': 1,
            'IVRT': 1,
            'LVEDP': 0.0
        }
        # add a single toggle
        self.loss_scaling_mode = 'cohort_minmax' # or   'hybrid_wls'

        # --- MAP soft-prior settings (calibrated from your GT vs. prediction pairs) ---
        # RMSE of your SWS->LVEDP regression in absolute mmHg:
        self.sigma_prior = 4.235  # mmHg

        # Optional: clinical tolerance dead-zone; no prior penalty within ±3 mmHg of the reg. mean
        self.prior_deadzone = 3.0  # mmHg
        self.use_prior_deadzone = True

        # Optional: tiny multiplicative "fudge" around the MAP value (keep at 1.0 unless you do CV tuning)
        self.kappa_prior = 1.0  # {0.5, 1.0, 2.0}, etc. 1.0 = pure MAP

        # For saving plots/results
        self.subject_dir = os.path.join(self.save_root, str(self.sub_id))
        os.makedirs(self.subject_dir, exist_ok=True)

        self._precompute_bounds_arrays()
    # === V4 NORMALIZATION: Parameter scaling helpers (θ ↔ z) ================
    def _precompute_bounds_arrays(self):
        """Cache bounds + scaling metadata."""
        self.lb = np.array([b[0] for b in self.bounds], dtype=float)
        self.ub = np.array([b[1] for b in self.bounds], dtype=float)
        self.span = self.ub - self.lb

        # Order must match self.param_names:
        # ['R_sys','Z_ao','C_sa','R_mv','E_max','E_min','t_peak','V_tot','C_sv']
        # log = strictly positive wide-range; lin = bounded time fraction
        self.scale_type = ['log', 'log', 'log', 'log', 'log', 'log', 'lin', 'log', 'log']
        self._eps = 1e-12  # numerical safety

    def z_from_theta(self, theta):
        """Map physical parameters θ -> z in [0,1]^d (log/lin per dimension)."""
        theta = np.asarray(theta, dtype=float)
        z = np.empty_like(theta)
        for i, st in enumerate(self.scale_type):
            lb, ub = self.lb[i], self.ub[i]
            if st == 'log':
                lb = max(lb, self._eps)
                ub = max(ub, lb * (1.0 + 1e-12))
                z[i] = (np.log(theta[i]) - np.log(lb)) / (np.log(ub) - np.log(lb))
            else:  # 'lin'
                z[i] = (theta[i] - lb) / (ub - lb)
        return np.clip(z, 0.0, 1.0)

    def theta_from_z(self, z):
        """Map z in [0,1]^d -> physical parameters θ (log/lin per dimension)."""
        z = np.asarray(z, dtype=float)
        z = np.clip(z, 0.0, 1.0)
        theta = np.empty_like(z)
        for i, st in enumerate(self.scale_type):
            lb, ub = self.lb[i], self.ub[i]
            if st == 'log':
                lb = max(lb, self._eps)
                ub = max(ub, lb * (1.0 + 1e-12))
                logθ = np.log(lb) + z[i] * (np.log(ub) - np.log(lb))
                theta[i] = np.exp(logθ)
            else:  # 'lin'
                theta[i] = lb + z[i] * (ub - lb)
        return theta

    def objective_z(self, z):
        """Objective in z-space: transform then reuse existing objective(θ)."""
        theta = self.theta_from_z(z)
        return self.objective(theta)

    def initial_state(self, p):#each simulation should use a new set of initial conditions
        V_sa0 = self.P_ao0 * p['C_sa']
        V_sv0 = p['V_tot'] - self.V_LV0 - V_sa0
        return [self.V_LV0, V_sa0, V_sv0]

    # -- Cardiovascular model and simulation methods --
    def elastance(self, t, p):
        tn = np.mod(t, p['T']) / p['t_peak']
        t1 = tn / self.alpha1
        t2 = tn / self.alpha2
        En = (t1 ** self.n1) / (1 + t1 ** self.n1)
        En *= 1.0 / (1 + t2 ** self.n2)
        return p['E_max'] * En * self.a + p['E_min']

    def circulation_odes(self, t, y, p):
        V_lv, V_sa, V_sv = y
        E_t = self.elastance(t, p)
        P_lv = E_t * V_lv + self.P_th
        P_sa = V_sa / p['C_sa']
        P_sv = V_sv / p['C_sv']
        Q_sv_lv = (P_sv - P_lv) / p['R_mv'] if P_sv > P_lv else 0.0
        Q_lv_ao = (P_lv - P_sa) / p['Z_ao'] if P_lv > P_sa else 0.0
        Q_sys   = (P_sa - P_sv) / p['R_sys'] if P_sa > P_sv else 0.0
        return [Q_sv_lv - Q_lv_ao,
                Q_lv_ao - Q_sys,
                Q_sys - Q_sv_lv]

    def run_simulation(self, params=None):
        """Simulate model for a set of parameters."""
        if params is None:
            params = self.params
        y0 = self.initial_state(params)  # <-- use params-consistent ICs
        num_steps = int(np.round(self.total / self.dt))
        t_eval = np.linspace(0, self.total, num_steps + 1)
        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, params),
                        [0, self.total], y0,
                        t_eval=t_eval, max_step=self.dt)
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
        """Cut out the last full cardiac cycle from simulation arrays."""
        dt = t[1] - t[0]
        T  = 60.0 / self.bpm
        spc = int(round(float(T / dt)))
        n   = len(t) // spc
        if n < 2:
            raise RuntimeError("Not enough full cycles to slice out")
        start = (n-2) * spc
        end   = (n-1) * spc
        return {
            't':    t[start:end] - t[start],
            'P_ao': P_ao[start:end],
            'P_lv': P_lv[start:end],
            'V_lv': V_lv[start:end],
            'Q_sv': Q_sv_lv[start:end],
            'Q_ao': Q_lv_ao[start:end]
        }

    def _cycle_period(self, t):
        t = np.asarray(t, dtype=float)
        dt = float(t[1] - t[0])
        return float(t[-1] - t[0] + dt)

    def _dt_cyclic(self, t1, t2, T):
        # positive time difference with wrap-around
        return float(t2 - t1) if t2 >= t1 else float(t2 + T - t1)

    def _flow_segments(self, Q, thr_frac=0.01, abs_thr=1e-3, min_len=3):
        Q = np.asarray(Q, dtype=float)
        if np.max(Q) <= 0:
            return []
        thr = max(thr_frac * np.max(Q), abs_thr)
        mask = Q > thr
        if not np.any(mask):
            return []
        d = np.diff(mask.astype(int))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0])
        if mask[0]:
            starts = [0] + starts
        if mask[-1]:
            ends = ends + [len(Q) - 1]
        segs = []
        for s, e in zip(starts, ends):
            if e - s + 1 >= min_len:
                segs.append((int(s), int(e)))
        return segs

    def _choose_main_segment(self, Q, segs):
        if not segs:
            return None, None
        Q = np.asarray(Q, dtype=float)
        peaks = [np.max(Q[s:e + 1]) for s, e in segs]
        k = int(np.argmax(peaks))
        return segs[k]

    def _valve_events_from_flows(self, t, Qao, Qmv, thr_ao=0.01, thr_mv=0.005):
        """
        Compute AVO/AVC from aortic flow, and MVO/MVC from mitral flow.
        Returns (avo, avc, mvo, mvc) indices. Some can be None if detection fails.
        """
        N = len(t)

        ao_segs = self._flow_segments(Qao, thr_frac=thr_ao, min_len=3)
        avo, avc = self._choose_main_segment(Qao, ao_segs)

        mv_segs = self._flow_segments(Qmv, thr_frac=thr_mv, min_len=3)
        if not mv_segs:
            return avo, avc, None, None

        # if no aortic events, fall back: mitral first start = MVO, last end = MVC
        if avo is None or avc is None:
            mvo = mv_segs[0][0]
            mvc = mv_segs[-1][1]
            return avo, avc, int(mvo), int(mvc)

        avo_i = int(avo)
        avc_i = int(avc)

        starts = np.array([s for s, e in mv_segs], dtype=int)
        ends = np.array([e for s, e in mv_segs], dtype=int)

        # MVO: first mitral segment start AFTER AVC (cyclic)
        starts_shift = np.where(starts >= avc_i, starts, starts + N)
        mvo = int(starts[np.argmin(starts_shift)])

        # MVC: last mitral segment end BEFORE AVO (cyclic)
        ends_shift = np.where(ends <= avo_i, ends, ends - N)
        mvc = int(ends[np.argmax(ends_shift)])

        return avo_i, avc_i, mvo, mvc

    def _idx_esp_max_P_over_V(self, P, V, avo=None, avc=None):
        eps = 1e-6
        ratio = P / np.maximum(V, eps)

        if avo is not None and avc is not None:
            if avc >= avo:
                idx_range = np.arange(avo, avc + 1)
            else:
                idx_range = np.concatenate([np.arange(avo, len(P)), np.arange(0, avc + 1)])
            return int(idx_range[np.argmax(ratio[idx_range])])

        return int(np.argmax(ratio))

    def extract_cycle_metrics(self, cyc):
        """
        Extract main cycle metrics from a cut cycle using paper-style definitions:
          ED  : dpdt_upstroke (onset of +dP/dt near MVC)
          ESP : max(P/V) within ejection window (AVO->AVC if available)
        Also computes IVRT = AVC->MVO from flows. [1](https://journals.physiology.org/doi/full/10.1152/ajpheart.00705.2019)[2](https://www.ahajournals.org/doi/pdf/10.1161/circimaging.110.961623)
        """

        t = np.asarray(cyc['t'], dtype=float)
        V = np.asarray(cyc['V_lv'], dtype=float)
        P = np.asarray(cyc['P_lv'], dtype=float)
        Qao = np.asarray(cyc['Q_ao'], dtype=float)
        Qmv = np.asarray(cyc['Q_sv'], dtype=float)

        Tcyc = self._cycle_period(t)

        # valve events
        avo_idx, avc_idx, mvo_idx, mvc_idx = self._valve_events_from_flows(
            t=t, Qao=Qao, Qmv=Qmv, thr_ao=0.01, thr_mv=0.005
        )

        # ---- ED index (dpdt_upstroke) ----
        dPdt = np.gradient(P, t)

        if self.ED_DEF == "dpdt_upstroke":
            if mvc_idx is not None:
                back_s = 0.25
                dt = float(t[1] - t[0])
                back_n = int(round(back_s / dt))
                i0 = max(0, int(mvc_idx) - back_n)
                i1 = int(mvc_idx)
                d = dPdt[i0:i1 + 1]
                crossings = np.where((d[:-1] <= 0) & (d[1:] > 0))[0] + 1
                ED_idx = int(i0 + crossings[-1]) if len(crossings) else int(np.argmax(V))
            else:
                i_peak = int(np.argmax(dPdt))
                crossings = np.where((dPdt[:-1] <= 0) & (dPdt[1:] > 0))[0] + 1
                crossings = crossings[crossings < i_peak]
                ED_idx = int(crossings[-1]) if len(crossings) else int(np.argmax(V))
        else:
            ED_idx = int(np.argmax(V))  # fallback

        EDV = float(V[ED_idx])
        LVEDP = float(P[ED_idx])

        # ---- ESP/ES index (max P/V) ----
        if self.ESP_DEF == "max_P_over_V":
            es_idx = self._idx_esp_max_P_over_V(P, V, avo=avo_idx, avc=avc_idx)
        else:
            es_idx = int(np.argmin(V))  # fallback

        ESP = float(P[es_idx])
        ESV = float(V[es_idx])

        # For reference (classic min-V ESV)
        ESV_minV = float(np.min(V))
        ESV_idx_minV = int(np.argmin(V))
        EDV_maxV = float(np.max(V))

        # ---- LVOT flow peak and time-to-peak ----
        peak_idx = int(np.argmax(Qao))
        Q_peak = float(Qao[peak_idx])
        t_peak = float(t[peak_idx])

        # ---- Ejection duration proxy (your old EDur) ----
        thresh = 0.01 * Q_peak
        mask = Qao > thresh
        edur_start_idx = int(np.where(mask)[0][0]) if mask.any() else ED_idx
        edur_end_idx = int(np.where(mask)[0][-1]) if mask.any() else ED_idx
        EDur = float(t[edur_end_idx] - t[edur_start_idx])

        # ---- IVRT (AVC -> MVO) ----  [1](https://journals.physiology.org/doi/full/10.1152/ajpheart.00705.2019)[2](https://www.ahajournals.org/doi/pdf/10.1161/circimaging.110.961623)
        if avc_idx is not None and mvo_idx is not None:
            IVRT = self._dt_cyclic(float(t[avc_idx]), float(t[mvo_idx]), Tcyc)
        else:
            IVRT = np.nan

        return (
            {
                'EDV': EDV, 'ESV': ESV, 'ESV_minV': ESV_minV, 'LVEDP': LVEDP,
                'ESP': ESP,'EDV_maxV': EDV_maxV,
                'Q_peak': Q_peak, 't_peak': t_peak, 'EDur': EDur,'ET': EDur,
                'IVRT': IVRT
            },
            {
                'EDV_idx': int(ED_idx),
                'ESV_idx': int(es_idx),
                'ESV_minV_idx': int(ESV_idx_minV),
                'peak_idx': int(peak_idx),
                'edur_start': int(edur_start_idx),
                'edur_end': int(edur_end_idx),
                'avo_idx': avo_idx, 'avc_idx': avc_idx, 'mvo_idx': mvo_idx, 'mvc_idx': mvc_idx
            }
        )

    # -- Loss function used for parameter estimation --
    def loss_all_matrices(self, theta):
        """Objective function: computes the weighted SSE loss for a candidate parameter vector."""
        params = self.params.copy()
        for name, val in zip(self.param_names, theta):
            params[name] = val
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = self.run_simulation(params)
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, _ = self.extract_cycle_metrics(cyc)
        gt = self.data_row  # ground-truth for this subject

        # Compute the main metrics
        sim = {
            # BEFORE:
            # 'EDV': mets['EDV'],
            # 'ESV': mets['ESV'],

            # AFTER (robust for loss; still report paper indices elsewhere):
            'EDV': mets.get('EDV_maxV', mets['EDV']),
            'ESV': mets.get('ESV_minV', mets['ESV']),

            'SV': mets['EDV'] - mets['ESV'],  # you can leave SV as-is (uses event ED/ES)
            'EF': (mets['EDV'] - mets['ESV']) / mets['EDV'] * 100 if mets['EDV'] > 0 else np.nan,
            'bSBP': np.max(cyc['P_ao']) * 1.1,
            'bDBP': np.min(cyc['P_ao']) * 1.1,
            'bMAP': (np.max(cyc['P_ao']) * 1.1 + 2 * np.min(cyc['P_ao']) * 1.1) / 3,
            'bPP': np.max(cyc['P_ao']) * 1.1 - np.min(cyc['P_ao']) * 1.1,
            'LVOT_Flow_Peak': mets['Q_peak'],
            'time_LVOT_Flow_Peak': mets['t_peak'],
            'ET': mets.get('ET', mets.get('EDur', np.nan)),
            'IVRT': mets.get('IVRT', np.nan)
        }
        #  (mean values of cohort)
        K_COHORT = {
            'bSBP': 138.43, 'bDBP': 75.15,
            'ESV': 48.56, 'EDV': 97.76, 'SV': 49.29 , 'EF': 52.57,
            'LVOT_Flow_Peak': 334.64,
            'ET': 0.29,'time_LVOT_Flow_Peak': 0.08,
            'IVRT': 0.1139
        }

        # inside loss_all_matrices()
        sse = 0.0
        for key, w in self.weights.items():
            if w == 0 or key == 'LVEDP':
                continue

            s_val = sim.get(key, np.nan)
            gt_val = gt.get(key, np.nan)

            # --- Robust guard: skip term if sim or GT is NaN/inf
            # (Optionally, add a small constant penalty instead of skipping)
            if not (np.isfinite(s_val) and np.isfinite(gt_val)):
                # e.g., to force some pressure to "try" to become computable:
                # sse += 0.5  # mild penalty
                continue

            if self.loss_scaling_mode == 'cohort_minmax':
                den = K_COHORT.get(key, 1.0)
            elif self.loss_scaling_mode == 'hybrid_wls':
                if key in ('bSBP', 'bDBP'):
                    den = 5.0
                elif key in ('EDV', 'ESV'):
                    den = max(12.0, 0.10 * max(abs(gt_val), 1e-9))
                elif key == 'LVOT_Flow_Peak':
                    den = max(0.10 * abs(gt_val), 1e-9)
                elif key in ('time_LVOT_Flow_Peak', 'ET', 'IVRT'):
                    den = 0.02
                else:
                    den = 1.0
            else:
                return 1e12

            sse += w * ((s_val - gt_val) / den) ** 2

        # Final guard so optimizer never sees NaN/inf:
        if not np.isfinite(sse):
            return 1e12

        LVEDP_sim = mets['LVEDP']

        # --- LVEDP soft prior (MAP, absolute units; optional ±3 mmHg dead-zone) ---
        #LVEDP_pred = 1.7206 * self.SWE_velocity + 8.1086  # regression mean (your current model)
        #LVEDP_sim = mets['LVEDP']

        #delta = LVEDP_sim - LVEDP_pred

        #if self.use_prior_deadzone and abs(delta) <= self.prior_deadzone:
        #    prior_penalty = 0.0
        #else:
        #    eff = abs(delta) - (self.prior_deadzone if self.use_prior_deadzone else 0.0)
        #    prior_penalty = (eff / self.sigma_prior) ** 2  # weight = 1/sigma_prior^2 (MAP)

        # Optional kappa around MAP (kept at 1.0 by default)
        #sse += self.kappa_prior * prior_penalty

        # Penalize unphysiological LVEDP modified to hinge to bound method.
        L, U = 4.0, 35.0
        s_scale = 5.0  # or (U-L)/2 ≈ 15.5
        viol = 0.0
        if LVEDP_sim < L:
            viol = (L - LVEDP_sim) / s_scale
        elif LVEDP_sim > U:
            viol = (LVEDP_sim - U) / s_scale
        sse += self.lambda_phys * (viol ** 2)
        #self._debug_loss_breakdown(sim, gt, K_COHORT)
        return sse

        #if LVEDP_sim < 4 or LVEDP_sim > 35:
        #    sse += self.lambda_phys * ((LVEDP_sim - 16) ** 2 / 16 ** 2)
        #return sse

    def _debug_loss_breakdown(self, sim, gt, K_COHORT):
        parts = {}
        for key, w in self.weights.items():
            if w == 0 or key == 'LVEDP':
                continue
            s_val = sim.get(key, np.nan)
            g_val = gt.get(key, np.nan)
            if not (np.isfinite(s_val) and np.isfinite(g_val)):
                continue
            if self.loss_scaling_mode == 'cohort_minmax':
                den = K_COHORT.get(key, 1.0)
            else:  # 'hybrid_wls'
                if key in ('bSBP', 'bDBP'):
                    den = 5.0
                elif key in ('EDV', 'ESV'):
                    den = max(12.0, 0.10 * max(abs(g_val), 1e-9))
                elif key == 'LVOT_Flow_Peak':
                    den = max(0.10 * abs(g_val), 1e-9)
                elif key in ('time_LVOT_Flow_Peak', 'ET', 'IVRT'):
                    den = 0.02
                else:
                    den = 1.0
            parts[key] = ((s_val - g_val) / den) ** 2
        # print once per subject
        if not hasattr(self, "_loss_once"):
            ordered = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)
            print(f"[sub {self.sub_id}] Loss breakdown (top 6): {ordered[:6]}")
            self._loss_once = True

    # -- Objective for scipy.optimize --
    def objective(self, theta):
        return self.loss_all_matrices(theta)

    # -- Run optimizer (DE + L-BFGS-B) in z-space, return physical x + meta --
    def optimize(self):
        np.random.seed(self.sub_id)
        t0 = time.perf_counter()

        dim = len(self.param_names)
        unit_bounds = [(0.0, 1.0)] * dim

        # ---- Differential Evolution in z-space ----
        result_de_z = differential_evolution(
            func=self.objective_z,
            bounds=unit_bounds,
            workers=cpu_count(),
            maxiter=25,
            popsize=5,
            tol=1e-2,
            disp=True,
            polish=False,  # we polish ourselves
            updating='deferred',  # better parallel behavior
            seed=self.sub_id
        )
        t1 = time.perf_counter()
        de_time = t1 - t0
        de_loss = float(result_de_z.fun)
        de_theta = self.theta_from_z(result_de_z.x)  # map to physical
        print(f"✓ DE complete in {de_time:.1f}s, best loss {de_loss:.6f}")

        # ---- L-BFGS-B in z-space ----
        print("Polishing with L-BFGS-B…")
        t2 = time.perf_counter()
        res_local_z = minimize(
            fun=self.objective_z,
            x0=result_de_z.x,
            method='L-BFGS-B',
            bounds=unit_bounds,
            options={'ftol': 1e-4, 'maxiter': 100, 'disp': False}
        )
        t3 = time.perf_counter()
        lbfgs_time = t3 - t2
        lbfgs_loss = float(res_local_z.fun)
        res_theta = self.theta_from_z(res_local_z.x)  # map to physical
        total_time = t3 - t0
        print(f"✓ L-BFGS-B complete in {lbfgs_time:.1f}s, final loss {lbfgs_loss:.6f}")
        print(f"Total optimization time: {total_time:.1f}s")

        # Build a SciPy-like result object with physical x for downstream code
        class _Res: pass

        res_local = _Res()
        res_local.x = res_theta
        res_local.fun = lbfgs_loss
        res_local.success = True  # or res_local_z.success

        # V4 meta with PHYSICAL x for DE and L-BFGS-B
        meta = {
            'de': {'x': de_theta, 'loss': de_loss, 'time_sec': de_time},
            'lbfgs': {'x': res_theta, 'loss': lbfgs_loss, 'time_sec': lbfgs_time}
        }
        return res_local, total_time, meta

    def simulate_and_metrics(self, best_theta):
        """Run the simulation with best-fit parameters and extract metrics and cycle."""
        params = self.params.copy()
        for name, val in zip(self.param_names, best_theta):
            params[name] = val
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = self.run_simulation(params)
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, idxs = self.extract_cycle_metrics(cyc)
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, params

    def save_current_figures(self):
        """Save all open matplotlib figures to subject's folder."""
        for i, fig_num in enumerate(plt.get_fignums(), 1):
            fig = plt.figure(fig_num)
            filename = os.path.join(self.subject_dir, f"fig_{i}.png")
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved: {filename}")

    def plot_all(self, t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, params_used):
        """Plot all basic simulation results and cycle metrics (and save them)."""
        # Elastance
        E_full = self.elastance(t, params_used)
        plt.figure(); plt.plot(t, E_full)
        plt.title(f'Time-Varying Elastance (Emax={params_used["E_max"]:.3f}, Emin={params_used["E_min"]:.3f})')
        plt.xlabel('Time [s]'); plt.ylabel('Elastance [mmHg/ml]'); plt.grid()
        # Pressures
        plt.figure()
        plt.plot(t, P_ao, label='Aortic P')
        plt.plot(t, P_lv, label='LV P')
        plt.title('Pressure Waveforms')
        plt.xlabel('Time [s]'); plt.ylabel('Pressure [mmHg]'); plt.legend(); plt.grid()
        # Flows
        plt.figure()
        plt.plot(t, Q_lv_ao, label='Ao Flow')
        plt.plot(t, Q_sv_lv, label='Mitral Flow')
        plt.plot(t, Q_sys, label='Sys Flow')
        plt.title('Flow Waveforms')
        plt.xlabel('Time [s]'); plt.ylabel('Flow Rate [ml/s]'); plt.legend(); plt.grid()
        # LV Volume
        plt.figure()
        plt.plot(t, V_lv, label='LV Volume')
        plt.title('LV Volume')
        plt.xlabel('Time [s]'); plt.ylabel('Volume [ml]'); plt.legend(); plt.grid()
        # PV loop
        plt.figure(); plt.plot(V_lv, P_lv)
        plt.title('LV PV Loop')
        plt.xlabel('Volume [ml]'); plt.ylabel('Pressure [mmHg]'); plt.grid()
        # Cycle Pressure
        t_c = cyc['t']
        plt.figure()
        plt.plot(t_c, cyc['P_ao'], label='P_ao')
        plt.plot(t_c, cyc['P_lv'], label='P_lv')
        plt.scatter([t_c[idxs['EDV_idx']]], [cyc['P_lv'][idxs['EDV_idx']]], c='r', marker='o', label='LVEDP@EDV')
        plt.legend(); plt.grid(); plt.title("Cycle Pressures")
        # Cycle Flow
        plt.figure()
        plt.plot(t_c, cyc['Q_ao'], label='Q_ao')
        plt.plot(t_c, cyc['Q_sv'], label='Q_sv')
        plt.scatter([t_c[idxs['peak_idx']]], [cyc['Q_ao'][idxs['peak_idx']]], c='g', marker='o', label='Q_peak')
        plt.axvline(t_c[idxs['edur_start']], color='m', linestyle='--', label='EDur start')
        plt.axvline(t_c[idxs['edur_end']], color='m', linestyle='--', label='EDur end')
        plt.legend(); plt.grid(); plt.title("Cycle Flows")
        # LV Volume (cycle)
        plt.figure()
        plt.plot(t_c, cyc['V_lv'], label='V_lv')
        plt.scatter([t_c[idxs['ESV_idx']]], [cyc['V_lv'][idxs['ESV_idx']]], c='b', marker='o', label='ESV')
        plt.scatter([t_c[idxs['EDV_idx']]], [mets['EDV']], c='r', marker='o', label='EDV')
        plt.title('LV Volume (Cycle)'); plt.xlabel('Time [s]'); plt.ylabel('Volume [ml]')
        plt.legend(); plt.grid()
        # PV loop (cycle)
        plt.figure()
        plt.plot(cyc['V_lv'], cyc['P_lv'])
        plt.scatter([mets['EDV']], [mets['LVEDP']], c='r', marker='o', label='EDV point')
        plt.title('LV PV Loop (Cycle)')
        plt.xlabel('Volume [ml]'); plt.ylabel('Pressure [mmHg]'); plt.legend(); plt.grid()

    def compare_with_gt(self, mets, cyc):
        """Print and return a dictionary comparing simulated and ground-truth metrics."""
        gt = self.data_row

        # Robust endpoints (what your loss uses)
        EDV_minmax = mets.get('EDV_maxV', np.max(cyc['V_lv']))
        ESV_minmax = mets.get('ESV_minV', np.min(cyc['V_lv']))
        SV_minmax = EDV_minmax - ESV_minmax
        EF_minmax = (SV_minmax / EDV_minmax * 100.0) if EDV_minmax > 0 else np.nan

        # Event-based endpoints (paper-aligned ED/ES for reporting only)
        EDV_event = mets['EDV']  # from dp/dt upstroke ED
        ESV_event = mets['ESV']  # from max(P/V) time
        SV_event = EDV_event - ESV_event
        EF_event = (SV_event / EDV_event * 100.0) if EDV_event > 0 else np.nan

        sim = {
            # --- volumes used in evaluation (min/max) ---
            'EDV': EDV_minmax,
            'ESV': ESV_minmax,
            'SV': SV_minmax,
            'EF': EF_minmax,

            # --- also expose event-based for plots/diagnostics ---
            'EDV_event': EDV_event,
            'ESV_event': ESV_event,
            'SV_event': SV_event,
            'EF_event': EF_event,

            # pressures & timing (unchanged)
            'bSBP': np.max(cyc['P_ao']) * 1.1,
            'bDBP': np.min(cyc['P_ao']) * 1.1,
            'bMAP': (np.max(cyc['P_ao']) * 1.1 + 2 * np.min(cyc['P_ao']) * 1.1) / 3,
            'bPP': np.max(cyc['P_ao']) * 1.1 - np.min(cyc['P_ao']) * 1.1,
            'LVOT_Flow_Peak': mets['Q_peak'],
            'time_LVOT_Flow_Peak': mets['t_peak'],
            'ET': mets.get('ET', mets.get('EDur', np.nan)),
            'IVRT': mets.get('IVRT', np.nan),
            'LVEDP': mets['LVEDP']
        }

        print(f"\n--- Comparing Simulation vs. Ground-Truth for sub_id={self.sub_id} ---")
        # print the min/max set (what you optimize and evaluate against)
        for key in ['EDV', 'ESV', 'SV', 'EF', 'bSBP', 'bDBP', 'bMAP', 'bPP',
                    'LVOT_Flow_Peak', 'time_LVOT_Flow_Peak', 'ET', 'IVRT', 'LVEDP']:
            s_val = sim.get(key, np.nan)
            gt_val = gt.get(key, np.nan)
            print(f"{key:>22} | {s_val:10.3f} | {gt_val:10.3f}")

        # also print event-based endpoints for reference
        print("Event-based (paper indices) -> "
              f"EDV_event={EDV_event:.1f}, ESV_event={ESV_event:.1f}, "
              f"SV_event={SV_event:.1f}, EF_event={EF_event:.1f}%")

        return sim

    # CHANGE V4: add meta to capture DE/L-BFGS-B info
    def collect_result_row(self, best, loss, opt_time, sim_matrices, meta=None):
        row = {
            'sub_id': self.sub_id,
            'loss': loss,
            'optimization_time_sec': opt_time,
            'soft_weight': self.soft_weight_LVEDP,
            'phys_weight': self.lambda_phys
        }
        # CHANGE V4: new fields (if meta is provided)
        if meta is not None:
            row.update({
                'de_loss': meta['de']['loss'],
                'de_time_sec': meta['de']['time_sec'],
                'lbfgs_loss': meta['lbfgs']['loss'],
                'lbfgs_time_sec': meta['lbfgs']['time_sec']
            })

        for name, val in zip(self.param_names, best.x):
            row[name] = val
        row.update(sim_matrices)
        return row


    # CHANGE V4: a separate row builder for DE-only CSV
    def collect_de_row(self, de_x, de_loss, de_time, sim_matrices):
        row = {
            'sub_id': self.sub_id,
            'de_loss': de_loss,
            'de_time_sec': de_time,
            'soft_weight': self.soft_weight_LVEDP,
            'phys_weight': self.lambda_phys
        }
        for name, val in zip(self.param_names, de_x):
            row[f"DE_{name}"] = val  # prefix to distinguish from final results
        row.update({f"DE_{k}": v for k, v in sim_matrices.items()})
        return row



# -- MAIN SCRIPT --
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    combined_df = pd.read_csv(
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Invasive_Study_Leuven_GT_matrices_all_subjects_V4.0.csv")
    #subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
                     #344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365]# half batch run

    #subject_list = [366, 367, 368, 369, 370, 371, 372, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 396,
                    #397, 398, 399, 400, 401, 402, 403, 404, 405, 409, 410, 411, 412, 413, 414, 415] # half batch run

    subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
                    344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365,
                    366, 367, 368, 369, 370, 371, 372, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 396,
                    397, 398, 399, 400, 401, 402, 403, 404, 405, 409, 410, 411, 412, 413, 414, 415] # full batch run

    #subject_list = [327] #test33

    #subject_list = [316, 327, 347, 354, 356, 361,362, 363, 366, 369, 390, 392, 414] # batch 1
    #subject_list = [328, 329, 331, 332, 334, 336] # batch 2
    #subject_list = [337, 338, 339, 341, 342, 343, 344, 346, 347, 349, 351, 352] # batch 3
    #subject_list = [354, 356, 357, 359, 360, 361, 362, 363, 364, 365] # batch 4

    save_root = (
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Results_Plots_Study_10_V4.1_T4_without_regression"
    )

    # CHANGE V4: define output CSVs (incremental append)
    results_path_final = os.path.join(save_root, "Study_10_V4.1_incremental.csv")
    results_path_deonly = os.path.join(save_root, "Study_10_V4.1_DE_only.csv")
    os.makedirs(save_root, exist_ok=True)

    sim_results = []


    for sub_id in subject_list:
        data_row = combined_df.loc[combined_df['sub_id'] == sub_id].squeeze()
        sim = SubjectSimulator(data_row, sub_id, save_root)

        # CHANGE V4: run optimizer and get DE/L-BFGS-B meta
        best, opt_time, meta = sim.optimize()
        print("Optimization success:", best.success)
        print("Best loss:", best.fun)
        for name, val in zip(sim.param_names, best.x):
            print(f"  {name:8s} = {val:.4f}")

        # ---- (A) DE-only simulation, metrics, and CSV append ----
        # simulate with DE parameters (before L-BFGS-B polish)
        t_de, P_ao_de, P_lv_de, V_lv_de, Q_sv_lv_de, Q_lv_ao_de, Q_sys_de = sim.run_simulation(
            params={**sim.params, **{k: v for k, v in zip(sim.param_names, meta['de']['x'])}}
        )
        cyc_de = sim.cycle_cutting_algo(t_de, P_ao_de, P_lv_de, V_lv_de, Q_sv_lv_de, Q_lv_ao_de)
        mets_de, _ = sim.extract_cycle_metrics(cyc_de)
        print("\nResults after DE-only \n")
        sim_matrices_de = sim.compare_with_gt(mets_de, cyc_de)  # prints and returns dict

        row_de = sim.collect_de_row(meta['de']['x'], meta['de']['loss'], meta['de']['time_sec'], sim_matrices_de)
        # append DE-only row
        de_exists = os.path.exists(results_path_deonly)
        pd.DataFrame([row_de]).to_csv(
            results_path_deonly, mode='a', header=not de_exists, index=False
        )

        # ---- (B) Final (post L-BFGS-B) simulation, plots, and CSV append ----
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, params_opt = sim.simulate_and_metrics(best.x)

        sim.plot_all(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, params_opt)  # Plot all
        sim.save_current_figures()  # Save all
        plt.close('all')  # Close all to avoid memory leak
        print("\nFinal Results after DE & L-BFGS-B\n")
        # Print metrics comparison and collect for CSV
        sim_matrices_final = sim.compare_with_gt(mets, cyc)

        # CHANGE V4: include DE/L-BFGS-B meta in the final row and append it
        row_final = sim.collect_result_row(best, best.fun, opt_time, sim_matrices_final, meta=meta)
        final_exists = os.path.exists(results_path_final)
        pd.DataFrame([row_final]).to_csv(
            results_path_final, mode='a', header=not final_exists, index=False
        )

#end main