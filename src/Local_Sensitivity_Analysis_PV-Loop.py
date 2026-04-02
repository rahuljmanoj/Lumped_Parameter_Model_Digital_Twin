import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import os
import time
from multiprocessing import cpu_count

class SubjectSimulator:
    def __init__(self, data_row,data_row_GT, sub_id, save_root, soft_weight=0.75, soft_weight_Emin=1, phys_weight=10):
        """Initialize all subject-specific and fixed simulation parameters."""
        self.sub_id = sub_id
        self.data_row = data_row
        self.data_row_GT = data_row_GT
        self.save_root = save_root
        # Weights for loss function
        self.soft_weight = soft_weight
        self.phys_weight = phys_weight
        self.soft_weight_Emin = soft_weight_Emin

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

        self.SWE_velocity = data_row['SWS (m/s)']
        self.GT_bMAP = data_row['GT bMAP (mmHg)']
        self.h_r = data_row['h/r']
        self.LAVI = data_row['LAVI (ml/m²)']
        self.BSA = data_row['BSA (m²)']
        self.age = data_row['age (yrs)']
        self.BMI = data_row['BMI (kg/m²)']
        self.weight = data_row['weight (kg)']


        self.bpm = data_row['HR (bpm)']
        self.T = 60.0 / self.bpm
        self.total = self.cycles * self.T
        self.dt = self.T / 500.0

        # === Paper-aligned PV definitions ===
        self.ED_DEF = "dpdt_upstroke"  # EDP at onset of +dP/dt
        self.ESP_DEF = "max_P_over_V"  # ESP at max(P/V)

        # Model parameter names and initial guess (for bounds)
        self.param_names = ['R_sys', 'Z_ao', 'C_sa', 'R_mv', 'E_max', 'E_min', 't_peak', 'V_tot', 'C_sv']
        self.params = {
            'R_sys': data_row['R_sys (mmHg s/ml)']* (1 + 0.00),
            'Z_ao': data_row['Z_ao (mmHg s/ml)']* (1 + 0.00),
            'C_sa': data_row['C_sa (ml/mmHg)']* (1 + 0.00),
            'R_mv': data_row['R_mv (mmHg s/ml)']* (1 + 0.00),
            'E_max': data_row['E_max (mmHg/ml)']* (1 + 0.00),
            'E_min': data_row['E_min (mmHg/ml)']* (1 + 0.00),
            't_peak': data_row['t_peak (s)']* (1 + 0.00),
            'V_tot': data_row['V_tot (ml)']* (1 + 0.00),
            'C_sv': data_row['C_sv (ml/mmHg)']* (1 + 0.00),
            'T': self.T
        }
        # Bounds for optimization (subject specific)
        self.bounds = [
            (self.params['R_sys'] * (1 - 0.20), self.params['R_sys'] * (1 + 0.20)),
            (self.params['Z_ao'] * (1 - 0.20), self.params['Z_ao'] * (1 + 0.20)),
            (self.params['C_sa'] * (1 - 0.20), self.params['C_sa'] * (1 + 0.20)),
            (0.01, 0.1),     # R_mv
            (0.9, 10.0),     # E_max
            (0.01, 1.0),     # E_min #(self.params['E_min'] * (1 - 0.30), self.params['E_min'] * (1 + 0.30)),
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
            'EDV': 1, 'ESV': 1, 'SV': 1, 'EF': 1,
            'bSBP': 1, 'bDBP': 1, 'bMAP': 1, 'bPP': 1,
            'LVOT_Flow_Peak': 1, 'time_LVOT_Flow_Peak': 1, 'ED': 1,
            'LVEDP': 0.0
        }
        # For saving plots/results
        self.subject_dir = os.path.join(self.save_root, str(self.sub_id))
        os.makedirs(self.subject_dir, exist_ok=True)

    def initial_state(self, p):
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

    def run_simulation(self, params=None, return_state=False):
        if params is None:
            params = self.params

        # (recommended) compute T/dt/total from params so this works even if params['T'] changes
        T = float(params['T'])
        dt = T / 500.0
        total = self.cycles * T

        y0 = self.initial_state(params)
        num_steps = int(np.round(total / dt))
        t_eval = np.linspace(0, total, num_steps + 1)

        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, params),
                        [0, total], y0, t_eval=t_eval, max_step=dt)

        t = sol.t
        V_lv, V_sa, V_sv = sol.y
        P_lv = self.elastance(t, params) * V_lv + self.P_th
        P_sa = V_sa / params['C_sa']
        P_ao = np.where(P_lv > P_sa, P_lv, P_sa)
        P_sv = V_sv / params['C_sv']
        Q_sv_lv = np.where(P_sv > P_lv, (P_sv - P_lv) / params['R_mv'], 0.0)
        Q_lv_ao = np.where(P_lv > P_sa, (P_lv - P_sa) / params['Z_ao'], 0.0)
        Q_sys = np.where(P_sa > P_sv, (P_sa - P_sv) / params['R_sys'], 0.0)

        if return_state:
            return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, sol.y[:, -1].copy()
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys

    def cycle_cutting_algo(self, t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, T_override=None):
        """Cut out the last full cardiac cycle from simulation arrays."""
        dt = t[1] - t[0]
        T = float(T_override) if T_override is not None else (60.0 / self.bpm)
        spc = int(round(float(T / dt)))
        n = len(t) // spc
        if n < 2:
            raise RuntimeError("Not enough full cycles to slice out")
        start = (n - 2) * spc
        end = (n - 1) * spc
        return {
            't': t[start:end] - t[start],
            'P_ao': P_ao[start:end],
            'P_lv': P_lv[start:end],
            'V_lv': V_lv[start:end],
            'Q_sv': Q_sv_lv[start:end],
            'Q_ao': Q_lv_ao[start:end]
        }

    def extract_cycle_metrics(self, cyc):
        """
        Extract cycle metrics + indices needed by plot_all().
        Keeps backward compatibility with existing plotting code.
        """
        t = np.asarray(cyc['t'], dtype=float)
        V = np.asarray(cyc['V_lv'], dtype=float)
        P = np.asarray(cyc['P_lv'], dtype=float)
        Q = np.asarray(cyc['Q_ao'], dtype=float)

        # --- Use your paper-aligned definitions if you want ---
        # ED index: by default use your current ED definition choice
        # Here we keep EDV_idx as max(V) for plotting consistency (still works visually),
        # and LVEDP at that same index (as old behaviour).
        EDV_idx = int(np.argmax(V))
        ESV_idx = int(np.argmin(V))  # true min-volume point for plotting

        EDV = float(V[EDV_idx])
        ESV = float(V[ESV_idx])
        LVEDP = float(P[EDV_idx])

        # Peak aortic flow
        peak_idx = int(np.argmax(Q))
        Q_peak = float(Q[peak_idx])
        t_peak = float(t[peak_idx])

        # Ejection duration indices
        thresh = 0.01 * Q_peak if Q_peak > 0 else 0.0
        mask = Q > thresh
        if np.any(mask):
            edur_start_idx = int(np.where(mask)[0][0])
            edur_end_idx = int(np.where(mask)[0][-1])
            EDur = float(t[edur_end_idx] - t[edur_start_idx])
        else:
            edur_start_idx = EDV_idx
            edur_end_idx = EDV_idx
            EDur = 0.0

        mets = {
            'EDV': EDV,
            'ESV': ESV,
            'LVEDP': LVEDP,
            'Q_peak': Q_peak,
            't_peak': t_peak,
            'EDur': EDur
        }
        idxs = {
            'EDV_idx': EDV_idx,
            'ESV_idx': ESV_idx,
            'peak_idx': peak_idx,
            'edur_start': edur_start_idx,
            'edur_end': edur_end_idx
        }
        return mets, idxs

    def run_simulation_with_HR(self, HR, params=None, cycles=None):
        """
        Run simulation with HR override.
        IMPORTANT: t_peak is kept FIXED (in seconds) as you requested.
        """
        if params is None:
            params = self.params
        if cycles is None:
            cycles = self.cycles

        p = params.copy()

        HR = float(HR)
        T_new = 60.0 / HR
        p['T'] = T_new  # override cycle length
        # NOTE: do NOT modify p['t_peak'] (kept fixed)

        dt_new = T_new / 500.0
        total_new = cycles * T_new

        y0 = self.initial_state(p)
        num_steps = int(np.round(total_new / dt_new))
        t_eval = np.linspace(0.0, total_new, num_steps + 1)

        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, p),
                        [0.0, total_new], y0, t_eval=t_eval, max_step=dt_new)

        t = sol.t
        V_lv, V_sa, V_sv = sol.y

        P_lv = self.elastance(t, p) * V_lv + self.P_th
        P_sa = V_sa / p['C_sa']

        # Keep your current physics definition for aortic pressure:
        P_ao = np.where(P_lv > P_sa, P_lv, P_sa)

        P_sv = V_sv / p['C_sv']
        Q_sv_lv = np.where(P_sv > P_lv, (P_sv - P_lv) / p['R_mv'], 0.0)
        Q_lv_ao = np.where(P_lv > P_sa, (P_lv - P_sa) / p['Z_ao'], 0.0)
        Q_sys = np.where(P_sa > P_sv, (P_sa - P_sv) / p['R_sys'], 0.0)

        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, p, dt_new

    def pv_full_sensitivity_table(self,
                                  base_params=None,
                                  param_list=None,
                                  steps=(-0.10, 0.10),
                                  include_hr=True,
                                  hr_steps=(-0.10, 0.10),
                                  n_beats=8,
                                  frac_drop=0.6,
                                  qc_plots=False):
        """
        Create rows for: baseline + +/- steps for each parameter + HR.
        Returns list of dict rows (long format).
        Each row includes all pv_full_report outputs + metadata (sub_id, param, delta, HR_bpm_case).
        """
        if base_params is None:
            base_params = self.params.copy()

        # Default: perturb all model params except T (HR handled separately via T)
        if param_list is None:
            param_list = [k for k in base_params.keys() if k != 'T']

        rows = []

        def run_case(param_name, delta, params_case):
            report, _ = self.pv_full_report(params_case, n_beats=n_beats, frac_drop=frac_drop)
            row = report.copy()
            row['sub_id'] = self.sub_id
            row['param'] = param_name
            row['delta'] = float(delta)
            row['HR_bpm_case'] = 60.0 / float(params_case['T'])
            row['occl_n_beats'] = int(n_beats)
            row['occl_frac_drop'] = float(frac_drop)
            if qc_plots:
                self.qc_plot_edpvr_case(params_case, case_name=f"{param_name}_{delta:+.0%}",
                                        n_beats=n_beats, frac_drop=frac_drop)
            return row

        # ---- baseline ----
        rows.append(run_case('baseline', 0.0, base_params))

        # ---- +/-10% for all model parameters ----
        for pname in param_list:
            if pname not in base_params:
                continue
            for s in steps:
                pvar = base_params.copy()
                pvar[pname] = base_params[pname] * (1.0 + s)
                rows.append(run_case(pname, s, pvar))

        # ---- HR perturbations: change T only; keep t_peak fixed ----
        if include_hr:
            HR0 = 60.0 / float(base_params['T'])
            for s in hr_steps:
                HR_new = HR0 * (1.0 + s)
                pvar = base_params.copy()
                pvar['T'] = 60.0 / HR_new
                rows.append(run_case('HR', s, pvar))

        return rows

    def pv_metrics_single_loop(self, cyc, HR_bpm=None, ed_def="maxV", esp_def="end_ejection"):
        """
        Compute single-loop PV metrics from one cardiac cycle dict under selectable
        ED/ESP definitions.

        Reference definitions:
          - EDP at onset of positive dP/dt deflection (ED corner). [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/Impact%20of%20Loading%20and%20Myocardial%20Mechanical%20Properties%20on%20Natural%20Shear%20Waves%20Comparison%20to%20Pressure-Volume%20Loops.pdf)[2](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/Continuous%20shear%20wave%20measurements%20for%20dynamic%20cardiac%20stiffness%20evaluation%20in%20pigs.pdf)
          - ESP at maximal P/V ratio. [2](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/Continuous%20shear%20wave%20measurements%20for%20dynamic%20cardiac%20stiffness%20evaluation%20in%20pigs.pdf)
        """
        t = np.asarray(cyc['t'], dtype=float)
        P = np.asarray(cyc['P_lv'], dtype=float)
        V = np.asarray(cyc['V_lv'], dtype=float)

        Qao = cyc.get('Q_ao', None)
        Qsv = cyc.get('Q_sv', None)
        Pao = cyc.get('P_ao', None)

        # ---------- Valve events (compute ONCE) ----------
        Tcyc = self._cycle_period(t)
        avo_idx = avc_idx = mvo_idx = mvc_idx = None
        if Qao is not None and Qsv is not None:
            avo_idx, avc_idx, mvo_idx, mvc_idx = self._valve_events_from_flows(
                t=t, Qao=Qao, Qmv=Qsv, thr_ao=0.01, thr_mv=0.005
            )

        # ---------- ED definition ----------
        if ed_def == "maxV":
            ED_idx = int(np.argmax(V))

        elif ed_def == "MVC_flow":
            EDV_mvc, _ = self.ed_point_at_mvc(cyc, thr_frac=0.005)
            ED_idx = int(np.argmin(np.abs(V - EDV_mvc)))

        elif ed_def == "dpdt_upstroke":
            # Paper-style ED: onset of positive deflection of dP/dt (ED corner). [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/Impact%20of%20Loading%20and%20Myocardial%20Mechanical%20Properties%20on%20Natural%20Shear%20Waves%20Comparison%20to%20Pressure-Volume%20Loops.pdf)[2](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/Continuous%20shear%20wave%20measurements%20for%20dynamic%20cardiac%20stiffness%20evaluation%20in%20pigs.pdf)
            dPdt = np.gradient(P, t)

            # Constrain search to a window before MVC (robust). If MVC unknown, fall back.
            if mvc_idx is not None:
                # search back 250 ms from MVC (tunable)
                back_s = 0.25
                dt = float(t[1] - t[0])
                back_n = int(round(back_s / dt))
                i0 = max(0, int(mvc_idx) - back_n)
                i1 = int(mvc_idx)

                d = dPdt[i0:i1 + 1]
                crossings = np.where((d[:-1] <= 0) & (d[1:] > 0))[0] + 1
                if len(crossings):
                    ED_idx = int(i0 + crossings[-1])
                else:
                    # fall back to MVC_flow if dP/dt crossing unclear
                    EDV_mvc, _ = self.ed_point_at_mvc(cyc, thr_frac=0.005)
                    ED_idx = int(np.argmin(np.abs(V - EDV_mvc)))
            else:
                # fallback: last crossing before dPdt max
                i_peak = int(np.argmax(dPdt))
                crossings = np.where((dPdt[:-1] <= 0) & (dPdt[1:] > 0))[0] + 1
                crossings = crossings[crossings < i_peak]
                ED_idx = int(crossings[-1]) if len(crossings) else int(np.argmax(V))

        else:
            raise ValueError(f"Unknown ed_def={ed_def}")

        EDV = float(V[ED_idx])
        EDP = float(P[ED_idx])  # LVEDP under chosen ED definition

        # ---------- ESP/ES definition ----------
        ESV_minV = float(np.min(V))  # keep classical reference ESV

        if esp_def == "end_ejection":
            es_idx = self._idx_es_end_ejection(Qao, thr_frac=0.01) if Qao is not None else None
            if es_idx is None:
                es_idx = int(np.argmin(V))
            ESP = float(P[es_idx])
            ESV = float(V[es_idx])

        elif esp_def == "max_P_over_V":
            # Paper-style ESP: maximal P/V. Restrict to ejection interval when possible.
            es_idx = self._idx_esp_max_P_over_V(P, V, Qao=Qao, avo=avo_idx, avc=avc_idx)
            ESP = float(P[es_idx])
            ESV = float(V[es_idx])  # definition-consistent "V at ESP"
        else:
            raise ValueError(f"Unknown esp_def={esp_def}")

        # ---------- Derived volumetrics ----------
        SV = EDV - ESV
        EF = (SV / EDV * 100.0) if EDV > 0 else np.nan

        # ---------- Pressures ----------
        Psys = float(np.max(P))
        Pmin = float(np.min(P))

        # ---------- Stroke work ----------
        SW = self.pv_loop_area_shoelace(V, P)

        # ---------- dP/dt ----------
        dPdt = np.gradient(P, t)
        dPdt_max = float(np.max(dPdt))
        dPdt_min = float(np.min(dPdt))

        out = dict(
            EDV=EDV, ESV=ESV, ESV_minV=ESV_minV, SV=SV, EF=EF,
            LVEDP=EDP, ESP=ESP, Psys_LV=Psys, Pmin_LV=Pmin,
            SW=SW, dPdt_max=dPdt_max, dPdt_min=dPdt_min
        )

        # ---------- Aortic pressures ----------
        if Pao is not None:
            aSBP = float(np.max(Pao))
            aDBP = float(np.min(Pao))
            aMAP = float((aSBP + 2 * aDBP) / 3.0)
            out.update(aSBP=aSBP, aDBP=aDBP, aMAP=aMAP)

        # ---------- CO and Ea ----------
        if HR_bpm is not None:
            HR = float(HR_bpm)
            out['CO_L_min'] = float((SV * HR) / 1000.0)
            out['Ea'] = float(ESP / SV) if SV > 0 else np.nan

        # ---------- Timing from valve events ----------
        out['ET'] = self._dt_cyclic(t[avo_idx], t[avc_idx], Tcyc) if (
                    avo_idx is not None and avc_idx is not None) else np.nan
        out['IVCT'] = self._dt_cyclic(t[mvc_idx], t[avo_idx], Tcyc) if (
                    mvc_idx is not None and avo_idx is not None) else np.nan
        out['IVRT'] = self._dt_cyclic(t[avc_idx], t[mvo_idx], Tcyc) if (
                    avc_idx is not None and mvo_idx is not None) else np.nan

        # ---------- Tau (fit on IVR segment) ----------
        if avc_idx is not None and mvo_idx is not None:
            t_iv, P_iv = self._get_segment_cyclic(t, P, avc_idx, mvo_idx)
            if t_iv is not None and len(t_iv) >= 10:
                dPdt_iv = np.gradient(P_iv, t_iv)
                i0 = int(np.argmin(dPdt_iv))
                out['tau'] = self._fit_tau_isovol_relaxation(t_iv[i0:], P_iv[i0:]) if (len(t_iv) - i0) >= 8 else np.nan
            else:
                out['tau'] = np.nan
        else:
            out['tau'] = np.nan

        return out

    def _fit_tau_isovol_relaxation(self, t_seg, P_seg):
        import numpy as np
        from scipy.optimize import curve_fit

        # Require net decrease in pressure over the segment
        if P_seg[-1] > P_seg[0] - 1e-3:
            return np.nan

        t_seg = np.asarray(t_seg, dtype=float)
        P_seg = np.asarray(P_seg, dtype=float)

        if len(t_seg) < 8:
            return np.nan

        t0 = t_seg[0]
        P0 = P_seg[0]

        # Estimate asymptote as the minimum in the segment
        P_inf0 = float(np.min(P_seg))

        # Estimate slope at start (use local derivative)
        dPdt0 = float(np.gradient(P_seg, t_seg)[0])

        # Tau initial guess from Weiss-type estimate with asymptote
        # tau ≈ -(P0 - P_inf)/dPdt0  (dPdt0 is negative)
        if dPdt0 < 0 and (P0 - P_inf0) > 0:
            tau0 = float(np.clip(-(P0 - P_inf0) / dPdt0, 0.01, 0.15))  # seconds
        else:
            tau0 = 0.04  # fallback 40 ms

        def model(t, P_inf, tau):
            return P_inf + (P0 - P_inf) * np.exp(-(t - t0) / tau)

        bounds = ([-np.inf, 1e-3], [np.inf, 0.20])  # 1 ms to 200 ms

        try:
            popt, _ = curve_fit(model, t_seg, P_seg, p0=[P_inf0, tau0], bounds=bounds, maxfev=20000)
            return float(popt[1])
        except Exception:
            return np.nan

    def definition_sensitivity_audit(self, params=None, n_beats=8, frac_drop=0.6):
        """
        Computes baseline + occlusion-derived fits under different ED/ESP definitions
        and returns a dict of deltas to quantify sensitivity.
        """
        if params is None:
            params = self.params

        # baseline run
        t, P_ao, P_lv, V_lv, Q_sv, Q_ao, Q_sys = self.run_simulation(params)
        cyc0 = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv, Q_ao, T_override=params['T'])
        HR_run = 60.0 / float(params['T'])

        # Compute baseline metrics under 4 combos
        combos = [
            ("old", "maxV", "end_ejection"),
            ("paperED_paperESP", "dpdt_upstroke", "max_P_over_V"),
            ("paperED_oldESP", "dpdt_upstroke", "end_ejection"),
            ("oldED_paperESP", "maxV", "max_P_over_V"),
        ]

        base = {}
        for tag, ed_def, esp_def in combos:
            base[tag] = self.pv_metrics_single_loop(cyc0, HR_bpm=HR_run, ed_def=ed_def, esp_def=esp_def)

        # occlusion family (for EDPVR, PRSW, ESPVR fits)
        family = self.preload_occlusion_family(params, n_beats=n_beats, frac_drop=frac_drop)

        # Recompute per-beat metrics under each combo (important!)
        # Replace each family's metrics with redefined metrics temporarily
        fits = {}
        for tag, ed_def, esp_def in combos:
            fam2 = []
            for d in family:
                cyc = d['cycle']
                mets = self.pv_metrics_single_loop(cyc, HR_bpm=HR_run, ed_def=ed_def, esp_def=esp_def)
                fam2.append(dict(cycle=cyc, metrics=mets, y_end=d.get('y_end', None)))
            fits[tag] = self.fit_pv_relations(fam2)

        # Build delta summary vs old
        out = {"sub_id": self.sub_id}
        ref_tag = "old"

        # baseline deltas
        for tag, _, _ in combos:
            if tag == ref_tag:
                continue
            for k in ["EDV", "LVEDP", "ESV", "ESP", "SV", "EF", "Ea"]:
                out[f"delta_{k}_{tag}"] = base[tag].get(k, np.nan) - base[ref_tag].get(k, np.nan)

        # relation deltas (occlusion derived)
        for tag, _, _ in combos:
            if tag == ref_tag:
                continue
            for k in ["beta", "rel_EDPVR_C", "rel_EDPVR_D", "rel_EDPVR_E", "PRSW_slope", "Ees", "V0"]:
                out[f"delta_{k}_{tag}"] = fits[tag].get(k, np.nan) - fits[ref_tag].get(k, np.nan)

        return out

    def _get_segment_cyclic(self, t, y, idx_start, idx_end):
        """
        Return (t_seg, y_seg) from idx_start to idx_end with wrap-around.
        If idx_end < idx_start, concatenates end->end and start->idx_end with time shifted by +T.
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        T = self._cycle_period(t)

        if idx_start is None or idx_end is None:
            return None, None

        if idx_end >= idx_start:
            return t[idx_start:idx_end + 1], y[idx_start:idx_end + 1]
        else:
            t1 = t[idx_start:]
            y1 = y[idx_start:]
            t2 = t[:idx_end + 1] + T
            y2 = y[:idx_end + 1]
            return np.concatenate([t1, t2]), np.concatenate([y1, y2])

    def _flow_segments(self, Q, thr_frac=0.01, abs_thr=1e-3, min_len=3):
        """
        Return list of (start_idx, end_idx) segments where Q > threshold.
        Works with multiple segments (e.g., mitral E + A waves).
        min_len: minimum number of samples in a segment to reject noise.
        """
        Q = np.asarray(Q, dtype=float)
        if np.max(Q) <= 0:
            return []

        thr = max(thr_frac * np.max(Q), abs_thr)
        mask = Q > thr
        if not np.any(mask):
            return []

        # Rising edges: 0->1; Falling edges: 1->0
        d = np.diff(mask.astype(int))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0])

        # Handle if mask begins True
        if mask[0]:
            starts = [0] + starts
        # Handle if mask ends True
        if mask[-1]:
            ends = ends + [len(Q) - 1]

        segs = []
        for s, e in zip(starts, ends):
            if e - s + 1 >= min_len:
                segs.append((int(s), int(e)))
        return segs

    def _choose_main_segment(self, Q, segs):
        """
        Pick the physiologically dominant segment: max peak (or max area).
        For aortic ejection there should be one dominant segment.
        """
        if not segs:
            return None, None
        Q = np.asarray(Q, dtype=float)
        # Choose by peak flow within each segment
        peaks = [np.max(Q[s:e + 1]) for s, e in segs]
        k = int(np.argmax(peaks))
        return segs[k]

    def _valve_events_from_flows(self, t, Qao, Qmv, thr_ao=0.01, thr_mv=0.005):
        """
        Compute AVO/AVC from Qao and MVO/MVC from Qmv using segment logic.
        Returns indices (avo, avc, mvo, mvc) (some may be None).
        """
        N = len(t)
        # --- Aortic: choose main ejection segment ---
        ao_segs = self._flow_segments(Qao, thr_frac=thr_ao, min_len=3)
        avo, avc = self._choose_main_segment(Qao, ao_segs)

        # --- Mitral: potentially multiple segments (E and A) ---
        mv_segs = self._flow_segments(Qmv, thr_frac=thr_mv, min_len=3)
        if not mv_segs:
            return avo, avc, None, None

        # If we don't have aortic events, fall back:
        if avo is None or avc is None:
            # choose the earliest as MVO and latest as MVC within the cycle
            mvo = mv_segs[0][0]
            mvc = mv_segs[-1][1]
            return avo, avc, int(mvo), int(mvc)

        # We want:
        #   MVO = first mitral segment start AFTER AVC
        #   MVC = last mitral segment end BEFORE AVO
        # Use wrap-around logic in index-space:
        avc_i = int(avc)
        avo_i = int(avo)

        starts = np.array([s for s, e in mv_segs], dtype=int)
        ends = np.array([e for s, e in mv_segs], dtype=int)

        # Find MVO candidate after AVC (cyclic)
        starts_shift = np.where(starts >= avc_i, starts, starts + N)
        mvo = int(starts[np.argmin(starts_shift)])

        # Find MVC candidate before AVO (cyclic)
        # We want the last inflow end occurring before AVO in cyclic time.
        ends_shift = np.where(ends <= avo_i, ends, ends - N)  # shift ends after AVO to negative
        mvc = int(ends[np.argmax(ends_shift)])

        return avo_i, avc_i, mvo, mvc

    def _cycle_period(self, t):
        t = np.asarray(t, dtype=float)
        dt = float(t[1] - t[0])
        return float(t[-1] - t[0] + dt)

    def _dt_cyclic(self, t1, t2, T):
        """Return (t2 - t1) with wrap-around on [0, T)."""
        return float(t2 - t1) if t2 >= t1 else float(t2 + T - t1)

    def preload_occlusion_family(self, params=None, n_beats=8, frac_drop=0.35,
                                 ed_def=None, esp_def=None):
        """
        Generate a family of PV loops by reducing venous volume V_sv between beats
        (vena cava occlusion analogue). Returns list of dicts with per-beat metrics.
        """
        if params is None:
            params = self.params
        p = params.copy()

        # Use paper-aligned defaults unless explicitly overridden
        if ed_def is None:  ed_def = self.ED_DEF
        if esp_def is None: esp_def = self.ESP_DEF

        # 1) steady-state baseline run + final state
        t, P_ao, P_lv, V_lv, Q_sv, Q_ao, Q_sys, y_end = self.run_simulation(p, return_state=True)

        cyc0 = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv, Q_ao, T_override=p['T'])
        HR_run = 60.0 / float(p['T'])

        base_metrics = self.pv_metrics_single_loop(
            cyc0,
            HR_bpm=HR_run,
            ed_def=self.ED_DEF,
            esp_def=self.ESP_DEF
        )

        family = [dict(cycle=cyc0, metrics=base_metrics, y_end=y_end.copy())]

        # 2) beat-by-beat integration
        T = float(p['T'])
        dt = T / 500.0
        t_eval = np.linspace(0.0, T, int(np.round(T / dt)) + 1)

        y_state = y_end.copy()
        total_remove = frac_drop * y_state[2]
        dV = total_remove / n_beats

        for k in range(n_beats):
            y_state[2] = max(1e-6, y_state[2] - dV)

            sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, p),
                            [0.0, T], y_state, t_eval=t_eval, max_step=dt)

            t_k = sol.t
            V_lv_k, V_sa_k, V_sv_k = sol.y

            P_lv_k = self.elastance(t_k, p) * V_lv_k + self.P_th
            P_sa_k = V_sa_k / p['C_sa']
            P_ao_k = np.where(P_lv_k > P_sa_k, P_lv_k, P_sa_k)
            P_sv_k = V_sv_k / p['C_sv']

            Q_ao_k = np.where(P_lv_k > P_sa_k, (P_lv_k - P_sa_k) / p['Z_ao'], 0.0)
            Q_sv_k = np.where(P_sv_k > P_lv_k, (P_sv_k - P_lv_k) / p['R_mv'], 0.0)

            cyc = dict(t=t_k, P_ao=P_ao_k, P_lv=P_lv_k, V_lv=V_lv_k, Q_ao=Q_ao_k, Q_sv=Q_sv_k)
            mets = self.pv_metrics_single_loop(
                cyc,
                HR_bpm=HR_run,
                ed_def=self.ED_DEF,
                esp_def=self.ESP_DEF
            )

            family.append(dict(cycle=cyc, metrics=mets, y_end=sol.y[:, -1].copy()))
            y_state = sol.y[:, -1].copy()

        return family

    def fit_pv_relations(self, family, prsw_skip_first: int = 1):
        """
        Fit ESPVR, EDPVR (beta), and PRSW from a preload-occlusion PV family.

        Parameters
        ----------
        family : list[dict]
            Output of preload_occlusion_family() (baseline + occlusion beats)
        prsw_skip_first : int
            Number of initial beats to skip *only* for PRSW regression and PRSW plotting.
            Defaults to 1 to remove the baseline point (family[0]).

        Returns
        -------
        dict
            Includes classical outputs + the arrays *used* for PRSW under:
            - 'EDV_prsw_used', 'SW_prsw_used'
        """
        import numpy as np
        from scipy.optimize import curve_fit

        # --- pull per-beat arrays (all beats, baseline included) ---
        EDV = np.array([d['metrics']['EDV'] for d in family], dtype=float)
        ESV = np.array([d['metrics']['ESV'] for d in family], dtype=float)
        ESP = np.array([d['metrics']['ESP'] for d in family], dtype=float)
        SW = np.array([d['metrics']['SW'] for d in family], dtype=float)

        # --- EDPVR ED points (MVC-based ED points) ---
        EDV_edpvr, EDP_edpvr = [], []
        for d in family:
            cyc = d['cycle']
            m = self.pv_metrics_single_loop(cyc, HR_bpm=None,
                                            ed_def=self.ED_DEF, esp_def=self.ESP_DEF)
            EDV_edpvr.append(m['EDV'])
            EDP_edpvr.append(m['LVEDP'])
        EDV_edpvr = np.asarray(EDV_edpvr, float)
        EDP_edpvr = np.asarray(EDP_edpvr, float)

        # ---------- ESPVR: ESP = Ees * ESV + b ----------
        A = np.vstack([ESV, np.ones_like(ESV)]).T
        Ees, b = np.linalg.lstsq(A, ESP, rcond=None)[0]
        V0 = -b / Ees if Ees != 0 else np.nan

        # ---------- PRSW: SW vs EDV (skip first beat[s]) ----------
        idx0 = int(max(0, prsw_skip_first))
        EDV_prsw = EDV[idx0:].copy()
        SW_prsw = SW[idx0:].copy()

        def _ols(x, y):
            A = np.vstack([x, np.ones_like(x)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(m), float(c)

        def _r2(y, yhat):
            y = np.asarray(y, float)
            yhat = np.asarray(yhat, float)
            ok = np.isfinite(y) & np.isfinite(yhat)
            y, yhat = y[ok], yhat[ok]
            if len(y) < 2:
                return np.nan
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        if len(EDV_prsw) >= 2:
            # OLS on the used points
            Mw_ols, c_ols = _ols(EDV_prsw, SW_prsw)
            Vw_ols = -c_ols / Mw_ols if Mw_ols != 0 else np.nan
            SW_hat = Mw_ols * EDV_prsw + c_ols
            R2_ols = _r2(SW_prsw, SW_hat)

            # Robustify with Cook's distance (on the used subset)
            X = np.vstack([EDV_prsw, np.ones_like(EDV_prsw)]).T
            XtX_inv = np.linalg.inv(X.T @ X)
            H = X @ XtX_inv @ X.T
            h = np.clip(np.diag(H), 1e-9, 0.999999)
            resid = SW_prsw - SW_hat
            p = 2
            mse = np.sum(resid ** 2) / max(len(EDV_prsw) - p, 1)
            cooks = (resid ** 2 / (p * mse)) * (h / (1 - h) ** 2)
            thr = 4.0 / len(EDV_prsw)
            remove_idx = [int(i) for i in np.argsort(cooks)[::-1] if cooks[i] > thr][:2]

            if len(remove_idx) > 0 and (len(EDV_prsw) - len(remove_idx)) >= 4:
                keep = np.ones(len(EDV_prsw), dtype=bool)
                keep[remove_idx] = False
                Mw_rob, c_rob = _ols(EDV_prsw[keep], SW_prsw[keep])
                Vw_rob = -c_rob / Mw_rob if Mw_rob != 0 else np.nan
                R2_rob = _r2(SW_prsw[keep], Mw_rob * EDV_prsw[keep] + c_rob)
            else:
                Mw_rob, Vw_rob, R2_rob = Mw_ols, Vw_ols, R2_ols
                remove_idx = []
            Mw, Vw = Mw_rob, Vw_rob
        else:
            # Not enough points after skipping
            Mw_ols = Vw_ols = R2_ols = np.nan
            Mw = Vw = np.nan
            R2_rob = np.nan
            remove_idx = []

        # ---------- EDPVR: P = C * exp(beta*(V - D)) + E (robust) ----------
        Cfit, betafit, Dfit, Efit, EDV_used, EDP_used = self._fit_edpvr_robust(EDV_edpvr, EDP_edpvr)

        # Optional energetics (keep as in your code)
        ESP0 = float(ESP[0]) if len(ESP) else np.nan
        ESV0 = float(ESV[0]) if len(ESV) else np.nan
        SW0 = float(SW[0]) if len(SW) else np.nan
        PE0 = float(0.5 * ESP0 * (ESV0 - V0)) if np.isfinite(V0) else np.nan
        PVA0 = float(SW0 + PE0) if np.isfinite(PE0) else np.nan
        Eff0 = float(SW0 / PVA0) if (np.isfinite(PVA0) and PVA0 > 0) else np.nan

        EDV0 = float(EDV_used[0]) if len(EDV_used) else np.nan
        dPdV0 = float(Cfit * betafit * np.exp(betafit * (EDV0 - Dfit))) if np.all(
            np.isfinite([Cfit, betafit, Dfit])) else np.nan

        return dict(
            # ESPVR
            Ees=float(Ees), V0=float(V0),

            # EDPVR
            beta=float(betafit),
            rel_EDPVR_C=float(Cfit), rel_EDPVR_D=float(Dfit), rel_EDPVR_E=float(Efit),
            dPdV_ED=float(dPdV0),
            EDPVR_params=(float(Cfit), float(betafit), float(Dfit), float(Efit)),

            # Energetics (baseline)
            PE=float(PE0), PVA=float(PVA0), efficiency=float(Eff0),

            # Full arrays (all beats)
            EDV=EDV, ESV=ESV, ESP=ESP, SW=SW,
            EDV_edpvr=EDV_used, EDP_edpvr=EDP_used,

            # PRSW outputs (regression on *used* subset)
            PRSW_slope=float(Mw), Vw=float(Vw),
            PRSW_slope_ols=float(Mw_ols), Vw_ols=float(Vw_ols),
            R2_PRSW_ols=float(R2_ols), R2_PRSW_rob=float(R2_rob),
            PRSW_outlier_idx=";".join(map(str, remove_idx)),

            # NEW: return the exact points used in the PRSW fit/plot
            EDV_prsw_used=EDV_prsw, SW_prsw_used=SW_prsw
        )

    def pv_fit_quality(self, family, fits):
        """
        R² for:
          - ESPVR (ESP vs ESV)
          - PRSW  (SW vs EDV) -> uses EDV_prsw_used / SW_prsw_used (subset used in the fit)
          - EDPVR (EDP vs EDV)
        """
        import numpy as np

        # Full arrays (for ESPVR)
        ESV = np.asarray(fits.get('ESV', []), dtype=float)
        ESP = np.asarray(fits.get('ESP', []), dtype=float)

        # PRSW uses the same subset used for the regression
        EDV_prsw = np.asarray(fits.get('EDV_prsw_used', fits.get('EDV', [])), dtype=float)
        SW_prsw = np.asarray(fits.get('SW_prsw_used', fits.get('SW', [])), dtype=float)

        # EDPVR arrays
        EDV_edpvr = np.asarray(fits.get('EDV_edpvr', []), dtype=float)
        EDP_edpvr = np.asarray(fits.get('EDP_edpvr', []), dtype=float)

        # Parameters
        Ees = float(fits.get('Ees', np.nan))
        V0 = float(fits.get('V0', np.nan))
        Mw = float(fits.get('PRSW_slope', np.nan))
        Vw = float(fits.get('Vw', np.nan))
        Cfit = float(fits.get('rel_EDPVR_C', np.nan))
        Dfit = float(fits.get('rel_EDPVR_D', np.nan))
        Efit = float(fits.get('rel_EDPVR_E', np.nan))
        betafit = float(fits.get('beta', np.nan))

        def r2(y, yhat):
            y = np.asarray(y, float);
            yhat = np.asarray(yhat, float)
            ok = np.isfinite(y) & np.isfinite(yhat)
            y, yhat = y[ok], yhat[ok]
            if len(y) < 2: return np.nan
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # ESPVR R² on full ES points
        if ESV.size and ESP.size and np.isfinite(Ees) and np.isfinite(V0):
            esp_hat = Ees * (ESV - V0)
            r2_espvr = r2(ESP, esp_hat)
        else:
            r2_espvr = np.nan

        # PRSW R² on the same subset used to fit Mw, Vw
        if EDV_prsw.size and SW_prsw.size and np.isfinite(Mw) and np.isfinite(Vw):
            sw_hat = Mw * (EDV_prsw - Vw)
            r2_prsw = r2(SW_prsw, sw_hat)
        else:
            r2_prsw = np.nan

        # EDPVR R² on MVC-based ED points
        if EDV_edpvr.size and EDP_edpvr.size and np.all(np.isfinite([Cfit, betafit, Dfit, Efit])):
            edp_hat = Cfit * np.exp(betafit * (EDV_edpvr - Dfit)) + Efit
            r2_edpvr = r2(EDP_edpvr, edp_hat)
        else:
            r2_edpvr = np.nan

        return {'R2_ESPVR': r2_espvr, 'R2_PRSW': r2_prsw, 'R2_EDPVR': r2_edpvr}

    def pv_full_report(self, params=None, n_beats=8, frac_drop=0.60):
        """
        Returns a dictionary with:
          - single-loop metrics at baseline steady state
          - ESPVR / EDPVR(beta) / PRSW / energetics from preload occlusion family
        """
        if params is None:
            params = self.params

        # baseline steady state cycle + metrics
        t, P_ao, P_lv, V_lv, Q_sv, Q_ao, Q_sys = self.run_simulation(params)
        cyc0 = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv, Q_ao, T_override=params['T'])
        HR_run = 60.0 / float(params['T'])

        single = self.pv_metrics_single_loop(cyc0, HR_bpm=HR_run, ed_def=self.ED_DEF, esp_def=self.ESP_DEF)
        family = self.preload_occlusion_family(params, n_beats=n_beats, frac_drop=frac_drop,
                                               ed_def=self.ED_DEF, esp_def=self.ESP_DEF)

        try:
            fits = self.fit_pv_relations(family)
        except Exception as e:
            print(f"[warn] fit_pv_relations failed for sub {self.sub_id}: {e}")
            fits = dict(
                Ees=np.nan, V0=np.nan, beta=np.nan,
                rel_EDPVR_C=np.nan, rel_EDPVR_D=np.nan, rel_EDPVR_E=np.nan,
                PRSW_slope=np.nan, Vw=np.nan,
                PE=np.nan, PVA=np.nan, efficiency=np.nan,
                EDV=np.array([]), ESV=np.array([]), ESP=np.array([]), SW=np.array([]),
                EDV_edpvr=np.array([]), EDP_edpvr=np.array([])
            )

        fitq = self.pv_fit_quality(family, fits)

        # merge
        out = {}
        # single-beat
        out.update({f"single_{k}": v for k, v in single.items()})
        # relations
        exclude = ['EDV', 'ESV', 'ESP', 'SW', 'EDV_edpvr', 'EDP_edpvr']
        out.update({f"rel_{k}": v for k, v in fits.items() if k not in exclude})
        out.update({f"fit_{k}": v for k, v in fitq.items()})

        return out, family

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
            'EDV': mets['EDV'],
            'ESV': mets['ESV'],
            'SV':  mets['EDV'] - mets['ESV'],
            'EF':  (mets['EDV'] - mets['ESV'])/mets['EDV']*100 if mets['EDV'] > 0 else np.nan,
            'bSBP': np.max(cyc['P_ao'])*1.1,
            'bDBP': np.min(cyc['P_ao'])*1.1,
            'bMAP': (np.max(cyc['P_ao'])*1.1 + 2*np.min(cyc['P_ao'])*1.1)/3,
            'bPP':  np.max(cyc['P_ao'])*1.1 - np.min(cyc['P_ao'])*1.1,
            'LVOT_Flow_Peak': mets['Q_peak'],
            'time_LVOT_Flow_Peak': mets['t_peak'],
            'ED': mets['EDur']
        }
        sse = 0.0
        for key, w in self.weights.items():
            if key == 'LVEDP':
                continue
            s_val = sim.get(key, np.nan)
            gt_val = gt.get(key, np.nan)
            sse += w * ((s_val - gt_val)/gt_val)**2 if gt_val != 0 else 0.0

        # Soft prior (1) (regression)
        #LVEDP_pred = 2.4033 * self.SWE_velocity + 6.3966
        LVEDP_pred = (
                             2.5645 * self.SWE_velocity + 0.1495 * self.GT_bMAP - 8.7409
        )
        LVEDP_sim = mets['LVEDP']
        sse += self.soft_weight * ((LVEDP_sim - LVEDP_pred) ** 2 / LVEDP_pred ** 2)


        interval = 0.2  # mmHg/ml
        Emin_pred = (self.SWE_velocity - 2.489) / 2.5691 #Pig Data before I/R injury
        Emin_sim = params['E_min']

        # Penalty if outside the interval:
        if Emin_sim < Emin_pred - interval:
            sse += self.soft_weight_Emin * ((Emin_sim - (Emin_pred - interval)) ** 2) / Emin_pred ** 2
        elif Emin_sim > Emin_pred + interval:
            sse += self.soft_weight_Emin * ((Emin_sim - (Emin_pred + interval)) ** 2) / Emin_pred ** 2
        #else: No penalty if within the interval

        # Penalize unphysiological LVEDP
        if LVEDP_sim < 4 or LVEDP_sim > 35:
            sse += self.phys_weight * ((LVEDP_sim - 16) ** 2 / 16 ** 2)
        return sse

    # -- Objective for scipy.optimize --
    def objective(self, theta):
        return self.loss_all_matrices(theta)

    # -- Run optimizer (DE + L-BFGS-B) --
    def optimize(self):
        np.random.seed(self.sub_id)
        t0 = time.perf_counter()
        result_de = differential_evolution(
            func=self.objective,
            bounds=self.bounds,
            workers=cpu_count(),
            maxiter=50,
            popsize=5,
            tol=1e-4,
            disp=True
        )
        t1 = time.perf_counter()
        print(f"✓ DE complete in {t1-t0:.1f}s, best loss {result_de.fun:.6f}")
        print("Polishing with L-BFGS-B…")
        t2 = time.perf_counter()
        res_local = minimize(
            fun=self.objective,
            x0=result_de.x,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'ftol': 1e-4, 'maxiter': 200, 'disp': False}
        )
        t3 = time.perf_counter()
        print(f"✓ L-BFGS-B complete in {t3-t2:.1f}s, final loss {res_local.fun:.6f}")
        print(f"Total optimization time: {t3-t0:.1f}s")
        return res_local, t3-t0

    def simulate_and_metrics(self, best_theta):
        """Run the simulation with best-fit parameters and extract metrics and cycle."""
        params = self.params.copy()
        for name, val in zip(self.param_names, best_theta):
            params[name] = val
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = self.run_simulation(params)
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, idxs = self.extract_cycle_metrics(cyc)
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, params

    def simulate_cycle_and_metrics_HR(self, params, HR):
        """
        Run simulation at a specified HR (t_peak fixed), then cut last cycle and compute metrics.
        """
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, p_used, dt_used = \
            self.run_simulation_with_HR(HR, params=params)

        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, T_override=p_used['T'])
        mets, idxs = self.extract_cycle_metrics(cyc)

        EDV = mets['EDV']
        ESV = mets['ESV']
        SV = EDV - ESV
        EF = (SV / EDV * 100.0) if EDV > 0 else np.nan

        aSBP = float(np.max(cyc['P_ao']))
        aDBP = float(np.min(cyc['P_ao']))
        bSBP = 1.1 * aSBP
        bDBP = 1.1 * aDBP
        bMAP = (bSBP + 2 * bDBP) / 3.0

        metrics = {
            'EDV': float(EDV),
            'ESV': float(ESV),
            'SV': float(SV),
            'EF': float(EF),
            'aSBP': float(aSBP),
            'bSBP': float(bSBP),
            'bMAP': float(bMAP),
            'LVEDP': float(mets['LVEDP']),
            'ESP_LV': float(np.max(cyc['P_lv'])),
            'HR_bpm': float(HR)
        }
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, metrics

    def pv_loop_area_shoelace(self, V, P):
        """
        Shoelace formula for polygon area.
        Returns |Area| in units of (mmHg·mL) if P is mmHg and V is mL.
        """
        V = np.asarray(V, dtype=float)
        P = np.asarray(P, dtype=float)

        # Ensure the polygon is closed
        if V[0] != V[-1] or P[0] != P[-1]:
            V = np.append(V, V[0])
            P = np.append(P, P[0])

        area = 0.5 * np.abs(np.dot(V[:-1], P[1:]) - np.dot(V[1:], P[:-1]))
        return float(area)

    def plot_pv_analysis_dashboard(self, family, fits, save_path=None, show=False):
        """
        Multi-panel PV diagnostics:
          (1) PV family overlay + ED/ES points
          (2) ESPVR: ESP vs ESV + fit
          (3) EDPVR: EDP vs EDV + exponential fit
          (4) PRSW: SW vs EDV + linear fit (paper style)
          (5) Beat-to-beat trends: EDV/ESV and EDP/ESP/SW
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # ---------- Pull arrays (DO NOT mix definitions) ----------
        # For PRSW: must match the EDV used in PRSW regression
        EDV_prsw = np.asarray(fits.get('EDV_prsw_used', fits.get('EDV', [])), dtype=float)
        SW = np.asarray(fits.get('SW_prsw_used', fits.get('SW', [])), dtype=float)

        # For EDPVR: use ED points used in EDPVR fit
        EDV_edp = np.asarray(fits.get('EDV_edpvr', []), dtype=float)
        EDP_edp = np.asarray(fits.get('EDP_edpvr', []), dtype=float)

        # For ESPVR: ES points from family metrics
        ESV = np.asarray(fits.get('ESV', []), dtype=float)
        ESP = np.asarray(fits.get('ESP', []), dtype=float)

        # Fit parameters
        Ees = float(fits.get('Ees', np.nan))
        V0 = float(fits.get('V0', np.nan))

        Mw = float(fits.get('PRSW_slope', np.nan))
        Vw = float(fits.get('Vw', np.nan))

        # EDPVR parameters tuple
        Cfit, betafit, Dfit, Efit = (np.nan, np.nan, np.nan, np.nan)
        if fits.get('EDPVR_params', None) is not None:
            Cfit, betafit, Dfit, Efit = [float(x) for x in fits['EDPVR_params']]

        # Fit quality (optional, if present in fits or compute externally)
        # If you already compute fitq in pv_full_report, pass it in. Here, compute quick R² for PRSW:
        def r2(y, yhat):
            y = np.asarray(y, float);
            yhat = np.asarray(yhat, float)
            ok = np.isfinite(y) & np.isfinite(yhat)
            y = y[ok];
            yhat = yhat[ok]
            if len(y) < 2:
                return np.nan
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # ---------- Fit functions ----------
        def edpvr_fun(V):
            return Cfit * np.exp(betafit * (V - Dfit)) + Efit

        def espvr_fun(V):
            return Ees * (V - V0)

        def prsw_fun(V):
            return Mw * (V - Vw)

        # Beat index for trends (use PRSW EDV length as reference)
        beat_idx = np.arange(len(EDV_prsw))

        # ---------- Figure layout ----------
        fig = plt.figure(figsize=(12, 10), dpi=200)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.35, wspace=0.25)

        # =========================
        # (1) PV family overlay
        # =========================
        ax0 = fig.add_subplot(gs[0, 0])
        for d in family:
            cyc = d['cycle']
            ax0.plot(cyc['V_lv'], cyc['P_lv'], lw=0.9, alpha=0.7)

        # ED points (for EDPVR) and ES points (for ESPVR)
        if EDV_edp.size and EDP_edp.size:
            ax0.scatter(EDV_edp, EDP_edp, s=18, c='tab:blue', label='ED points')
        if ESV.size and ESP.size:
            ax0.scatter(ESV, ESP, s=18, c='tab:red', label='ES points')

        ax0.set_xlabel("LV Volume [mL]")
        ax0.set_ylabel("LV Pressure [mmHg]")
        ax0.set_title("Preload-occlusion PV loop family")
        ax0.legend(frameon=False)

        # =========================
        # (2) ESPVR
        # =========================
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.scatter(ESV, ESP, s=22, label="ES points")

        if np.all(np.isfinite([Ees, V0])) and ESV.size:
            Vgrid = np.linspace(np.min(ESV), np.max(ESV), 200)
            ax1.plot(Vgrid, espvr_fun(Vgrid), lw=2.0, label=f"Fit: Ees={Ees:.3f}, V0={V0:.2f}")

        ax1.set_xlabel("ESV [mL]")
        ax1.set_ylabel("ESP [mmHg]")
        ax1.set_title("ESPVR (contractility)")
        ax1.legend(frameon=False)

        # =========================
        # (3) EDPVR
        # =========================
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(EDV_edp, EDP_edp, s=22, label="ED points")

        if np.all(np.isfinite([Cfit, betafit, Dfit, Efit])) and EDV_edp.size:
            Vgrid = np.linspace(np.min(EDV_edp), np.max(EDV_edp), 200)
            ax2.plot(Vgrid, edpvr_fun(Vgrid), lw=2.0, label=f"Fit: beta={betafit:.4f}")

        ax2.set_xlabel("EDV [mL]")
        ax2.set_ylabel("EDP (LVEDP) [mmHg]")
        ax2.set_title("EDPVR (diastolic stiffness)")
        ax2.legend(frameon=False)

        # =========================
        # (4) PRSW (paper-style)
        # =========================
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(EDV_prsw, SW, s=22, color="0.3", label="Beats")

        if np.all(np.isfinite([Mw, Vw])) and EDV_prsw.size:
            Vgrid = np.linspace(np.min(EDV_prsw), np.max(EDV_prsw), 200)
            SW_hat = prsw_fun(EDV_prsw)
            r2_prsw = r2(SW, SW_hat)

            ax3.plot(Vgrid, prsw_fun(Vgrid), color="tab:orange", lw=2.0,
                     label=f"PRSW={Mw:.1f} (R²={r2_prsw:.3f})")

        ax3.set_xlabel("EDV [mL]")
        ax3.set_ylabel("Stroke Work [mmHg·mL]")
        ax3.set_title("PRSW (stroke work vs preload)")
        ax3.legend(frameon=False)

        # =========================
        # (5) Beat-to-beat trends
        # =========================
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(beat_idx, EDV_prsw, marker='o', lw=1.5, label="EDV")
        ax4.plot(beat_idx, ESV[:len(beat_idx)], marker='o', lw=1.5, label="ESV")
        ax4.set_xlabel("Beat index")
        ax4.set_ylabel("Volume [mL]")
        ax4.set_title("Volumes during preload reduction")
        ax4.legend(frameon=False)

        ax5 = fig.add_subplot(gs[2, 1])
        # If you want beat-to-beat EDP, use EDP_edp (same length as occlusion beats) when available
        if EDP_edp.size == beat_idx.size:
            ax5.plot(beat_idx, EDP_edp, marker='o', lw=1.5, label="LVEDP")
        ax5.plot(beat_idx, ESP[:len(beat_idx)], marker='o', lw=1.5, label="ESP")
        ax5.plot(beat_idx, SW, marker='o', lw=1.5, label="SW")
        ax5.set_xlabel("Beat index")
        ax5.set_ylabel("Pressure / Work")
        ax5.set_title("Pressures and work during preload reduction")
        ax5.legend(frameon=False)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Saved PV analysis dashboard: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_current_figures(self):
        """Save all open matplotlib figures to subject's folder."""
        for i, fig_num in enumerate(plt.get_fignums(), 1):
            fig = plt.figure(fig_num)
            filename = os.path.join(self.subject_dir, f"fig_{i}.png")
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved: {filename}")

    def _style_axes(self, ax, xlabel, ylabel, add_legend=False):
        """Common formatting: Times New Roman, 9 pt, black, no grid, no title."""
        import matplotlib as mpl

        # Prefer Times New Roman; fall back if not installed
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

        ax.set_xlabel(xlabel, fontname="Times New Roman", fontsize=9, color="black")
        ax.set_ylabel(ylabel, fontname="Times New Roman", fontsize=9, color="black")

        ax.tick_params(axis='both', labelsize=9, colors="black")

        # Ensure tick labels exist and are final
        ax.figure.canvas.draw()

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontsize(9)
            lbl.set_color("black")

        if add_legend:
            leg = ax.legend(fontsize=9, frameon=False)
            for txt in leg.get_texts():
                txt.set_fontfamily("Times New Roman")
                txt.set_color("black")

        ax.grid(False)
        ax.set_title("")

    def plot_single_loop_summary(self, single_report, save_path=None, show=False):
        import matplotlib.pyplot as plt

        keys = ['SV', 'EF', 'CO_L_min', 'LVEDP', 'ESP', 'SW', 'Ea']
        vals = [single_report.get(k, float('nan')) for k in keys]

        fig = plt.figure(figsize=(8, 3), dpi=200)
        ax = fig.add_subplot(111)
        ax.bar(keys, vals)
        ax.set_ylabel("Value")
        ax.set_title("Baseline single-loop PV metrics")
        ax.tick_params(axis='x', rotation=45)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Saved single-loop summary: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_all(self, t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, params_used):
        """Plot all basic simulation results and cycle metrics (9 figs with specific formatting)."""

        # 1) Time-varying elastance (Fig 1) – 15 x 5 cm
        E_full = self.elastance(t, params_used)
        fig1 = plt.figure(figsize=(20 / 2.54, 5 / 2.54), dpi=300)
        ax1 = fig1.add_subplot(111)
        ax1.plot(t, E_full,linewidth=0.75)
        self._style_axes(ax1, "Time [s]", "Elastance [mmHg/ml]", add_legend=False)

        # 2) Pressures (Fig 2) – 15 x 5 cm
        fig2 = plt.figure(figsize=(20 / 2.54, 5 / 2.54), dpi=300)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 175)
        ax2.set_xticks(np.arange(0, 8.1, 0.5))
        ax2.set_yticks(np.arange(0, 176, 25))
        ax2.plot(t, P_ao, label="Aortic Pressure",linewidth=0.75)
        ax2.plot(t, P_lv, label="LV Pressure",linewidth=0.75)
        self._style_axes(ax2, "Time [s]", "Pressure [mmHg]", add_legend=True)

        # 3) Flows (Fig 3) – 15 x 5 cm
        fig3 = plt.figure(figsize=(20 / 2.54, 5 / 2.54), dpi=300)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlim(0, 8)
        ax3.set_ylim(0, 500)
        ax3.set_xticks(np.arange(0, 8.1, 0.5))
        ax3.set_yticks(np.arange(0, 501, 100))
        ax3.plot(t, Q_lv_ao, label="Aortic Flow",linewidth=0.75,color="crimson")
        ax3.plot(t, Q_sv_lv, label="Mitral Flow",linewidth=0.75,color="aqua")
        ax3.plot(t, Q_sys, label="Systemic Flow",linewidth=0.75,color="green")
        self._style_axes(ax3, "Time [s]", "Flow rate [ml/s]", add_legend=True)

        # 4) LV volume vs time (Fig 4) – 15 x 5 cm
        fig4 = plt.figure(figsize=(20 / 2.54, 5 / 2.54), dpi=300)
        ax4 = fig4.add_subplot(111)
        ax4.plot(t, V_lv, label="LV Volume",linewidth=0.75,color="darkblue")
        ax4.set_xlim(0, 8)
        ax4.set_ylim(20, 100)
        ax4.set_xticks(np.arange(0, 8.1, 0.5))
        ax4.set_yticks(np.arange(20, 101, 20))
        self._style_axes(ax4, "Time [s]", "Volume [ml]", add_legend=True)

        # 5) Global PV loop (full simulation) (Fig 5) – 5 x 5 cm
        fig5 = plt.figure(figsize=(6 / 2.54, 6 / 2.54), dpi=600)
        ax5 = fig5.add_subplot(111)
        ax5.plot(V_lv, P_lv,linewidth=0.75,color="indigo")
        self._style_axes(ax5, "LV Volume [ml]", "LV Pressure [mmHg]", add_legend=False)

        # Cycle arrays
        t_c = cyc['t']
        P_ao_c = cyc['P_ao']
        P_lv_c = cyc['P_lv']
        V_lv_c = cyc['V_lv']
        Q_ao_c = cyc['Q_ao']
        Q_sv_c = cyc['Q_sv']

        # 6) Cycle pressures (Fig 6) – 5 x 5 cm
        fig6 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax6 = fig6.add_subplot(111)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 120)
        ax6.set_xticks(np.arange(0, 1.1, 0.25))
        ax6.set_yticks(np.arange(0, 151, 20))
        ax6.plot(t_c, P_ao_c,linewidth=0.75)
        ax6.plot(t_c, P_lv_c,linewidth=0.75)
        #ax6.scatter([t_c[idxs['EDV_idx']]], [P_lv_c[idxs['EDV_idx']]], c="r", marker="o",
                    #label="LVEDP@EDV")
        self._style_axes(ax6, "Time [s]", "Pressure [mmHg]", add_legend=True)

        # 7) Cycle flows (Fig 7) – 5 x 5 cm
        fig7 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax7 = fig7.add_subplot(111)
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 500)
        ax7.set_xticks(np.arange(0, 1.1, 0.25))
        ax7.set_yticks(np.arange(0, 401, 100))
        ax7.plot(t_c, Q_ao_c,color="crimson",linewidth=0.75)
        ax7.plot(t_c, Q_sv_c,color="aqua",linewidth=0.75)
        #ax7.scatter([t_c[idxs['peak_idx']]], [Q_ao_c[idxs['peak_idx']]], c="g", marker="o",
                    #label="Q_peak")
        ax7.axvline(t_c[idxs['edur_start']], linestyle="--", color="black", linewidth=0.75)
        ax7.axvline(t_c[idxs['edur_end']], linestyle="--", color="black", linewidth=0.75)
        self._style_axes(ax7, "Time [s]", "Flow rate [ml/s]", add_legend=True)

        # 8) LV volume over cycle (Fig 8) – 5 x 5 cm
        fig8 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax8 = fig8.add_subplot(111)
        ax8.plot(t_c, V_lv_c,linewidth=0.75)
        ax8.scatter([t_c[idxs['ESV_idx']]], [V_lv_c[idxs['ESV_idx']]], color="blue", marker="o",
                    label="ESV")
        ax8.scatter([t_c[idxs['EDV_idx']]], [mets['EDV']], c="r", marker="o",
                    label="EDV")
        self._style_axes(ax8, "Time [s]", "Volume [ml]", add_legend=True)

        # 9) PV loop for last cycle (Fig 9) – 5 x 5 cm
        fig9 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax9 = fig9.add_subplot(111)
        ax9.set_xlim(20, 70)
        ax9.set_ylim(0, 120)
        ax9.set_xticks(np.arange(20, 81, 10))
        ax9.set_yticks(np.arange(0, 151, 20))
        ax9.plot(V_lv_c, P_lv_c,linewidth=0.75)
        #ax9.scatter([mets['EDV']], [mets['LVEDP']], c="g", marker="o", label="LVEDP")
        self._style_axes(ax9, "LV Volume [ml]", "LV Pressure [mmHg]", add_legend=True)



    def compare_with_gt(self, mets, cyc):
        """Print and return a dictionary comparing simulated and ground-truth metrics."""
        gt = self.data_row_GT
        sim = {
            'EDV': mets['EDV'],
            'ESV': mets['ESV'],
            'SV':  mets['EDV'] - mets['ESV'],
            'EF':  (mets['EDV'] - mets['ESV'])/mets['EDV']*100 if mets['EDV']>0 else np.nan,
            'bSBP': np.max(cyc['P_ao'])*1.1,
            'bDBP': np.min(cyc['P_ao'])*1.1,
            'bMAP': (np.max(cyc['P_ao'])*1.1 + 2*np.min(cyc['P_ao'])*1.1)/3,
            'bPP':  np.max(cyc['P_ao'])*1.1 - np.min(cyc['P_ao'])*1.1,
            'LVOT_Flow_Peak': mets['Q_peak'],
            'time_LVOT_Flow_Peak': mets['t_peak'],
            'ED': mets['EDur'],
            'LVEDP': mets['LVEDP']
        }
        print(f"\n--- Comparing Simulation vs. Ground-Truth for sub_id={self.sub_id} ---")
        print(f"{'Metric':>22} | {'Sim':>10} | {'GT':>10}")
        print('-'*46)
        for key in [
            'EDV','ESV','SV','EF',
            'bSBP','bDBP','bMAP','bPP',
            'LVOT_Flow_Peak','time_LVOT_Flow_Peak',
            'ED','LVEDP'
        ]:
            s_val = sim.get(key, np.nan)
            gt_val = gt.get(key, np.nan)
            print(f"{key:>22} | {s_val:10.3f} | {gt_val:10.3f}")
        return sim

    def collect_result_row(self, best, loss, opt_time, sim_matrices):
        row = {'sub_id': self.sub_id, 'loss': loss, 'optimization_time_sec': opt_time,
               'soft_weight': self.soft_weight, 'phys_weight': self.phys_weight}
        for name, val in zip(self.param_names, best.x):
            row[name] = val
        row.update(sim_matrices)
        return row

    # === NEW: convenience function to simulate one cycle and compute metrics ===
    def simulate_cycle_and_metrics(self, params):
        """Run simulation with given params, cut last full cycle, return cycle and key metrics."""
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = self.run_simulation(params)
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, idxs = self.extract_cycle_metrics(cyc)

        # Basic volumetric metrics
        EDV = mets['EDV']
        ESV = mets['ESV']
        SV = EDV - ESV
        EF = (SV / EDV * 100.0) if EDV > 0 else np.nan

        # Pressure metrics
        aSBP = np.max(cyc['P_ao'])             # central / aortic SBP
        aDBP = np.min(cyc['P_ao'])             # central / aortic DBP
        aMAP = (aSBP + 2 * aDBP) / 3.0         # central MAP (if you ever need it)

        bSBP = 1.1 * aSBP                      # brachial SBP (your convention)
        bDBP = 1.1 * aDBP                      # brachial DBP
        bMAP = (bSBP + 2 * bDBP) / 3.0         # brachial MAP

        LVEDP = mets['LVEDP']

        metrics = {
            'EDV': EDV,
            'ESV': ESV,
            'SV': SV,
            'EF': EF,
            'aSBP': aSBP,
            'bSBP': bSBP,
            'bMAP': bMAP,
            'LVEDP': LVEDP
        }
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, metrics

    def run_sensitivity_analysis(self,
                                 preload_params=('V_tot', 'C_sv', 'R_mv'),
                                 afterload_params=('R_sys', 'C_sa', 'Z_ao'),
                                 elastance_params=('E_min', 'E_max', 't_peak'),
                                 steps=None,
                                 fix_axes=True,
                                 include_hr=True,
                                 hr_steps=None):
        """
        Integrated sensitivity analysis:
          - OAT for preload/afterload/elastance (including t_peak if provided)
          - HR sensitivity (± steps) with t_peak fixed (uses simulate_cycle_and_metrics_HR)

        Produces:
          - One PV overlay figure per parameter (baseline + each step)
          - One PV overlay figure for HR (baseline + each HR step)
          - One CSV with metrics and % changes vs baseline
        """
        if steps is None:
            steps = [-0.10, 0.10]
        if hr_steps is None:
            hr_steps = [-0.10, 0.10]

        baseline_params = self.params.copy()

        # --- baseline metrics ---
        (_, _, _, _, _, _, _,
         cyc_base, mets_base, _, metrics_base) = self.simulate_cycle_and_metrics(baseline_params)

        base_esp = float(np.max(cyc_base['P_lv']))

        rows = []

        def pct_change(val, base):
            return 100.0 * (val - base) / base if base != 0 else np.nan

        # =========================
        # 1) OAT PARAMETER BLOCKS
        # =========================
        blocks = [
            ('Preload', preload_params),
            ('Afterload', afterload_params),
            ('Elastance', elastance_params),
        ]

        for block_name, params_list in blocks:
            for pname in params_list:

                if pname not in baseline_params:
                    print(f"[warn] {pname} not in params; skipping.")
                    continue

                fig = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
                ax = fig.add_subplot(111)

                # baseline PV
                ax.plot(cyc_base['V_lv'], cyc_base['P_lv'], label="Baseline", linewidth=1.0, color='k')
                #ax.scatter([mets_base['EDV']], [mets_base['LVEDP']], s=15, color='k')

                # OAT perturbations: vary only pname
                for pc in steps:
                    params_var = baseline_params.copy()
                    params_var[pname] = baseline_params[pname] * (1.0 + pc)
                    label = f"{pname} {int(pc * 100)}%"

                    (_, _, _, _, _, _, _,
                     cyc, mets, _, metrics) = self.simulate_cycle_and_metrics(params_var)

                    line, = ax.plot(cyc['V_lv'], cyc['P_lv'], label=label, linewidth=0.75)
                    #ax.scatter([mets['EDV']], [mets['LVEDP']], s=15, color=line.get_color())

                    esp_lv = float(np.max(cyc['P_lv']))

                    rows.append({
                        'sub_id': self.sub_id,
                        'Scenario': block_name,
                        'Changed_param': pname,
                        'Change': f"{int(pc * 100)}%",

                        'SV [ml]': metrics['SV'],
                        'EF [%]': metrics['EF'],
                        'aSBP [mmHg]': metrics['aSBP'],
                        'bSBP [mmHg]': metrics['bSBP'],
                        'bMAP [mmHg]': metrics['bMAP'],
                        'LVEDP [mmHg]': metrics['LVEDP'],
                        'ESP_LV [mmHg]': esp_lv,

                        'SV_change_%': pct_change(metrics['SV'], metrics_base['SV']),
                        'EF_change_%': pct_change(metrics['EF'], metrics_base['EF']),
                        'aSBP_change_%': pct_change(metrics['aSBP'], metrics_base['aSBP']),
                        'bSBP_change_%': pct_change(metrics['bSBP'], metrics_base['bSBP']),
                        'bMAP_change_%': pct_change(metrics['bMAP'], metrics_base['bMAP']),
                        'LVEDP_change_%': pct_change(metrics['LVEDP'], metrics_base['LVEDP']),
                        'ESP_LV_change_%': pct_change(esp_lv, base_esp),
                    })

                # axis formatting
                if fix_axes:
                    ax.set_xlim(30, 90)
                    ax.set_ylim(0, 150)
                    ax.set_xticks(np.arange(30, 91, 10))
                    ax.set_yticks(np.arange(0, 151, 25))
                ax.set_xlabel("LV Volume [ml]", fontname="Times New Roman", fontsize=9, color="black")
                ax.set_ylabel("LV Pressure [mmHg]", fontname="Times New Roman", fontsize=9, color="black")
                #ax.legend(fontsize=8, frameon=False)
                ax.grid(False)

                fig_path = os.path.join(self.subject_dir, f"PV_OAT_{block_name}_{pname}_sub_{self.sub_id}.png")
                fig.savefig(fig_path, dpi=600, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {fig_path}")

        # =========================
        # 2) HR SENSITIVITY BLOCK
        # =========================
        if include_hr:
            HR0 = float(self.bpm)

            fig = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
            ax = fig.add_subplot(111)

            ax.plot(cyc_base['V_lv'], cyc_base['P_lv'], label=f"Baseline HR={HR0:.1f}", linewidth=1.0, color='k')
            #ax.scatter([mets_base['EDV']], [mets_base['LVEDP']], s=15, color='k')

            for pc in hr_steps:
                HR_var = HR0 * (1.0 + pc)
                label = f"HR {int(pc * 100)}%"

                (_, _, _, _, _, _, _,
                 cyc, mets, _, metrics) = self.simulate_cycle_and_metrics_HR(baseline_params, HR_var)

                line, = ax.plot(cyc['V_lv'], cyc['P_lv'], label=label, linewidth=0.75)
                #ax.scatter([mets['EDV']], [mets['LVEDP']], s=15, color=line.get_color())

                rows.append({
                    'sub_id': self.sub_id,
                    'Scenario': 'HR',
                    'Changed_param': 'HR',
                    'Change': f"{int(pc * 100)}%",
                    'HR [bpm]': metrics['HR_bpm'],

                    'SV [ml]': metrics['SV'],
                    'EF [%]': metrics['EF'],
                    'aSBP [mmHg]': metrics['aSBP'],
                    'bSBP [mmHg]': metrics['bSBP'],
                    'bMAP [mmHg]': metrics['bMAP'],
                    'LVEDP [mmHg]': metrics['LVEDP'],
                    'ESP_LV [mmHg]': metrics['ESP_LV'],

                    'SV_change_%': pct_change(metrics['SV'], metrics_base['SV']),
                    'EF_change_%': pct_change(metrics['EF'], metrics_base['EF']),
                    'LVEDP_change_%': pct_change(metrics['LVEDP'], metrics_base['LVEDP']),
                    'ESP_LV_change_%': pct_change(metrics['ESP_LV'], base_esp),
                })

            if fix_axes:
                ax.set_xlim(30, 90)
                ax.set_ylim(0, 150)
                ax.set_xticks(np.arange(30, 91, 10))
                ax.set_yticks(np.arange(0, 151, 25))
            ax.set_xlabel("LV Volume [ml]", fontname="Times New Roman", fontsize=9, color="black")
            ax.set_ylabel("LV Pressure [mmHg]", fontname="Times New Roman", fontsize=9, color="black")
            #ax.legend(fontsize=8, frameon=False)
            ax.grid(False)

            fig_path = os.path.join(self.subject_dir, f"PV_HR_sub_{self.sub_id}.png")
            fig.savefig(fig_path, dpi=600, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {fig_path}")

        # =========================
        # 3) SAVE CSV
        # =========================
        df = pd.DataFrame(rows)
        metrics_path = os.path.join(self.subject_dir, f"sensitivity_metrics_sub_{self.sub_id}.csv")
        df.to_csv(metrics_path, index=False)
        print(f"Saved sensitivity metrics table: {metrics_path}")

    def _idx_ed_maxV(self, V):
        return int(np.argmax(V))

    def _idx_ed_dpdt_upstroke(self, t, P, V, mvc_idx=None, search_back_s=0.25):
        """
        Paper-style ED: onset of positive deflection of dP/dt (ED corner).

        If mvc_idx provided, constrain search to [mvc_idx - search_back_s, mvc_idx].
        Fallback: MVC_flow (if available in caller) or max(V).
        """
        t = np.asarray(t, dtype=float)
        P = np.asarray(P, dtype=float)
        V = np.asarray(V, dtype=float)

        dPdt = np.gradient(P, t)

        # Constrained search near MVC (preferred)
        if mvc_idx is not None:
            dt = float(t[1] - t[0])
            back_n = int(round(search_back_s / dt))
            i0 = max(0, int(mvc_idx) - back_n)
            i1 = int(mvc_idx)

            d = dPdt[i0:i1 + 1]
            crossings = np.where((d[:-1] <= 0) & (d[1:] > 0))[0] + 1
            if len(crossings):
                return int(i0 + crossings[-1])

        # Unconstrained fallback: last crossing before peak +dP/dt
        i_peak = int(np.argmax(dPdt))
        crossings = np.where((dPdt[:-1] <= 0) & (dPdt[1:] > 0))[0] + 1
        crossings = crossings[crossings < i_peak]
        if len(crossings):
            return int(crossings[-1])

        # Final fallback
        return int(np.argmax(V))

    def _idx_es_end_ejection(self, Qao, thr_frac=0.01):
        if Qao is None or np.max(Qao) <= 0:
            return None
        thr = thr_frac * np.max(Qao)
        mask = Qao > thr
        if not mask.any():
            return None
        return int(np.where(mask)[0][-1])

    def _idx_esp_max_P_over_V(self, P, V, Qao=None, avo=None, avc=None):
        """
        Paper-style ESP: pressure at maximal P/V.
        Prefer to restrict to the ejection window (AVO->AVC) if available.
        """
        eps = 1e-6
        ratio = P / np.maximum(V, eps)

        if avo is not None and avc is not None:
            # handle possible wrap is not expected in one cycle, but be safe
            if avc >= avo:
                idx_range = np.arange(avo, avc + 1)
            else:
                idx_range = np.concatenate([np.arange(avo, len(P)), np.arange(0, avc + 1)])
            i = idx_range[np.argmax(ratio[idx_range])]
            return int(i)

        # else: global max
        return int(np.argmax(ratio))

    def plot_pv_preload_with_relations(self,
                                       params=None,
                                       n_beats=8,
                                       frac_drop=0.60,
                                       beat_for_Ea_SW=0,  # usually baseline = 0
                                       figsize_cm=(6, 6),
                                       dpi=600,
                                       xlim=None,
                                       ylim=None,
                                       annotate=True,
                                       save_path=None,
                                       show=False):
        """
        Single-panel PV figure (journal-ready):
          - preload occlusion PV family overlay
          - ED points (MVC-based) and ES points (end-ejection)
          - ESPVR (Ees, V0)
          - EDPVR (C, beta, D, E)
          - Ea line for chosen beat
          - SW shown as shaded PV loop area for chosen beat

        Uses:
          family = preload_occlusion_family()
          fits   = fit_pv_relations(family)

        Parameters:
          beat_for_Ea_SW: which beat to use for Ea and SW display (default baseline beat=0)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if params is None:
            params = self.params

        # --- compute family and fits ---
        family = self.preload_occlusion_family(params=params, n_beats=n_beats, frac_drop=frac_drop)
        fits = self.fit_pv_relations(family)

        # --- pull arrays ---
        EDV_all = np.asarray(fits.get('EDV', []), dtype=float)
        ESV_all = np.asarray(fits.get('ESV', []), dtype=float)
        ESP_all = np.asarray(fits.get('ESP', []), dtype=float)

        EDV_edp = np.asarray(fits.get('EDV_edpvr', []), dtype=float)  # MVC-based ED points
        EDP_edp = np.asarray(fits.get('EDP_edpvr', []), dtype=float)

        # --- fitted params ---
        Ees = float(fits.get('Ees', np.nan))
        V0 = float(fits.get('V0', np.nan))

        Cfit, betafit, Dfit, Efit = (np.nan, np.nan, np.nan, np.nan)
        if fits.get("EDPVR_params", None) is not None:
            Cfit, betafit, Dfit, Efit = [float(v) for v in fits["EDPVR_params"]]

        # --- choose beat for Ea and SW ---
        beat_for_Ea_SW = int(np.clip(beat_for_Ea_SW, 0, len(family) - 1))
        cyc_sel = family[beat_for_Ea_SW]["cycle"]
        met_sel = family[beat_for_Ea_SW]["metrics"]

        EDV0 = float(met_sel["EDV"])
        ESV0 = float(met_sel["ESV"])
        ESP0 = float(met_sel["ESP"])
        LVEDP0 = float(met_sel["LVEDP"])
        SW0 = float(met_sel["SW"])

        SV0 = EDV0 - ESV0
        Ea0 = float(met_sel.get("Ea", np.nan))
        if (not np.isfinite(Ea0)) and SV0 > 0:
            Ea0 = ESP0 / SV0

        # --- figure setup ---
        fig_w, fig_h = figsize_cm[0] / 2.54, figsize_cm[1] / 2.54
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(20, 90)
        ax.set_ylim(0, 150)
        ax.set_xticks(np.arange(20, 91, 10))
        ax.set_yticks(np.arange(0, 151, 20))

        # --- PV family in gray ---
        Vmin, Vmax, Pmin, Pmax = np.inf, -np.inf, np.inf, -np.inf
        for d in family:
            V = np.asarray(d["cycle"]["V_lv"], dtype=float)
            P = np.asarray(d["cycle"]["P_lv"], dtype=float)
            ax.plot(V, P, color="0.75", lw=0.8, alpha=0.9, zorder=1)
            Vmin = min(Vmin, np.min(V));
            Vmax = max(Vmax, np.max(V))
            Pmin = min(Pmin, np.min(P));
            Pmax = max(Pmax, np.max(P))

        # --- highlight chosen beat (baseline by default) ---
        V_sel = np.asarray(cyc_sel["V_lv"], dtype=float)
        P_sel = np.asarray(cyc_sel["P_lv"], dtype=float)
        ax.plot(V_sel, P_sel, color="k", lw=1.4, zorder=3, label="Selected loop")

        # --- stroke work as shaded PV area for selected beat ---
        #ax.fill(V_sel, P_sel, color="0.5", alpha=0.18, zorder=2,
                #label=f"SW = {SW0:.0f} mmHg·mL")

        # --- ED & ES points ---
        # ED points used for EDPVR (MVC-based)
        if EDV_edp.size > 0 and EDP_edp.size > 0:
            ax.scatter(EDV_edp, EDP_edp, s=18, color="tab:blue",
                       edgecolor="none", zorder=5, label="ED points")

        # ES points used for ESPVR
        if ESV_all.size > 0 and ESP_all.size > 0:
            ax.scatter(ESV_all, ESP_all, s=18, color="tab:red",
                       edgecolor="none", zorder=5, label="ES points")

        # mark selected beat ED/ES
        ax.scatter([EDV0], [LVEDP0], s=28, color="tab:blue", edgecolor="k", linewidth=0.3, zorder=6)
        ax.scatter([ESV0], [ESP0], s=28, color="tab:red", edgecolor="k", linewidth=0.3, zorder=6)

        # --- ESPVR line: P = Ees (V - V0) ---
        if np.isfinite(Ees) and np.isfinite(V0):
            Vgrid_esp = np.linspace(min(Vmin, V0) - 10, Vmax + 10, 400)
            Pgrid_esp = Ees * (Vgrid_esp - V0)
            ax.plot(Vgrid_esp, Pgrid_esp, color="tab:red", lw=1.2, zorder=2,
                    label=f"ESPVR: Ees={Ees:.2f}, V0={V0:.1f}")

        # --- EDPVR curve: P = C exp(beta (V - D)) + E ---
        if np.all(np.isfinite([Cfit, betafit, Dfit, Efit])) and (EDV_edp.size > 0):
            Vgrid_edp = np.linspace(min(EDV_edp.min(), Vmin) - 10, Vmax + 10, 400)
            Pgrid_edp = Cfit * np.exp(betafit * (Vgrid_edp - Dfit)) + Efit
            ax.plot(Vgrid_edp, Pgrid_edp, color="tab:orange", lw=1.2, ls="--", zorder=2,
                    label=f"EDPVR: β={betafit:.4f}")

        # --- Ea line (recommended): connect (ED at MVC) -> (ES at end-ejection) ---
        # Use the same ED definition you use for ED points on the plot:
        EDV_ea, EDP_ea = self.ed_point_at_mvc(cyc_sel, thr_frac=0.005)

        ESV_ea = float(met_sel["ESV"])
        ESP_ea = float(met_sel["ESP"])

        SV_ea = EDV_ea - ESV_ea
        if SV_ea > 0:
            Ea_plot = (ESP_ea - EDP_ea) / SV_ea  # mmHg/mL

            Vgrid_ea = np.linspace(ESV_ea, EDV_ea, 100)
            # Line through (EDV_ea, EDP_ea) with slope -Ea_plot in PV-plane
            Pgrid_ea = EDP_ea + Ea_plot * (EDV_ea - Vgrid_ea)

            ax.plot(Vgrid_ea, Pgrid_ea,
                    color="tab:green", lw=1.2, ls="-.", zorder=2,
                    label=f"Ea={Ea_plot:.2f} mmHg/mL")

        # --- axis limits ---
        #if xlim is None:
        #    Vpad = 0.06 * (Vmax - Vmin + 1e-9)
        #    ax.set_xlim(Vmin - Vpad, Vmax + Vpad)
        #else:
        #    ax.set_xlim(*xlim)

        #if ylim is None:
        #    Ppad = 0.10 * (Pmax - Pmin + 1e-9)
        #    ax.set_ylim(max(0, Pmin - Ppad), Pmax + Ppad)
        #else:
        #    ax.set_ylim(*ylim)

        # --- formatting (uses your helper) ---
        self._style_axes(ax, "LV Volume [mL]", "LV Pressure [mmHg]", add_legend=False)

        # legend (journal-friendly)
        #leg = ax.legend(loc="best", frameon=False, fontsize=8)
        #for txt in leg.get_texts():
        #    txt.set_fontname("Times New Roman")
        #    txt.set_color("black")

        # optional annotation block
        #if annotate:
        #    txt = (f"Ees={Ees:.2f} mmHg/mL\n"
        #           f"β={betafit:.4f}\n"
        #           f"Ea={Ea0:.2f} mmHg/mL\n"
        #           f"SW={SW0:.0f} mmHg·mL")
        #    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
        #            fontsize=8, fontname="Times New Roman",
        #            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))

        # save/show
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved PV preload+relations figure: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return family, fits

    def qc_plot_edpvr_case(self, params_case, case_name, n_beats=8, frac_drop=0.6, save_dir=None):
        """
        QC plot for one sensitivity case:
          - PV family overlay (preload occlusion)
          - ED points (EDV, LVEDP) and EDPVR fit curve
          - Reports beta, (C,D,E) and R² of EDPVR fit

        case_name: string like 'baseline', 'R_sys_-10', etc.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = self.subject_dir

        # Run occlusion family for this case
        family = self.preload_occlusion_family(params_case, n_beats=n_beats, frac_drop=frac_drop)

        # Fit relations (includes EDPVR params)
        fits = self.fit_pv_relations(family)
        fitq = self.pv_fit_quality(family, fits)

        EDV = np.asarray(fits.get('EDV_edpvr', fits['EDV']), dtype=float)
        EDP = np.asarray(fits.get('EDP_edpvr', np.nan), dtype=float)
        if not np.any(np.isfinite(EDP)):
            # plot only PV family and write "EDPVR fit failed" in title
            ...

        Cfit = float(fits['rel_EDPVR_C'])
        Dfit = float(fits['rel_EDPVR_D'])
        Efit = float(fits['rel_EDPVR_E'])
        betafit = float(fits['beta'])

        def edpvr_fun(V):
            return Cfit * np.exp(betafit * (V - Dfit)) + Efit

        # ---- Make figure ----
        fig = plt.figure(figsize=(10, 4), dpi=200)
        gs = fig.add_gridspec(1, 2, wspace=0.30)

        # (A) PV family
        ax0 = fig.add_subplot(gs[0, 0])
        for d in family:
            cyc = d['cycle']
            ax0.plot(cyc['V_lv'], cyc['P_lv'], lw=0.8, alpha=0.7)
        ax0.set_xlabel("LV Volume [mL]")
        ax0.set_ylabel("LV Pressure [mmHg]")
        ax0.set_title(f"{case_name}: PV family (preload occlusion)")

        # (B) EDPVR ED points + fit
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.scatter(EDV, EDP, s=22, label="ED points")

        Vgrid = np.linspace(np.min(EDV), np.max(EDV), 200)
        ax1.plot(Vgrid, edpvr_fun(Vgrid), lw=2.0, label="EDPVR fit")

        r2_edpvr = float(fitq.get('R2_EDPVR', np.nan))
        ax1.set_title(
            f"EDPVR fit: β={betafit:.3g}  R²={r2_edpvr:.3f}\n"
            f"C={Cfit:.3g}, D={Dfit:.2f}, E={Efit:.2f}"
        )
        ax1.set_xlabel("EDV [mL]")
        ax1.set_ylabel("LVEDP [mmHg]")
        ax1.legend(frameon=False)

        out_path = os.path.join(save_dir, f"QC_EDPVR_{case_name}_sub_{self.sub_id}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved EDPVR QC plot: {out_path}")

        return fits, fitq

    def ed_point_at_mvc(self, cyc, thr_frac=0.005):
        """
        Return (EDV_mvc, EDP_mvc) using mitral valve closure (MVC) timing.
        MVC is approximated as the last time index where mitral inflow Q_sv is above threshold.

        thr_frac: fraction of peak Q_sv used as threshold (smaller than Q_ao threshold is often better).
        Fallback: if Q_sv not available or no mask, use max(V) point.
        """
        import numpy as np

        t = cyc['t']
        V = cyc['V_lv']
        P = cyc['P_lv']
        Qsv = cyc.get('Q_sv', None)

        # Default fallback (old definition)
        EDV_idx = int(np.argmax(V))
        EDV_fallback = float(V[EDV_idx])
        EDP_fallback = float(P[EDV_idx])

        if Qsv is None:
            return EDV_fallback, EDP_fallback

        Qsv = np.asarray(Qsv, dtype=float)
        if np.max(Qsv) <= 0:
            return EDV_fallback, EDP_fallback

        thr = max(thr_frac * np.max(Qsv), 1e-3)  # absolute floor threshold (ml/s)
        mask = Qsv > thr
        if not mask.any():
            return EDV_fallback, EDP_fallback

        mvc_idx = int(np.where(mask)[0][-1])  # last inflow sample = MVC
        return float(V[mvc_idx]), float(P[mvc_idx])

    def _fit_edpvr_robust(self, EDV_edpvr, EDP_edpvr, fix_E='min'):
        """
        Robust EDPVR fit.

        Default: fix_E='min'  -> E is fixed to min(EDP points),
                                 fit C, beta, D to (P - E).
        Alternative: fix_E=0  -> E fixed to 0 mmHg (if you pre-reference pressure).
        Returns: (Cfit, betafit, Dfit, Efit, EDV_used, EDP_used)
        """
        import numpy as np
        from scipy.optimize import curve_fit

        x = np.asarray(EDV_edpvr, dtype=float)
        y = np.asarray(EDP_edpvr, dtype=float)

        # remove NaNs
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 5:
            return np.nan, np.nan, np.nan, np.nan, x, y

        # sort + unique x
        order = np.argsort(x)
        x, y = x[order], y[order]
        x_unique, idx_unique = np.unique(x, return_index=True)
        y = y[idx_unique]
        x = x_unique
        if len(x) < 5:
            return np.nan, np.nan, np.nan, np.nan, x, y

        # robust outlier removal (MAD)
        med = np.median(y)
        mad = np.median(np.abs(y - med)) + 1e-9
        z = 0.6745 * (y - med) / mad
        keep = np.abs(z) < 3.5
        if keep.sum() >= 5:
            x, y = x[keep], y[keep]

        Vmin, Vmax = float(np.min(x)), float(np.max(x))
        Pmin, Pmax = float(np.min(y)), float(np.max(y))

        # ---- Fix E ----
        if fix_E == 'min':
            Efit = Pmin
        else:
            Efit = float(fix_E)  # e.g., 0.0

        y_shift = y - Efit
        y_shift = np.clip(y_shift, 1e-6, np.inf)  # ensure positive for exp fit

        # 3-parameter model (no E)
        def edpvr3(V, C, beta, D):
            return C * np.exp(beta * (V - D))

        # initial guesses
        C0 = max(np.max(y_shift), 1e-3)
        D0 = Vmin
        beta0_list = [0.005, 0.02, 0.05, 0.1]

        bounds = (
            [1e-6, 0.0, Vmin - 50.0],
            [np.inf, 1.0, Vmax + 50.0]
        )

        best = None
        best_sse = np.inf

        for b0 in beta0_list:
            p0 = [C0, b0, D0]
            try:
                popt, _ = curve_fit(edpvr3, x, y_shift, p0=p0, bounds=bounds, maxfev=200000)
                yhat = edpvr3(x, *popt)
                sse = float(np.sum((y_shift - yhat) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best = popt
            except RuntimeError:
                pass

        if best is None:
            return np.nan, np.nan, np.nan, np.nan, x, y

        Cfit, betafit, Dfit = [float(v) for v in best]
        return Cfit, betafit, Dfit, float(Efit), x, y

    def plot_wiggers_threepanel(self, cyc, HR_bpm=None, save_path=None, show=False,
                                dpi=600, figsize_cm=(22, 15)):
        """
        Wiggers-like 3-panel figure (no PV loop here):
          (1) Pressures: P_ao (tab:blue) and P_lv (tab:orange)
          (2) Flows: Q_ao and Q_sv (mitral)
          (3) LV volume

        Time axis: within-cycle time 0..T (not re-referenced to MVC).
        Valve event lines: MVC, AVO, AVC, MVO.
        Intervals shown as shaded spans: IVCT (MVC->AVO), ET (AVO->AVC), IVRT (AVC->MVO).
        Also overlays tau exponential fit within IVRT on LV pressure curve and marks fit start at -dP/dtmin.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        if HR_bpm is None:
            HR_bpm = 60.0 / float(self.params['T'])
        HR_bpm = float(HR_bpm)

        t = np.asarray(cyc['t'], dtype=float)
        P_lv = np.asarray(cyc['P_lv'], dtype=float)
        V_lv = np.asarray(cyc['V_lv'], dtype=float)
        P_ao = np.asarray(cyc.get('P_ao', np.full_like(P_lv, np.nan)), dtype=float)
        Qao = np.asarray(cyc.get('Q_ao', np.full_like(P_lv, np.nan)), dtype=float)
        Qmv = np.asarray(cyc.get('Q_sv', np.full_like(P_lv, np.nan)), dtype=float)

        Tcyc = self._cycle_period(t)

        # Valve events from flows
        avo_idx, avc_idx, mvo_idx, mvc_idx = self._valve_events_from_flows(
            t=t, Qao=Qao, Qmv=Qmv, thr_ao=0.01, thr_mv=0.005
        )
        if any(x is None for x in [mvc_idx, avo_idx, avc_idx, mvo_idx]):
            raise RuntimeError(f"Valve events missing: MVC={mvc_idx}, AVO={avo_idx}, AVC={avc_idx}, MVO={mvo_idx}")

        t_mvc = float(t[mvc_idx])
        t_avo = float(t[avo_idx])
        t_avc = float(t[avc_idx])
        t_mvo = float(t[mvo_idx])

        # Compute intervals using your existing metric extractor (paper defs)
        m = self.pv_metrics_single_loop(cyc, HR_bpm=HR_bpm, ed_def=self.ED_DEF, esp_def=self.ESP_DEF)
        ET = m.get("ET", np.nan)
        IVCT = m.get("IVCT", np.nan)
        IVRT = m.get("IVRT", np.nan)
        tau = m.get("tau", np.nan)

        # ---- tau overlay prep: fit exponential on IVRT segment (AVC->MVO) ----
        # Get IVRT segment using cyclic extraction (may wrap across cycle end)
        t_iv, P_iv = self._get_segment_cyclic(t, P_lv, avc_idx, mvo_idx)

        tau_overlay = None  # (t_fit, P_hat, t0, P0, P_inf, tau_fit)
        if t_iv is not None and len(t_iv) >= 10:
            dPdt_iv = np.gradient(P_iv, t_iv)
            i0 = int(np.argmin(dPdt_iv))  # start at most negative dP/dt

            if (len(t_iv) - i0) >= 8:
                t_fit = t_iv[i0:]
                P_fit = P_iv[i0:]

                t0 = float(t_fit[0])
                P0 = float(P_fit[0])

                # initial guesses
                P_inf0 = float(np.min(P_fit))
                dPdt0 = float(np.gradient(P_fit, t_fit)[0])
                if dPdt0 < 0 and (P0 - P_inf0) > 0:
                    tau0 = float(np.clip(-(P0 - P_inf0) / dPdt0, 0.01, 0.15))
                else:
                    tau0 = 0.04

                # nonzero-asymptote model: P(t)=P_inf+(P0-P_inf)*exp(-(t-t0)/tau)
                def model(tt, P_inf, tau_local):
                    return P_inf + (P0 - P_inf) * np.exp(-(tt - t0) / tau_local)

                bounds = ([-np.inf, 1e-3], [np.inf, 0.20])  # 1ms..200ms
                try:
                    popt, _ = curve_fit(model, t_fit, P_fit, p0=[P_inf0, tau0],
                                        bounds=bounds, maxfev=20000)
                    P_inf = float(popt[0])
                    tau_fit = float(popt[1])

                    P_hat = model(t_fit, P_inf, tau_fit)
                    tau_overlay = (t_fit, P_hat, t0, P0, P_inf, tau_fit)
                except Exception:
                    tau_overlay = None

        # Helper: draw cyclic span on 0..T
        def span_cyclic(ax, t1, t2, color, alpha=0.15, label=None):
            if t2 >= t1:
                ax.axvspan(t1, t2, color=color, alpha=alpha, label=label)
            else:
                ax.axvspan(t1, Tcyc, color=color, alpha=alpha, label=label)
                ax.axvspan(0.0, t2, color=color, alpha=alpha)

        # Helper: vertical event markers
        def mark_event(ax, tx, txt, color='k', ls='-', lw=1.0):
            ax.axvline(tx, color=color, lw=lw, ls=ls)
            ax.text(tx, 0.97, txt, transform=ax.get_xaxis_transform(),
                    ha='left', va='top', fontsize=8, color=color)

        # Figure layout with good spacing
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

        fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0], hspace=0.28)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        # (1) Pressures
        ax1.plot(t, P_ao, color='tab:blue', lw=1.1, label="Aortic Pressure")
        ax1.plot(t, P_lv, color='tab:orange', lw=1.1, label="LV Pressure")

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 140)
        ax1.set_xticks(np.arange(0, 1, 0.1))
        ax1.set_yticks(np.arange(0, 141, 30))

        # interval shading
        span_cyclic(ax1, t_mvc, t_avo, color='tab:green', alpha=0.12, label="IVCT")
        span_cyclic(ax1, t_avo, t_avc, color='tab:red', alpha=0.10, label="ET")
        span_cyclic(ax1, t_avc, t_mvo, color='tab:purple', alpha=0.10, label="IVRT")

        # event markers (MVC not black anymore)
        mark_event(ax1, t_mvc, " ", 'tab:brown')
        mark_event(ax1, t_avo, " ", 'tab:pink')
        mark_event(ax1, t_avc, " ", 'tab:pink', ls='--')
        mark_event(ax1, t_mvo, " ", 'tab:purple')

        # ---- Overlay tau fit curve within IVRT on pressure plot ----
        if tau_overlay is not None:
            t_fit, P_hat, t0, P0, P_inf, tau_fit = tau_overlay

            t_plot = np.asarray(t_fit, float)
            P_plot = np.asarray(P_hat, float)

            m1 = t_plot <= Tcyc
            m2 = t_plot > Tcyc

            labeled = False

            if np.any(m1):
                ax1.plot(t_plot[m1], P_plot[m1],
                         color='k', lw=1.3, ls='--', alpha=0.9,
                         label=("LV relaxation time constant" if not labeled else "_nolegend_"))
                labeled = True

            if np.any(m2):
                ax1.plot(t_plot[m2] - Tcyc, P_plot[m2],
                         color='k', lw=1.3, ls='--', alpha=0.9,
                         label=("_nolegend_" if labeled else "τ fit (IVRT)"))
                labeled = True

            # black dot marking start at -dP/dtmin
            ax1.scatter([t0 if t0 <= Tcyc else t0 - Tcyc],
                        [P0],
                        s=22, color='k', zorder=6,
                        label="−dP/dt min")

            # asymptote line
            ax1.axhline(P_inf, color='k', lw=0.8, ls=':', alpha=0.6)

        ax1.set_ylabel("Pressure [mmHg]", fontsize=9)

        # ---- Legend ordering: Aortic, LV, tau-fit, dot, then IVCT/ET/IVRT ----
        handles, labels = ax1.get_legend_handles_labels()

        def pick(name):
            if name in labels:
                i = labels.index(name)
                return handles[i], labels[i]
            return None

        order_names = ["Aortic Pressure", "LV Pressure", "τ fit (IVRT)", "−dP/dtmin (fit start)", "IVCT", "ET", "IVRT"]
        ordered = []
        for nm in order_names:
            item = pick(nm)
            if item is not None:
                ordered.append(item)

        # Add any remaining labeled items (safety)
        for h, lab in zip(handles, labels):
            if (lab not in [x[1] for x in ordered]) and (not lab.startswith("_")):
                ordered.append((h, lab))

        ax1.legend([h for h, _ in ordered], [lab for _, lab in ordered],
                   frameon=False, fontsize=8, loc='upper right')

        ax1.grid(False)

        # (2) Flows
        ax2.plot(t, Qao, color='crimson', lw=1.0, label="Aortic Flow")
        ax2.plot(t, Qmv, color='cyan', lw=1.0, label="Mitral Flow")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 400)
        ax2.set_xticks(np.arange(0, 1, 0.1))
        ax2.set_yticks(np.arange(0, 401, 100))

        span_cyclic(ax2, t_mvc, t_avo, color='tab:green', alpha=0.12)
        span_cyclic(ax2, t_avo, t_avc, color='tab:red', alpha=0.10)
        span_cyclic(ax2, t_avc, t_mvo, color='tab:purple', alpha=0.10)

        mark_event(ax2, t_mvc, "", 'tab:brown')
        mark_event(ax2, t_avo, "", 'tab:pink')
        mark_event(ax2, t_avc, "", 'tab:pink', ls='--')
        mark_event(ax2, t_mvo, "", 'tab:purple')

        ax2.set_ylabel("Flow [mL/s]", fontsize=9)
        ax2.legend(frameon=False, fontsize=8, loc='upper right')
        ax2.grid(False)

        # (3) Volume
        ax3.plot(t, V_lv, color='navy', lw=1.0, label="LV Volume")
        span_cyclic(ax3, t_mvc, t_avo, color='tab:green', alpha=0.12)
        span_cyclic(ax3, t_avo, t_avc, color='tab:red', alpha=0.10)
        span_cyclic(ax3, t_avc, t_mvo, color='tab:purple', alpha=0.10)

        mark_event(ax3, t_mvc, "", 'tab:brown')
        mark_event(ax3, t_avo, "", 'tab:pink')
        mark_event(ax3, t_avc, "", 'tab:pink', ls='--')
        mark_event(ax3, t_mvo, "", 'tab:purple')

        ax3.set_ylabel("Volume [mL]", fontsize=9)
        ax3.set_xlabel("Time within cardiac cycle [s]", fontsize=9)
        ax3.legend(frameon=False, fontsize=8, loc='upper right')
        ax3.grid(False)

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 80)
        ax3.set_xticks(np.arange(0, 1, 0.1))
        ax3.set_yticks(np.arange(0, 81, 20))

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved Wiggers 3-panel figure: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return m, dict(mvc=mvc_idx, avo=avo_idx, avc=avc_idx, mvo=mvo_idx)

    def plot_pv_loop_with_valve_events(self, cyc, HR_bpm=None, save_path=None, show=False,
                                       dpi=600, figsize_cm=(6, 6),
                                       xlim=(20, 90), xtick_step=10,
                                       ylim=(0, 150), ytick_step=20):
        """
        PV loop figure with:
          - ED point (EDV, EDP) and ES point (ESV, ESP)
          - Valve events MVC, AVO, AVC, MVO marked as small dashes (not dots)
          - IVCT segment (MVC->AVO) highlighted transparently
          - IVRT segment (AVC->MVO) highlighted transparently
          - No title, journal-style axes
        """

        import numpy as np
        import matplotlib.pyplot as plt

        if HR_bpm is None:
            HR_bpm = 60.0 / float(self.params['T'])
        HR_bpm = float(HR_bpm)

        # --- signals ---
        t = np.asarray(cyc['t'], dtype=float)
        P = np.asarray(cyc['P_lv'], dtype=float)
        V = np.asarray(cyc['V_lv'], dtype=float)
        Qao = np.asarray(cyc.get('Q_ao', np.full_like(P, np.nan)), dtype=float)
        Qmv = np.asarray(cyc.get('Q_sv', np.full_like(P, np.nan)), dtype=float)

        Tcyc = self._cycle_period(t)

        # --- valve events (indices) ---
        avo_idx, avc_idx, mvo_idx, mvc_idx = self._valve_events_from_flows(
            t=t, Qao=Qao, Qmv=Qmv, thr_ao=0.01, thr_mv=0.005
        )
        if any(x is None for x in [mvc_idx, avo_idx, avc_idx, mvo_idx]):
            raise RuntimeError(f"Valve events missing: MVC={mvc_idx}, AVO={avo_idx}, AVC={avc_idx}, MVO={mvo_idx}")

        # --- ED/ES points (paper defs you chose) ---
        m = self.pv_metrics_single_loop(cyc, HR_bpm=HR_bpm, ed_def=self.ED_DEF, esp_def=self.ESP_DEF)
        EDV = float(m['EDV']);
        EDP = float(m['LVEDP'])
        ESV = float(m['ESV']);
        ESP = float(m['ESP'])

        # --- cyclic segments for overlays ---
        def seg_xy(idx_start, idx_end):
            tseg, Pseg = self._get_segment_cyclic(t, P, idx_start, idx_end)
            _, Vseg = self._get_segment_cyclic(t, V, idx_start, idx_end)
            return Vseg, Pseg

        V_ivct, P_ivct = seg_xy(mvc_idx, avo_idx)  # MVC->AVO
        V_ivrt, P_ivrt = seg_xy(avc_idx, mvo_idx)  # AVC->MVO

        # --- styling ---
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

        fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
        ax = fig.add_subplot(111)

        # Base PV loop
        ax.plot(V, P, color='0.15', lw=1.4, zorder=3)

        # Transparent IVCT / IVRT overlays (keep PV loop visible beneath)
        if V_ivct is not None and len(V_ivct) > 1:
            ax.plot(V_ivct, P_ivct, color='tab:green', lw=6.0, alpha=0.28,
                    solid_capstyle='round', label="IVCT", zorder=2)
        if V_ivrt is not None and len(V_ivrt) > 1:
            ax.plot(V_ivrt, P_ivrt, color='tab:purple', lw=6.0, alpha=0.28,
                    solid_capstyle='round', label="IVRT", zorder=2)

        # --- ED/ES points as circles (dominant points) ---
        ax.scatter([EDV], [EDP], s=55, color='tab:blue', edgecolor='k', linewidth=0.35, zorder=6)
        ax.scatter([ESV], [ESP], s=55, color='tab:red', edgecolor='k', linewidth=0.35, zorder=6)

        # Text labels for ED/ES (offset so they won't be blocked)
        #ax.annotate("(EDV,EDP)", xy=(EDV, EDP), xytext=(6, 6), textcoords="offset points",
        #            fontsize=9, color='tab:blue', ha='left', va='bottom')
        #ax.annotate("(ESV,ESP)", xy=(ESV, ESP), xytext=(6, 6), textcoords="offset points",
        #            fontsize=9, color='tab:red', ha='left', va='bottom')

        # --- Valve event markers as small dashes (instead of dots) ---
        # Use marker '_' (horizontal dash). If you prefer vertical dash use marker '|'.
        '''def mark_dash(i, label, color, dx=6, dy=0):
            ax.plot([V[i]], [P[i]], marker='_', markersize=14, markeredgewidth=2.2,
                    color=color, linestyle='None', zorder=7)
            ax.annotate(label, xy=(V[i], P[i]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=9, color=color, ha='left', va='center')

        # Offset labels slightly so they don't overlap ED/ES texts
        mark_dash(mvc_idx, "MVC", color='k', dx=6, dy=-10)
        mark_dash(avo_idx, "AVO", color='tab:red', dx=6, dy=-2)
        mark_dash(avc_idx, "AVC", color='tab:red', dx=6, dy=10)
        mark_dash(mvo_idx, "MVO", color='tab:purple', dx=6, dy=-2)'''

        # --- axes formatting (journal) ---
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, xtick_step))

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
            if ytick_step is not None:
                ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, ytick_step))

        ax.set_xlabel("LV Volume [mL]", fontsize=9)
        ax.set_ylabel("LV Pressure [mmHg]", fontsize=9)
        ax.grid(False)

        # No title (as requested)
        ax.set_title("")

        # Legend placed outside to avoid overlapping texts
        ax.legend(ncol=2, frameon=False, fontsize=8, loc='upper center')

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved PV loop with valve events: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return m, dict(mvc=mvc_idx, avo=avo_idx, avc=avc_idx, mvo=mvo_idx)

    def tau_fit_details(self, cyc, HR_bpm=None):
        """
        Returns details for tau fitting on the IVRT segment:
          - t_iv, P_iv: IVRT segment (AVC->MVO)
          - i0: index of most negative dP/dt within IVRT (fit start)
          - popt: (P_inf, tau) from curve fit
          - idxs: dict of event indices
        Tau model uses nonzero asymptote: P(t) = P_inf + (P0-P_inf)*exp(-(t-t0)/tau) [1](https://www.heartlungcirc.org/article/S1443-9506%2823%2903397-8/fulltext)[2](https://ieeexplore.ieee.org/document/1106362)
        """
        import numpy as np
        from scipy.optimize import curve_fit

        if HR_bpm is None:
            HR_bpm = 60.0 / float(self.params['T'])

        t = np.asarray(cyc['t'], dtype=float)
        P = np.asarray(cyc['P_lv'], dtype=float)
        Qao = cyc.get('Q_ao', None)
        Qmv = cyc.get('Q_sv', None)

        Tcyc = self._cycle_period(t)

        avo_idx = avc_idx = mvo_idx = mvc_idx = None
        if Qao is not None and Qmv is not None:
            avo_idx, avc_idx, mvo_idx, mvc_idx = self._valve_events_from_flows(
                t=t, Qao=np.asarray(Qao, float), Qmv=np.asarray(Qmv, float),
                thr_ao=0.01, thr_mv=0.005
            )

        if avc_idx is None or mvo_idx is None:
            return None, None, None, None, dict(avo=avo_idx, avc=avc_idx, mvo=mvo_idx, mvc=mvc_idx)

        # IVRT segment
        t_iv, P_iv = self._get_segment_cyclic(t, P, avc_idx, mvo_idx)
        if t_iv is None or len(t_iv) < 10:
            return t_iv, P_iv, None, None, dict(avo=avo_idx, avc=avc_idx, mvo=mvo_idx, mvc=mvc_idx)

        # Start fit at most negative dP/dt within IVRT
        dPdt_iv = np.gradient(P_iv, t_iv)
        i0 = int(np.argmin(dPdt_iv))
        if (len(t_iv) - i0) < 8:
            return t_iv, P_iv, i0, None, dict(avo=avo_idx, avc=avc_idx, mvo=mvo_idx, mvc=mvc_idx)

        t_fit = t_iv[i0:]
        P_fit = P_iv[i0:]
        t0 = float(t_fit[0])
        P0 = float(P_fit[0])
        P_inf0 = float(np.min(P_fit))  # initial asymptote guess

        # Model with nonzero asymptote [1](https://www.heartlungcirc.org/article/S1443-9506%2823%2903397-8/fulltext)[2](https://ieeexplore.ieee.org/document/1106362)
        def model(tt, P_inf, tau):
            return P_inf + (P0 - P_inf) * np.exp(-(tt - t0) / tau)

        # bounds: tau 1–200 ms
        bounds = ([-np.inf, 1e-3], [np.inf, 0.20])

        # initial tau guess based on Weiss-type estimate
        dPdt0 = float(np.gradient(P_fit, t_fit)[0])
        if dPdt0 < 0 and (P0 - P_inf0) > 0:
            tau0 = float(np.clip(-(P0 - P_inf0) / dPdt0, 0.01, 0.15))
        else:
            tau0 = 0.04

        try:
            popt, _ = curve_fit(model, t_fit, P_fit, p0=[P_inf0, tau0], bounds=bounds, maxfev=20000)
            P_inf, tau = float(popt[0]), float(popt[1])
            return t_iv, P_iv, i0, (P_inf, tau), dict(avo=avo_idx, avc=avc_idx, mvo=mvo_idx, mvc=mvc_idx)
        except Exception:
            return t_iv, P_iv, i0, None, dict(avo=avo_idx, avc=avc_idx, mvo=mvo_idx, mvc=mvc_idx)

    def plot_tau_calculation(self, cyc, HR_bpm=None, save_path=None, show=False,
                             dpi=600, figsize_cm=(16, 10)):
        """
        Plot how tau is computed:
          Panel 1: LV pressure over one cycle with AVC and MVO marked + IVRT shaded
          Panel 2: Pressure decay in IVRT, mark -dP/dtmin, show exponential fit + tau

        IVRT: AVC->MVO
        Tau model: mono-exponential with nonzero asymptote [1](https://www.heartlungcirc.org/article/S1443-9506%2823%2903397-8/fulltext)[2](https://ieeexplore.ieee.org/document/1106362)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if HR_bpm is None:
            HR_bpm = 60.0 / float(self.params['T'])
        HR_bpm = float(HR_bpm)

        t = np.asarray(cyc['t'], float)
        P = np.asarray(cyc['P_lv'], float)
        Qao = np.asarray(cyc.get('Q_ao', np.full_like(P, np.nan)), float)
        Qmv = np.asarray(cyc.get('Q_sv', np.full_like(P, np.nan)), float)
        Tcyc = self._cycle_period(t)

        t_iv, P_iv, i0, popt, idxs = self.tau_fit_details(cyc, HR_bpm=HR_bpm)

        # event times
        avc_idx = idxs.get('avc', None)
        mvo_idx = idxs.get('mvo', None)
        t_avc = float(t[avc_idx]) if avc_idx is not None else np.nan
        t_mvo = float(t[mvo_idx]) if mvo_idx is not None else np.nan

        # style
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

        fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.28)

        # -------- Panel 1: LV pressure vs time with IVRT shaded --------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, P, color='tab:orange', lw=1.1, label='LV pressure')

        if np.isfinite(t_avc) and np.isfinite(t_mvo):
            # Shade IVRT cyclically
            if t_mvo >= t_avc:
                ax1.axvspan(t_avc, t_mvo, color='tab:purple', alpha=0.12, label='IVRT (AVC→MVO)')
            else:
                ax1.axvspan(t_avc, Tcyc, color='tab:purple', alpha=0.12, label='IVRT (wrap)')
                ax1.axvspan(0.0, t_mvo, color='tab:purple', alpha=0.12)

            ax1.axvline(t_avc, color='tab:red', lw=1.0, ls='--')
            ax1.axvline(t_mvo, color='tab:blue', lw=1.0)
            ax1.text(t_avc, 0.95, "AVC", transform=ax1.get_xaxis_transform(), fontsize=8, color='tab:red', va='top')
            ax1.text(t_mvo, 0.95, "MVO", transform=ax1.get_xaxis_transform(), fontsize=8, color='tab:blue', va='top')

        ax1.set_ylabel("LV Pressure [mmHg]", fontsize=9)
        ax1.set_xlabel("Time within cardiac cycle [s]", fontsize=9)
        ax1.legend(frameon=False, fontsize=8, loc='upper right')
        ax1.grid(False)

        # -------- Panel 2: IVRT segment + tau fit --------
        ax2 = fig.add_subplot(gs[1, 0])
        if t_iv is None or P_iv is None or len(t_iv) < 10:
            ax2.text(0.5, 0.5, "IVRT segment not available for tau fit", ha='center', va='center')
        else:
            ax2.plot(t_iv, P_iv, color='0.2', lw=1.2, label='P(t) in IVRT')

            # Mark start of fit at -dP/dtmin
            if i0 is not None and 0 <= i0 < len(t_iv):
                ax2.scatter([t_iv[i0]], [P_iv[i0]], s=35, color='tab:purple',
                            edgecolor='k', linewidth=0.3, zorder=5, label='Start fit at min dP/dt')

            # Fit curve
            if popt is not None:
                P_inf, tau = popt
                t_fit = t_iv[i0:]
                t0 = float(t_fit[0])
                P0 = float(P_iv[i0])

                P_hat = P_inf + (P0 - P_inf) * np.exp(-(t_fit - t0) / tau)
                ax2.plot(t_fit, P_hat, color='tab:orange', lw=2.0,
                         label=f"Fit: $\\tau$={tau * 1000:.1f} ms, $P_\\infty$={P_inf:.1f} mmHg")

            ax2.set_xlabel("Time within IVRT [s]", fontsize=9)
            ax2.set_ylabel("LV Pressure [mmHg]", fontsize=9)
            ax2.legend(frameon=False, fontsize=8, loc='upper right')
            ax2.grid(False)

        # no title (journal style)
        for ax in [ax1, ax2]:
            ax.set_title("")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved tau calculation plot: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_prsw_like_paper(self, family, fits=None, save_path=None, show=False,
                             dpi=600, figsize_cm=(6, 6)):
        """
        Paper-style PRSW plot: Stroke Work vs EDV during preload reduction.
        PRSW is the slope of SW vs EDV during preload reduction. [3](https://www.ahajournals.org/doi/pdf/10.1161/circheartfailure.110.941773)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if fits is None:
            fits = self.fit_pv_relations(family)  # will default prsw_skip_first=1

        EDV = np.asarray(fits.get('EDV_prsw_used', fits.get('EDV', [])), dtype=float)
        SW = np.asarray(fits.get('SW_prsw_used', fits.get('SW', [])), dtype=float)
        Mw = float(fits.get('PRSW_slope', np.nan))
        Vw = float(fits.get('Vw', np.nan))

        # R² for PRSW
        def r2(y, yhat):
            y = np.asarray(y, float);
            yhat = np.asarray(yhat, float)
            ok = np.isfinite(y) & np.isfinite(yhat)
            y = y[ok];
            yhat = yhat[ok]
            if len(y) < 2:
                return np.nan
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        SW_hat = Mw * (EDV - Vw) if np.isfinite(Mw) and np.isfinite(Vw) else np.full_like(SW, np.nan)
        r2_prsw = r2(SW, SW_hat)

        # style
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

        fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)

        ax = fig.add_subplot(111)
        ax.set_xlim(40, 90)
        ax.set_ylim(1500, 4000)
        ax.set_xticks(np.arange(40, 91, 10))
        ax.set_yticks(np.arange(1500, 4001, 500))

        # points
        ax.scatter(EDV, SW, s=22, color='0.25', alpha=0.9, edgecolor='none')

        # fit line
        if np.isfinite(Mw) and np.isfinite(Vw) and len(EDV) > 1:
            xg = np.linspace(np.min(EDV), np.max(EDV), 200)
            yg = Mw * (xg - Vw)
            ax.plot(xg, yg, color='tab:orange', lw=1.0)

        ax.set_xlabel("EDV [mL]", fontsize=9)
        ax.set_ylabel("Stroke Work [mmHg·mL]", fontsize=9)
        ax.grid(False)
        ax.set_title("")  # no title (journal style)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved PRSW plot: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

# ============================================================
# Preload/Afterload: full PV-report style outputs for 4 conditions
# - baseline + {preload↑, preload↓, afterload↑, afterload↓}
# - per condition: single-loop haemodynamics + occlusion-derived relations
# - writes CSV (absolute values + %Δ vs baseline) and a PV overlay figure
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def _preafter_paramsets(sim):
    b = sim.params.copy()

    # Preload modulation
    preload_inc = {
        'V_tot': b['V_tot'] * 1.20,
        'C_sv' : b['C_sv']  * 1.20,
        'R_mv' : b['R_mv']  * 0.85
    }
    preload_dec = {
        'V_tot': b['V_tot'] * 0.75,
        'C_sv' : b['C_sv']  * 0.70,
        'R_mv' : b['R_mv']  * 1.20
    }

    # Afterload modulation
    after_inc = {
        'R_sys': b['R_sys'] * 1.35,
        'C_sa' : b['C_sa']  * 0.65,
        'Z_ao' : b['Z_ao']  * 1.20
    }
    after_dec = {
        'R_sys': b['R_sys'] * 0.70,
        'C_sa' : b['C_sa']  * 1.30,
        'Z_ao' : b['Z_ao']  * 0.80
    }

    return {
        'preload_increase': preload_inc,
        'preload_decrease': preload_dec,
        'afterload_increase': after_inc,
        'afterload_decrease': after_dec
    }

def _single_cycle_full_metrics(sim, cyc, params):
    """
    Use your pv_metrics_single_loop(...) to get the full single-beat metrics
    and then derive brachial values and extras (no class edits).
    """
    # HR from period T (keeps your timing)
    HR = 60.0 / float(params['T']) if 'T' in params and params['T'] > 0 else float(sim.bpm)

    # Your paper-aligned definitions are already stored on the class:
    m = sim.pv_metrics_single_loop(
        cyc, HR_bpm=HR, ed_def=sim.ED_DEF, esp_def=sim.ESP_DEF
    )  # returns EDV, ESV, SV, EF, LVEDP, ESP, Psys_LV, Pmin_LV, SW, dP/dt, aSBP/aDBP/aMAP, CO_L_min, Ea, ET/IVCT/IVRT, tau  [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/LPM_V4.1_Local_Sensitivity_Analysis_PV-Loop.py)

    # Brachial from your own convention: 1.1 × aortic values (as in your code)  [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/LPM_V4.1_Local_Sensitivity_Analysis_PV-Loop.py)
    aSBP = float(m.get('aSBP', np.nan))
    aDBP = float(m.get('aDBP', np.nan))
    bSBP = 1.1 * aSBP if np.isfinite(aSBP) else np.nan
    bDBP = 1.1 * aDBP if np.isfinite(aDBP) else np.nan
    bPP  = bSBP - bDBP if np.all(np.isfinite([bSBP, bDBP])) else np.nan
    bMAP = (bSBP + 2.0 * bDBP) / 3.0 if np.all(np.isfinite([bSBP, bDBP])) else np.nan

    # Add ESP_LV explicitly for convenience
    ESP_LV = float(m.get('ESP', np.nan))

    out = {
        # Pressures (brachial + aortic)
        'bSBP': bSBP, 'bDBP': bDBP, 'bPP': bPP, 'bMAP': bMAP,
        'aSBP': aSBP, 'aDBP': aDBP, 'aMAP': float(m.get('aMAP', np.nan)),

        # Volumes & derived
        'EDV': float(m.get('EDV', np.nan)),
        'ESV': float(m.get('ESV', np.nan)),
        'SV':  float(m.get('SV',  np.nan)),
        'EF':  float(m.get('EF',  np.nan)),

        # Pressures (LV)
        'ESP': ESP_LV,
        'EDP': float(m.get('LVEDP', np.nan)),
        'Max_LV_P': float(m.get('Psys_LV', np.nan)),
        'Min_LV_P': float(m.get('Pmin_LV', np.nan)),

        # Work/dynamics
        'Stroke_Work': float(m.get('SW', np.nan)),
        'dPdt_max':    float(m.get('dPdt_max', np.nan)),
        'dPdt_min':    float(m.get('dPdt_min', np.nan)),

        # Flow-like summaries
        'CO_L_min': float(m.get('CO_L_min', np.nan)),
        'Ea':       float(m.get('Ea', np.nan)),

        # Timings
        'Ejection_Time': float(m.get('ET',   np.nan)),
        'IVRT':          float(m.get('IVRT', np.nan)),
        'IVCT':          float(m.get('IVCT', np.nan)),
        'Tau':           float(m.get('tau',  np.nan)),
    }
    return out


def _occlusion_relations(sim, params, n_beats=8, frac_drop=0.60):
    """
    Use your preload occlusion + fit pipeline to compute:
      - ESPVR: Ees, V0
      - EDPVR: beta and dP/dV at ED
      - PRSW slope
    (Same code path you already use for the baseline PV report.)
    """
    family = sim.preload_occlusion_family(params=params, n_beats=n_beats, frac_drop=frac_drop)  # [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/LPM_V4.1_Local_Sensitivity_Analysis_PV-Loop.py)
    fits   = sim.fit_pv_relations(family)  # returns Ees, V0, beta, dPdV_ED, PRSW_slope  [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/LPM_V4.1_Local_Sensitivity_Analysis_PV-Loop.py)

    return {
        'Ees':        float(fits.get('Ees', np.nan)),
        'V0':         float(fits.get('V0',  np.nan)),
        'EDPVR_beta': float(fits.get('beta', np.nan)),
        'EDPVR_dPdV': float(fits.get('dPdV_ED', np.nan)),
        'PRSW':       float(fits.get('PRSW_slope', np.nan))
    }


def _simulate_full_condition(sim, deltas=None, n_beats=8, frac_drop=0.60):
    """
    Run one condition → full single-loop metrics + occlusion-derived relations,
    and compute the *operating* dP/dV at the condition's EDV (paper-style).
    Uses your existing APIs; no class changes.
    """
    import numpy as np

    # -- condition parameters
    params = sim.params.copy()
    if deltas:
        params.update(deltas)

    # -- one steady-state beat (cycle) under this condition
    (_, _, _, _, _, _, _,
     cyc, _mets, _idxs, _metrics) = sim.simulate_cycle_and_metrics(params)  # your class method  [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/_layouts/15/Doc.aspx?sourcedoc=%7B3E93ACEA-AC6F-4F02-9C2C-0B15F844FCA0%7D&file=preafterload_metrics_FULL_sub_999.csv&action=default&mobileredirect=true)

    # -- full single-beat metrics (includes EDV, EDP, ESP, SW, dP/dt, aSBP/aDBP/aMAP, ET/IVCT/IVRT, tau, CO, Ea)
    single = _single_cycle_full_metrics(sim, cyc, params)

    # -- preload-occlusion family & PV fits **for this condition**
    family = sim.preload_occlusion_family(params=params, n_beats=n_beats, frac_drop=frac_drop)  # [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/_layouts/15/Doc.aspx?sourcedoc=%7B3E93ACEA-AC6F-4F02-9C2C-0B15F844FCA0%7D&file=preafterload_metrics_FULL_sub_999.csv&action=default&mobileredirect=true)
    fits   = sim.fit_pv_relations(family)  # returns Ees, V0, beta, (C,D,E) tuple, PRSW_slope, dPdV_ED ﻿ [1](https://studntnu-my.sharepoint.com/personal/rahulma_ntnu_no/_layouts/15/Doc.aspx?sourcedoc=%7B3E93ACEA-AC6F-4F02-9C2C-0B15F844FCA0%7D&file=preafterload_metrics_FULL_sub_999.csv&action=default&mobileredirect=true)

    # -- relations packed
    rels = {
        'Ees':        float(fits.get('Ees', np.nan)),
        'V0':         float(fits.get('V0',  np.nan)),
        'EDPVR_beta': float(fits.get('beta', np.nan)),
        'EDPVR_dPdV': float(fits.get('dPdV_ED', np.nan)),  # fit's reference-point slope
        'PRSW':       float(fits.get('PRSW_slope', np.nan))
    }

    # -- MERGE single + relations
    full = {**single, **rels}

    # -- NEW: operating stiffness at THIS condition’s operating EDV
    # pull the EDPVR parameters (C, beta, D, E) and the operating EDV from 'single'
    Cfit, betafit, Dfit, Efit = fits.get("EDPVR_params", (np.nan, np.nan, np.nan, np.nan))
    EDV_op = single.get('EDV', np.nan)
    if np.all(np.isfinite([Cfit, betafit, Dfit, EDV_op])):
        full['Operating_dPdV_at_ED'] = float(Cfit * betafit * np.exp(betafit * (EDV_op - Dfit)))
    else:
        full['Operating_dPdV_at_ED'] = np.nan

    return cyc, full


def run_preafter_save_full(sim,
                           csv_name=None,
                           fig_name=None,
                           n_beats=8,
                           frac_drop=0.60,
                           xlim=(20, 120),
                           ylim=(0, 150)):
    """
    Baseline + four conditions:
      - CSV: absolute values for the full PV-report metrics + %Δ vs baseline
      - Figure: PV overlay (baseline + 4 curves)
    """
    if csv_name is None:
        csv_name = f"preafterload_metrics_FULL_sub_{sim.sub_id}.csv"
    if fig_name is None:
        fig_name = f"PV_preafter_overlay_sub_{sim.sub_id}.png"

    # --- Baseline
    base_cyc, base_full = _simulate_full_condition(sim, None, n_beats=n_beats, frac_drop=frac_drop)

    # --- Four conditions
    sets = _preafter_paramsets(sim)
    results = {}
    for label, dlt in sets.items():
        cyc, full = _simulate_full_condition(sim, dlt, n_beats=n_beats, frac_drop=frac_drop)
        results[label] = {'cyc': cyc, 'full': full}

    # --- CSV: absolute + %Δ vs baseline for all requested fields
    fields = [
        'bSBP', 'bDBP', 'bPP', 'bMAP', 'aSBP', 'aDBP', 'aMAP',
        'ESV', 'EDV', 'SV', 'EF',
        'ESP', 'EDP', 'Max_LV_P', 'Min_LV_P',
        'Stroke_Work', 'dPdt_max', 'dPdt_min',
        'CO_L_min', 'Ea',
        'Ees', 'V0', 'EDPVR_beta', 'EDPVR_dPdV', 'PRSW',
        'Ejection_Time', 'IVRT', 'IVCT', 'Tau',
        'Operating_dPdV_at_ED'  # <-- add this line
    ]

    rows = []
    for label, res in results.items():
        f = res['full']
        row = {'Condition': label}
        for k in fields:
            base_val = base_full.get(k, np.nan)
            cond_val = f.get(k, np.nan)
            row[k] = cond_val
            row[f"{k}_pct_change"] = (
                100.0 * (cond_val - base_val) / base_val
                if np.isfinite(base_val) and base_val != 0.0 else np.nan
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(sim.subject_dir, csv_name)
    df.to_csv(out_csv, index=False)
    print(f"[pre/after FULL] Saved CSV → {out_csv}")

    # --- PV overlay plot: baseline + 4 conditions
    plt.figure(figsize=(6, 3), dpi=300)
    plt.plot(base_cyc['V_lv'], base_cyc['P_lv'], color='k', lw=2.0, label="Baseline")
    color_map = {
        "preload_increase":  "tab:blue",
        "preload_decrease":  "tab:cyan",
        "afterload_increase":"tab:red",
        "afterload_decrease":"tab:orange"
    }
    for label, res in results.items():
        cyc = res['cyc']
        plt.plot(cyc['V_lv'], cyc['P_lv'], lw=1.4, color=color_map.get(label, None),
                 label=label.replace("_", " ").title())

    plt.xlabel("LV Volume [mL]")
    plt.ylabel("LV Pressure [mmHg]")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(frameon=False)
    plt.grid(False)
    out_fig = os.path.join(sim.subject_dir, fig_name)
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[pre/after FULL] Saved overlay figure → {out_fig}")

# -- MAIN SCRIPT --
if __name__ == '__main__':
    combined_GT_df = pd.read_csv(
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Invasive_Study_Leuven_GT_matrices_all_subjects_V4.0.csv")

    save_root = (
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Figures_V4\PV_LOOP_999"
    )
    excel_path = (r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
                  r"\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx")
    sheet_name = "Study_9_V4.1_T7"
    combined_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, nrows=69)

    #subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
    #                344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365,
    #                366, 367, 368, 369, 370, 371, 372, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 396,
    #                397, 398, 399, 400, 401, 402, 403, 404, 405, 409, 410, 411, 412, 413, 414, 415]  # full batch run
    subject_list = [999] #cohort average


    sim_results = []
    cohort_rows = []  # <-- NEW: list of per-subject report dicts
    sens_rows = []  # NEW: baseline + +/-10% for all params + HR (long-format table)
    audit_rows = []

    for sub_id in subject_list:
        row_df = combined_df.loc[combined_df['Sub ID'] == sub_id]
        if row_df.empty:
            print(f"[skip] Sub ID {sub_id} not found in Excel")
            continue
        data_row = row_df.iloc[0]

        row_gt_df = combined_GT_df.loc[combined_GT_df['sub_id'] == sub_id]
        if row_gt_df.empty:
            print(f"[skip] Sub ID {sub_id} not found in GT CSV")
            continue
        data_row_GT = row_gt_df.iloc[0]
        sim = SubjectSimulator(data_row, data_row_GT, sub_id, save_root)

        print(f"\nParameters for subject {sub_id}:")
        for k, v in sim.params.items():
            print(f"  {k:8} = {v}")

        # ---- Baseline simulation + plotting (optional) ----
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = sim.run_simulation(sim.params)
        cyc = sim.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, T_override=sim.params['T'])
        mets, idxs = sim.extract_cycle_metrics(cyc)
        sim.compare_with_gt(mets, cyc)
        HR_run = 60.0 / float(sim.params['T'])

        # Wiggers-like 3-panel (no PV loop)
        wiggers_path = os.path.join(sim.subject_dir, f"Wiggers_3panel_sub_{sub_id}.png")
        sim.plot_wiggers_threepanel(cyc, HR_bpm=HR_run, save_path=wiggers_path, show=False)

        # PV loop with events and isovolumic segments
        pv_events_path = os.path.join(sim.subject_dir, f"PV_loop_events_sub_{sub_id}.png")
        sim.plot_pv_loop_with_valve_events(cyc, HR_bpm=HR_run, save_path=pv_events_path, show=False)

        # Tau calculation plot (baseline cycle 'cyc' that you already have)
        tau_path = os.path.join(sim.subject_dir, f"Tau_calculation_sub_{sub_id}.png")
        sim.plot_tau_calculation(cyc, HR_bpm=HR_run, save_path=tau_path, show=False)

        # Optional baseline plots:
        sim.plot_all(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, sim.params)
        sim.save_current_figures()
        plt.close('all')

        # --- PV loop overlays for ±10% parameter changes (this is what you're missing) ---
        sim.run_sensitivity_analysis(
            preload_params=('V_tot', 'C_sv', 'R_mv'),
            afterload_params=('R_sys', 'C_sa', 'Z_ao'),
            elastance_params=('E_min', 'E_max', 't_peak'),
            steps=[-0.10, 0.10],
            fix_axes=True,
            include_hr=True,
            hr_steps=[-0.10, 0.10]
        )

        # ---- NEW: PV full sensitivity table for this subject ----
        rows_sub = sim.pv_full_sensitivity_table(
            base_params=sim.params.copy(),
            param_list=['R_sys', 'Z_ao', 'C_sa', 'R_mv', 'E_max', 'E_min', 't_peak', 'V_tot', 'C_sv'],
            steps=(-0.10, 0.10),
            include_hr=True,
            hr_steps=(-0.10, 0.10),
            n_beats=8,
            frac_drop=0.6,
            qc_plots=True  # <--- THIS triggers QC plotting
        )
        sens_rows.extend(rows_sub)

        # ---- Full PV analysis report (single loop + occlusion-derived ESPVR/EDPVR/PRSW) ----
        report, family = sim.pv_full_report(sim.params, n_beats=8, frac_drop=0.6)
        # --- compute fits BEFORE printing EDPVR parameters ---
        fits = sim.fit_pv_relations(family)

        prsw_path = os.path.join(sim.subject_dir, f"PRSW_SW_vs_EDV_sub_{sub_id}.png")        # PRSW plot using preload occlusion family and fits
        sim.plot_prsw_like_paper(family, fits=fits, save_path=prsw_path, show=False)

        # ---- Add to combined cohort table ----
        report_row = report.copy()
        report_row['sub_id'] = sub_id
        report_row['HR_bpm'] = float(sim.bpm)  # optional metadata
        report_row['occl_n_beats'] = 8  # optional metadata
        report_row['occl_frac_drop'] = 0.6  # optional metadata
        cohort_rows.append(report_row)

        report_df = pd.DataFrame([report])
        report_path = os.path.join(sim.subject_dir, f"pv_full_report_sub_{sub_id}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Saved PV full report: {report_path}")
        # ---- Plot single-loop summary (baseline beat = family[0]) ----
        # Baseline beat metrics
        single = family[0]['metrics']

        sum_path = os.path.join(sim.subject_dir, f"PV_single_summary_sub_{sub_id}.png")
        sim.plot_single_loop_summary(single, save_path=sum_path, show=False)

        # Save occlusion beat-by-beat metrics
        family_rows = []
        for i, d in enumerate(family):
            family_rows.append({'beat_index': i, **d['metrics']})
        family_df = pd.DataFrame(family_rows)
        family_path = os.path.join(sim.subject_dir, f"preload_occlusion_family_metrics_sub_{sub_id}.csv")
        family_df.to_csv(family_path, index=False)
        print(f"Saved preload occlusion family metrics: {family_path}")

        # (plotting code from section 3 can go here)
        dash_path = os.path.join(sim.subject_dir, f"PV_analysis_dashboard_sub_{sub_id}.png")
        sim.plot_pv_analysis_dashboard(family, fits, save_path=dash_path, show=False)

        # --- PV preload + relations figure per subject ---
        fig_path = os.path.join(sim.subject_dir, f"PV_preload_EDPVR_ESPVR_Ea_SW_sub_{sub_id}.png")
        sim.plot_pv_preload_with_relations(
            params=sim.params,
            n_beats=8,
            frac_drop=0.6,
            beat_for_Ea_SW=0,
            figsize_cm=(6, 6),
            dpi=600,
            save_path=fig_path,
            show=False
        )

        # --- Definition sensitivity audit per subject ---
        audit = sim.definition_sensitivity_audit(sim.params, n_beats=8, frac_drop=0.6)
        audit_rows.append(audit)

        # --- Full PV-report style outputs for preload/afterload (+ CSV + overlay)
        run_preafter_save_full(sim, n_beats=8, frac_drop=0.60)

    audit_df = pd.DataFrame(audit_rows)
    audit_out = os.path.join(save_root, "pv_definition_sensitivity_audit.csv")
    audit_df.to_csv(audit_out, index=False)
    print(f"Saved definition sensitivity audit: {audit_out}")

    # ---- Save combined cohort CSV ----
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_out = os.path.join(save_root, "pv_full_report_selected_subjects.csv")
    cohort_df.to_csv(cohort_out, index=False)
    print(f"Saved combined PV report table: {cohort_out}")

    if len(sens_rows) > 0:
        sens_df = pd.DataFrame(sens_rows)
        sens_out = os.path.join(save_root, "pv_full_sensitivity_selected_subjects.csv")
        sens_df.to_csv(sens_out, index=False)
        print(f"Saved PV sensitivity table: {sens_out}")
    else:
        print("Sensitivity table not generated (sens_rows empty).")

#end main