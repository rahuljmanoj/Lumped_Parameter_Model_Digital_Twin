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
        """Extract main cycle metrics from a cut cycle."""
        t = cyc['t']
        V = cyc['V_lv']
        P = cyc['P_lv']
        Q = cyc['Q_ao']
        EDV_idx = np.argmax(V)
        ESV_idx = np.argmin(V)
        EDV = V[EDV_idx]
        ESV = V[ESV_idx]
        LVEDP = P[EDV_idx]
        peak_idx = np.argmax(Q)
        Q_peak = Q[peak_idx]
        t_peak = t[peak_idx]
        thresh = 0.01 * Q_peak
        mask = Q > thresh
        edur_start_idx = np.where(mask)[0][0] if mask.any() else EDV_idx
        edur_end_idx   = np.where(mask)[0][-1] if mask.any() else EDV_idx
        EDur = t[edur_end_idx] - t[edur_start_idx]
        return (
            {'EDV': EDV, 'ESV': ESV, 'LVEDP': LVEDP,
             'Q_peak': Q_peak, 't_peak': t_peak, 'EDur': EDur},
            {'EDV_idx': EDV_idx, 'ESV_idx': ESV_idx,
             'peak_idx': peak_idx,
             'edur_start': edur_start_idx, 'edur_end': edur_end_idx}
        )

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

    def pv_metrics_single_loop(self, cyc, HR_bpm=None):
        """
        Compute single-loop PV metrics from one cardiac cycle dict.
        cyc keys: 't','P_lv','V_lv','Q_ao' (and optionally 'P_ao')
        """
        t = cyc['t']
        P = cyc['P_lv']
        V = cyc['V_lv']
        Qao = cyc.get('Q_ao', None)
        Pao = cyc.get('P_ao', None)

        EDV_idx = int(np.argmax(V))
        ESV_idx = int(np.argmin(V))

        EDV = float(V[EDV_idx])
        ESV = float(V[ESV_idx])
        SV = EDV - ESV
        EF = (SV / EDV * 100.0) if EDV > 0 else np.nan

        EDP = float(P[EDV_idx])  # LVEDP
        ESP = float(P[ESV_idx])  # end-systolic pressure at ESV (simple definition)
        Psys = float(np.max(P))
        Pmin = float(np.min(P))

        # Stroke work: area inside PV loop
        SW = self.pv_loop_area_shoelace(V, P)

        # Pressure derivatives
        dPdt = np.gradient(P, t)
        dPdt_max = float(np.max(dPdt))
        dPdt_min = float(np.min(dPdt))

        out = dict(
            EDV=EDV, ESV=ESV, SV=SV, EF=EF,
            LVEDP=EDP, ESP=ESP, Psys_LV=Psys, Pmin_LV=Pmin,
            SW=SW, dPdt_max=dPdt_max, dPdt_min=dPdt_min
        )

        # Aortic pressures if available
        if Pao is not None:
            aSBP = float(np.max(Pao))
            aDBP = float(np.min(Pao))
            aMAP = float((aSBP + 2 * aDBP) / 3.0)
            out.update(aSBP=aSBP, aDBP=aDBP, aMAP=aMAP)

        # CO and Ea if HR provided
        if HR_bpm is not None:
            HR = float(HR_bpm)
            CO_L_min = (SV * HR) / 1000.0  # SV in mL -> L/min
            out['CO_L_min'] = float(CO_L_min)
            out['Ea'] = float(ESP / SV) if SV > 0 else np.nan  # effective arterial elastance

        # Optional timing based on Qao threshold (if Qao available)
        if Qao is not None and np.max(Qao) > 0:
            thresh = 0.01 * np.max(Qao)
            mask = Qao > thresh
            if mask.any():
                t_ej = t[mask]
                out['ET'] = float(t_ej[-1] - t_ej[0])  # ejection time

        return out

    def preload_occlusion_family(self, params=None, n_beats=8, frac_drop=0.35):
        """
        Generate a family of PV loops by reducing venous volume V_sv between beats
        (vena cava occlusion analogue). Returns list of dicts with per-beat metrics.
        """
        if params is None:
            params = self.params
        p = params.copy()

        # 1) steady-state baseline run + final state
        t, P_ao, P_lv, V_lv, Q_sv, Q_ao, Q_sys, y_end = self.run_simulation(p, return_state=True)

        # Extract baseline last cycle
        cyc0 = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv, Q_ao, T_override=p['T'])
        base_metrics = self.pv_metrics_single_loop(cyc0, HR_bpm=self.bpm)

        family = []
        family.append(dict(cycle=cyc0, metrics=base_metrics, y_end=y_end.copy()))

        # 2) prepare beat-by-beat integration
        T = float(p['T'])
        dt = T / 500.0
        t_eval = np.linspace(0.0, T, int(np.round(T / dt)) + 1)

        # We will remove volume from venous compartment state y[2]
        y_state = y_end.copy()
        total_remove = frac_drop * y_state[2]
        dV = total_remove / n_beats

        for k in range(n_beats):
            # Remove volume from venous compartment (preload reduction)
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
            mets = self.pv_metrics_single_loop(cyc, HR_bpm=self.bpm)

            family.append(dict(cycle=cyc, metrics=mets, y_end=sol.y[:, -1].copy()))

            # update for next beat
            y_state = sol.y[:, -1].copy()

        return family

    def fit_pv_relations(self, family):
        """
        Fit ESPVR, EDPVR (beta), PRSW from a preload-occlusion PV family.
        family: output of preload_occlusion_family()
        """
        import numpy as np
        from scipy.optimize import curve_fit

        EDV = np.array([d['metrics']['EDV'] for d in family], dtype=float)
        ESV = np.array([d['metrics']['ESV'] for d in family], dtype=float)
        EDP = np.array([d['metrics']['LVEDP'] for d in family], dtype=float)
        ESP = np.array([d['metrics']['ESP'] for d in family], dtype=float)
        SW = np.array([d['metrics']['SW'] for d in family], dtype=float)

        # --- ESPVR: linear fit ESP = Ees*ESV + b, then V0 = -b/Ees
        A = np.vstack([ESV, np.ones_like(ESV)]).T
        Ees, b = np.linalg.lstsq(A, ESP, rcond=None)[0]
        V0 = -b / Ees if Ees != 0 else np.nan

        # --- PRSW: SW = Mw*EDV + c, then Vw = -c/Mw
        A2 = np.vstack([EDV, np.ones_like(EDV)]).T
        Mw, c = np.linalg.lstsq(A2, SW, rcond=None)[0]
        Vw = -c / Mw if Mw != 0 else np.nan

        # --- EDPVR: P = C*exp(beta*(V-D)) + E
        def edpvr(V, C, beta, D, E):
            return C * np.exp(beta * (V - D)) + E

        # Initial guesses: keep stable
        p0 = [1.0, 0.05, np.min(EDV), 0.0]
        bounds = ([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, 1.0, np.inf, np.inf])

        popt, _ = curve_fit(edpvr, EDV, EDP, p0=p0, bounds=bounds, maxfev=50000)
        Cfit, betafit, Dfit, Efit = popt

        # Optional energetics: PE, PVA, efficiency (needs Ees & V0, classic approach)
        # A common approximation for potential energy is triangular area:
        # PE ≈ 0.5 * ESP * (ESV - V0)
        # Use baseline (first) beat values for reporting (or average)
        ESP0 = float(ESP[0])
        ESV0 = float(ESV[0])
        SW0 = float(SW[0])
        PE0 = float(0.5 * ESP0 * (ESV0 - V0)) if np.isfinite(V0) else np.nan
        PVA0 = float(SW0 + PE0) if np.isfinite(PE0) else np.nan
        Eff0 = float(SW0 / PVA0) if (np.isfinite(PVA0) and PVA0 > 0) else np.nan

        return dict(
            Ees=float(Ees), V0=float(V0),

            beta=float(betafit),
            rel_EDPVR_C=float(Cfit),
            rel_EDPVR_D=float(Dfit),
            rel_EDPVR_E=float(Efit),

            EDPVR_params=(float(Cfit), float(betafit), float(Dfit), float(Efit)),  # keep if you want
            PRSW_slope=float(Mw), Vw=float(Vw),

            PE=float(PE0), PVA=float(PVA0), efficiency=float(Eff0),

            EDV=EDV, ESV=ESV, EDP=EDP, ESP=ESP, SW=SW
        )

    def pv_fit_quality(self, family, fits):
        """
        Compute R² goodness-of-fit for:
          - ESPVR (ESP vs ESV)
          - PRSW (SW vs EDV)
          - EDPVR (EDP vs EDV) using fitted exponential parameters

        Returns: dict with keys R2_ESPVR, R2_PRSW, R2_EDPVR
        """
        import numpy as np

        # Extract data series from fits (already arrays)
        EDV = np.asarray(fits['EDV'], dtype=float)
        ESV = np.asarray(fits['ESV'], dtype=float)
        EDP = np.asarray(fits['EDP'], dtype=float)
        ESP = np.asarray(fits['ESP'], dtype=float)
        SW = np.asarray(fits['SW'], dtype=float)

        # Parameters
        Ees = float(fits['Ees'])
        V0 = float(fits['V0'])
        Mw = float(fits['PRSW_slope'])
        Vw = float(fits['Vw'])

        # EDPVR parameters (we stored flat keys too)
        Cfit = float(fits['rel_EDPVR_C'])
        Dfit = float(fits['rel_EDPVR_D'])
        Efit = float(fits['rel_EDPVR_E'])
        betafit = float(fits['beta'])

        # Helper: R²
        def r2(y, yhat):
            y = np.asarray(y, dtype=float)
            yhat = np.asarray(yhat, dtype=float)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Predictions
        esp_hat = Ees * (ESV - V0)  # ESPVR
        sw_hat = Mw * (EDV - Vw)  # PRSW
        edp_hat = Cfit * np.exp(betafit * (EDV - Dfit)) + Efit  # EDPVR

        return {
            'R2_ESPVR': r2(ESP, esp_hat),
            'R2_PRSW': r2(SW, sw_hat),
            'R2_EDPVR': r2(EDP, edp_hat)
        }

    def pv_full_report(self, params=None, n_beats=8, frac_drop=0.35):
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
        single = self.pv_metrics_single_loop(cyc0, HR_bpm=self.bpm)

        # preload occlusion family + fits
        family = self.preload_occlusion_family(params, n_beats=n_beats, frac_drop=frac_drop)
        fits = self.fit_pv_relations(family)
        fitq = self.pv_fit_quality(family, fits)

        # merge
        out = {}
        # single-beat
        out.update({f"single_{k}": v for k, v in single.items()})
        # relations
        out.update({f"rel_{k}": v for k, v in fits.items() if k not in ['EDV', 'ESV', 'EDP', 'ESP', 'SW']})
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
        Creates a multi-panel figure:
          (1) PV family overlay + ED/ES points
          (2) ESPVR: ESP vs ESV + fitted line
          (3) EDPVR: EDP vs EDV + fitted exponential (beta)
          (4) PRSW: SW vs EDV + fitted line
          (5) Beat-to-beat trends: EDV, EDP, ESV, ESP, SW

        family: list from preload_occlusion_family()
        fits: dict from fit_pv_relations(family)
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # --- extract arrays from fits ---
        EDV = fits['EDV'];
        EDP = fits['EDP']
        ESV = fits['ESV'];
        ESP = fits['ESP']
        SW = fits['SW']

        Ees = fits['Ees'];
        V0 = fits['V0']
        beta = fits['beta']
        Cfit, betafit, Dfit, Efit = fits['EDPVR_params']
        Mw = fits['PRSW_slope'];
        Vw = fits['Vw']

        beat_idx = np.arange(len(EDV))

        # --- define fit functions for plotting ---
        def edpvr_fun(V):
            return Cfit * np.exp(betafit * (V - Dfit)) + Efit

        def espvr_fun(V):
            return Ees * (V - V0)

        def prsw_fun(V):
            return Mw * (V - Vw)

        # --- figure layout ---
        fig = plt.figure(figsize=(12, 10), dpi=200)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.35, wspace=0.25)

        # =========================
        # (1) PV family overlay
        # =========================
        ax0 = fig.add_subplot(gs[0, 0])
        for i, d in enumerate(family):
            cyc = d['cycle']
            ax0.plot(cyc['V_lv'], cyc['P_lv'], lw=0.9, alpha=0.7)
        # mark ED/ES points
        ax0.scatter(EDV, EDP, s=18, c='tab:blue', label='ED points')
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
        Vgrid = np.linspace(min(ESV), max(ESV), 200)
        ax1.plot(Vgrid, espvr_fun(Vgrid), lw=2.0, label=f"Fit: Ees={Ees:.3f}, V0={V0:.2f}")
        ax1.set_xlabel("ESV [mL]")
        ax1.set_ylabel("ESP [mmHg]")
        ax1.set_title("ESPVR (contractility)")
        ax1.legend(frameon=False)

        # =========================
        # (3) EDPVR
        # =========================
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(EDV, EDP, s=22, label="ED points")
        Vgrid = np.linspace(min(EDV), max(EDV), 200)
        ax2.plot(Vgrid, edpvr_fun(Vgrid), lw=2.0, label=f"Fit: beta={beta:.4f}")
        ax2.set_xlabel("EDV [mL]")
        ax2.set_ylabel("EDP (LVEDP) [mmHg]")
        ax2.set_title("EDPVR (diastolic stiffness)")
        ax2.legend(frameon=False)

        # =========================
        # (4) PRSW
        # =========================
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(EDV, SW, s=22, label="Beats")
        Vgrid = np.linspace(min(EDV), max(EDV), 200)
        ax3.plot(Vgrid, prsw_fun(Vgrid), lw=2.0, label=f"Fit: PRSW={Mw:.3f}, Vw={Vw:.2f}")
        ax3.set_xlabel("EDV [mL]")
        ax3.set_ylabel("Stroke Work [mmHg·mL]")
        ax3.set_title("PRSW (stroke work vs preload)")
        ax3.legend(frameon=False)

        # =========================
        # (5) Beat-to-beat trends
        # =========================
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(beat_idx, EDV, marker='o', lw=1.5, label="EDV")
        ax4.plot(beat_idx, ESV, marker='o', lw=1.5, label="ESV")
        ax4.set_xlabel("Beat index")
        ax4.set_ylabel("Volume [mL]")
        ax4.set_title("Volumes during preload reduction")
        ax4.legend(frameon=False)

        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(beat_idx, EDP, marker='o', lw=1.5, label="LVEDP")
        ax5.plot(beat_idx, ESP, marker='o', lw=1.5, label="ESP")
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
        fig1 = plt.figure(figsize=(15 / 2.54, 5 / 2.54), dpi=300)
        ax1 = fig1.add_subplot(111)
        ax1.plot(t, E_full,linewidth=0.75)
        self._style_axes(ax1, "Time [s]", "Elastance [mmHg/ml]", add_legend=False)

        # 2) Pressures (Fig 2) – 15 x 5 cm
        fig2 = plt.figure(figsize=(15 / 2.54, 5 / 2.54), dpi=300)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 180)
        ax2.set_xticks(np.arange(0, 8.1, 0.5))
        ax2.set_yticks(np.arange(0, 181, 25))
        ax2.plot(t, P_ao, label="Aortic Pressure",linewidth=0.75)
        ax2.plot(t, P_lv, label="LV Pressure",linewidth=0.75)
        self._style_axes(ax2, "Time [s]", "Pressure [mmHg]", add_legend=True)

        # 3) Flows (Fig 3) – 15 x 5 cm
        fig3 = plt.figure(figsize=(15 / 2.54, 5 / 2.54), dpi=300)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlim(0, 8)
        ax3.set_ylim(0, 700)
        ax3.set_xticks(np.arange(0, 8.1, 0.5))
        ax3.set_yticks(np.arange(0, 701, 100))
        ax3.plot(t, Q_lv_ao, label="Aortic Flow",linewidth=0.75)
        ax3.plot(t, Q_sv_lv, label="Mitral Flow",linewidth=0.75)
        ax3.plot(t, Q_sys, label="Systemic Flow",linewidth=0.75)
        self._style_axes(ax3, "Time [s]", "Flow rate [ml/s]", add_legend=True)

        # 4) LV volume vs time (Fig 4) – 15 x 5 cm
        fig4 = plt.figure(figsize=(15 / 2.54, 5 / 2.54), dpi=300)
        ax4 = fig4.add_subplot(111)
        ax4.plot(t, V_lv, label="LV Volume",linewidth=0.75)
        self._style_axes(ax4, "Time [s]", "Volume [ml]", add_legend=True)

        # 5) Global PV loop (full simulation) (Fig 5) – 5 x 5 cm
        fig5 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax5 = fig5.add_subplot(111)
        ax5.plot(V_lv, P_lv,linewidth=0.75)
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
        ax6.set_yticks(np.arange(0, 121, 20))
        ax6.plot(t_c, P_ao_c,linewidth=0.75)
        ax6.plot(t_c, P_lv_c,linewidth=0.75)
        #ax6.scatter([t_c[idxs['EDV_idx']]], [P_lv_c[idxs['EDV_idx']]], c="r", marker="o",
                    #label="LVEDP@EDV")
        self._style_axes(ax6, "Time [s]", "Pressure [mmHg]", add_legend=True)

        # 7) Cycle flows (Fig 7) – 5 x 5 cm
        fig7 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax7 = fig7.add_subplot(111)
        ax7.plot(t_c, Q_ao_c, label="Aortic Flow",linewidth=0.75)
        ax7.plot(t_c, Q_sv_c, label="Systemic Venous Flow",linewidth=0.75)
        ax7.scatter([t_c[idxs['peak_idx']]], [Q_ao_c[idxs['peak_idx']]], c="g", marker="o",
                    label="Q_peak")
        ax7.axvline(t_c[idxs['edur_start']], linestyle="--", color="black", linewidth=0.75)
        ax7.axvline(t_c[idxs['edur_end']], linestyle="--", color="black", linewidth=0.75)
        self._style_axes(ax7, "Time [s]", "Flow rate [ml/s]", add_legend=True)

        # 8) LV volume over cycle (Fig 8) – 5 x 5 cm
        fig8 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax8 = fig8.add_subplot(111)
        ax8.plot(t_c, V_lv_c)
        ax8.scatter([t_c[idxs['ESV_idx']]], [V_lv_c[idxs['ESV_idx']]], c="b", marker="o",
                    label="ESV")
        ax8.scatter([t_c[idxs['EDV_idx']]], [mets['EDV']], c="r", marker="o",
                    label="EDV")
        self._style_axes(ax8, "Time [s]", "Volume [ml]", add_legend=True)

        # 9) PV loop for last cycle (Fig 9) – 5 x 5 cm
        fig9 = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
        ax9 = fig9.add_subplot(111)
        ax9.set_xlim(20, 70)
        ax9.set_ylim(0, 120)
        ax9.set_xticks(np.arange(20, 71, 10))
        ax9.set_yticks(np.arange(0, 121, 20))
        ax9.plot(V_lv_c, P_lv_c,linewidth=0.75)
        ax9.scatter([mets['EDV']], [mets['LVEDP']], c="g", marker="o", label="LVEDP")
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
                ax.scatter([mets_base['EDV']], [mets_base['LVEDP']], s=15, color='k')

                # OAT perturbations: vary only pname
                for pc in steps:
                    params_var = baseline_params.copy()
                    params_var[pname] = baseline_params[pname] * (1.0 + pc)
                    label = f"{pname} {int(pc * 100)}%"

                    (_, _, _, _, _, _, _,
                     cyc, mets, _, metrics) = self.simulate_cycle_and_metrics(params_var)

                    line, = ax.plot(cyc['V_lv'], cyc['P_lv'], label=label, linewidth=0.75)
                    ax.scatter([mets['EDV']], [mets['LVEDP']], s=15, color=line.get_color())

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
                    ax.set_xlim(20, 80)
                    ax.set_ylim(0, 160)
                ax.set_xlabel("LV Volume [ml]", fontname="Times New Roman", fontsize=9, color="black")
                ax.set_ylabel("LV Pressure [mmHg]", fontname="Times New Roman", fontsize=9, color="black")
                ax.legend(fontsize=8, frameon=False)
                ax.grid(False)

                fig_path = os.path.join(self.subject_dir, f"PV_OAT_{block_name}_{pname}_sub_{self.sub_id}.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
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
            ax.scatter([mets_base['EDV']], [mets_base['LVEDP']], s=15, color='k')

            for pc in hr_steps:
                HR_var = HR0 * (1.0 + pc)
                label = f"HR {int(pc * 100)}%"

                (_, _, _, _, _, _, _,
                 cyc, mets, _, metrics) = self.simulate_cycle_and_metrics_HR(baseline_params, HR_var)

                line, = ax.plot(cyc['V_lv'], cyc['P_lv'], label=label, linewidth=0.75)
                ax.scatter([mets['EDV']], [mets['LVEDP']], s=15, color=line.get_color())

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
                ax.set_xlim(20, 80)
                ax.set_ylim(0, 160)
            ax.set_xlabel("LV Volume [ml]", fontname="Times New Roman", fontsize=9, color="black")
            ax.set_ylabel("LV Pressure [mmHg]", fontname="Times New Roman", fontsize=9, color="black")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(False)

            fig_path = os.path.join(self.subject_dir, f"PV_HR_sub_{self.sub_id}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {fig_path}")

        # =========================
        # 3) SAVE CSV
        # =========================
        df = pd.DataFrame(rows)
        metrics_path = os.path.join(self.subject_dir, f"sensitivity_metrics_sub_{self.sub_id}.csv")
        df.to_csv(metrics_path, index=False)
        print(f"Saved sensitivity metrics table: {metrics_path}")

# -- MAIN SCRIPT --
if __name__ == '__main__':
    combined_GT_df = pd.read_csv(
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Invasive_Study_Leuven_GT_matrices_all_subjects_V4.0.csv")

    save_root = (
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\PV_LOOP_Analysis_Figure_V4.1"
    )
    excel_path = (r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
                  r"\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx")
    sheet_name = "Study_3_V4.1_T6"
    combined_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, nrows=69)

    #subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
    #                344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365,
    #                366, 367, 368, 369, 370, 371, 372, 382, 383, 385, 386, 387, 388, 389, 390, 391, 392, 396,
    #                397, 398, 399, 400, 401, 402, 403, 404, 405, 409, 410, 411, 412, 413, 414, 415]  # full batch run
    subject_list = [999] #test


    sim_results = []
    cohort_rows = []  # <-- NEW: list of per-subject report dicts

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

        # Optional baseline plots:
        sim.plot_all(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, sim.params)
        sim.save_current_figures()
        plt.close('all')

        # ---- Full PV analysis report (single loop + occlusion-derived ESPVR/EDPVR/PRSW) ----
        report, family = sim.pv_full_report(sim.params, n_beats=8, frac_drop=0.5)
        # ---- Add to combined cohort table ----
        report_row = report.copy()
        report_row['sub_id'] = sub_id
        report_row['HR_bpm'] = float(sim.bpm)  # optional metadata
        report_row['occl_n_beats'] = 8  # optional metadata
        report_row['occl_frac_drop'] = 0.5  # optional metadata
        cohort_rows.append(report_row)

        report_df = pd.DataFrame([report])
        report_path = os.path.join(sim.subject_dir, f"pv_full_report_sub_{sub_id}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Saved PV full report: {report_path}")
        # ---- Plot single-loop summary (baseline beat = family[0]) ----
        single = family[0]['metrics']  # baseline beat metrics from pv_metrics_single_loop
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

        # Optional: plots for family and fits
        fits = sim.fit_pv_relations(family)
        # (plotting code from section 3 can go here)
        dash_path = os.path.join(sim.subject_dir, f"PV_analysis_dashboard_sub_{sub_id}.png")
        sim.plot_pv_analysis_dashboard(family, fits, save_path=dash_path, show=False)

    # ---- Save combined cohort CSV ----
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_out = os.path.join(save_root, "pv_full_report_selected_subjects.csv")
    cohort_df.to_csv(cohort_out, index=False)
    print(f"Saved combined PV report table: {cohort_out}")

#end main