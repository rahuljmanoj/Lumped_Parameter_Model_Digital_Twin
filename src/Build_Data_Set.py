import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import os
import time
from multiprocessing import cpu_count
from scipy.stats import qmc   # for Latin Hypercube
from joblib import Parallel, delayed

# --- Global parameter bounds for synthetic dataset (figure values) ---
GLOBAL_BOUNDS = {
    "R_sys":  (0.85, 4.29),     # mmHg·s/ml
    "Z_ao":   (0.03, 0.10),     # mmHg·s/ml
    "C_sa":   (0.24, 1.55),    # ml/mmHg
    "R_mv":   (0.01, 0.10),     # mmHg·s/ml
    "E_max":  (0.90, 9.99),    # mmHg/ml
    "E_min":  (0.01, 2.00),     # mmHg/ml
    "t_peak": (0.16, 0.37),     # s
    "V_tot":  (174.45, 887.37),   # ml
    "C_sv":   (2.42, 29.74),     # ml/mmHg
    "T":      (0.78, 1.75),     # s (beat period)
}
PHYS_LIMITS = {
    "LVEDP": (4.0, 35.0),   # mmHg
    "aSBP":  (80.0, 200.0), # mmHg
    "SV":    (20.0, 120.0), # ml
    "EF":    (20.0, 80.0),  # %
}

class SubjectSimulator:
    def __init__(self, data_row, sub_id, save_root, soft_weight=0.75, soft_weight_Emin=1, phys_weight=10):
        """Initialize all subject-specific and fixed simulation parameters."""
        self.sub_id = sub_id
        self.data_row = data_row
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
        self.GT_bMAP = data_row['bMAP']
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
            'R_sys': data_row['R_sys (mmHg s/ml)'],
            'Z_ao': data_row['Z_ao (mmHg s/ml)'],
            'C_sa': data_row['C_sa (ml/mmHg)'],
            'R_mv': data_row['R_mv (mmHg s/ml)'],
            'E_max': data_row['E_max (mmHg/ml)'],
            'E_min': data_row['E_min (mmHg/ml)'],
            't_peak': data_row['t_peak (s)'],
            'V_tot': data_row['V_tot (ml)'],
            'C_sv': data_row['C_sv (ml/mmHg)'],
            'T': self.T
        }
        # Bounds for optimization (subject specific)
        self.bounds = [
            (self.params['R_sys'] * (1 - 0.20), self.params['R_sys'] * (1 + 0.20)),
            (self.params['Z_ao'] * (1 - 0.20), self.params['Z_ao'] * (1 + 0.20)),
            (self.params['C_sa'] * (1 - 0.20), self.params['C_sa'] * (1 + 0.20)),
            (0.01, 0.1),     # R_mv
            (0.9, 10.0),     # E_max
            (self.params['E_min'] * (1 - 0.20), self.params['E_min'] * (1 + 0.20)),
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
        V_sv0 = max(V_sv0, 1.0)
        V_sa0 = max(V_sa0, 1.0)
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
        """Simulate model for a set of parameters (uses per-sample T)."""
        if params is None:
            params = self.params

        # use the sample's T for time grid
        T_local = float(params.get('T', self.T))
        total = self.cycles * T_local
        dt = T_local / 500.0

        y0 = self.initial_state(params)  # params-consistent ICs
        num_steps = int(np.round(total / dt))
        t_eval = np.linspace(0, total, num_steps + 1)

        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, params),
                        [0, total], y0,
                        t_eval=t_eval, max_step=dt)

        t = sol.t
        V_lv, V_sa, V_sv = sol.y

        P_lv = self.elastance(t, params) * V_lv + self.P_th
        P_sa = V_sa / params['C_sa']
        P_ao = np.where(P_lv > P_sa, P_lv, P_sa)
        P_sv = V_sv / params['C_sv']
        Q_sv_lv = np.where(P_sv > P_lv, (P_sv - P_lv) / params['R_mv'], 0.0)
        Q_lv_ao = np.where(P_lv > P_sa, (P_lv - P_sa) / params['Z_ao'], 0.0)
        Q_sys = np.where(P_sa > P_sv, (P_sa - P_sv) / params['R_sys'], 0.0)
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, T_local

    def cycle_cutting_algo(self, t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, T_local):
        """Cut the last full cardiac cycle using the current sample's period T_local."""
        dt = t[1] - t[0]
        spc = int(round(float(T_local / dt)))
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

    def compute_outputs_from_params(self, params):
        """
        Runs the solver for a given parameter dict and returns scalar outputs you need
        for the synthetic dataset + a small cycle waveform (for optional saving).
        """
        # unpack ALL returns incl. T_local
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, T_local = self.run_simulation(params)
        # pass T_local to cutter
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, T_local)
        mets, _ = self.extract_cycle_metrics(cyc)

        # scalars
        EDV = float(mets['EDV'])
        ESV = float(mets['ESV'])
        SV = float(EDV - ESV)
        EF = float((SV / EDV) * 100.0) if EDV > 0 else float('nan')
        LVEDP = float(mets['LVEDP'])
        aSBP = float(np.max(cyc['P_ao']))  # aortic SBP (you scale to brachial elsewhere)
        CO = float(SV / T_local * 60.0 / 1000.0)  # L/min, using this sample's period
        bSBP = float(np.max(cyc['P_ao']) * 1.1)
        bDBP = float(np.min(cyc['P_ao']))
        bMAP = float((bSBP + 2 * bDBP) / 3.0)
        bPP = float(bSBP - bDBP)

        outputs = {
            # core ones used by the classifier
            "LVEDP": LVEDP,
            "aSBP": aSBP,  # aortic SBP (unscaled)
            "SV": SV,
            "EF": EF,
            "CO": CO,

            # add the ones you want in CSV
            "EDV": EDV,
            "ESV": ESV,

            # convenience: brachial metrics (remove if you don’t want them)
            "bSBP": bSBP,
            "bDBP": bDBP,
            "MAP": bMAP,
            "PP": bPP,

            # ejection timing metrics from mets
            "Q_peak": float(mets["Q_peak"]),
            "t_peak": float(mets["t_peak"]),
            "ED": float(mets["EDur"]),
        }

        # waveform slice (optional)
        wave = {
            "t": cyc['t'],
            "Pao": cyc['P_ao'],
            "Plv": cyc['P_lv'],
            "Vlv": cyc['V_lv'],
            "Qao": cyc['Q_ao'],
        }
        return outputs, wave

    def physiological_label(self, outputs):
        """
        Return 1 if the sample is physiologically valid, else 0.
        Uses PHYS_LIMITS defined at the top of the file.
        """
        try:
            lv_ok = PHYS_LIMITS["LVEDP"][0] <= outputs["LVEDP"] <= PHYS_LIMITS["LVEDP"][1]
            sbp_ok = PHYS_LIMITS["aSBP"][0] <= outputs["aSBP"] <= PHYS_LIMITS["aSBP"][1]
            sv_ok = PHYS_LIMITS["SV"][0] <= outputs["SV"] <= PHYS_LIMITS["SV"][1]
            ef_ok = PHYS_LIMITS["EF"][0] <= outputs["EF"] <= PHYS_LIMITS["EF"][1]
            return int(lv_ok and sbp_ok and sv_ok and ef_ok)
        except Exception:
            # if any key is missing or non-finite
            return 0

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

    def simulate_and_save_dataset(self, params, sample_id, save_dir):
        """Run sim, extract metrics + save both CSV row and NPZ waveform."""
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, T_local = self.run_simulation(params)
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, T_local)
        mets, idxs = self.extract_cycle_metrics(cyc)

        sim = {
            'EDV': mets['EDV'],
            'ESV': mets['ESV'],
            'SV': mets['EDV'] - mets['ESV'],
            'EF': (mets['EDV'] - mets['ESV']) / mets['EDV'] * 100 if mets['EDV'] > 0 else np.nan,
            'LVEDP': mets['LVEDP'],
            # brachial-scaled pressures
            'bSBP': np.max(cyc['P_ao']) * 1.1,
            'bDBP': np.min(cyc['P_ao']) * 1.1,
            'bMAP': (np.max(cyc['P_ao']) * 1.1 + 2 * np.min(cyc['P_ao']) * 1.1) / 3,
            'bPP': (np.max(cyc['P_ao']) - np.min(cyc['P_ao'])) * 1.1,
            'LVOT_Flow_Peak': mets['Q_peak'],
            'time_LVOT_Flow_Peak': mets['t_peak'],
            'ED': mets['EDur'],
            'CO': ((mets['EDV'] - mets['ESV']) / T_local * 60.0 / 1000.0)
        }

        # Row for dataset (include T explicitly)
        row = {
            'sample_id': sample_id,
            **{k: params[k] for k in self.param_names},  # your 9 params
            'T': float(params.get('T', self.T)),
            **sim
        }

        npz_path = os.path.join(save_dir, f"sample_{sample_id}.npz")
        np.savez(
            npz_path,
            t=cyc['t'], P_lv=cyc['P_lv'], P_ao=cyc['P_ao'],
            V_lv=cyc['V_lv'], Q_sv=cyc['Q_sv'], Q_ao=cyc['Q_ao']
        )
        return row


def lhs_sample(N, bounds_dict, seed=0):
    names = list(bounds_dict.keys())
    lo = np.array([bounds_dict[n][0] for n in names], float)
    hi = np.array([bounds_dict[n][1] for n in names], float)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    U = sampler.random(N)
    X = qmc.scale(U, lo, hi)
    return names, X

def make_params_from_vector(sim, names, vec):
    """
    Build a full params dict for sim.run_simulation from a vector sampled over GLOBAL_BOUNDS.
    Keeps your field names untouched and includes T.
    """
    p = sim.params.copy()
    for n, v in zip(names, vec):
        p[n] = float(v)
    return p

def build_synthetic_dataset(sim, N=10000, out_dir="synthetic_dataset",
                            save_waveforms_subset=100, seed=42, n_jobs=-1,
                            debug_first=5):
    os.makedirs(out_dir, exist_ok=True)
    names, X = lhs_sample(N, GLOBAL_BOUNDS, seed=seed)

    def eval_one(i):
        xi = X[i, :]
        params = make_params_from_vector(sim, names, xi)
        outputs, wave = sim.compute_outputs_from_params(params)
        label = sim.physiological_label(outputs)
        prow = {n: float(params[n]) for n in names}
        prow["sample_id"] = i
        orow = {"sample_id": i, **outputs, "is_valid": int(label)}
        return prow, orow, wave

    # Debug a few samples without parallel/try-except
    for i in range(min(debug_first, N)):
        try:
            _ = eval_one(i)
        except Exception as e:
            print(f"[DEBUG] sample {i} failed with: {type(e).__name__}: {e}")
            raise

    # Then run the rest in parallel (you can restore try/except if you prefer)
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(eval_one)(i) for i in range(N))

    df_params = pd.DataFrame([r[0] for r in results])
    df_out    = pd.DataFrame([r[1] for r in results])
    df_merged = df_params.merge(df_out, on="sample_id")
    df_params.to_csv(os.path.join(out_dir, "synthetic_params.csv"), index=False)
    df_out.to_csv(os.path.join(out_dir, "synthetic_outputs.csv"), index=False)
    df_merged.to_csv(os.path.join(out_dir, "synthetic_dataset_merged.csv"), index=False)

    wave_dir = os.path.join(out_dir, "waveforms_subset")
    os.makedirs(wave_dir, exist_ok=True)
    kept = 0
    for i, (_, _, wave) in enumerate(results):
        if wave is not None and kept < save_waveforms_subset:
            np.savez_compressed(os.path.join(wave_dir, f"sample_{i}.npz"),
                                t=wave["t"], Pao=wave["Pao"], Plv=wave["Plv"],
                                Vlv=wave["Vlv"], Qao=wave["Qao"])
            kept += 1

    print(f"\nSynthetic dataset saved to: {out_dir}")
    print(f"  N = {N} | valid rate = {df_merged['is_valid'].mean()*100:.1f}%")
    print("  Files: synthetic_params.csv, synthetic_outputs.csv, synthetic_dataset_merged.csv, waveforms_subset/")
    return df_merged



# -- MAIN SCRIPT --
if __name__ == '__main__':
    save_root = (
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Dataset_Python\Dataset_Creation_1k_samples"
    )
    # Create a dummy row for initialization (values won't matter, will be overwritten)
    dummy_row = {
        'SWS (m/s)': 2.0,
        'bMAP': 90,
        'h/r': 0.5,
        'LAVI (ml/m²)': 25,
        'BSA (m²)': 1.8,
        'age (yrs)': 50,
        'BMI (kg/m²)': 25,
        'weight (kg)': 75,
        'HR (bpm)': 75,
        'R_sys (mmHg s/ml)': 1.0,
        'Z_ao (mmHg s/ml)': 0.1,
        'C_sa (ml/mmHg)': 2.0,
        'R_mv (mmHg s/ml)': 0.05,
        'E_max (mmHg/ml)': 2.0,
        'E_min (mmHg/ml)': 0.1,
        't_peak (s)': 0.3,
        'V_tot (ml)': 500,
        'C_sv (ml/mmHg)': 10.0,
        'Sub ID': 0  # dummy ID
    }

    sim_for_dataset = SubjectSimulator(dummy_row, 0, save_root)

    SYN_OUT = os.path.join(save_root, "SYNTHETIC")  # choose your folder
    df_syn = build_synthetic_dataset(
        sim_for_dataset,
        N=10000,  # adjust size
        out_dir=SYN_OUT,
        save_waveforms_subset=10000,  # optional
        seed=123,
        n_jobs=-1
    )

#end main