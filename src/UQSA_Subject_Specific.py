import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import os
import time
from multiprocessing import cpu_count
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
#from scipy.stats import lognorm, beta as beta_dist, qmc
from scipy.stats import norm, qmc
from joblib import Parallel, delayed

PHYS_LVEDP_RANGE = (4.0, 35.0)

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
            (0.5, 3.0),     # R_sys (self.params['R_sys'] * (1 - 0.20), self.params['R_sys'] * (1 + 0.20)),
            (0.01, 1.0),     # Z_ao  (self.params['Z_ao'] * (1 - 0.20), self.params['Z_ao'] * (1 + 0.20)),
            (0.5, 10.0),     # C_sa  (self.params['C_sa'] * (1 - 0.20), self.params['C_sa'] * (1 + 0.20)),
            (0.01, 0.1),     # R_mv
            (0.9, 10.0),     # E_max
            (0.01, 2.5),     # E_min #(self.params['E_min'] * (1 - 0.30), self.params['E_min'] * (1 + 0.30)),
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

    def run_simulation(self, params=None):
        """Simulate model for a set of parameters."""
        if params is None:
            params = self.params
        y0 = self.initial_state(params)  # <-- use params-consistent ICs
        num_steps = int(np.round(self.total / self.dt))
        t_eval = np.linspace(0, self.total, num_steps + 1)
        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, params),
                        [0, self.total], y0,
                        t_eval=t_eval, max_step=self.dt, rtol=1e-6, atol=1e-8)
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

    def run_one_lvedp(self, params=None):
        """Return LVEDP for a given parameter dict (non-intrusive)."""
        if params is None:
            params = self.params
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = self.run_simulation(params)
        cyc = self.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, _ = self.extract_cycle_metrics(cyc)
        return float(mets['LVEDP'])

#end class SubjectSimulator

def safe_lvedp_eval(sim, params):
    """Run once; return LVEDP or NaN if failure/unphysiologic."""
    try:
        y = sim.run_one_lvedp(params)
        if not np.isfinite(y):
            return np.nan
        if not (PHYS_LVEDP_RANGE[0] <= y <= PHYS_LVEDP_RANGE[1]):
            return np.nan
        return float(y)
    except Exception:
        return np.nan

def eval_matrix_parallel(sim, names, X, save_dir=None, label="mc", replace_invalid=True):
    def one_row(xi):
        p = sim.params.copy()
        for j, n in enumerate(names):
            p[n] = float(xi[j])
        return safe_lvedp_eval(sim, p)

    Y = Parallel(n_jobs=-1, verbose=10)(delayed(one_row)(xi) for xi in X)
    Y = np.array(Y, float)
    mask_bad = ~np.isfinite(Y)

    if replace_invalid and mask_bad.any():
        med = np.nanmedian(Y)
        Y[mask_bad] = med
        print(f"[warn] replaced {mask_bad.mean()*100:.1f}% invalid LVEDP with median {med:.2f} mmHg")
    elif not replace_invalid and mask_bad.any():
        print(f"[info] found {mask_bad.mean()*100:.1f}% invalid LVEDP values - leaving them as NaN")

    if save_dir is not None and mask_bad.any():
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(X[mask_bad], columns=names).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_{label}_invalid_params.csv"),
            index=False
        )
    return Y, mask_bad



def make_subject_bounds(sim, pct=None):
    """Hard bounds (from your model) intersected with a ±% window around best-fit."""
    if pct is None:
        pct = {"R_sys":0.10,"Z_ao":0.10,"C_sa":0.10,"R_mv":0.10,
               "E_max":0.10,"E_min":0.10,"t_peak":0.10,"V_tot":0.10,"C_sv":0.10}
    names = sim.param_names
    lb, ub = [], []
    for n in names:
        r = pct.get(n, 0.2)
        base = float(sim.params[n])
        lo_w, hi_w = (1-r)*base, (1+r)*base
        hard_lo, hard_hi = sim.bounds[names.index(n)]
        lb.append(max(lo_w, hard_lo))
        ub.append(min(hi_w, hard_hi))

    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), "Non-finite bounds"
    assert np.all(lb < ub), "Lower bound not < upper bound for at least one parameter"
    return names, np.array(lb, float), np.array(ub, float)

def default_prior_spec(sim):
    """
    Per-parameter prior spec relative to sim.params (means at best-fits).
    - normal: specify coefficient of variation (cv), mean = best-fit
    """
    p = sim.params
    return {
        "R_sys":  ("normal", {"cv": 0.12}),
        "Z_ao":   ("normal", {"cv": 0.12}),
        "C_sa":   ("normal", {"cv": 0.15}),
        "R_mv":   ("normal", {"cv": 0.15}),
        "E_max":  ("normal", {"cv": 0.15}),
        "E_min":  ("normal", {"cv": 0.15}),
        # t_peak now also normal, centered at best-fit, with a reasonably tight cv
        "t_peak": ("normal", {"cv": 0.10}),
        "V_tot":  ("normal", {"cv": 0.12}),
        "C_sv":   ("normal", {"cv": 0.15})
    }


def inverse_cdf_from_unit(u, name, best, lo, hi, prior):
    """Map U in (0,1) to param value with prior; then truncate to [lo, hi]."""
    kind, cfg = prior

    if kind == "normal":
        cv = cfg["cv"]
        sigma = cv * abs(best) if best != 0 else cv
        # Avoid zero sigma
        if sigma <= 0:
            x = best
        else:
            x = norm.ppf(u, loc=best, scale=sigma)

    else:
        # fallback uniform if an unknown prior type sneaks in
        x = lo + (hi - lo) * u

    # truncate to hard window
    return float(np.minimum(np.maximum(x, lo), hi))


def save_subject_conditions(sim, names, lb, ub, prior_spec, save_dir):
    """Save bounds and priors for reproducibility."""
    df = pd.DataFrame({
        "param": names,
        "lower": lb,
        "upper": ub,
        "best_fit": [float(sim.params[n]) for n in names],
        "prior_type": [prior_spec[n][0] for n in names],
        "prior_config": [str(prior_spec[n][1]) for n in names]
    })
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{sim.sub_id}_uqsa_conditions.csv")
    df.to_csv(out_path, index=False)
    print(f"[{sim.sub_id}] Saved conditions → {out_path}")

def sobol_with_ci(problem, Y, calc_second_order=False, n_resamples=500, conf_level=0.95, seed=0):
    """
    Wrapper around SALib's Sobol analyze() to add bootstrap confidence intervals.
    Works with SALib >=1.5 (no built-in CI).
    """
    rng = np.random.default_rng(seed)

    # Core Sobol analysis
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)

    # Bootstrap resampling
    n = len(Y)
    S1_samples, ST_samples = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)  # sample with replacement
        Si_resamp = sobol_analyze.analyze(problem, Y[idx], calc_second_order=calc_second_order,
                                          print_to_console=False)
        S1_samples.append(Si_resamp["S1"])
        ST_samples.append(Si_resamp["ST"])

    S1_samples = np.array(S1_samples)
    ST_samples = np.array(ST_samples)

    alpha = (1 - conf_level) / 2
    Si["S1_conf"] = np.nanpercentile(S1_samples, [100*alpha, 100*(1-alpha)], axis=0).T
    Si["ST_conf"] = np.nanpercentile(ST_samples, [100*alpha, 100*(1-alpha)], axis=0).T

    return Si

def mc_uq_lvedp(sim, N=2000, seed=0, pct=None, prior_spec=None, save_dir=None):
    names, lb, ub = make_subject_bounds(sim, pct)
    if prior_spec is None:
        prior_spec = default_prior_spec(sim)

    d = len(names)
    U = qmc.LatinHypercube(d=d, seed=seed).random(N)
    X = np.empty_like(U)
    for j, n in enumerate(names):
        for i in range(N):
            X[i, j] = inverse_cdf_from_unit(
                U[i, j], n, best=float(sim.params[n]),
                lo=lb[j], hi=ub[j], prior=prior_spec[n]
            )

    # Do NOT replace invalid values for MC - we want to drop them
    Y_raw, mask_bad = eval_matrix_parallel(sim, names, X, save_dir=save_dir,
                                           label="mc", replace_invalid=False)

    valid = ~mask_bad
    n_total = N
    n_valid = int(valid.sum())
    invalid_rate = (n_total - n_valid) / n_total * 100.0
    print(f"[{sim.sub_id}] UQ invalid LVEDP rate: {invalid_rate:.1f}%, "
          f"kept {n_valid} / {n_total} samples")

    if n_valid == 0:
        raise RuntimeError(f"No valid LVEDP samples for subject {sim.sub_id} in MC UQ")

    X_valid = X[valid]
    Y_valid = Y_raw[valid]

    mean = float(np.mean(Y_valid))
    sd   = float(np.std(Y_valid, ddof=1))
    p025, p975 = np.percentile(Y_valid, [2.5, 97.5])

    return {
        "names": names,
        "lb": lb,
        "ub": ub,
        "X": X_valid,
        "Y": Y_valid,
        "mask_bad": mask_bad,
        "summary": {
            "mean": mean,
            "sd": sd,
            "PI95": (float(p025), float(p975)),
            "N": n_valid,
            "N_total": int(n_total)
        }
    }

def sobol_sa_lvedp(sim, N=1024, seed=0, pct=None, prior_spec=None,
                   second_order=False, n_resamples=500, save_dir=None):
    names, lb, ub = make_subject_bounds(sim, pct)
    if prior_spec is None:
        prior_spec = default_prior_spec(sim)

    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]]*len(names)}
    U = sobol_sample.sample(problem, N, calc_second_order=second_order)

    X = np.empty_like(U)
    for j, n in enumerate(names):
        for i in range(U.shape[0]):
            X[i, j] = inverse_cdf_from_unit(
                U[i, j], n,
                best=float(sim.params[n]),
                lo=lb[j], hi=ub[j],
                prior=prior_spec[n]
            )

    # For Sobol we still need a full vector with no NaNs, so we keep replacement on
    Y, mask_bad = eval_matrix_parallel(sim, names, X, save_dir=save_dir,
                                       label="sobol", replace_invalid=True)
    print(f"[{sim.sub_id}] Sobol invalid LVEDP rate (before replacement): "
          f"{np.mean(mask_bad) * 100:.1f}%")

    # Run SA + bootstrap wrapper
    Si = sobol_with_ci(problem, Y, calc_second_order=second_order, n_resamples=n_resamples)

    out = {
        "names": names,
        "S1": Si["S1"].tolist(),
        "ST": Si["ST"].tolist(),
        "S1_conf": Si["S1_conf"],
        "ST_conf": Si["ST_conf"],
        "N_base": int(N),
        "n_eval": int(len(Y)),
        "mask_bad": mask_bad,
        "X": X,
        "U": U
    }
    if second_order:
        out["S2"] = Si["S2"].tolist()
        out["S2_conf"] = Si["S2_conf"]
    return out

def sobol_sa_validity(sim, N=1024, seed=0, pct=None, prior_spec=None,
                      n_resamples=500, save_dir=None):
    """
    Sobol sensitivity analysis on VALIDITY indicator:
        Y_valid = 1 if LVEDP is valid (within PHYS_LVEDP_RANGE and simulation ok)
        Y_valid = 0 if invalid
    This tells you which parameters mostly control model unphysiology/failure.
    """
    names, lb, ub = make_subject_bounds(sim, pct)
    if prior_spec is None:
        prior_spec = default_prior_spec(sim)

    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]] * len(names)}
    U = sobol_sample.sample(problem, N, calc_second_order=False)

    # Map Sobol U ∈ [0,1] to parameter space using the same priors as LVEDP SA
    X = np.empty_like(U)
    for j, n in enumerate(names):
        for i in range(U.shape[0]):
            X[i, j] = inverse_cdf_from_unit(
                U[i, j], n,
                best=float(sim.params[n]),
                lo=lb[j], hi=ub[j],
                prior=prior_spec[n]
            )

    # Evaluate LVEDP once, but we only care about mask_bad (valid vs invalid)
    # IMPORTANT: no replacement, we just need mask_bad.
    Y_lvedp, mask_bad = eval_matrix_parallel(
        sim, names, X, save_dir=save_dir,
        label="sobol_validity", replace_invalid=False
    )

    # Build validity indicator: 1 = valid, 0 = invalid
    Y_valid = (~mask_bad).astype(float)
    invalid_rate = mask_bad.mean() * 100.0
    print(f"[{sim.sub_id}] Sobol VALIDITY invalid rate: {invalid_rate:.1f}% "
          f"(Y_valid mean ≈ probability of validity: {Y_valid.mean():.3f})")

    # Run Sobol on Y_valid
    Si = sobol_with_ci(problem, Y_valid,
                       calc_second_order=False,
                       n_resamples=n_resamples)

    out = {
        "names": names,
        "S1": Si["S1"].tolist(),
        "ST": Si["ST"].tolist(),
        "S1_conf": Si["S1_conf"],
        "ST_conf": Si["ST_conf"],
        "N_base": int(N),
        "n_eval": int(len(Y_valid)),
        "mask_bad": mask_bad,
        "X": X,
        "U": U,
    }
    return out



def run_uq_sa_for_subject(sim, N_mc=2000, N_sobol=1024, pct=None,
                          prior_spec=None, save_dir=None, n_resamples=500):

    names, lb, ub = make_subject_bounds(sim, pct)

    if prior_spec is None:
        prior_spec = default_prior_spec(sim)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_subject_conditions(sim, names, lb, ub, prior_spec, save_dir)

    # === UQ on LVEDP ===
    mc = mc_uq_lvedp(sim, N=N_mc, seed=0, pct=pct,
                     prior_spec=prior_spec, save_dir=save_dir)

    # === Sobol SA on LVEDP ===
    sa_lvedp = sobol_sa_lvedp(sim, N=N_sobol, seed=0, pct=pct,
                              prior_spec=prior_spec, second_order=False,
                              n_resamples=n_resamples, save_dir=save_dir)

    # === Sobol SA on VALIDITY (probability of being valid) ===
    sa_valid = sobol_sa_validity(sim, N=N_sobol, seed=0, pct=pct,
                                 prior_spec=prior_spec, n_resamples=n_resamples,
                                 save_dir=save_dir)

    # UQ samples dataframe (LVEDP only, valid cases)
    df_uq = pd.DataFrame({"sub_id": [sim.sub_id] * len(mc["Y"]), "LVEDP": mc["Y"]})

    # SA on LVEDP
    S1_conf_L = np.asarray(sa_lvedp["S1_conf"])
    ST_conf_L = np.asarray(sa_lvedp["ST_conf"])
    df_sa_lvedp = pd.DataFrame({
        "sub_id": sim.sub_id,
        "metric": "LVEDP",
        "param": sa_lvedp["names"],
        "S1": sa_lvedp["S1"],
        "S1_low": S1_conf_L[:, 0],
        "S1_high": S1_conf_L[:, 1],
        "ST": sa_lvedp["ST"],
        "ST_low": ST_conf_L[:, 0],
        "ST_high": ST_conf_L[:, 1],
    })

    # SA on VALIDITY (probability of valid LVEDP)
    S1_conf_V = np.asarray(sa_valid["S1_conf"])
    ST_conf_V = np.asarray(sa_valid["ST_conf"])
    df_sa_valid = pd.DataFrame({
        "sub_id": sim.sub_id,
        "metric": "VALIDITY",
        "param": sa_valid["names"],
        "S1": sa_valid["S1"],
        "S1_low": S1_conf_V[:, 0],
        "S1_high": S1_conf_V[:, 1],
        "ST": sa_valid["ST"],
        "ST_low": ST_conf_V[:, 0],
        "ST_high": ST_conf_V[:, 1],
    })

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # UQ params & outputs (only valid samples)
        pd.DataFrame(mc["X"], columns=mc["names"]).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_uq_params.csv"), index=False
        )
        pd.DataFrame({"LVEDP": mc["Y"]}).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_uq_samples.csv"), index=False
        )
        pd.Series(mc["summary"]).to_json(
            os.path.join(save_dir, f"{sim.sub_id}_uq_summary.json")
        )

        # Sobol params (same design for LVEDP and VALIDITY)
        pd.DataFrame(sa_lvedp["X"], columns=sa_lvedp["names"]).to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_sobol_params.csv"), index=False
        )

        # Sobol results (two metrics in one file or separate as you prefer)
        df_sa_lvedp.to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_sobol_lvedp.csv"),
            index=False
        )
        df_sa_valid.to_csv(
            os.path.join(save_dir, f"{sim.sub_id}_sobol_validity.csv"),
            index=False
        )

    return df_uq, df_sa_lvedp, df_sa_valid, mc, sa_lvedp, sa_valid



# -- MAIN SCRIPT --
if __name__ == '__main__':
    save_root = (
    r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
    r"\3_Codes\Python\Data_Results\Results_UQSA_Population_Level")

    uqsa_root = os.path.join(save_root, "UQ_SA")
    excel_path = (r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
                 r"\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx")
    sheet_name = "Study_3_V4.1_T6"
    combined_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, nrows=69)


    #subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
                     #344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365]# full batch run
    subject_list = [999]

    #subject_list = [316, 319, 323, 325, 326, 327] # batch 1
    #subject_list = [328, 329, 331, 332, 334, 336] # batch 2
    #subject_list = [337, 338, 339, 341, 342, 343, 344, 346, 347, 349, 351, 352] # batch 3
    #subject_list = [354, 356, 357, 359, 360, 361, 362, 363, 364, 365] # batch 4

    for sub_id in subject_list:
        data_row = combined_df.loc[combined_df['Sub ID'] == sub_id].squeeze()
        sim = SubjectSimulator(data_row, sub_id, save_root)
        print(f"\nParameters for subject {sub_id}:")
        for k, v in sim.params.items():
            print(f"  {k:8} = {v}")

        names, lb, ub = make_subject_bounds(sim, pct=None)
        print(f"\n[{sim.sub_id}] Parameter bounds used for UQ/SA:")
        for n, lo, hi in zip(names, lb, ub):
            print(f"  {n:7s}: {lo:.6g} → {hi:.6g}   (best={float(sim.params[n]):.6g})")
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
                print(f"  [warn] Bad interval for {n}: [{lo}, {hi}]")

        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = sim.run_simulation(sim.params)
        cyc = sim.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, idxs = sim.extract_cycle_metrics(cyc)
        sim_matrices = sim.compare_with_gt(mets, cyc)
        sim.plot_all(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, sim.params)  # Plot all
        sim.save_current_figures()  # Save all
        plt.close('all')  # Close all to avoid memory leak
        #plt.show()

     # === UQ + SA ===
        save_dir = os.path.join(uqsa_root, str(sub_id))
        df_uq, df_sa_lvedp, df_sa_valid, mc, sa_lvedp, sa_valid = run_uq_sa_for_subject(
            #sim, N_mc=75000, N_sobol=2048, pct=None, save_dir=save_dir, n_resamples=2000
            sim, N_mc=1000, N_sobol=64, pct=None, save_dir=save_dir, n_resamples=64
        )

        print(f"[{sub_id}] UQ LVEDP: mean={mc['summary']['mean']:.2f}, "
              f"sd={mc['summary']['sd']:.2f}, 95%PI={mc['summary']['PI95']}")
        print(f"[{sub_id}] Top ST (LVEDP):",
              sorted(zip(sa_lvedp["names"], sa_lvedp["ST"]), key=lambda x: -x[1])[:3])
        print(f"[{sub_id}] Top ST (VALIDITY):",
              sorted(zip(sa_valid["names"], sa_valid["ST"]), key=lambda x: -x[1])[:3])

#end main