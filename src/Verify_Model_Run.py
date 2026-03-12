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

    # === NEW: sensitivity analysis for preload, afterload, elastance ===
    def run_sensitivity_analysis(self):
        """
        Sensitivity analysis for:
          1) Preload:    V_sv, R_mv, C_sv
          2) Afterload:  R_sys, Z_ao, C_sa
          3) Elastance:  E_min, E_max

        For each scenario:
          - Overplot PV loops for +5%, +10%, +20%
          - Mark LVEDP on each loop
          - Build table containing absolute values and percent change from baseline
        """
        baseline_params = self.params.copy()
        perc_changes = [0.05, 0.10, 0.20]

        scenarios = [
            {'name': 'Preload', 'params': ['V_tot', 'R_mv', 'C_sv']},
            {'name': 'Afterload', 'params': ['R_sys', 'Z_ao', 'C_sa']},
            {'name': 'Elastance', 'params': ['E_min', 'E_max']}
        ]

        # --- baseline metrics (used for % change) ---
        (_, _, _, _, _, _, _,
         cyc_base, mets_base, _, metrics_base) = self.simulate_cycle_and_metrics(baseline_params)

        rows = []

        def pct_change(val, base):
            return 100.0 * (val - base) / base if base != 0 else np.nan

        for scen in scenarios:
            scen_name = scen['name']
            param_list = scen['params']

            # formatted PV-loop figure
            fig = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
            ax = fig.add_subplot(111)

            for pc in perc_changes:
                factor = 1.0 + pc
                label = f"+{int(pc * 100)}%"

                # varied parameters
                params_var = baseline_params.copy()
                for pk in param_list:
                    params_var[pk] = baseline_params[pk] * factor

                # simulate
                (t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys,
                 cyc, mets, idxs, metrics) = self.simulate_cycle_and_metrics(params_var)

                # PV loop
                line, = ax.plot(cyc['V_lv'], cyc['P_lv'], label=label, linewidth=0.75)
                color = line.get_color()

                # LVEDP marker
                ax.scatter([mets['EDV']], [mets['LVEDP']], s=15, color=color)

                # store row for table (absolute + % change)
                row = {
                    'sub_id': self.sub_id,
                    'Scenario': scen_name,
                    'Changed_params': ','.join(param_list),
                    'Change': label,

                    'SV [ml]': metrics['SV'],
                    'EF [%]': metrics['EF'],
                    'aSBP [mmHg]': metrics['aSBP'],
                    'bSBP [mmHg]': metrics['bSBP'],
                    'bMAP [mmHg]': metrics['bMAP'],
                    'LVEDP [mmHg]': metrics['LVEDP'],

                    'SV_change_%': pct_change(metrics['SV'], metrics_base['SV']),
                    'EF_change_%': pct_change(metrics['EF'], metrics_base['EF']),
                    'aSBP_change_%': pct_change(metrics['aSBP'], metrics_base['aSBP']),
                    'bSBP_change_%': pct_change(metrics['bSBP'], metrics_base['bSBP']),
                    'bMAP_change_%': pct_change(metrics['bMAP'], metrics_base['bMAP']),
                    'LVEDP_change_%': pct_change(metrics['LVEDP'], metrics_base['LVEDP']),
                }
                rows.append(row)

            # axis formatting (like your example figure)
            ax.set_xlim(20, 60)
            ax.set_ylim(0, 120)
            ax.set_xticks(np.arange(20, 61, 10))
            ax.set_yticks(np.arange(0, 121, 20))

            ax.set_xlabel("LV Volume [ml]", fontname="Times New Roman", fontsize=9, color="black")
            ax.set_ylabel("LV Pressure [mmHg]", fontname="Times New Roman", fontsize=9, color="black")

            ax.tick_params(axis='both', labelsize=9, colors="black")
            for lbl in ax.get_xticklabels():
                lbl.set_fontname("Times New Roman")
            for lbl in ax.get_yticklabels():
                lbl.set_fontname("Times New Roman")

            legend = ax.legend(fontsize=9, frameon=False)
            for txt in legend.get_texts():
                txt.set_fontname("Times New Roman")
                txt.set_color("black")

            ax.grid(False)

            fig_filename = f"PV_sensitivity_{scen_name}_sub_{self.sub_id}.png"
            fig_path = os.path.join(self.subject_dir, fig_filename)
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved formatted sensitivity PV plot: {fig_path}")

        # --- build and save table ---
        df = pd.DataFrame(rows)

        print("\n=== Sensitivity analysis summary including percent change from baseline ===")
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 150,
                               'display.float_format', '{:.2f}'.format):
            print(df)

        metrics_path = os.path.join(self.subject_dir,
                                    f"sensitivity_metrics_sub_{self.sub_id}.csv")
        df.to_csv(metrics_path, index=False)
        print(f"Saved sensitivity metrics table: {metrics_path}")


# -- MAIN SCRIPT --
if __name__ == '__main__':
    combined_GT_df = pd.read_csv(
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Invasive_Study_Leuven_GT_matrices_all_subjects_V4.0.csv")

    save_root = (
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Data_Results\Local_SA_Figure_3_4"
    )
    excel_path = (r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
                  r"\3_Codes\Python\Data_Results\Results_Validation_Paper_all_subjects_V4.xlsx")
    sheet_name = "Study_8_V4"
    combined_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3, nrows=69)


    #subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
                    # 344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365]# full batch run
    subject_list = [999] #test

    #subject_list = [316, 319, 323, 325, 326, 327] # batch 1
    #subject_list = [328, 329, 331, 332, 334, 336] # batch 2
    #subject_list = [337, 338, 339, 341, 342, 343, 344, 346, 347, 349, 351, 352] # batch 3
    #subject_list = [354, 356, 357, 359, 360, 361, 362, 363, 364, 365] # batch 4


    sim_results = []

    for sub_id in subject_list:
        data_row_GT = combined_GT_df.loc[combined_GT_df['sub_id'] == sub_id].squeeze()
        data_row = combined_df.loc[combined_df['Sub ID'] == sub_id].squeeze()
        sim = SubjectSimulator(data_row, data_row_GT, sub_id, save_root)

        print(f"\nParameters for subject {sub_id}:")
        for k, v in sim.params.items():
            print(f"  {k:8} = {v}")

        # Optional: run and compare baseline (keeps your existing behavior)
        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = sim.run_simulation(sim.params)
        cyc = sim.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        mets, idxs = sim.extract_cycle_metrics(cyc)
        sim_matrices = sim.compare_with_gt(mets, cyc)  # GT not used for optimization here, just printing

        # Optional: plot and save baseline figures
        sim.plot_all(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs, sim.params)
        sim.save_current_figures()
        plt.close('all')

        # NEW: run parameter sensitivity analysis
        sim.run_sensitivity_analysis()

        # If you want to save PV-loop sensitivity figures as well:
        sim.save_current_figures()
        plt.close('all')

#end main