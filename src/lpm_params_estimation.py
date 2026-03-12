import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import os
import time
from multiprocessing import cpu_count

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

        self.SWE_velocity = data_row['SWE_vel_MVC']
        self.GT_bMAP = data_row['bMAP']
        self.h_r = data_row['h_r']
        self.LAVI = data_row['LAVI']
        self.BSA = data_row['BSA']
        self.age = data_row['age']
        self.BMI = data_row['BMI']
        self.weight = data_row['weight']


        self.bpm = data_row['HR']
        self.T = 60.0 / self.bpm
        self.total = self.cycles * self.T
        self.dt = self.T / 500.0

        # Model parameter names and initial guess (for bounds)
        self.param_names = ['R_sys', 'Z_ao', 'C_sa', 'R_mv', 'E_max', 'E_min', 't_peak', 'V_tot', 'C_sv']
        self.params = {
            'R_sys': data_row['R_sys'],
            'Z_ao': data_row['Z_ao'],
            'C_sa': data_row['C_sa'],
            'R_mv': 0.015,
            'E_max': 4.176,
            'E_min': 0.342, #'E_min': data_row['pred_E_min'],
            't_peak': 0.304,
            'V_tot': 1648.611,
            'C_sv': 9.705,
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
        num_steps = int(np.round(self.total / self.dt))
        t_eval = np.linspace(0, self.total, num_steps + 1)
        sol = solve_ivp(lambda tt, yy: self.circulation_odes(tt, yy, params),
                        [0, self.total], self.y0,
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
        return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs

    def save_current_figures(self):
        """Save all open matplotlib figures to subject's folder."""
        for i, fig_num in enumerate(plt.get_fignums(), 1):
            fig = plt.figure(fig_num)
            filename = os.path.join(self.subject_dir, f"fig_{i}.png")
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved: {filename}")

    def plot_all(self, t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs):
        """Plot all basic simulation results and cycle metrics (and save them)."""
        # Elastance
        E_full = self.elastance(t, self.params)
        plt.figure(); plt.plot(t, E_full)
        plt.title('Time-Varying Elastance')
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

# -- MAIN SCRIPT --
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    combined_df = pd.read_csv(
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Dataset_Python\Invasive_Study_Leuven_GT_matrices.csv")
    #subject_list = [316, 319, 323, 325, 326, 327, 328, 329, 331, 332, 334, 336, 337, 338, 339, 341, 342, 343,
                     #344, 346, 347, 349, 351, 352, 354, 356, 357, 359, 360, 361, 362, 363, 364, 365]# full batch run
    subject_list = [325] #test

    #subject_list = [316, 319, 323, 325, 326, 327] # batch 1
    #subject_list = [328, 329, 331, 332, 334, 336] # batch 2
    #subject_list = [337, 338, 339, 341, 342, 343, 344, 346, 347, 349, 351, 352] # batch 3
    #subject_list = [354, 356, 357, 359, 360, 361, 362, 363, 364, 365] # batch 4

    save_root = (
        r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
        r"\3_Codes\Python\Dataset_Python\Results_Plots_Study_23_test"
    )
    sim_results = []

    for sub_id in subject_list:
        data_row = combined_df.loc[combined_df['sub_id'] == sub_id].squeeze()
        sim = SubjectSimulator(data_row, sub_id, save_root)
        best, opt_time = sim.optimize()
        print("Optimization success:", best.success)
        print("Best loss:", best.fun)
        for name, val in zip(sim.param_names, best.x):
            print(f"  {name:8s} = {val:.4f}")

        # Final simulation with best-fit parameters
        #t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = sim.run_simulation()
        #cyc = sim.cycle_cutting_algo(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
        #mets, idxs = sim.extract_cycle_metrics(cyc)

        t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs = sim.simulate_and_metrics(best.x)
        sim.plot_all(t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys, cyc, mets, idxs) # Plot all
        sim.save_current_figures() # Save all
        plt.show()
        #plt.close('all')  # Close all to avoid memory leak

        #Print metrics comparison and collect for CSV
        sim_matrices = sim.compare_with_gt(mets, cyc)
        row = sim.collect_result_row(best, best.fun, opt_time, sim_matrices)
        sim_results.append(row)

    # Save all results as a CSV
    results_df = pd.DataFrame(sim_results)
    results_path = os.path.join(save_root, "Study_22_Study_11_with_pred_LVEDP_MVR_weights_1_no_fixed_params.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nAll subject results saved to:\n{results_path}")
#end main