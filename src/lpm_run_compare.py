"""
lumped_model_base_params_compare_matrices.py

Standalone script:
- Hard-coded model & IC parameters
- Simulate a 0D cardiovascular model over multiple cycles
- Plot full-simulation results and time-varying elastance
- Extract single cycle (valley-to-valley)
- Compute key metrics: EDV, ESV, LVEDP, Q_ao peak, time to peak, ejection duration
- Plot cycle waveforms and PV loop, marking metric points
- Compare simulated metrics against ground-truth CSV in a separate function
- Compute weighted SSE loss across metrics (excluding LVEDP)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import pandas as pd

gt_csv_path = (
    r"C:\Workspace\Post_Doc_Works_NTNU"
    r"\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin"
    r"\3_Codes\Python\Dataset_Python\Data Annotation\Sub_3267_325_gt_metrics.csv"
)

# ----------------------------------------
# Simulation settings (hard-coded)
# ----------------------------------------
bpm       = 65         # heart rate [beats/min], to be changed for each subject
cycles    = 20           # number of cardiac cycles to simulate, fixed
P_ao0     = 70           # initial aortic pressure [mmHg], fixed
V_LV0     = 100.0        # initial LV volume [ml], fixed
P_th      = -4.0         # intrathoracic pressure [mmHg], fixed

a         = 1.55         # double-hill time-varying elastance shape function scaling factor, fixed
alpha1    = 0.7          # double-hill time-varying elastance shape function time-scale α₁, fixed
alpha2    = 1.17         # double-hill time-varying elastance shape function time-scale α₂, fixed
n1        = 1.9          # double-hill time-varying elastance shape function term 1 power, fixed
n2        = 21.9         # double-hill time-varying elastance shape function term 2 power, fixed

use_cycle = 'second_last'       # which cycle to extract: 'last' or 'second_last'

# ----------------------------------------
# Model base parameters
# ----------------------------------------
params = {
    'R_mv':   0.05,   # mitral valve resistance (mmHg·s/ml)
    'R_sys':  2.67,   # systemic resistance
    'Z_ao':   0.081,  # aortic impedance
    'C_sa':   0.45,   # arterial compliance (ml/mmHg)
    'C_sv':   15,   # venous compliance (ml/mmHg)
    'E_max':  5,   # max elastance (mmHg/ml)
    'E_min':  0.05,  # min elastance (mmHg/ml)
    't_peak': 0.35,  # time to peak elastance (s)
    'V_tot': 300  # total blood volume (ml)
}

# ----------------------------------------
# Elastance function
# ----------------------------------------
def elastance(t, p):
    tn = np.mod(t, p['T']) / p['t_peak']
    t1 = tn / alpha1
    t2 = tn / alpha2
    En = (t1 ** n1) / (1 + t1 ** n1)
    En *= 1.0 / (1 + t2 ** n2)
    return p['E_max'] * En * a + p['E_min']

# ----------------------------------------
# ODE system
# ----------------------------------------
def circulation_odes(t, y, p):
    V_lv, V_sa, V_sv = y
    E_t = elastance(t, p)
    P_lv = E_t * V_lv + P_th
    P_sa = V_sa / p['C_sa']
    P_sv = V_sv / p['C_sv']
    Q_sv_lv = (P_sv - P_lv)/p['R_mv'] if P_sv > P_lv else 0.0
    Q_lv_ao = (P_lv - P_sa)/p['Z_ao'] if P_lv > P_sa else 0.0
    Q_sys   = (P_sa - P_sv)/p['R_sys'] if P_sa > P_sv else 0.0
    return [Q_sv_lv - Q_lv_ao,
            Q_lv_ao - Q_sys,
            Q_sys - Q_sv_lv]

# ----------------------------------------
# Simulation runner
# ----------------------------------------
def run_simulation(params, total_time, dt, y0):
    t_eval = np.arange(0, total_time + dt, dt)
    sol = solve_ivp(lambda tt, yy: circulation_odes(tt, yy, params),
                    [0, total_time], y0,
                    t_eval=t_eval, max_step=dt)
    t = sol.t
    V_lv, V_sa, V_sv = sol.y
    P_lv = elastance(t, params) * V_lv + P_th
    P_sa = V_sa / params['C_sa']
    # aortic pressure follows valve: when LV > arterial, P_ao = P_lv else P_sa
    P_ao = np.where(P_lv > P_sa, P_lv, P_sa)
    # venous pressure
    P_sv = V_sv / params['C_sv']
    # flows remain unchanged
    Q_sv_lv = np.where(P_sv > P_lv, (P_sv - P_lv)/params['R_mv'], 0.0)
    Q_lv_ao = np.where(P_lv > P_sa, (P_lv - P_sa)/params['Z_ao'], 0.0)
    Q_sys   = np.where(P_sa > P_sv, (P_sa - P_sv)/params['R_sys'], 0.0)
    return t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys

# ----------------------------------------
# Cycle-cutting: slicing out the Nth cycle by index
# ----------------------------------------
def cycle_cutting_algo(t, bpm, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao):
    dt = t[1] - t[0]
    T  = 60.0 / bpm
    spc = int(round(float(T / dt)))
    n   = len(t) // spc  # number of full cycles
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

# ----------------------------------------
# Extract cycle metrics
# ----------------------------------------
def extract_cycle_metrics(cyc):
    t = cyc['t']
    V = cyc['V_lv']
    P = cyc['P_lv']
    Q = cyc['Q_ao']
    # EDV & ESV
    EDV_idx = np.argmax(V)
    ESV_idx = np.argmin(V)
    EDV = V[EDV_idx]
    ESV = V[ESV_idx]
    LVEDP = P[EDV_idx]
    # Q_peak & t_peak
    peak_idx = np.argmax(Q)
    Q_peak = Q[peak_idx]
    t_peak = t[peak_idx]
    # ejection duration
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

# ----------------------------------------
# Plotting single cycle with metrics
# ----------------------------------------
def plot_cycle_with_metrics(cyc, mets, idxs):
    t_c = cyc['t']
    # Pressure: mark LVEDP@EDV
    plt.figure()
    plt.plot(t_c, cyc['P_ao'], label='P_ao')
    plt.plot(t_c, cyc['P_lv'], label='P_lv')
    t_edv = t_c[idxs['EDV_idx']]
    p_lvedp = cyc['P_lv'][idxs['EDV_idx']]
    plt.scatter([t_edv], [p_lvedp], c='r', marker='o', label='LVEDP@EDV')
    plt.legend(); plt.grid()
    # Flow: mark Q_peak and ejection duration
    plt.figure()
    plt.plot(t_c, cyc['Q_ao'], label='Q_ao')
    plt.plot(t_c, cyc['Q_sv'], label='Q_sv')
    t_pk = t_c[idxs['peak_idx']]
    q_pk = cyc['Q_ao'][idxs['peak_idx']]
    plt.scatter([t_pk], [q_pk], c='g', marker='o', label='Q_peak')
    t_start = t_c[idxs['edur_start']]
    t_end   = t_c[idxs['edur_end']]
    plt.axvline(t_start, color='m', linestyle='--', label='EDur start')
    plt.axvline(t_end,   color='m', linestyle='--', label='EDur end')
    plt.legend(); plt.grid()
    # Volume: mark EDV and ESV
    plt.figure()
    plt.plot(t_c, cyc['V_lv'], label='V_lv')
    t_esv = t_c[idxs['ESV_idx']]
    v_esv = cyc['V_lv'][idxs['ESV_idx']]
    plt.scatter([t_esv], [v_esv], c='b', marker='o', label='ESV')
    t_edv = t_c[idxs['EDV_idx']]
    v_edv = mets['EDV']
    plt.scatter([t_edv], [v_edv], c='r', marker='o', label='EDV')
    plt.title('Cycle LV Volume'); plt.xlabel('s'); plt.ylabel('ml'); plt.legend(); plt.grid()
    # PV loop: mark EDV point
    plt.figure()
    plt.plot(cyc['V_lv'], cyc['P_lv'])
    plt.scatter([mets['EDV']], [mets['LVEDP']], c='r', marker='o', label='EDV point')
    plt.title('Cycle PV Loop'); plt.xlabel('V_lv'); plt.ylabel('P_lv'); plt.legend(); plt.grid()

# ----------------------------------------
# Compare with ground-truth CSV
# ----------------------------------------
def compare_with_gt(mets, cyc, gt_path, ppa=1.1):
    gt_df = pd.read_csv(gt_path)
    gt = dict(zip(gt_df['Metric'], gt_df['Value']))
    sim = {}
    sim['EDV'] = mets['EDV']
    sim['ESV'] = mets['ESV']
    sim['SV']  = mets['EDV'] - mets['ESV']
    sim['EF']  = (sim['SV']/sim['EDV']*100) if sim['EDV']>0 else np.nan
    sim['bSBP'] = np.max(cyc['P_ao'])*ppa
    sim['bDBP'] = np.min(cyc['P_ao'])*ppa
    sim['bMAP'] = (sim['bSBP'] + 2*sim['bDBP'])/3
    sim['bPP']  = sim['bSBP'] - sim['bDBP']
    sim['LVOT_Flow_Peak']      = mets['Q_peak']
    sim['time_LVOT_Flow_Peak'] = mets['t_peak']
    sim['ED']   = mets['EDur']
    sim['LVEDP']= mets['LVEDP']
    print("\nComparing simulation vs. ground-truth:")
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


# ----------------------------------------
# Compute weighted SSE loss (excluding LVEDP)
# ----------------------------------------
def loss_all_matrices(mets, cyc, gt_path, weights, ppa=1.1):
    # load ground-truth
    gt_df = pd.read_csv(gt_path)
    gt = dict(zip(gt_df['Metric'], gt_df['Value']))
    # assemble sim metrics
    sim = {
        'EDV': mets['EDV'],
        'ESV': mets['ESV'],
        'SV':  mets['EDV'] - mets['ESV'],
        'EF':  (mets['EDV'] - mets['ESV'])/mets['EDV']*100,
        'bSBP': np.max(cyc['P_ao'])*ppa,
        'bDBP': np.min(cyc['P_ao'])*ppa,
        'bMAP': (np.max(cyc['P_ao'])*ppa + 2*np.min(cyc['P_ao'])*ppa)/3,
        'bPP':  np.max(cyc['P_ao'])*ppa - np.min(cyc['P_ao'])*ppa,
        'LVOT_Flow_Peak':      mets['Q_peak'],
        'time_LVOT_Flow_Peak': mets['t_peak'],
        'ED': mets['EDur']
    }
    # compute weighted SSE
    sse = 0.0
    for key, w in weights.items():
        if key == 'LVEDP':
            continue
        s_val = sim.get(key, np.nan)
        gt_val = gt.get(key, np.nan)
        sse += w * ((s_val - gt_val)/gt_val)**2
    return sse

# ----------------------------------------
# Main entry
# ----------------------------------------
if __name__ == '__main__':
    # timing
    T = 60.0 / bpm           # time period of the cardiac cycle from bpm
    total = cycles * T       # total cardiac cycles for the simulation
    dt = T / 500.0           # sampling frequency of 500 Hz, dt = 2 ms

    # update params
    params['T'] = T          # T is added as the 10th model parameter

    # initial conditions
    V_sa0 = P_ao0 * params['C_sa']
    V_sv0 = params['V_tot'] - V_LV0 - V_sa0
    y0 = [V_LV0, V_sa0, V_sv0]

    # run sim
    t, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao, Q_sys = run_simulation(params, total, dt, y0)

    # plot elastance
    E_full = elastance(t, params)
    plt.figure()
    plt.plot(t, E_full)
    plt.title('Time-Varying Elastance (All Cycles)')
    plt.xlabel('Time [s]'); plt.ylabel('Elastance [mmHg/ml]'); plt.grid()

    # plot pressures
    plt.figure()
    plt.plot(t, P_ao, label='Aortic P')
    plt.plot(t, P_lv, label='LV P')
    plt.title('Pressure Waveforms')
    plt.xlabel('Time [s]'); plt.ylabel('mmHg'); plt.legend(); plt.grid()

    # plot flows
    plt.figure()
    plt.plot(t, Q_lv_ao, label='Ao Flow')
    plt.plot(t, Q_sv_lv, label='Mitral Flow')
    plt.plot(t, Q_sys, label='Sys Flow')
    plt.title('Flow Waveforms')
    plt.xlabel('Time [s]'); plt.ylabel('ml/s'); plt.legend(); plt.grid()

    # LV volume
    plt.figure()
    plt.plot(t, V_lv, label='LV Volume')
    plt.title('LV Volume')
    plt.xlabel('Time [s]'); plt.ylabel('ml'); plt.legend(); plt.grid()

    # PV loop
    plt.figure(); plt.plot(V_lv, P_lv)
    plt.title('Pressure-Volume Loop')
    plt.xlabel('V_lv [ml]'); plt.ylabel('P_lv [mmHg]'); plt.grid()

    # extract and plot metrics on cycle
    cyc = cycle_cutting_algo(t,bpm, P_ao, P_lv, V_lv, Q_sv_lv, Q_lv_ao)
    mets, idxs = extract_cycle_metrics(cyc)
    plot_cycle_with_metrics(cyc, mets, idxs)


    # print and compare
    print("Cycle Metrics:")
    for k, v in mets.items(): print(f"{k}: {v:.3f}")
    compare_with_gt(mets, cyc, gt_csv_path)
    # define weights for loss (example)
    weights = {
        'EDV': 0.5, 'ESV': 0.5, 'SV': 0.5, 'EF': 0.5,
        'bSBP': 1.5, 'bDBP': 1.5, 'bMAP': 1.5, 'bPP': 1.5,
        'LVOT_Flow_Peak': 1.0, 'time_LVOT_Flow_Peak': 1.0, 'ED': 0.5,
        'LVEDP': 0.0
    }
    loss = loss_all_matrices(mets, cyc, gt_csv_path, weights)
    print(f"Weighted SSE loss (excluding LVEDP): {loss:.3f}")
    plt.show()



