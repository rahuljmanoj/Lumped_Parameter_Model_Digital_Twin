
#!/usr/bin/env python3
"""
compute_and_compare_metrics.py

Simulate a 0D cardiovascular model over multiple cycles,
visualize full-simulation results and time-varying elastance.
Uses only P_AO0, V_LV0, and V_tot as inputs, with no V0 offset.
Intrathoracic pressure fixed at -4 mmHg.
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------
# Model parameters (can override V_tot)
# ----------------------------------------
BASE_PARAMS = {
    'R_mv':   0.0547,   # mitral valve resistance (mmHg·s/ml)
    'R_sys':  2.1396,   # systemic resistance
    'Z_ao':   0.0671,  # aortic impedance
    'C_sa':   0.3601,   # arterial compliance (ml/mmHg)
    'C_sv':   16.6097,   # venous compliance (ml/mmHg)
    'E_max':  7.8354,   # max elastance (mmHg/ml)
    'E_min':  0.3584,  # min elastance (mmHg/ml)
    'a':      1.55,   # elastance scaling factor
    'alpha1': 0.7,    # time scale α₁
    'alpha2': 1.17,   # time scale α₂
    'n1':     1.9,
    'n2':     21.9,
    't_peak':0.2972,     # time to peak elastance (s)
    'V_tot': 1021.6298    # total blood volume (ml)
}
P_th = -4.0  # intrathoracic pressure (mmHg)

# ----------------------------------------
# Elastance function
# ----------------------------------------
def elastance(t, p):
    """
    Time-varying elastance including alpha1, alpha2:
      E(t) = E_max * a * En + E_min
    with
      tn = (t mod T)/t_peak,
      t1 = tn/alpha1,
      t2 = tn/alpha2,
      En = (t1**n1)/(1+t1**n1) * 1/(1+(t2**n2))
    where t_peak = p['t_peak'], a = p['a'], alpha1 = p['alpha1'], alpha2 = p['alpha2']
    """
    t_peak = p['t_peak']
    a       = p['a']
    T       = p['T']
    # time normalized to t_peak
    tn      = np.mod(t, T) / t_peak
    alpha1  = p['alpha1']
    alpha2  = p['alpha2']
    n1      = p['n1']
    n2      = p['n2']
    # apply time scales
    t1 = tn / alpha1
    t2 = tn / alpha2
    # activation
    En = (t1**n1) / (1 + t1**n1)
    En *= 1.0 / (1 + t2**n2)
    # elastance
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
# Main entry
# ----------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--bpm', '65',
            '--cycles', '10',
            '--P_AO0', '100',
            '--V_LV0', '100',
            '--V_tot', '300'
        ])
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpm',    type=float, default=65,  help='Heart rate [bpm]')
    parser.add_argument('--cycles', type=int,   default=10,  help='Number of cycles')
    parser.add_argument('--P_AO0',  type=float, required=True, help='Initial aortic P [mmHg]')
    parser.add_argument('--V_LV0',  type=float, required=True, help='Initial LV volume [ml]')
    parser.add_argument('--V_tot',  type=float, required=True, help='Total blood volume [ml]')
    args = parser.parse_args()

    # timing
    T = 60.0 / args.bpm
    total = T * args.cycles
    dt = T / 500.0

    # update params
    params = BASE_PARAMS.copy()
    params['T']     = T
    params['V_tot'] = args.V_tot

    # initial volumes
    V_lv0 = args.V_LV0
    V_sa0 = args.P_AO0 * params['C_sa']
    V_sv0 = params['V_tot'] - V_lv0 - V_sa0
    y0 = [V_lv0, V_sa0, V_sv0]

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
    plt.figure()
    plt.plot(V_lv, P_lv)
    plt.title('Pressure-Volume Loop')
    plt.xlabel('V_lv [ml]'); plt.ylabel('P_lv [mmHg]'); plt.grid()

    plt.show()

