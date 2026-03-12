import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) SWS – Emin data (your first dataset)
# ---------------------------
SWS_emin = np.array([
    3.6, 4.733333333, 2.9, 2.966666667, 3.033333333, 2.7, 2.466666667, 2.966666667,
    3.483333333, 3.45, 5.3, 2.877777778, 2.1, 2.6, 2.066666667, 3.066666667,
    2.666666667, 2.4, 2.2, 2.15, 3.966666667, 2.916666667, 2.833333333, 2.15,
    6, 7.233333333, 3.666666667, 4.666666667, 4.2, 4.966666667, 3.894444444,
    3.55, 5.85, 4.733333333, 5.466666667, 6.733333333, 4.233333333, 3.966666667,
    4.4, 3.233333333, 3.866666667, 4.255555556, 3.466666667, 5.05, 4.683333333
])

Emin = np.array([
    0.504844894, 0.757422212, 0.310895144, 0.453671425, 0.20140014, 0.326118538,
    0.444050989, 0.47539305, 0.781839745, 0.585894883, 0.273242597, 0.537275184,
    0.077643247, 0.067316693, 0.038808455, 0.057018827, 0.071908533, 0.097780814,
    0.129414683, 0.09779066, 0.081983902, 0.03051376, 0.109185134, 0.093653953,
    0.651671917, 1.203725779, 0.659224425, 0.676985555, 0.580876238, 0.563677495,
    0.791283129, 0.777059192, 0.392970692, 0.709794442, 1.104491773, 1.737011132,
    0.628344781, 0.620331039, 0.748212651, 0.53959498, 0.62889814, 0.991644367,
    0.560979891, 0.386987555, 0.903532195
])

# sort for smooth plotting
idx_emin = np.argsort(SWS_emin)
SWS_emin_sorted = SWS_emin[idx_emin]

# your Excel regression for Emin:
Emin_pred = 0.2107 * SWS_emin_sorted - 0.2955
emin_interval = 0.5
emin_upper = Emin_pred + emin_interval
emin_lower = Emin_pred - emin_interval

# ---------------------------
# 2) SWS – LVEDP data (new dataset)
# ---------------------------
SWS_lvedp = np.array([
    3.595, 5.815, 3.707, 3.489, 4.808, 3.775, 9.263, 5.348, 2.902, 3.86,
    3.326, 4.005, 7.004, 5.15, 7.282, 3.369, 6.986, 5.875, 6.493, 6.509,
    3.89, 8.886, 7.867, 5.009, 3.492, 2.397, 8.471, 2.96, 3.122, 3.522,
    3.534, 3.181, 5.202, 2.557, 4.089, 3.086, 3.826, 1.732, 2.773, 5.476,
    4.505, 8.472, 7.823, 3.35, 8.208, 6.658, 7.877, 11.52, 2.833, 5.94,
    2.667, 10.195, 6.266, 12.407, 5.633, 5.795, 3.093, 3.907, 4.484, 10.923,
    4.548, 8.805, 10.508, 5.378, 11.352, 3.682, 1.203, 4.321
])

LVEDP = np.array([
    10.00, 24.00, 20.00, 14.00, 24.00, 7.00, 28.00, 16.00, 16.00, 26.00,
    14.00, 12.00, 24.00, 16.00, 36.00, 13.00, 18.00, 27.00, 23.00, 22.00,
    10.00, 22.00, 22.00, 20.00, 12.00, 12.00, 24.00, 14.00, 14.00, 12.00,
    12.00, 13.00, 26.00, 15.00, 10.00, 14.00, 12.00, 10.00, 12.00, 16.00,
    12.00, 24.00, 20.00, 12.00, 20.00, 22.00, 18.00, 24.00, 8.00, 16.00,
    8.00, 30.00, 20.00, 28.00, 18.00, 16.00, 12.00, 14.00, 14.00, 24.00,
    14.00, 16.00, 20.00, 25.00, 24.00, 14.00, 9.00, 14.00
])

# fit regression from THIS data
# (you could hardcode 1.7206*x + 8.1086 instead, but this keeps it tied to your data)
coef = np.polyfit(SWS_lvedp, LVEDP, 1)   # slope, intercept
slope_lv, intercept_lv = coef
# sort x for plotting
idx_lv = np.argsort(SWS_lvedp)
SWS_lvedp_sorted = SWS_lvedp[idx_lv]
LVEDP_pred = slope_lv * SWS_lvedp_sorted + intercept_lv

lvedp_interval = 2.5  # mmHg
lvedp_upper = LVEDP_pred + lvedp_interval
lvedp_lower = LVEDP_pred - lvedp_interval

# ---------------------------
# 3) Plot both
# ---------------------------
plt.figure(figsize=(8,6))
plt.scatter(SWS_emin, Emin, color='teal', label='Emin data')
plt.plot(SWS_emin_sorted, Emin_pred, 'k', lw=2, label='Emin prior: 0.2107x - 0.2955')
plt.fill_between(SWS_emin_sorted, emin_lower, emin_upper, alpha=0.15, label='Emin ±0.5')
plt.xlabel('SWE velocity (m/s)')
plt.ylabel('Emin (mmHg/ml)')
plt.title('SWS–Emin prior model from Pig study')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(SWS_lvedp, LVEDP, color='tab:blue', label='LVEDP data')
plt.plot(SWS_lvedp_sorted, LVEDP_pred, 'k', lw=2,
         label=f'LVEDP prior: y = {slope_lv:.3f}x + {intercept_lv:.3f}')
plt.fill_between(SWS_lvedp_sorted, lvedp_lower, lvedp_upper, alpha=0.15,
                 label='LVEDP ±2.5 mmHg')
plt.xlabel('SWE velocity (m/s)')
plt.ylabel('LVEDP (mmHg)')
plt.title('SWS–LVEDP prior with tolerance')
plt.legend()
plt.tight_layout()
plt.show()

print(f"SWS-LVEDP Prior model from human study: LVEDP = {slope_lv:.4f} * SWS + {intercept_lv:.4f}")
