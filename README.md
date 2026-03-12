# Lumped Parameter Model for LV Pressure–Volume Loop Simulation and LVEDP Estimation (V4.1)

This repository contains the **V4.1 implementation of the Lumped Parameter Model (LPM)** used for:

- **Left‑ventricular (LV) pressure–volume (PV) loop simulation**
- **Parameter estimation (PE)** to match clinical or experimental ground‑truth hemodynamics  
- **Uncertainty Quantification & Sensitivity Analysis (UQSA)**
- **Local Sensitivity Analysis (SA)** including preload/afterload/elastance perturbations
- **Pulse Wave and cardiovascular mechanics analysis**
- **Statistical validation and performance assessments**

This model is used to **estimate LV End‑Diastolic Pressure (LVEDP)**, derive **contractility/diastolic indices**, and explore parameter identifiability and physiological variability.

---

# 1. Overview of the Model

The cardiovascular model is a **0D lumped‑parameter representation** of the LV–systemic circulation, using:

- **Time‑varying elastance** representation of LV contraction  
- **Tri-compartment flow model** (LV, systemic arterial, venous compartments)  
- **Valve dynamics represented by flow-dependent resistances**  
- **Literature-aligned definitions** for ED, ES, EDP, ESP  from Bezy et al (2025), Caenen et al. (2025) & Burkoff et al. (2015) 
- **Identifiable hemodynamic parameters**:

| Parameter | Meaning |
|----------|---------|
| R_sys | Systemic vascular resistance |
| Z_ao | Aortic characteristic impedance |
| C_sa | Systemic arterial compliance |
| R_mv | Mitral valve resistance |
| E_max | Peak LV elastance |
| E_min | Baseline LV elastance |
| t_peak | Peak systolic timing |
| V_tot | Total blood volume (LV + arterial + venous) |
| C_sv | Systemic venous compliance |

The model integrates forward in time using `solve_ivp`, and extracts physiologically aligned metrics including:

- EDV, ESV, SV, EF  
- LVEDP (dP/dt upstroke definition)  
- ESP (max P/V during AVO→AVC)  
- IVRT, ET, Ejection Duration  
- Peak aortic flow, timing, pressures  

---

### Main Scripts

---

## **2.1 Parameter Estimation (PE)**
**File:** `LPM_V4.1_Parameter_Estimation.py`  
**Purpose:**
- Subject‑specific parameter estimation using **Differential Evolution (DE)** + **L‑BFGS‑B**  
- Multi‑metric weighted loss with:
  - Cohort-based normalization  
  - Physiological range penalty  
  - LVEDP soft prior (MAP-based)  
  - Paper-aligned ED/ES definitions  
- Saves:
  - DE-only results  
  - Final optimized results  
  - Plots (flows, pressures, PV loops)  
  - Incremental CSV logs

---

## **2.2 Uncertainty Quantification & Sensitivity Analysis (UQSA)**
**File:** `LPM_V4.1_UQSA_Subject_Specific.py`  

Includes:
- **Monte-Carlo UQ (75k draws)**  
- Sampling in **z‑space** (normalized 0–1 transform)  
- **Physiological validity filtering**  
- **Saltelli Sobol analysis** for:
  - LVEDP (median replacement for invalids)
  - Model validity probability  
- Subject-specific parameter bounds using ±10% windows  
- Saves:
  - full Theta, Z samples  
  - LVEDP distributions  
  - Sobol indices + bootstrap CIs  
  - invalid parameter traces  

---

## **2.3 Local Sensitivity: Preload / Afterload / Elastance**
**File:** `LPM_V4.1_Local_Sensitivity_Analysis_PV-Loop.py`  

Features:
- Single‑subject PV loop sensitivity under:
  - Preload perturbations (V_tot, C_sv, R_mv)  
  - Afterload perturbations (R_sys, C_sa, Z_ao)  
  - Elastance axis perturbations (E_min, E_max, t_peak)  
  - HR sensitivity (HR ±10% with fixed t_peak)  
- Quantifies:
  - Changes in EDV, ESV, SV, EF  
  - Aortic pressures (central/brachial)  
  - LVEDP, ESP  
- Produces PV overlays & CSV metric tables.

---

# 3. V4.1 Major Updates

The V4.1 release includes significant methodological improvements:

### **Parameter Estimation (PE)**
1. Save subject-wise CSV after each optimization (append mode)  
2. Save DE and L‑BFGS‑B losses + time taken  
3. Save DE‑only optimized values and simulation output  
4. Integration of new R_sys, Z_ao, C_sa parameters from **KNN Virtual Patient Matching**  
5. New **hinge‑to‑bound physiological penalty** instead of hard penalties  
6. MAP-based LVEDP soft prior using SWE velocity (regression RMSE-based)  
7. Cohort mean/minmax normalization of metric error; Hybrid WLS option  
8. **E_min fixed per subject** to isotonic calibrated value before PE  
9. Cleanup of redundant variables; add **IVRT** and use **ET** in loss  
10. Removal of LVOT peak flow/time from loss  
11. Full parameter normalization to z-space during optimization  
12. Paper-aligned ED/ES definitions:
    - ED: dp/dt upstroke  
    - ES: max(P/V)

---

### **Additional Codes in V4.1**
- **k-fold validation** of isotonic-calibrated E_min  
- **KNN-based feature matching** between in-silico and clinical datasets  
- **Local sensitivity analysis** of preload/afterload on PV loops  
- **Visualisation of prior distributions** for UQSA  
- **Pulse wave analysis** from simulated dataset  
- **Statistical analysis:** ROC, correlations, Bland‑Altman  
- **UQSA per subject & population-level**  
- **Visualization tools** for UQSA results

---

Outputs:

*_incremental.csv
*_DE_only.csv
Diagnostic plots
Cycle metrics
PV loops

**For UQSA**
Saves UQ summaries and Sobol indices per subject
Requires Excel input (set EXCEL_PATH)

**For Local Sensitivity Analysis and Preload/Afterload Effects**Generates:
PV overlays for each parameter perturbation
Sensitivity tables
HR perturbation effects

