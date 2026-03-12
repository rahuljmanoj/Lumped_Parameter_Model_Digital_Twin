#!/usr/bin/env python3
"""
generate_gt_metrics.py

Generates a ground-truth metrics CSV for your cost-function evaluation.
"""

import os
import pandas as pd

def main():
    # ----------------------------
    # User-set ground-truth values
    # ----------------------------
    EDV = 90    # End-diastolic volume [ml]
    ESV = 37  # End-systolic volume [ml]
    bSBP = 174 # Brachial Systolic BP [mmHg]
    bDBP = 91     # Brachial Diastolic BP [mmHg]

    LVOT_Flow_Peak = 357.184 # Peak LVOT flow rate [ml/s]
    time_LVOT_Flow_Peak = 0.078  # Time of peak LVOT flow [s]
    ED = 0.283152 # Diastolic duration [s]

    LVEDP = 14    # LV end-diastolic pressure [mmHg]

    HR = 82       # Heart Rate [bpm]

    PWV = 13.872 # Aortic PWV [m/s]
    Z_ao = 0.0901646 # Characteristric Impedance [mmHg s /ml]
    R_sys = 1.63829 # Systemic Resistance [mmHg s /ml]
    C_sa = 0.387568  # Aoritc Compliance [ml/mmHg]

    SWE_vel_MVC = 4.3214 # SWE Velocity after MVC [m/s]

    # ----------------------------
    # Derived quantities
    # ----------------------------
    SV   = EDV - ESV                    # Stroke volume [ml]
    EF   = SV / EDV * 100               # Ejection fraction [%]
    bMAP = (bSBP + 2 * bDBP) / 3        # Mean arterial pressure approximation [mmHg]
    bPP  = bSBP - bDBP                  # Pulse pressure [mmHg]

    # ----------------------------
    # Assemble into a DataFrame
    # ----------------------------
    metrics = {
        'Metric': [
            'EDV', 'ESV', 'SV', 'EF',
            'bSBP', 'bDBP', 'bMAP', 'bPP',
            'LVOT_Flow_Peak', 'time_LVOT_Flow_Peak', 'ED', 'LVEDP', 'HR', 'PWV', 'Z_ao', 'R_sys', 'C_sa', 'SWE_vel_MVC'
        ],
        'Value': [
            EDV, ESV, SV, EF,
            bSBP, bDBP, bMAP, bPP,
            LVOT_Flow_Peak, time_LVOT_Flow_Peak, ED, LVEDP, HR, PWV, Z_ao, R_sys, C_sa, SWE_vel_MVC
        ]
    }
    df = pd.DataFrame(metrics)

    # ----------------------------
    # Save to CSV in the target directory
    # ----------------------------
    output_dir = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Missing_Data_Imputation_GT_matrices"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "Sub_4141_415_gt_metrics.csv")

    df.to_csv(output_file, index=False)
    print(f"Ground-truth metrics written to:\n  {output_file}")

if __name__ == '__main__':
    main()
