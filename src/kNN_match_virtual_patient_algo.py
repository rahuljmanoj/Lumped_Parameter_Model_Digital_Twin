
import pandas as pd, numpy as np

# Files you provided

v = pd.read_csv(r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\pwdb_haemod_params.csv", skipinitialspace=True)
p = pd.read_csv(r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\Invasive_Study_Leuven_GT_matrices_all_subjects.csv", skipinitialspace=True)


v.columns = [c.strip() for c in v.columns]
p.columns = [c.strip() for c in p.columns]

# ----- choose variables (brachial SBP/DBP) -----
# Virtual columns (fall back to SBP_a/DBP_a if needed)
vu = v.copy()
if "SBP" not in vu.columns:
    for alt in ["SBP_a [mmHg]", "SBP_a"]:
        if alt in vu.columns: vu.rename(columns={alt: "SBP"}, inplace=True)
if "DBP" not in vu.columns:
    for alt in ["DBP_a [mmHg]", "DBP_a"]:
        if alt in vu.columns: vu.rename(columns={alt: "DBP"}, inplace=True)
vu = vu[["ref_sub_id","age","HR","SV","SBP","DBP"]].dropna()

# Patient columns (map brachial bSBP/bDBP to SBP/DBP)
p_map = {"sub_id":"sub_id","ref_sub_id":"ref_sub_id","age":"age","HR":"HR","SV":"SV","bSBP":"SBP","bDBP":"DBP"}
src = [s for s in p_map if s in p.columns]
pu = p[src].copy().rename(columns={s:d for s,d in p_map.items() if s in p.columns})
pu = pu.dropna(subset=["sub_id","age","SBP","DBP","SV","HR"])

cols = ["age","SBP","DBP","SV","HR"]

# ----- z-score standardization on the virtual cohort -----
center = vu[cols].mean()
scale  = vu[cols].std(ddof=0).replace(0, 1.0)
Vz = (vu[cols] - center) / scale
Pz = (pu[cols] - center) / scale

# ----- Euclidean distance in z-space -----
VV = (Vz.values**2).sum(axis=1)
PP = (Pz.values**2).sum(axis=1)
D2 = PP[:,None] + VV[None,:] - 2*(Pz.values @ Vz.values.T)
D = np.sqrt(np.maximum(D2, 0.0))

# ----- retrieve top-3 (reuse allowed) -----
K = 1
idx  = np.argsort(D, axis=1)[:, :K]
dist = np.take_along_axis(D, idx, axis=1)

# ----- diagnostics -----
v_min, v_max = vu[cols].min(), vu[cols].max()
rows = []
for i, prow in pu.reset_index(drop=True).iterrows():
    outside = [c for c in cols if (prow[c] < v_min[c]) or (prow[c] > v_max[c])]
    for r in range(K):
        j = idx[i, r]
        vrow = vu.iloc[j]
        asd  = ((vrow[cols] - prow[cols]).abs() / scale).to_dict()
        diffs = (vrow[cols] - prow[cols]).to_dict()
        rows.append({
            "patient_sub_id": prow["sub_id"],
            "patient_ref_sub_id": prow.get("ref_sub_id", np.nan),
            "virtual_ref_sub_id": vrow["ref_sub_id"],
            "rank": r+1,
            "distance_z": float(dist[i, r]),
            "mean_ASD": float(np.mean(list(asd.values()))),
            "max_ASD": float(np.max(list(asd.values()))),
            "patient_out_of_support": bool(len(outside)>0),
            "vars_outside_support": ";".join(outside) if outside else ""
        } | {f"patient_{c}": prow[c] for c in cols}
          | {f"virtual_{c}": vrow[c] for c in cols}
          | {f"diff_{c}": diffs[c] for c in cols}
          | {f"ASD_{c}":  asd[c] for c in cols})

matches = pd.DataFrame(rows)
matches.to_csv(r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Data_Results\patient_virtual_1NN_matches.csv", index=False)
