import os, json
import numpy as np
import pandas as pd

def _safe_int(s):
    try: return int(s)
    except Exception: return None


def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None

def load_subject_artifacts(uqsa_root):
    """
    Load UQ and Sobol artifacts for all subjects in uqsa_root.

    Expected files per subject folder:
      <id>_uq_summary.json
      <id>_uq_params.csv
      <id>_mc_invalid_params.csv

      <id>_sobol_params.csv
      <id>_sobol_invalid_params.csv        (Sobol on LVEDP)
      <id>_sobol_lvedp.csv                 (Sobol indices on LVEDP)

      <id>_sobol_validity_invalid_params.csv (Sobol on VALIDITY)
      <id>_sobol_validity.csv               (Sobol indices on VALIDITY)
    """
    rows_summary = []
    rows_fail = []
    sa_frames = []

    for entry in os.scandir(uqsa_root):
        if not entry.is_dir():
            continue
        sub_id = _safe_int(entry.name)
        if sub_id is None:
            continue

        folder = entry.path

        # UQ files
        f_summary = os.path.join(folder, f"{sub_id}_uq_summary.json")
        f_uq_X    = os.path.join(folder, f"{sub_id}_uq_params.csv")
        f_uq_inv  = os.path.join(folder, f"{sub_id}_mc_invalid_params.csv")

        # Sobol common params
        f_sa_X        = os.path.join(folder, f"{sub_id}_sobol_params.csv")
        # Sobol outputs
        f_sa_lvedp    = os.path.join(folder, f"{sub_id}_sobol_lvedp.csv")
        f_sa_valid    = os.path.join(folder, f"{sub_id}_sobol_validity.csv")
        # Sobol invalid param logs
        f_sa_inv_lvedp = os.path.join(folder, f"{sub_id}_sobol_invalid_params.csv")
        f_sa_inv_valid = os.path.join(folder, f"{sub_id}_sobol_validity_invalid_params.csv")

        # ---- UQ summary ----
        if os.path.isfile(f_summary):
            try:
                with open(f_summary, "r") as fp:
                    s = json.load(fp)
                pi = s.get("PI95") or [np.nan, np.nan]
                rows_summary.append({
                    "sub_id": sub_id,
                    "N": s.get("N", np.nan),
                    "mean": s.get("mean", np.nan),
                    "sd": s.get("sd", np.nan),
                    "PI95_low": pi[0],
                    "PI95_high": pi[1],
                })
            except Exception as e:
                print(f"[warn] could not parse summary for {sub_id}: {e}")

        # ---- Sobol results (LVEDP and VALIDITY) ----
        # LVEDP Sobol file
        if os.path.isfile(f_sa_lvedp):
            try:
                df_sa_l = pd.read_csv(f_sa_lvedp)
                if "sub_id" not in df_sa_l.columns:
                    df_sa_l["sub_id"] = sub_id
                if "metric" not in df_sa_l.columns:
                    df_sa_l["metric"] = "LVEDP"
                sa_frames.append(df_sa_l)
            except Exception as e:
                print(f"[warn] could not read sobol LVEDP csv for {sub_id}: {e}")

        # VALIDITY Sobol file
        if os.path.isfile(f_sa_valid):
            try:
                df_sa_v = pd.read_csv(f_sa_valid)
                if "sub_id" not in df_sa_v.columns:
                    df_sa_v["sub_id"] = sub_id
                if "metric" not in df_sa_v.columns:
                    df_sa_v["metric"] = "VALIDITY"
                sa_frames.append(df_sa_v)
            except Exception as e:
                print(f"[warn] could not read sobol VALIDITY csv for {sub_id}: {e}")

        # ---- invalid accounting: UQ ----
        if os.path.isfile(f_uq_X):
            n_all = max(sum(1 for _ in open(f_uq_X)) - 1, 0)
            n_bad = max(sum(1 for _ in open(f_uq_inv)) - 1, 0) if os.path.isfile(f_uq_inv) else 0
            rows_fail.append({
                "sub_id": sub_id,
                "phase": "UQ",
                "n_all": n_all,
                "n_bad": n_bad,
                "bad_pct": (100.0 * n_bad / n_all) if n_all > 0 else np.nan
            })

        # ---- invalid accounting: Sobol LVEDP ----
        if os.path.isfile(f_sa_X):
            n_all_sa = max(sum(1 for _ in open(f_sa_X)) - 1, 0)

            if os.path.isfile(f_sa_inv_lvedp):
                n_bad_l = max(sum(1 for _ in open(f_sa_inv_lvedp)) - 1, 0)
                rows_fail.append({
                    "sub_id": sub_id,
                    "phase": "SA_LVEDP",
                    "n_all": n_all_sa,
                    "n_bad": n_bad_l,
                    "bad_pct": (100.0 * n_bad_l / n_all_sa) if n_all_sa > 0 else np.nan
                })

            if os.path.isfile(f_sa_inv_valid):
                n_bad_v = max(sum(1 for _ in open(f_sa_inv_valid)) - 1, 0)
                rows_fail.append({
                    "sub_id": sub_id,
                    "phase": "SA_VALIDITY",
                    "n_all": n_all_sa,
                    "n_bad": n_bad_v,
                    "bad_pct": (100.0 * n_bad_v / n_all_sa) if n_all_sa > 0 else np.nan
                })

    # ---- build frames ----
    df_summary = (pd.DataFrame(
        rows_summary,
        columns=["sub_id", "N", "mean", "sd", "PI95_low", "PI95_high"]
    ).sort_values("sub_id"))

    if sa_frames:
        df_sa = pd.concat(sa_frames, ignore_index=True)
    else:
        df_sa = pd.DataFrame(
            columns=["sub_id", "metric", "param",
                     "S1", "S1_low", "S1_high",
                     "ST", "ST_low", "ST_high"]
        )

    if not df_sa.empty:
        # Sort by subject, then metric, then parameter
        sort_cols = [c for c in ["sub_id", "metric", "param"] if c in df_sa.columns]
        df_sa = df_sa.sort_values(sort_cols)

    df_fail = (pd.DataFrame(
        rows_fail,
        columns=["sub_id", "phase", "n_all", "n_bad", "bad_pct"]
    ).sort_values(["sub_id", "phase"]))

    return df_summary, df_sa, df_fail


if __name__ == "__main__":
    uqsa_root = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Results_UQSA_Population_Level\UQ_SA"

    df_summary, df_sa, df_fail = load_subject_artifacts(uqsa_root)

    print("summary shape:", df_summary.shape, "| columns:", list(df_summary.columns))
    print("sobol   shape:", df_sa.shape, "| columns:", list(df_sa.columns))
    print("fail    shape:", df_fail.shape, "| columns:", list(df_fail.columns))

    # invalid rate wide table now has UQ, SA_LVEDP, SA_VALIDITY
    fail_wide = (df_fail
                 .pivot_table(index="sub_id", columns="phase", values="bad_pct", aggfunc="mean")
                 .rename_axis(None, axis=1)
                 .reset_index())

    df_report = (df_summary
                 .merge(fail_wide, on="sub_id", how="left")
                 .merge(df_sa, on="sub_id", how="left"))

    # save if needed
    out_dir = uqsa_root
    df_summary.to_csv(os.path.join(out_dir, "_summary_per_subject.csv"), index=False)
    df_sa.to_csv(os.path.join(out_dir, "_sobol_long.csv"), index=False)
    df_fail.to_csv(os.path.join(out_dir, "_invalid_rates.csv"), index=False)
    df_report.to_csv(os.path.join(out_dir, "_report_joined.csv"), index=False)

