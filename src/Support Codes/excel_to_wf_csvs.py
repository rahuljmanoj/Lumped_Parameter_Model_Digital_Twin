#!/usr/bin/env python3
import os
import pandas as pd

# ----------------------------------------
# Configuration: set your paths here
# ----------------------------------------
# Full path to the Excel file you want to convert
input_excel = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Sub_3199_316.xlsx"

# (Optional) sheet name or index, if not the first sheet
sheet_name = None  # e.g. 'Sheet1' or 0 for first sheet

# Output directory where CSVs will be saved (defaults to folder of the Excel file)
output_dir = os.path.dirname(input_excel)
# ----------------------------------------


def excel_to_csvs(excel_path, sheet_name, output_dir):
    """
    Reads the specified Excel file and writes out three CSVs:
      - time_wf.csv      (Time values)
      - flow_wf.csv      (Flow Rate values)
      - pressure_wf.csv  (Pressure values)
    """
    # Read Excel (may return dict if multiple sheets)
    data = pd.read_excel(excel_path, sheet_name=sheet_name)
    if isinstance(data, dict):
        df = data[next(iter(data))]
    else:
        df = data

    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()

    # Required columns
    required_cols = ['Time (s)', 'Flow Rate (ml/s)', 'Pressure (mmHg)']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print("Error: the following required columns are missing:", missing)
        print("Available columns in sheet:", list(df.columns))
        raise ValueError(f"Missing columns: {missing}")

    # Extract series
    time_series     = df['Time (s)']
    flow_series     = df['Flow Rate (ml/s)']
    pressure_series = df['Pressure (mmHg)']

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    time_path     = os.path.join(output_dir, 'Sub_3199_316_time_wf.csv')
    flow_path     = os.path.join(output_dir, 'Sub_3199_316_flow_wf.csv')
    pressure_path = os.path.join(output_dir, 'Sub_3199_316_pressure_wf.csv')

    # Save CSVs (no header, no index)
    time_series.to_csv(time_path,     index=False, header=False)
    flow_series.to_csv(flow_path,     index=False, header=False)
    pressure_series.to_csv(pressure_path, index=False, header=False)

    print("Generated CSV files:")
    print(" -", time_path)
    print(" -", flow_path)
    print(" -", pressure_path)


if __name__ == '__main__':
    excel_to_csvs(input_excel, sheet_name, output_dir)
