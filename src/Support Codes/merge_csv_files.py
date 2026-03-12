import pandas as pd
import glob
import os

# Folder with your 34 CSV files
input_folder = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Missing_Data_Imputation_GT_matrices"   # <-- change if your folder is different
output_file = r"C:\Workspace\Post_Doc_Works_NTNU\Projects\2_SWE_Velocity_LV_Filling_Pressure_Digital_Twin\3_Codes\Python\Dataset_Python\Invasive_Study_Leuven_GT_matrices_all_subjects.csv"

# Find all csv files in the folder
file_list = glob.glob(os.path.join(input_folder, '*.csv'))

# This will store each subject's row as a dict
all_data = []

for file in file_list:
    df = pd.read_csv(file)
    # convert Metric/Value columns to a row-dict
    row = {row['Metric']: row['Value'] for _, row in df.iterrows()}
    # Get subject ID from filename
    sub_id = os.path.splitext(os.path.basename(file))[0]
    row['sub_id'] = sub_id
    all_data.append(row)

# Build the merged DataFrame
merged_df = pd.DataFrame(all_data)

# Move sub_id to the first column
cols = list(merged_df.columns)
cols.insert(0, cols.pop(cols.index('sub_id')))
merged_df = merged_df[cols]

# Save to csv
merged_df.to_csv(output_file, index=False)

print(f"Saved merged data to: {output_file}")
