import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os

# --- SETTINGS (Your specific file paths) ---
# Main directory where the code runs (PW2025)
base_dir = os.path.dirname(os.path.abspath(__file__))

# FULL paths where the files are located (Using r"..." for Windows paths)
path_TD_raw = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\results_cones_2D_TD_unlabelled_20_10_2025_after.xlsx"
path_ASD_raw = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\results_cones_2D_ASD_unlabelled_31_10_2025_after.xlsx"


def preprocess_data_silva(df_input, group_name):
    print(f"[{group_name}] Processing...")
    df = df_input.copy()

    # 1. Remove Duplicates
    df.drop_duplicates(inplace=True)

    # 2. Confidence Filtering (Remove low confidence data)
    if 'confidence_yaw' in df.columns:
        df.loc[df['confidence_yaw'] < 0.15, 'yaw'] = np.nan

    # 3. Remove Zeros (Sensor error)
    if 'child_keypoint_x' in df.columns:
        mask_zeros = df['child_keypoint_x'] == 0
        cols_3d = ['child_keypoint_x', 'child_keypoint_y', 'child_keypoint_z']
        df.loc[mask_zeros, [c for c in cols_3d if c in df.columns]] = np.nan

    # 4. Interpolation (Fill gaps - Max 60 frames)
    cols = ['yaw', 'child_keypoint_x', 'child_keypoint_z']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=60)

    # 5. Cleaning (Drop remaining NaNs)
    # Yaw and x/z coordinates are mandatory for Silva analysis
    df_clean = df.dropna(subset=['yaw', 'child_keypoint_x']).copy()

    # 6. Smoothing (Median Filter)
    for col in cols:
        if col in df_clean.columns:
            try:
                df_clean[col] = medfilt(df_clean[col].values, 5)
            except:
                pass

    print(f"-> Dropped from {len(df_input)} rows to {len(df_clean)} rows.")
    return df_clean


# --- EXECUTION ---
print("Searching for files...")

if os.path.exists(path_TD_raw) and os.path.exists(path_ASD_raw):
    print("Files found! Loading...")
    df_TD = pd.read_excel(path_TD_raw)
    df_ASD = pd.read_excel(path_ASD_raw)

    df_TD_clean = preprocess_data_silva(df_TD, "TD")
    df_ASD_clean = preprocess_data_silva(df_ASD, "ASD")

    # Save clean files to the MAIN script location (inside PW2025)
    # So 2nd and 3rd codes can find them easily.
    out_td = os.path.join(base_dir, "TD_cleaned.xlsx")
    out_asd = os.path.join(base_dir, "ASD_cleaned.xlsx")

    df_TD_clean.to_excel(out_td, index=False)
    df_ASD_clean.to_excel(out_asd, index=False)

    print("\n[STEP 1 SUCCESSFUL]")
    print(f"Clean files saved to:\n1. {out_td}\n2. {out_asd}")
    print("You can now run Code 2.")
else:
    print("ERROR: Files still not found. Please ensure the file path/name is exactly correct.")
    print(f"Path searched 1: {path_TD_raw}")
