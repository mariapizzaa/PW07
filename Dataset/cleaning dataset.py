import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os
<<<<<<< Updated upstream
# --- 1. DATASET LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_dir, "dataset iniziali e risultati")
filename_TD = "results_cones_2D_TD_unlabelled_20_10_2025_after.xlsx"
filename_ASD = "results_cones_2D_ASD_unlabelled_31_10_2025_after.xlsx"
# Define file paths (ensure these match your local directory)
path_TD = os.path.join(data_folder, filename_TD)
path_ASD = os.path.join(data_folder, filename_ASD)
=======

# --- 1. PATH CONFIGURATION ---
# Specific path requested for OneDrive
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Construct full file paths
path_TD = os.path.join(BASE_PATH, "results_cones_2D_TD_unlabelled_20_10_2025_after.xlsx")
path_ASD = os.path.join(BASE_PATH, "results_cones_2D_ASD_unlabelled_31_10_2025_after.xlsx")

# Safety check to prevent crashes
if not os.path.exists(path_TD) or not os.path.exists(path_ASD):
    print(f"ERROR: Files not found in folder: {BASE_PATH}")
    print("Please check the filenames and directory path.")
    exit()

>>>>>>> Stashed changes
try:
    print("Loading datasets...")
    df_TD = pd.read_excel(path_TD)
    df_ASD = pd.read_excel(path_ASD)
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading Excel files: {e}")
    exit()


# --- 2. ADVANCED PREPROCESSING FUNCTION ---
def preprocess_data_advanced(df_input, group_name):
    """
    Advanced Preprocessing Pipeline:
    1. Duplicate Removal (Data Cleaning)
    2. Confidence Filtering (Noise Reduction)
    3. Sensor Error Removal (Zero filtering)
    4. Z-Score Filter (Outlier Detection)
    5. Linear Interpolation (Missing Values Imputation)
    6. Smoothing (Signal Denoising)
    """
    df = df_input.copy()
    initial_len = len(df)

    print(f"\n--- Processing Group: {group_name} ({initial_len} rows) ---")

    # A. DATA CLEANING: Remove Duplicates
    # Essential for correct velocity calculations (dt > 0)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_len:
        print(f"   - Duplicates removed: {initial_len - len(df)}")

    # B. NOISE REDUCTION: Confidence Filter
    # Set unreliable data to NaN so it can be interpolated later
    CONFIDENCE_THRESHOLD = 0.15
    mask_low_conf = (df['confidence_yaw'] < CONFIDENCE_THRESHOLD) | (df['confidence_pitch'] < CONFIDENCE_THRESHOLD)

    cols_gaze = ['yaw', 'pitch']
    df.loc[mask_low_conf, cols_gaze] = np.nan

    # C. SENSOR ERROR FILTERING: Remove Zeros
    # Specific sensor error where 3D coordinates default to exactly 0
    cols_3d = ['child_keypoint_x', 'child_keypoint_y', 'child_keypoint_z']
    mask_zeros = df['child_keypoint_x'] == 0
    df.loc[mask_zeros, cols_3d] = np.nan

    # D. OUTLIER DETECTION: Z-Score Method
    # Removes statistically improbable values (e.g., head teleporting)
    # Threshold: 3 Standard Deviations
    for col in cols_3d:
        if col in df.columns:
            col_data = df[col]
            # Calculate Z-score ignoring NaNs
            z_scores = np.abs((col_data - np.nanmean(col_data)) / np.nanstd(col_data))
            outliers = z_scores > 3
            # If outliers are found, set them to NaN
            if outliers.sum() > 0:
                df.loc[outliers, col] = np.nan
                # print(f"   - Outliers removed in {col}: {outliers.sum()}")

    # E. MISSING VALUES IMPUTATION: Linear Interpolation
    # We limit interpolation to 60 frames (approx. 2 seconds) to avoid inventing data
    LIMIT_FRAMES = 60
    cols_to_process = cols_gaze + cols_3d

    for col in cols_to_process:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=LIMIT_FRAMES)

    # F. DATA REDUCTION:
    # We keep the row ONLY if we have valid gaze data.
    # If 3D keypoints are missing but gaze is valid, the row is still useful.
    df_clean = df.dropna(subset=cols_gaze).copy()

    # G. SMOOTHING: Median Filter
    # Reduces sensor jitter while preserving signal edges (saccades)
    KERNEL_SIZE = 5
    for col in cols_to_process:
        if col in df_clean.columns:
            try:
                df_clean.loc[:, col] = medfilt(df_clean[col].values, KERNEL_SIZE)
            except Exception as e:
                print(f"   ! Smoothing error on {col}: {e}")

    # Final Statistics
    final_len = len(df_clean)
    perc_kept = (final_len / initial_len) * 100
    print(f"   > Final rows: {final_len} ({perc_kept:.1f}% of original data)")

    return df_clean


# --- 3. EXECUTION ---
df_TD_clean = preprocess_data_advanced(df_TD, "TD")
df_ASD_clean = preprocess_data_advanced(df_ASD, "ASD")

# --- 4. SAVING ---
# Save files in the same directory as the originals
output_TD = os.path.join(BASE_PATH, "TD_cleaned_advanced.xlsx")
output_ASD = os.path.join(BASE_PATH, "ASD_cleaned_advanced.xlsx")

print("\nSaving files...")
try:
    df_TD_clean.to_excel(output_TD, index=False)
    df_ASD_clean.to_excel(output_ASD, index=False)
    print(f"Done! Files saved in: {BASE_PATH}")
    print(f"   - {os.path.basename(output_TD)}")
    print(f"   - {os.path.basename(output_ASD)}")
except Exception as e:
    print(f"Error saving files: {e}")
