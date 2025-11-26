import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os
# --- 1. DATASET LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_dir, "dataset iniziali e risultati")
filename_TD = "results_cones_2D_TD_unlabelled_20_10_2025_after.xlsx"
filename_ASD = "results_cones_2D_ASD_unlabelled_31_10_2025_after.xlsx"
# Define file paths (ensure these match your local directory)
path_TD = os.path.join(data_folder, filename_TD)
path_ASD = os.path.join(data_folder, filename_ASD)
try:
    df_TD = pd.read_excel(path_TD)
    df_ASD = pd.read_excel(path_ASD)
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"File error: {e}")
    exit()


# --- 2. PREPROCESSING FUNCTION ---
def preprocess_data_final(df_input, group_name):
    """
    Applies the preprocessing pipeline: Duplicate removal -> Confidence Filtering ->
    Invalid 3D point removal -> Linear Interpolation -> Smoothing.
    """
    # Create an explicit copy to avoid SettingWithCopyWarning
    df = df_input.copy()

    # --- PARAMETERS (Rationale explained below) ---
    # Threshold lowered to 0.15 because the diagnosis showed mean confidence is ~0.26.
    # A stricter threshold (e.g., 0.5) would discard >80% of data.
    CONFIDENCE_THRESHOLD = 0.15

    # Interpolation limit set to 60 frames (approx. 2 seconds at 30fps).
    # Allows recovering short data losses due to movement or occlusion.
    LIMIT_FRAMES = 60

    # Kernel size for Median Filter. Must be odd.
    # Used to reduce jitter/noise from low-confidence data.
    KERNEL_SIZE = 5

    print(f"\nProcessing group: {group_name}...")

    # A. REMOVE DUPLICATES
    # Removes identical rows which might result from logging errors.
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} duplicate rows.")

    # B. CONFIDENCE FILTERING
    # Set gaze angles to NaN if the algorithm is not confident enough[cite: 1406, 1407].
    mask_yaw = df['confidence_yaw'] < CONFIDENCE_THRESHOLD
    mask_pitch = df['confidence_pitch'] < CONFIDENCE_THRESHOLD

    df.loc[mask_yaw, 'yaw'] = np.nan
    df.loc[mask_pitch, 'pitch'] = np.nan

    # C. SENSOR ERROR FILTERING (Zero Removal)
    # A head position of exactly (0,0,0) usually indicates tracking loss, not real position.
    mask_zeros = df['child_keypoint_x'] == 0
    df.loc[mask_zeros, ['child_keypoint_x', 'child_keypoint_y', 'child_keypoint_z']] = np.nan

    # D. LINEAR INTERPOLATION
    # Fills the NaN gaps created above using a linear method.
    cols_to_interpolate = ['yaw', 'pitch', 'child_keypoint_x', 'child_keypoint_y', 'child_keypoint_z']
    for col in cols_to_interpolate:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=LIMIT_FRAMES)

    # E. STRATEGIC DROPNA
    # We drop a row ONLY if gaze data (yaw/pitch) is still missing after interpolation.
    # We keep the row even if 3D keypoints are missing, as gaze is the primary metric.
    df_clean = df.dropna(subset=['yaw', 'pitch']).copy()

    # F. SMOOTHING (Median Filter)
    # Applies a median filter to remove "spikes" and noise.
    for col in cols_to_interpolate:
        if col in df_clean.columns:
            try:
                df_clean.loc[:, col] = medfilt(df_clean[col].values, KERNEL_SIZE)
            except Exception as e:
                print(f"Smoothing error on column {col}: {e}")

    # Final Statistics
    lost_rows = initial_len - len(df_clean)
    perc_lost = (lost_rows / initial_len) * 100
    print(f"Original rows: {initial_len} -> Cleaned rows: {len(df_clean)}")
    print(f"Data lost: {lost_rows} ({perc_lost:.1f}%)")

    return df_clean


# --- 3. EXECUTION ---
df_TD_clean = preprocess_data_final(df_TD, "TD")
df_ASD_clean = preprocess_data_final(df_ASD, "ASD")

df_TD_clean.to_excel("TD_cleaned.xlsx", index=False)
df_ASD_clean.to_excel("ASD_cleaned.xlsx", index=False)
