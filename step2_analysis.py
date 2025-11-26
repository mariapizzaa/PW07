import pandas as pd
import numpy as np
import os
from scipy.ndimage import median_filter

# --- SETTINGS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")

# Silva 2024 Targets (in meters)
# NOTE: We assume the Robot's Z coordinate is 1.0 meter.
TARGETS = {
    'NAO_Robot': {'x': 0.0, 'z': 1.0, 'width_m': 0.90},
    'Therapist': {'x': -1.0, 'z': 2.0, 'width_m': 0.45},
    'Screen': {'x': 1.0, 'z': 2.0, 'width_m': 0.50}
}

GAZE_NOISE_RAD = 0.069
MIN_FIXATION_MS = 400


def calculate_attention(df_input, group_label):
    df = df_input.copy()

    # 1. MM -> Meter Conversion
    if 'child_keypoint_x' in df.columns:
        df['x_m'] = df['child_keypoint_x'] / 1000.0
        df['z_m'] = df['child_keypoint_z'] / 1000.0

    # 2. Yaw Conversion (Radian Check)
    # If data is in degrees, convert to radians
    if df['yaw'].abs().max() > 3.20:
        df['yaw_rad'] = np.deg2rad(df['yaw'])
    else:
        df['yaw_rad'] = df['yaw']

    # 3. FPS Calculation
    if 'timestamp' in df.columns and len(df) > 1:
        mean_dt = df['timestamp'].diff().mean()
        fps = 30 if pd.isna(mean_dt) or mean_dt == 0 else 1.0 / mean_dt
    else:
        fps = 30

    min_frames = int((MIN_FIXATION_MS / 1000.0) * fps)
    if min_frames < 1: min_frames = 1

    # 4. Silva Geometric Analysis
    results_list = []
    subjects = df['id_soggetto'].unique()

    for subj in subjects:
        sub_df = df[df['id_soggetto'] == subj].sort_values('timestamp').copy()

        # Gaze check for each target
        for t_name, info in TARGETS.items():
            # Vector to target
            delta_x = info['x'] - sub_df['x_m']
            delta_z = info['z'] - sub_df['z_m']

            # --- CRITICAL FIX: 180 Degree (Pi) Shift ---
            # Since the robot is "in front" of the child (closer to the camera), the vector is in the -Z direction.
            # Normal arctan2 sees this as +/- 3.14 (behind).
            # However, the gaze estimator sees 0 as "looking at the camera".
            # Therefore, we subtract (or add) PI from the calculated angle so the coordinates align.
            target_angle = np.arctan2(delta_x, delta_z)

            # Normalize the angle (fit between -pi and +pi)
            # Here we rotate the target angle by 180 degrees so "0" points to the camera.
            target_angle_corrected = target_angle - np.pi
            target_angle_corrected = np.arctan2(np.sin(target_angle_corrected), np.cos(target_angle_corrected))

            # Dynamic Window (AOI)
            dist = np.sqrt(delta_x ** 2 + delta_z ** 2).replace(0, 0.1)
            aoi_width = np.arctan((info['width_m'] / 2.0) / dist) + GAZE_NOISE_RAD

            # Is looking?
            angle_diff = np.abs(sub_df['yaw_rad'] - target_angle_corrected)
            # Find the shortest path for the difference (e.g., difference between 350 and 10 degrees is 20)
            angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

            is_looking = (angle_diff <= aoi_width).astype(int)

            # 400ms Filter
            sub_df[f'attn_{t_name}'] = median_filter(is_looking, size=min_frames)

        # Aggregate statistics (TFD %)
        total_frames = len(sub_df)
        stats = {'Subject_ID': subj, 'Group': group_label}

        for t_name in TARGETS.keys():
            count = sub_df[f'attn_{t_name}'].sum()
            stats[f'TFD_{t_name}_%'] = (count / total_frames) * 100.0

        results_list.append(stats)

    return pd.DataFrame(results_list)


# --- EXECUTION ---
if os.path.exists(path_TD_clean) and os.path.exists(path_ASD_clean):
    print("Running analysis (Coordinate system corrected)...")
    df_TD = pd.read_excel(path_TD_clean)
    df_ASD = pd.read_excel(path_ASD_clean)

    summary_TD = calculate_attention(df_TD, "TD")
    summary_ASD = calculate_attention(df_ASD, "ASD")

    final_results = pd.concat([summary_TD, summary_ASD], ignore_index=True)

    output_path = os.path.join(base_dir, "Silva_Analysis_Results.xlsx")
    final_results.to_excel(output_path, index=False)

    print("\n[STEP 2 COMPLETED]")
    # Quick check: Show mean values
    print(final_results.groupby('Group')[['TFD_NAO_Robot_%']].mean())
    print("\nYou can now run the STEP 3 code to see the graph.")
else:
    print("ERROR: Cleaned files not found.")
