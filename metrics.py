import pandas as pd
import numpy as np
import os

# --- 1. SETUP PATHS ---
# Uses relative paths assuming this script is in the same folder as the cleaned files
base_dir = os.path.dirname(os.path.abspath(__file__))
# Note: If cleaned files are in a subfolder, adjust this line (e.g., os.path.join(base_dir, "Dataset", "dataset iniziali..."))
# For now, we assume they are where we saved them in the previous step
path_TD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned.xlsx")


# --- 2. ENERGY FUNCTION (Single Subject) ---
def calculate_energy_for_subject(df_subject):
    """
    Calculates Anzalone Energy for a single subject's dataframe.
    """
    df = df_subject.copy()

    # Constants from Anzalone et al. (2019)
    TOTAL_MASS_KG = 25.0
    HEAD_MASS_KG = TOTAL_MASS_KG * 0.0668
    HEAD_RADIUS_M = 0.0835
    HEAD_INERTIA = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)

    # 1. Conversions
    # Convert mm to meters (Essential for Joule calculation)
    # If columns are missing (e.g. dropped during cleaning), fill with NaN to avoid errors
    for col in ['child_keypoint_x', 'child_keypoint_y', 'child_keypoint_z']:
        if col not in df.columns:
            df[col] = np.nan

    df['x_m'] = df['child_keypoint_x'] / 1000.0
    df['y_m'] = df['child_keypoint_y'] / 1000.0
    df['z_m'] = df['child_keypoint_z'] / 1000.0

    # 2. Delta Time
    df['dt'] = df['timestamp'].diff()

    # Filter valid time steps (avoid division by zero)
    # We use a temporary mask to calculate velocities only on valid rows
    valid_mask = df['dt'] > 0

    # 3. Translational Energy
    # Calculate diffs only
    dx = df['x_m'].diff()
    dy = df['y_m'].diff()
    dz = df['z_m'].diff()

    dist_sq = dx ** 2 + dy ** 2 + dz ** 2

    # Calculate velocity only where dt > 0
    velocity_sq = np.zeros(len(df)) * np.nan  # Initialize with NaNs
    velocity_sq[valid_mask] = dist_sq[valid_mask] / (df.loc[valid_mask, 'dt'] ** 2)

    df['energy_translational'] = 0.5 * HEAD_MASS_KG * velocity_sq

    # 4. Rotational Energy
    d_yaw = df['yaw'].diff()
    d_pitch = df['pitch'].diff()

    # Unwrap yaw (handle -pi to pi transition)
    d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

    ang_dist_sq = d_yaw ** 2 + d_pitch ** 2

    ang_vel_sq = np.zeros(len(df)) * np.nan
    ang_vel_sq[valid_mask] = ang_dist_sq[valid_mask] / (df.loc[valid_mask, 'dt'] ** 2)

    df['energy_rotational'] = 0.5 * HEAD_INERTIA * ang_vel_sq

    # 5. Total
    df['energy_total'] = df['energy_translational'] + df['energy_rotational']

    return df


# --- 3. BATCH PROCESSING FUNCTION ---
def process_group_dataset(file_path, group_label):
    """
    Loads a dataset, iterates through each subject, calculates energy,
    and returns a summary table.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None, None

    print(f"Loading {group_label} data...")
    df_all = pd.read_excel(file_path)

    # List to store processed frames for all subjects
    processed_dfs = []
    # List to store summary statistics (one row per subject)
    summary_data = []

    # Get unique subjects
    subjects = df_all['id_soggetto'].unique()
    print(f"Found {len(subjects)} subjects in {group_label} group.")

    for subj_id in subjects:
        # Extract data for ONE subject
        df_subj = df_all[df_all['id_soggetto'] == subj_id].sort_values('timestamp')

        # Calculate Energy
        df_subj = calculate_energy_for_subject(df_subj)

        # Store full data
        processed_dfs.append(df_subj)

        # Calculate Median for this subject (ignoring NaNs automatically)
        median_energy = df_subj['energy_total'].median()

        # Save to summary list
        summary_data.append({
            'Subject_ID': subj_id,
            'Group': group_label,
            'Median_Energy_J': median_energy
        })

    # Recombine all processed data into one big dataframe
    df_full_with_energy = pd.concat(processed_dfs)

    # Create summary dataframe
    df_summary = pd.DataFrame(summary_data)

    return df_full_with_energy, df_summary


# --- 4. EXECUTION ---
# Process TD
df_TD_full, df_TD_summary = process_group_dataset(path_TD_clean, "TD")

# Process ASD
df_ASD_full, df_ASD_summary = process_group_dataset(path_ASD_clean, "ASD")

# Combine Summaries into one final dataset for statistical analysis
if df_TD_summary is not None and df_ASD_summary is not None:
    final_stats_dataset = pd.concat([df_TD_summary, df_ASD_summary], ignore_index=True)

    # --- 5. SAVING OUTPUTS ---
    output_summary_path = os.path.join(base_dir, "Final_Energy_Statistics.xlsx")
    final_stats_dataset.to_excel(output_summary_path, index=False)

    print("\n" + "=" * 40)
    print("PROCESSING COMPLETE")
    print("=" * 40)
    print(f"Summary dataset saved to:\n{output_summary_path}")
    print("\nPreview of the final dataset:")
    print(final_stats_dataset)

    #Save the full time-series with energy columns if needed for plotting
    df_TD_full.to_csv(os.path.join(base_dir, "TD_time_series_energy.csv"), index=False)
    df_ASD_full.to_csv(os.path.join(base_dir, "ASD_time_series_energy.csv"), index=False)

else:
    print("Error processing datasets. Check file paths.")