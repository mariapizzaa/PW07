import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.ndimage import median_filter
import os

# --- SETTINGS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")

# Silva 2024 Targets (Relevant AOIs)
TARGETS = {
    'NAO_Robot': {'x': 0.0, 'z': 1.0, 'width_m': 0.90},
    'Therapist': {'x': -1.0, 'z': 2.0, 'width_m': 0.45},
    'Screen': {'x': 1.0, 'z': 2.0, 'width_m': 0.50}
}

GAZE_NOISE_RAD = 0.069
MIN_FIXATION_MS = 400


def calculate_avc(df_input, group_label):
    df = df_input.copy()

    # --- 1. PREPARATION (Coordinates & Angles) ---
    if 'child_keypoint_x' in df.columns:
        df['x_m'] = df['child_keypoint_x'] / 1000.0
        df['z_m'] = df['child_keypoint_z'] / 1000.0

    if df['yaw'].abs().max() > 3.20:
        df['yaw_rad'] = np.deg2rad(df['yaw'])
    else:
        df['yaw_rad'] = df['yaw']

    # FPS Calculation
    if 'timestamp' in df.columns and len(df) > 1:
        mean_dt = df['timestamp'].diff().mean()
        fps = 30 if pd.isna(mean_dt) or mean_dt == 0 else 1.0 / mean_dt
    else:
        fps = 30

    min_frames = int((MIN_FIXATION_MS / 1000.0) * fps)
    if min_frames < 1: min_frames = 1

    results_list = []
    subjects = df['id_soggetto'].unique()

    for subj in subjects:
        sub_df = df[df['id_soggetto'] == subj].sort_values('timestamp').copy()
        total_frames = len(sub_df)

        # This matrix will track "Is looking at anything relevant?" for each frame
        # Initially all False (0)
        is_looking_at_relevant = np.zeros(len(sub_df), dtype=int)

        # --- 2. CHECK FOR EACH TARGET ---
        for t_name, info in TARGETS.items():
            delta_x = info['x'] - sub_df['x_m']
            delta_z = info['z'] - sub_df['z_m']

            # Silva Angle
            target_angle = np.arctan2(delta_x, delta_z)
            target_angle_corrected = target_angle - np.pi
            target_angle_corrected = np.arctan2(np.sin(target_angle_corrected), np.cos(target_angle_corrected))

            dist = np.sqrt(delta_x ** 2 + delta_z ** 2).replace(0, 0.1)
            aoi_width = np.arctan((info['width_m'] / 2.0) / dist) + GAZE_NOISE_RAD

            angle_diff = np.abs(sub_df['yaw_rad'] - target_angle_corrected)
            angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

            # Is looking at this target?
            looking_at_this = (angle_diff <= aoi_width).astype(int)

            # Update "Relevant" matrix (OR operation: Robot OR Therapist OR Screen)
            is_looking_at_relevant = np.bitwise_or(is_looking_at_relevant, looking_at_this)

        # --- 3. AVC CALCULATION (Vacancy Analysis) ---
        # Vacancy = Places that are not Relevant (Inverse)
        is_vacant = 1 - is_looking_at_relevant

        # Remove noise (400ms filter)
        is_vacant_filtered = median_filter(is_vacant, size=min_frames)

        # METRIC A: AVC (Count) - Number of times gaze goes vacant
        # Count transitions from 0 to 1 (Engaged -> Disengaged)
        transitions = np.diff(is_vacant_filtered, prepend=0)
        avc_count = np.sum(transitions == 1)

        # METRIC B: Vacancy Rate (%) - Total disengagement time
        vacancy_rate = (np.sum(is_vacant_filtered) / total_frames) * 100.0

        results_list.append({
            'Subject_ID': subj,
            'Group': group_label,
            'AVC_Count': avc_count,  # "Vacancy Counts" in literature
            'Vacancy_Rate_Percent': vacancy_rate  # Disengagement Rate
        })

    return pd.DataFrame(results_list)


# --- EXECUTION ---
if os.path.exists(path_TD_clean) and os.path.exists(path_ASD_clean):
    print("AVC (AOI Vacancy Count) Analysis Starting...")

    df_TD = pd.read_excel(path_TD_clean)
    df_ASD = pd.read_excel(path_ASD_clean)

    avc_TD = calculate_avc(df_TD, "TD")
    avc_ASD = calculate_avc(df_ASD, "ASD")

    final_avc = pd.concat([avc_TD, avc_ASD], ignore_index=True)

    # --- STATISTICS ---
    td_vals = final_avc[final_avc['Group'] == 'TD']['AVC_Count']
    asd_vals = final_avc[final_avc['Group'] == 'ASD']['AVC_Count']

    stat, p_val = mannwhitneyu(td_vals, asd_vals)

    print("=" * 50)
    print("RESULT: AVC (AOI Vacancy Counts)")
    print("=" * 50)
    print(f"TD Mean: {td_vals.mean():.1f} counts")
    print(f"ASD Mean: {asd_vals.mean():.1f} counts")
    print(f"P-Value: {p_val:.5f}")

    if p_val < 0.05:
        print("✅ SIGNIFICANT DIFFERENCE FOUND (Consistent with literature).")
        if asd_vals.mean() > td_vals.mean():
            print("   -> ASD Group experienced more 'Social Disengagement'.")
    else:
        print("❌ No significant difference found.")

    # --- PLOT ---
    plt.figure(figsize=(9, 7))
    sns.set_style("whitegrid")

    # Boxplot
    my_pal = {"TD": "#2ecc71", "ASD": "#e74c3c"}  # Green (Good), Red (Disengaged)
    sns.boxplot(x='Group', y='AVC_Count', data=final_avc, palette=my_pal, showfliers=False)
    sns.swarmplot(x='Group', y='AVC_Count', data=final_avc, color=".25", size=6)

    plt.title('AVC: AOI Vacancy Counts (Social Disengagement)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Disengagements (Count)', fontsize=12)

    # Write P-Value
    y_max = final_avc['AVC_Count'].max()
    plt.text(0.5, y_max + 2, f"p = {p_val:.4f}", ha='center', fontsize=12, color='black')

    save_path = os.path.join(base_dir, "Result_Graph_AVC.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nGraph saved: {save_path}")

    # Save to Excel
    final_avc.to_excel(os.path.join(base_dir, "AVC_Metrics.xlsx"), index=False)
    plt.show()

else:
    print("Cleaned files not found.")
