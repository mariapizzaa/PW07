import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import mannwhitneyu
# ============================================================
# 1. SETUP PATHS
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

# File names produced by the cleaning script
TD_FILENAME = "TD_cleaned_advanced.xlsx"
ASD_FILENAME = "ASD_cleaned_advanced.xlsx"

path_TD = os.path.join(DATA_DIR, TD_FILENAME)
path_ASD= os.path.join(DATA_DIR, ASD_FILENAME)

# ============================================================
# 2. PHYSICAL CONSTANTS (as in Anzalone)
# ============================================================

TOTAL_MASS_KG  = 25.0                 # approximate body mass
HEAD_MASS_KG   = TOTAL_MASS_KG * 0.0668
HEAD_RADIUS_M  = 0.0835               # 8.35 cm
HEAD_INERTIA   = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)


# ============================================================
# 3. METRICS FOR ONE SUBJECT
# ============================================================

def compute_metrics_for_subject(df_subj):
    """
    Compute JA-related kinematic metrics for a single subject:

    - Gazing std (magnitude, yaw, pitch)
    - Displacement std (magnitude, left-right, front-back)
    - Median head kinetic energy (translational + rotational)

    This follows the spirit of Anzalone et al. (2019),
    where these kinematic markers are used to characterize
    JA-related behavior (without using AOIs explicitly).
    """

    # Sort by time
    df = df_subj.sort_values("timestamp").copy()

    # -----------------------------
    # 3.1 GAZING STD (yaw, pitch)
    # ----------------------------
    yaw   = df["yaw"].astype(float)
    pitch = df["pitch"].astype(float)

    # Gaze magnitude as in the paper: magnitude of (yaw, pitch) in angle space
    gaze_mag = np.sqrt(yaw**2 + pitch**2)

    gazing_std_yaw   = float(np.nanstd(yaw))
    gazing_std_pitch = float(np.nanstd(pitch))
    gazing_std_mag   = float(np.nanstd(gaze_mag))

    # -----------------------------
    # 3.2 DISPLACEMENT STD (child)
    # -----------------------------
    # Convert child keypoints from mm to meters
    x_m = df["child_keypoint_x"].astype(float) / 1000.0
    y_m = df["child_keypoint_y"].astype(float) / 1000.0
    z_m = df["child_keypoint_z"].astype(float) / 1000.0

    # Center around subject-specific mean (personal barycenter)
    x0 = x_m - np.nanmean(x_m)
    y0 = y_m - np.nanmean(y_m)
    z0 = z_m - np.nanmean(z_m)

    # Displacement magnitude (3D)
    disp_mag = np.sqrt(x0**2 + y0**2 + z0**2)

    displacement_std_mag = float(np.nanstd(disp_mag))
    displacement_std_LR  = float(np.nanstd(x0))   # Left-Right
    displacement_std_FB  = float(np.nanstd(z0))   # Front-Back (depth)

    # -----------------------------
    # 3.3 HEAD KINETIC ENERGY
    # -----------------------------
    # Time difference between frames
    dt = df["timestamp"].diff().astype(float)
    valid = dt > 0

    # Translational velocity in 3D (head position)
    dx = x_m.diff()
    dy = y_m.diff()
    dz = z_m.diff()

    dist_sq = dx**2 + dy**2 + dz**2

    vel_sq = np.full(len(df), np.nan)
    vel_sq[valid] = dist_sq[valid] / (dt[valid] ** 2)

    energy_trans = 0.5 * HEAD_MASS_KG * vel_sq

    # Rotational velocity from yaw & pitch
    d_yaw   = yaw.diff()
    d_pitch = pitch.diff()

    # Unwrap yaw to avoid jumps over ±pi
    d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

    ang_dist_sq = d_yaw**2 + d_pitch**2

    ang_vel_sq = np.full(len(df), np.nan)
    ang_vel_sq[valid] = ang_dist_sq[valid] / (dt[valid] ** 2)

    energy_rot = 0.5 * HEAD_INERTIA * ang_vel_sq

    energy_total = energy_trans + energy_rot

    median_head_energy = float(np.nanmedian(energy_total))

    # Pack everything into a dict
    metrics = {
        "GazingStd_mag": gazing_std_mag,
        "GazingStd_yaw": gazing_std_yaw,
        "GazingStd_pitch": gazing_std_pitch,
        "DisplacementStd_mag": displacement_std_mag,
        "DisplacementStd_LR": displacement_std_LR,
        "DisplacementStd_FB": displacement_std_FB,
        "MedianHeadEnergy": median_head_energy
    }

    return metrics


# ============================================================
# 4. PROCESS WHOLE GROUP
# ============================================================

def process_group(path, group_label):
    """
    Load one group (TD or ASD), split by subject,
    compute metrics for each subject.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path)

    summary_rows = []

    for subj_id, df_subj in df.groupby("id_soggetto"):
        metrics = compute_metrics_for_subject(df_subj)
        metrics["Subject_ID"] = subj_id
        metrics["Group"] = group_label
        summary_rows.append(metrics)

    return pd.DataFrame(summary_rows)


# ============================================================
# 5. RUN FOR TD + ASD AND SAVE SUMMARY
# ============================================================

df_TD  = process_group(path_TD, "TD")
df_ASD = process_group(path_ASD, "ASD")

df_summary = pd.concat([df_TD, df_ASD], ignore_index=True)

out_summary = os.path.join(base_dir, "Anzalone_like_metrics_summary.xlsx")
df_summary.to_excel(out_summary, index=False)

print("Saved per-subject metrics to:", out_summary)
print(df_summary.head())


# ============================================================
# 6. STATISTICAL TESTS (ASD vs TD) – MANN–WHITNEY
# ============================================================

metrics_to_test = [
    "GazingStd_mag",
    "GazingStd_yaw",
    "GazingStd_pitch",
    "DisplacementStd_mag",
    "DisplacementStd_LR",
    "DisplacementStd_FB",
    "MedianHeadEnergy"
]

results = []

for metric in metrics_to_test:
    # Drop NaNs (if any)
    df_valid = df_summary.dropna(subset=[metric])

    ASD_values = df_valid[df_valid["Group"] == "ASD"][metric].values
    TD_values  = df_valid[df_valid["Group"] == "TD"][metric].values

    if len(ASD_values) == 0 or len(TD_values) == 0:
        print(f"\n[{metric}] Not enough data for test.")
        continue

    u_stat, p_val = mannwhitneyu(ASD_values, TD_values, alternative="two-sided")

    print(f"\n===== {metric} =====")
    print("ASD n:", len(ASD_values), " TD n:", len(TD_values))
    print("Mann–Whitney U:", u_stat)
    print("p-value:", p_val)

    results.append({
        "Metric": metric,
        "U_statistic": u_stat,
        "p_value": p_val,
        "ASD_n": len(ASD_values),
        "TD_n": len(TD_values),
        "ASD_median": float(np.median(ASD_values)),
        "TD_median": float(np.median(TD_values))
    })

# Save stats summary
if results:
    df_stats = pd.DataFrame(results)
    out_stats = os.path.join(base_dir, "Anzalone_like_metrics_stats.xlsx")
    df_stats.to_excel(out_stats, index=False)
    print("\nStatistical results saved to:", out_stats)
else:
    print("\nNo statistical results produced.")