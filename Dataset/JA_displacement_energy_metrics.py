import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ============================================================
# 1. PATHS
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

path_TD  = os.path.join(DATA_DIR, "TD_cleaned_advanced.xlsx")
path_ASD = os.path.join(DATA_DIR, "ASD_cleaned_advanced.xlsx")


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_pose(path, group_label):
    """
    Load TD/ASD_cleaned_advanced and add Group column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    df["Group"] = group_label
    df = df.sort_values(["id_soggetto", "timestamp"])
    return df


df_TD  = load_pose(path_TD,  "TD")
df_ASD = load_pose(path_ASD, "ASD")

df_all = pd.concat([df_TD, df_ASD], ignore_index=True)


# ============================================================
# 3. CONSTANTS FOR HEAD ENERGY (Anzalone)
# ============================================================

TOTAL_MASS_KG = 25.0
HEAD_MASS_KG  = TOTAL_MASS_KG * 0.0668
HEAD_RADIUS_M = 0.0835
HEAD_INERTIA  = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)


# ============================================================
# 4. METRICS PER SUBJECT: DISPLACEMENT + ENERGY
# ============================================================

def compute_metrics_for_subject(df_subj):
    """
    Compute:
      - DisplacementStd_mag, DisplacementStd_LR, DisplacementStd_FB
      - MedianHeadEnergy_total, MedianHeadEnergy_trans, MedianHeadEnergy_rot
    for a single subject.
    """

    df = df_subj.sort_values("timestamp").copy()

    # -----------------------------
    # 4.1 DISPLACEMENT METRICS
    # -----------------------------

    # Convert from mm to meters
    x_m = df["child_keypoint_x"].astype(float) / 1000.0
    y_m = df["child_keypoint_y"].astype(float) / 1000.0
    z_m = df["child_keypoint_z"].astype(float) / 1000.0

    # Center around subject mean (personal barycenter)
    x0 = x_m - np.nanmean(x_m)
    y0 = y_m - np.nanmean(y_m)
    z0 = z_m - np.nanmean(z_m)

    # 3D displacement magnitude
    disp_mag = np.sqrt(x0**2 + y0**2 + z0**2)

    # Standard deviations
    displacement_std_mag = float(np.nanstd(disp_mag))
    displacement_std_LR  = float(np.nanstd(x0))   # Left-Right
    displacement_std_FB  = float(np.nanstd(z0))   # Front-Back

    # -----------------------------
    # 4.2 HEAD ENERGY METRICS
    # -----------------------------

    # Time difference
    dt = df["timestamp"].diff().astype(float)
    # accept positive steps up to 10 seconds, ignore only VERY large gaps
    valid = (dt > 0) & (dt < 10.0)

    # Translational velocity
    dx = x_m.diff()
    dy = y_m.diff()
    dz = z_m.diff()

    dist_sq = dx**2 + dy**2 + dz**2

    DT_SEC = 1.0 / 9.0

    vel_sq = dist_sq / (DT_SEC ** 2)
    vel_sq[valid] = dist_sq[valid] / (dt[valid] ** 2)

    energy_trans = 0.5 * HEAD_MASS_KG * vel_sq

    # Rotational velocity from yaw & pitch
    yaw   = df["yaw"].astype(float)
    pitch = df["pitch"].astype(float)

    d_yaw   = yaw.diff()
    d_pitch = pitch.diff()

    # unwrap yaw
    d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

    ang_dist_sq = d_yaw**2 + d_pitch**2

    ang_vel_sq  = ang_dist_sq / (DT_SEC ** 2)
    ang_vel_sq[valid] = ang_dist_sq[valid] / (dt[valid] ** 2)

    energy_rot = 0.5 * HEAD_INERTIA * ang_vel_sq

    energy_total = energy_trans + energy_rot

    median_energy_trans = float(np.nanmedian(energy_trans))
    median_energy_rot   = float(np.nanmedian(energy_rot))
    median_energy_total = float(np.nanmedian(energy_total))

    metrics = {
        # displacement
        "DisplacementStd_mag": displacement_std_mag,
        "DisplacementStd_LR": displacement_std_LR,
        "DisplacementStd_FB": displacement_std_FB,
        # energy
        "MedianHeadEnergy_trans": median_energy_trans,
        "MedianHeadEnergy_rot": median_energy_rot,
        "MedianHeadEnergy_total": median_energy_total,
    }

    return metrics


# ============================================================
# 5. LOOP OVER SUBJECTS AND SAVE SUMMARY
# ============================================================

rows = []

for subj_id, df_subj in df_all.groupby("id_soggetto"):
    metrics = compute_metrics_for_subject(df_subj)
    metrics["Subject_ID"] = subj_id
    metrics["Group"] = df_subj["Group"].iloc[0]
    rows.append(metrics)

df_metrics = pd.DataFrame(rows)

out_summary = os.path.join(base_dir, "Displacement_Energy_metrics_summary.xlsx")
df_metrics.to_excel(out_summary, index=False)

print("Saved metrics (displacement + energy) to:", out_summary)
print(df_metrics.head())


# ============================================================
# 6. STATISTICAL TESTS (ASD vs TD) – MANN–WHITNEY
# ============================================================

metrics_to_test = [
    "DisplacementStd_mag",
    "DisplacementStd_LR",
    "DisplacementStd_FB",
    "MedianHeadEnergy_total",
    "MedianHeadEnergy_trans",
    "MedianHeadEnergy_rot"
]

results = []

for metric in metrics_to_test:
    df_valid = df_metrics.dropna(subset=[metric])

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
    print("ASD median:", np.median(ASD_values), " TD median:", np.median(TD_values))

    results.append({
        "Metric": metric,
        "U_statistic": u_stat,
        "p_value": p_val,
        "ASD_n": len(ASD_values),
        "TD_n": len(TD_values),
        "ASD_median": float(np.median(ASD_values)),
        "TD_median": float(np.median(TD_values))
    })

if results:
    df_stats = pd.DataFrame(results)
    out_stats = os.path.join(base_dir, "Displacement_Energy_metrics_stats.xlsx")
    df_stats.to_excel(out_stats, index=False)
    print("\nStatistical results saved to:", out_stats)
else:
    print("\nNo statistical results produced.")
# ============================================================
# 7. BOXPLOTS FOR ALL METRICS
# ============================================================

import matplotlib.pyplot as plt

# Metrics to visualize (same names as in df_metrics)
metrics_to_plot = [
    "DisplacementStd_mag",
    "DisplacementStd_LR",
    "DisplacementStd_FB",
    "MedianHeadEnergy_total",
    "MedianHeadEnergy_trans",
    "MedianHeadEnergy_rot",
]

# Create a subfolder for plots (optional)
plots_dir = os.path.join(base_dir, "JA_plots")
os.makedirs(plots_dir, exist_ok=True)

for metric in metrics_to_plot:
    df_valid = df_metrics.dropna(subset=[metric])

    td_vals  = df_valid[df_valid["Group"] == "TD"][metric].values
    asd_vals = df_valid[df_valid["Group"] == "ASD"][metric].values

    if len(td_vals) == 0 or len(asd_vals) == 0:
        print(f"[{metric}] Not enough data to plot.")
        continue

    plt.figure(figsize=(5, 5))
    plt.boxplot(
        [td_vals, asd_vals],
        labels=["TD", "ASD"],
        showfliers=True
    )
    plt.title(metric)
    plt.ylabel(metric)
    plt.tight_layout()

    # save figure
    out_png = os.path.join(plots_dir, f"boxplot_{metric}.png")
    plt.savefig(out_png, dpi=300)
    print(f"Saved boxplot for {metric} to: {out_png}")

    # and show it (if you run the script interactively)
    # comment this out if ti dà fastidio aprire le finestre
    plt.show()
