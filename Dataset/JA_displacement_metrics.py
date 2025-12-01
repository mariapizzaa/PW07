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
    # make sure it's time-ordered per subject
    df = df.sort_values(["id_soggetto", "timestamp"])
    return df


df_TD  = load_pose(path_TD,  "TD")
df_ASD = load_pose(path_ASD, "ASD")

df_all = pd.concat([df_TD, df_ASD], ignore_index=True)


# ============================================================
# 3. DISPLACEMENT METRICS PER SUBJECT
# ============================================================

def displacement_metrics_for_subject(df_subj):
    """
    Compute displacement-based metrics for one subject:

      - DisplacementStd_mag : std of 3D displacement magnitude
      - DisplacementStd_LR  : std along x (Left-Right)
      - DisplacementStd_FB  : std along z (Front-Back)

    Positions are taken from child_keypoint_x/y/z (mm) and converted to meters.
    We center each axis around the subject's mean position to get
    a "personal reference frame".
    """

    # sort by time just to be safe
    df = df_subj.sort_values("timestamp").copy()

    # Convert from mm to meters
    x_m = df["child_keypoint_x"].astype(float) / 1000.0
    y_m = df["child_keypoint_y"].astype(float) / 1000.0
    z_m = df["child_keypoint_z"].astype(float) / 1000.0

    # Center around the personal barycenter (mean position)
    x0 = x_m - np.nanmean(x_m)
    y0 = y_m - np.nanmean(y_m)
    z0 = z_m - np.nanmean(z_m)

    # 3D displacement magnitude
    disp_mag = np.sqrt(x0**2 + y0**2 + z0**2)

    # Standard deviations (nan-aware)
    displacement_std_mag = float(np.nanstd(disp_mag))
    displacement_std_LR  = float(np.nanstd(x0))   # Left-Right
    displacement_std_FB  = float(np.nanstd(z0))   # Front-Back

    return {
        "DisplacementStd_mag": displacement_std_mag,
        "DisplacementStd_LR": displacement_std_LR,
        "DisplacementStd_FB": displacement_std_FB
    }


# ============================================================
# 4. LOOP OVER SUBJECTS AND SAVE SUMMARY
# ============================================================

rows = []

for subj_id, df_subj in df_all.groupby("id_soggetto"):
    metrics = displacement_metrics_for_subject(df_subj)
    metrics["Subject_ID"] = subj_id
    metrics["Group"] = df_subj["Group"].iloc[0]
    rows.append(metrics)

df_disp = pd.DataFrame(rows)

out_summary = os.path.join(base_dir, "Displacement_metrics_summary.xlsx")
df_disp.to_excel(out_summary, index=False)

print("Saved displacement metrics to:", out_summary)
print(df_disp.head())


# ============================================================
# 5. STATISTICAL TESTS (ASD vs TD) – MANN–WHITNEY
# ============================================================

metrics_to_test = [
    "DisplacementStd_mag",
    "DisplacementStd_LR",
    "DisplacementStd_FB"
]

results = []

for metric in metrics_to_test:
    df_valid = df_disp.dropna(subset=[metric])

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
    out_stats = os.path.join(base_dir, "Displacement_metrics_stats.xlsx")
    df_stats.to_excel(out_stats, index=False)
    print("\nStatistical results saved to:", out_stats)
else:
    print("\nNo statistical results produced.")
