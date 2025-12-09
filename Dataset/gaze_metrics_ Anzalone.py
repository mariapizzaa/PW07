import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# 1. PATH SETUP
# ======================================================
# The script automatically finds its own directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Input datasets for TD and ASD groups (cleaned datasets)
path_TD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")

# Output path for summary statistics
summary_output_path = os.path.join(base_dir, "Final_Gaze_Statistics.xlsx")

# Directory for saving boxplots
plot_dir = os.path.join(base_dir, "gaze_plots")
os.makedirs(plot_dir, exist_ok=True)



# ======================================================
# 2. METRIC COMPUTATION FOR A SINGLE SUBJECT
# ======================================================
def compute_gaze_metrics_for_subject(df_subject):
    """
    Computes the 3 gaze variability metrics defined in Anzalone et al. (2019):
        1. Yaw_STD   = std of gaze left–right displacement
        2. Pitch_STD = std of gaze up–down displacement
        3. Magnitude_STD = std of radial magnitude (combined yaw + pitch)

    Steps:
    - Extract yaw and pitch angles.
    - Center them (subtract each subject’s mean) to compute displacement.
    - Compute radial magnitude sqrt(yaw² + pitch²).
    """

    df = df_subject.copy()

    # Extract raw yaw and pitch (angles in degrees)
    yaw = df["yaw"].astype(float)
    pitch = df["pitch"].astype(float)

    # Center the signals around subject’s own mean
    # (This is exactly as done in Anzalone 2019)
    yaw_centered = yaw - yaw.mean()
    pitch_centered = pitch - pitch.mean()

    # Radial magnitude = combination of yaw and pitch displacement
    magnitude = np.sqrt(yaw_centered**2 + pitch_centered**2)

    # Return the three variability metrics
    return {
        "gaze_yaw_std": np.std(yaw_centered),
        "gaze_pitch_std": np.std(pitch_centered),
        "gaze_magnitude_std": np.std(magnitude)
    }



# ======================================================
# 3. PROCESS ONE GROUP (ASD or TD)
# ======================================================
def process_group_dataset(file_path, group_label):
    """
    Loads a dataset, processes each subject individually, computes gaze metrics,
    and returns a dataframe containing:
        Subject_ID, Group, Yaw_STD, Pitch_STD, Magnitude_STD.
    """

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    print(f"\nLoading group: {group_label}")
    df_all = pd.read_excel(file_path)

    summary_list = []

    # Identify unique subjects
    subjects = df_all["id_soggetto"].unique()
    print(f"Found {len(subjects)} subjects.")

    # Compute metrics for each subject
    for subj in subjects:

        # Sort by timestamp → ensures chronological order
        df_sub = df_all[df_all["id_soggetto"] == subj].sort_values("timestamp")

        # Compute gaze metrics
        metrics = compute_gaze_metrics_for_subject(df_sub)

        # Store in summary table
        summary_list.append({
            "Subject_ID": subj,
            "Group": group_label,
            "Gaze_Yaw_STD": metrics["gaze_yaw_std"],
            "Gaze_Pitch_STD": metrics["gaze_pitch_std"],
            "Gaze_Magnitude_STD": metrics["gaze_magnitude_std"]
        })

    # Return summary dataframe for this group
    return pd.DataFrame(summary_list)



# ======================================================
# 4. RUN PIPELINE TO COMPUTE METRICS FOR BOTH GROUPS
# ======================================================
df_TD_summary  = process_group_dataset(path_TD_clean,  "TD")
df_ASD_summary = process_group_dataset(path_ASD_clean, "ASD")

# Merge into a single table
final_summary = pd.concat([df_TD_summary, df_ASD_summary], ignore_index=True)
final_summary.to_excel(summary_output_path, index=False)

print("\n======================================")
print("GAZE METRICS COMPUTED AND SAVED")
print("======================================")
print(final_summary.head())

# Reload (optional safety)
df = pd.read_excel(summary_output_path)



# ======================================================
# 5. STATISTICAL ANALYSIS + BEAUTIFUL BOXPLOTS
# ======================================================
metrics = [
    ("Gaze_Yaw_STD",       "Yaw STD"),
    ("Gaze_Pitch_STD",     "Pitch STD"),
    ("Gaze_Magnitude_STD", "Magnitude STD"),
]

sns.set_style("whitegrid")


for metric_key, metric_label in metrics:

    print("\n" + "="*60)
    print(f" STATISTICAL ANALYSIS: {metric_label}")
    print("="*60)

    # Extract per-group distributions
    td_vals  = df[df["Group"] == "TD"][metric_key].dropna()
    asd_vals = df[df["Group"] == "ASD"][metric_key].dropna()

    # Mann–Whitney U-test (correct test for non-Normal small samples)
    stat, p_value = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")

    print(f"TD (n={len(td_vals)}), Median = {td_vals.median():.6f}")
    print(f"ASD (n={len(asd_vals)}), Median = {asd_vals.median():.6f}")
    print(f"U-statistic = {stat}")
    print(f"P-value     = {p_value:.6f}")

    # ------------------------------------------------------
    # PLOT (Nature-style)
    # ------------------------------------------------------
    plt.figure(figsize=(8, 6))

    # Boxplot (group comparison)
    ax = sns.boxplot(x="Group", y=metric_key, data=df,
                     palette="Set2", showfliers=False)

    # Overlay individual points
    sns.swarmplot(x="Group", y=metric_key, data=df,
                  color=".25", size=5)

    # Title with spacing above
    plt.title(f"{metric_label}: TD vs ASD", pad=25, fontsize=16)

    plt.ylabel(metric_label, fontsize=13)
    plt.xlabel("Group", fontsize=12)

    # ----- Calculate bracket height -----
    y_min = df[metric_key].min()
    y_max = df[metric_key].max()

    offset = (y_max - y_min) * 0.15
    bar_height = offset * 0.25

    # Coordinates for the bracket
    x1, x2 = 0, 1
    y = y_max + offset

    # Draw the bracket line
    ax.plot([x1, x1, x2, x2],
            [y, y + bar_height, y + bar_height, y],
            lw=1.4, c="black")

    # Asterisks based on p-value
    significance = "ns"
    if p_value < 0.001: significance = "***"
    elif p_value < 0.01: significance = "**"
    elif p_value < 0.05: significance = "*"

    # Add text label (p-value + stars)
    plt.text(
        0.5,
        y + bar_height + offset * 0.05,
        f"p = {p_value:.3f}\n{significance}",
        ha="center",
        va="bottom",
        fontsize=12
    )

    # Expand vertical limits for clarity
    plt.ylim(y_min - offset * 0.3, y + offset * 1.3)

    plt.tight_layout()

    # Save the figure
    out_path = os.path.join(plot_dir, f"{metric_key}_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Plot saved to: {out_path}")
