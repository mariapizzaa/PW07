import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================================================
# 1. PATH SETUP
# ======================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

path_TD  = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")

output_path = os.path.join(base_dir, "Final_Gaze_Variability_Statistics.xlsx")


# ======================================================
# 2. FPS DETECTION (for Unix second timestamps)
# ======================================================
def estimate_fps_from_unix_timestamps(timestamps):
    ts = pd.Series(timestamps).astype(int)
    counts = ts.value_counts()  # frames per second
    return int(counts.median())


# ======================================================
# 3. SLIDING-WINDOW STD
# ======================================================
def sliding_window_std(signal, window):
    signal = np.asarray(signal)

    if len(signal) < window or window <= 1:
        return np.array([np.nan])

    return np.array([
        np.std(signal[i:i + window])
        for i in range(len(signal) - window + 1)
    ])


# ======================================================
# 4. COMPUTE GAZE VARIABILITY WITH CONFIDENCE FILTERING
# ======================================================
def compute_variability(df_child):

    # Confidence filter
    df_child = df_child[
        (df_child["confidence_yaw"]   >= 0.5) &
        (df_child["confidence_pitch"] >= 0.5)
    ].copy()

    if len(df_child) < 5:
        return np.nan

    yaw_raw = df_child["yaw"].astype(float).values
    pitch_raw = df_child["pitch"].astype(float).values
    ts = df_child["timestamp"].astype(int).values

    # Center signals
    yaw = yaw_raw - np.mean(yaw_raw)
    pitch = pitch_raw - np.mean(pitch_raw)

    # Estimate FPS and window
    fps = estimate_fps_from_unix_timestamps(ts)
    window = max(2, fps)

    # Sliding-window STD
    yaw_std = sliding_window_std(yaw, window)
    pitch_std = sliding_window_std(pitch, window)

    # Combine radially
    combined = np.sqrt(yaw_std**2 + pitch_std**2)

    return np.nanmean(combined)


# ======================================================
# 5. PROCESS ONE GROUP
# ======================================================
def process_group(path, label):
    df = pd.read_excel(path)
    results = []

    for subject in df["id_soggetto"].unique():
        df_sub = df[df["id_soggetto"] == subject].sort_values("timestamp")
        metric = compute_variability(df_sub)

        results.append({
            "Subject_ID": subject,
            "Group": label,
            "Gaze_Variability_STD": metric
        })

    return pd.DataFrame(results)


# ======================================================
# 6. RUN PIPELINE (COMPUTE + SAVE)
# ======================================================
td_results = process_group(path_TD, "TD")
asd_results = process_group(path_ASD, "ASD")

final = pd.concat([td_results, asd_results], ignore_index=True)
final.to_excel(output_path, index=False)

print("\nSaved:", output_path)
print(final.head())


# ======================================================
# 7. LOAD RESULTS FOR STATISTICS
# ======================================================
df = pd.read_excel(output_path)
print("\nData loaded successfully for statistical analysis.\n")


# ======================================================
# 8. STATISTICAL ANALYSIS + PLOT
# ======================================================
metric_key = "Gaze_Variability_STD"
metric_label = "Sliding-Window Gaze Variability"

plot_dir = os.path.join(base_dir, "gaze_variability_plots")
os.makedirs(plot_dir, exist_ok=True)

print("\n" + "="*60)
print(f" STATISTICAL ANALYSIS: {metric_label}")
print("="*60)

td_vals  = df[df["Group"] == "TD"][metric_key].dropna()
asd_vals = df[df["Group"] == "ASD"][metric_key].dropna()

stat, p_value = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")

print(f"TD (n={len(td_vals)}), Median = {td_vals.median():.6f}")
print(f"ASD (n={len(asd_vals)}), Median = {asd_vals.median():.6f}")
print(f"U-statistic = {stat}")
print(f"P-value     = {p_value:.6f}")

if p_value < 0.05:
    print("✔ SIGNIFICANT DIFFERENCE (p < 0.05)")
else:
    print("✘ NOT SIGNIFICANT (p ≥ 0.05)")


# ======================================================
# 9. PLOT (Title not overlapping p-value)
# ======================================================
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Boxplot + swarm
ax = sns.boxplot(x="Group", y=metric_key, data=df, palette="Set2", showfliers=False)
sns.swarmplot(x="Group", y=metric_key, data=df, color=".25", size=5)

# Title with proper padding
plt.title("Sliding-Window Gaze Variability: TD vs ASD", pad=25, fontsize=16)

plt.ylabel(metric_label, fontsize=13)
plt.xlabel("Group", fontsize=12)

# ---- POSITIONING THE P-VALUE BRACKET ----

# Get y limits
y_min = df[metric_key].min()
y_max = df[metric_key].max()

# Vertical spacing above boxplot
offset = (y_max - y_min) * 0.10
bar_height = (y_max - y_min) * 0.02

# Coordinates
x1, x2 = 0, 1
y = y_max + offset

# Draw the bracket
ax.plot([x1, x1, x2, x2],
        [y, y + bar_height, y + bar_height, y],
        lw=1.4, c="black")

# Significance stars
sig = "ns"
if p_value < 0.001: sig = "***"
elif p_value < 0.01: sig = "**"
elif p_value < 0.05: sig = "*"

# Add the p-value text ABOVE the bracket
plt.text((x1 + x2)/2,
         y + bar_height + (offset * 0.15),
         f"p = {p_value:.3f}\n{sig}",
         ha="center",
         va="bottom",
         fontsize=12)

# Expand y-limit
plt.ylim(y_min - offset*0.3, y + offset * 1.5)

plt.tight_layout()

plot_path = os.path.join(plot_dir, f"{metric_key}_comparison.png")
plt.savefig(plot_path, dpi=300)
plt.show()
