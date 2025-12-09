import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================================================
# 1. SETUP PATHS
# ======================================================
# Base directory = folder where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Input datasets (TD and ASD groups)
path_TD_pose    = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD_pose   = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")

# Output Excel file
output_path = os.path.join(base_dir, "Final_MTI_Filtered.xlsx")

# Nominal frame rate (used only for choosing smoothing window)
FPS = 9.0
DT  = 1.0 / FPS   # Time between frames for speed-like operations


# ======================================================
# 2. LOAD DATA
# ======================================================
print("Loading ASD and TD datasets...\n")

# Check that files exist
if not (os.path.exists(path_TD_pose) and os.path.exists(path_ASD_pose)):
    print("❌ ERROR: Input files not found.")
    exit()

# Load Excel datasets
df_TD_pose  = pd.read_excel(path_TD_pose)
df_ASD_pose = pd.read_excel(path_ASD_pose)

print("Files loaded successfully.\n")


# ======================================================
# 3. HELPER FUNCTIONS
# ======================================================

def _clean_coords(df_sub):
    """
    Extracts head coordinates (x, y, z) for one subject.
    Ensures minimum length for meaningful analysis.
    """
    required_cols = {"child_keypoint_x", "child_keypoint_y", "child_keypoint_z"}
    if not required_cols.issubset(df_sub.columns):
        return None

    coords = df_sub[["child_keypoint_x", "child_keypoint_y", "child_keypoint_z"]].to_numpy(dtype=float)

    # Require at least 10 frames to compute MTI
    if len(coords) < 10:
        return None

    return coords


def _moving_average(x, window):
    """
    Simple moving average with a defined window size.
    Used both for:
    - noise reduction (short window)
    - slow drift removal (long window)
    """
    if len(x) < window:
        return np.full_like(x, np.nan)

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ======================================================
# 4. MTI (Micro-Tremor Instability) WITH PRE-FILTERING
# ======================================================
# MTI measures the residual high-frequency instability of head movement.
# Pipeline:
#   1) Pre-filter → removes camera jitter
#   2) Slow component → captures intentional movement
#   3) Fast residual = pre-filtered signal − slow trend
#   4) MTI = RMS energy of the fast residual (instability)


def compute_MTI(df_sub):
    coords = _clean_coords(df_sub)
    if coords is None:
        return np.nan

    # Window for pre-filtering (removes camera noise)
    PRE_WINDOW = 5

    # Window for slow component (~2 seconds of motion)
    SLOW_WINDOW = int(FPS * 2)
    if SLOW_WINDOW < 3:
        SLOW_WINDOW = 3

    residuals = []

    for axis in range(3):
        sig = coords[:, axis]

        # --------------------------------------------------
        # 1. PRE-FILTER — reduces camera noise and jitter
        # --------------------------------------------------
        sig_pre = _moving_average(sig, PRE_WINDOW)

        # --------------------------------------------------
        # 2. SLOW TREND — captures voluntary movement
        # --------------------------------------------------
        slow = _moving_average(sig_pre, SLOW_WINDOW)

        # --------------------------------------------------
        # 3. FAST RESIDUAL — isolates micro-tremor component
        # --------------------------------------------------
        fast = sig_pre - slow

        residuals.append(fast)

    # Convert residuals to (N × 3) matrix
    residuals = np.stack(residuals, axis=1)

    # Compute norm of the residual vector at each frame
    res_norm = np.linalg.norm(residuals, axis=1)

    # Remove NaN points (edges of moving average)
    res_norm = res_norm[~np.isnan(res_norm)]

    if len(res_norm) == 0:
        return np.nan

    # ------------------------------------------------------
    # 4. MTI = Root Mean Square (RMS) of residual micro-tremor
    # ------------------------------------------------------
    return float(np.sqrt(np.mean(res_norm**2)))


# ======================================================
# 5. PROCESS GROUP (TD or ASD)
# ======================================================
def compute_metrics_for_group(df_pose, group_label):
    """
    Computes MTI for each subject in the group.
    Returns a DataFrame with one row per subject.
    """

    results = []

    for sid in df_pose["id_soggetto"].unique():

        # Extract subject's time series sorted by frame number
        df_sub = df_pose[df_pose["id_soggetto"] == sid].sort_values("frame")

        # Compute MTI
        mti = compute_MTI(df_sub)

        results.append({
            "Subject_ID": sid,
            "Group": group_label,
            "MTI": mti
        })

    return pd.DataFrame(results)


# ======================================================
# 6. RUN PIPELINE
# ======================================================
print("Computing MTI...\n")

td_metrics  = compute_metrics_for_group(df_TD_pose,  "TD")
asd_metrics = compute_metrics_for_group(df_ASD_pose, "ASD")

# Final combined dataset
df_all = pd.concat([td_metrics, asd_metrics], ignore_index=True)
df_all.to_excel(output_path, index=False)

print("Saved MTI results to:", output_path)
print(df_all.head(), "\n")


# ======================================================
# 7. STATISTICAL ANALYSIS + PLOTS
# ======================================================
metric_key = "MTI"
metric_label = "Micro-Tremor Instability"

plot_dir = os.path.join(base_dir, "MTI_plots")
os.makedirs(plot_dir, exist_ok=True)

print("\n" + "="*60)
print(f" STATISTICAL ANALYSIS: {metric_label}")
print("="*60)

# Extract valid MTI values
td_vals  = df_all[df_all["Group"] == "TD"][metric_key].dropna()
asd_vals = df_all[df_all["Group"] == "ASD"][metric_key].dropna()

# Run Mann–Whitney test only if enough data is available
if len(td_vals) >= 2 and len(asd_vals) >= 2:
    stat, p_value = mannwhitneyu(td_vals, asd_vals)
else:
    stat, p_value = np.nan, np.nan
    print("⚠ Not enough valid samples for statistical test.")

print(f"TD (n={len(td_vals)}), Median = {td_vals.median():.6f}")
print(f"ASD (n={len(asd_vals)}), Median = {asd_vals.median():.6f}")
print(f"U-statistic = {stat}")
print(f"P-value     = {p_value}")

if p_value < 0.05:
        print("✔ SIGNIFICANT DIFFERENCE (p < 0.05)")
else:
        print("✘ NOT SIGNIFICANT (p ≥ 0.05)")


# ======================================================
# 8. PLOT MTI WITH OUTLIER-SAFE VISUALIZATION
# ======================================================
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Remove extreme outliers ONLY for visualization (not for statistics)
df_plot = df_all[df_all["MTI"] < 150]

# Create boxplot + swarmplot
sns.boxplot(x="Group", y="MTI", data=df_plot, palette="Set2", showfliers=False)
sns.swarmplot(x="Group", y="MTI", data=df_plot, color=".25", size=6)

plt.title("MTI Metric (ASD vs TD)", pad=40)
plt.ylabel("Micro-Tremor Instability")
plt.xlabel("Group")

# Expanded limits for aesthetics
plt.ylim(df_plot["MTI"].min() - 5, df_plot["MTI"].max() + 10)

# Add significance bar if valid
if len(td_vals) >= 2 and len(asd_vals) >= 2:
    sig = "ns"
    if p_value < 0.001: sig = "***"
    elif p_value < 0.01: sig = "**"
    elif p_value < 0.05: sig = "*"

    x1, x2 = 0, 1
    y = df_plot["MTI"].max() + 5

    # Significance bracket
    plt.plot([x1, x1, x2, x2], [y, y+5, y+5, y], lw=1.5, c="k")

    # Text annotation
    plt.text((x1+x2)/2, y+6, f"p = {p_value:.4f}\n{sig}",
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
