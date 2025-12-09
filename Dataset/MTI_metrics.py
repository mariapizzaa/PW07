import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================================================
# 1. SETUP PATHS
# ======================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

path_TD_pose    = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD_pose   = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")

output_path = os.path.join(base_dir, "Final_MTI_Filtered.xlsx")

FPS = 9.0
DT  = 1.0 / FPS


# ======================================================
# 2. LOAD DATA
# ======================================================

print("Loading ASD and TD datasets...\n")

if not (os.path.exists(path_TD_pose) and os.path.exists(path_ASD_pose)):
    print("❌ ERROR: Input files not found.")
    exit()

df_TD_pose  = pd.read_excel(path_TD_pose)
df_ASD_pose = pd.read_excel(path_ASD_pose)

print("Files loaded successfully.\n")


# ======================================================
# 3. HELPER FUNCTIONS
# ======================================================

def _clean_coords(df_sub):
    if not {"child_keypoint_x", "child_keypoint_y", "child_keypoint_z"}.issubset(df_sub.columns):
        return None
    coords = df_sub[["child_keypoint_x", "child_keypoint_y", "child_keypoint_z"]].to_numpy(dtype=float)
    if len(coords) < 10:
        return None
    return coords


def _moving_average(x, window):
    if len(x) < window:
        return np.full_like(x, np.nan)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ======================================================
# 4. MTI (Micro-Tremor Instability) with PRE-FILTER
# ======================================================

def compute_MTI(df_sub):
    coords = _clean_coords(df_sub)
    if coords is None:
        return np.nan

    PRE_WINDOW = 5         # filtro leggero anti-rumore telecamera
    SLOW_WINDOW = int(FPS * 2)  # ~18 frame
    if SLOW_WINDOW < 3:
        SLOW_WINDOW = 3

    residuals = []

    for axis in range(3):
        sig = coords[:, axis]

        # 1) Pre-filtro: rimuove rumore della camera (come suggerito dalla prof)
        sig_pre = _moving_average(sig, PRE_WINDOW)

        # 2) Componente lenta
        slow = _moving_average(sig_pre, SLOW_WINDOW)

        # 3) Residuo ad alta frequenza (micro tremore)
        fast = sig_pre - slow

        residuals.append(fast)

    residuals = np.stack(residuals, axis=1)
    res_norm = np.linalg.norm(residuals, axis=1)
    res_norm = res_norm[~np.isnan(res_norm)]

    if len(res_norm) == 0:
        return np.nan

    # RMS = Micro Tremor Instability
    return float(np.sqrt(np.mean(res_norm**2)))


# ======================================================
# 5. PROCESS GROUP
# ======================================================

def compute_metrics_for_group(df_pose, group_label):
    results = []

    for sid in df_pose["id_soggetto"].unique():
        df_sub = df_pose[df_pose["id_soggetto"] == sid].sort_values("frame")

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

td_vals  = df_all[df_all["Group"] == "TD"][metric_key].dropna()
asd_vals = df_all[df_all["Group"] == "ASD"][metric_key].dropna()

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

# Plot
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Remove extreme outliers for visualization only
df_plot = df_all[df_all["MTI"] < 150]

sns.boxplot(x="Group", y="MTI", data=df_plot, palette="Set2", showfliers=False)
sns.swarmplot(x="Group", y="MTI", data=df_plot, color=".25", size=6)

plt.title("MTI Metric (ASD vs TD)", pad=40)
plt.ylabel("Micro-Tremor Instability")
plt.xlabel("Group")

plt.ylim(df_plot["MTI"].min() - 5, df_plot["MTI"].max() + 10)

# (Optional) add significance if valid
if len(td_vals) >= 2 and len(asd_vals) >= 2:
    sig = "ns"
    if p_value < 0.001: sig = "***"
    elif p_value < 0.01: sig = "**"
    elif p_value < 0.05: sig = "*"

    x1, x2 = 0, 1
    y = df_plot["MTI"].max() + 5
    plt.plot([x1, x1, x2, x2], [y, y+5, y+5, y], lw=1.5, c="k")
    plt.text((x1+x2)/2, y+6, f"p = {p_value:.4f}\n{sig}",
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
