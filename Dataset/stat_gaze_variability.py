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
file_path = os.path.join(base_dir, "Final_Gaze_Variability_Statistics.xlsx")

# ======================================================
# 2. LOAD DATA
# ======================================================
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Data loaded successfully.\n")
else:
    print(f"Error: File not found in {file_path}")
    exit()

# ======================================================
# 3. METRIC TO ANALYZE
# ======================================================
metrics = [
    ("Gaze_Variability_STD", "Sliding-Window Gaze Variability")
]

plot_dir = os.path.join(base_dir, "gaze_variability_plots")
os.makedirs(plot_dir, exist_ok=True)

# ======================================================
# 4. STATISTICAL ANALYSIS
# ======================================================
for metric_key, metric_label in metrics:

    print("\n" + "="*60)
    print(f" STATISTICAL ANALYSIS: {metric_label}")
    print("="*60)

    # Extract data
    td_vals = df[df["Group"] == "TD"][metric_key].dropna()
    asd_vals = df[df["Group"] == "ASD"][metric_key].dropna()

    # Mann-Whitney U (two-sided)
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
    # 5. PLOT
    # ======================================================
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    sns.boxplot(x="Group", y=metric_key, data=df, palette="Set2", showfliers=False)
    sns.swarmplot(x="Group", y=metric_key, data=df, color=".25", size=6)

    # Annotation
    x1, x2 = 0, 1
    y_max = df[metric_key].max()
    y_min = df[metric_key].min()

    padding = (y_max - y_min) * 0.2 if y_max != y_min else 0.001
    height = padding * 0.5
    y_line = y_max + padding

    plt.plot([x1, x1, x2, x2],
             [y_line, y_line + height, y_line + height, y_line],
             lw=1.5, c='k')

    significance = "ns"
    if p_value < 0.001: significance = "***"
    elif p_value < 0.01: significance = "**"
    elif p_value < 0.05: significance = "*"

    plt.text((x1 + x2) / 2,
             y_line + height,
             f"p = {p_value:.4f}\n{significance}",
             ha='center', va='bottom', fontsize=12)

    plt.title(f"{metric_label}: TD vs ASD", fontsize=14)
    plt.ylabel(metric_label)
    plt.xlabel("Group")

    plt.ylim(y_min - padding, y_max + padding * 5)

    plt.tight_layout()

    output_path = os.path.join(plot_dir, f"{metric_key}_comparison.png")
    plt.savefig(output_path, dpi=300)

    print(f"Plot saved to: {output_path}")

    plt.show()
