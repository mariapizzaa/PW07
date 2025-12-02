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

# Create output directory for plots
plot_dir = os.path.join(base_dir, "gaze_variability_plots")
os.makedirs(plot_dir, exist_ok=True)

# ======================================================
# 4. STATISTICAL ANALYSIS FOR EACH METRIC
# ======================================================
for metric_key, metric_label in metrics:

    print("\n" + "="*60)
    print(f" STATISTICAL ANALYSIS: {metric_label}")
    print("="*60)

    # Extract TD and ASD values
    td_vals = df[df["Group"] == "TD"][metric_key]
    asd_vals = df[df["Group"] == "ASD"][metric_key]

    # Mann-Whitney U test (non-parametric alternative to independent t-test)
    stat, p_value = mannwhitneyu(td_vals, asd_vals)

    # Print results
    print(f"TD (n={len(td_vals)}), Median = {td_vals.median():.6f}")
    print(f"ASD (n={len(asd_vals)}), Median = {asd_vals.median():.6f}")
    print(f"U-statistic = {stat}")
    print(f"P-value     = {p_value:.6f}")

    if p_value < 0.05:
        print("✔ SIGNIFICANT DIFFERENCE (p < 0.05)")
    else:
        print("✘ NOT SIGNIFICANT (p ≥ 0.05)")

    # ======================================================
    # 5. PLOTTING
    # ======================================================
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    # Boxplot without outliers
    sns.boxplot(x="Group", y=metric_key, data=df, palette="Set2", showfliers=False)

    # Add individual data points
    sns.swarmplot(x="Group", y=metric_key, data=df, color=".25", size=6)

    # ------------------------------------------------------
    # Significance annotation (horizontal bar + p-value)
    # ------------------------------------------------------
    x1, x2 = 0, 1
    y_max = df[metric_key].max()
    height = y_max * 0.05
    y_line = y_max + height

    # Horizontal bracket
    plt.plot([x1, x1, x2, x2],
             [y_line, y_line + height, y_line + height, y_line],
             lw=1.5, c='k')

    # Convert p-value to asterisks
    significance = "ns"
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"

    # Add p-value text above the bracket
    plt.text((x1 + x2) / 2, y_line + height,
             f"p = {p_value:.4f}\n{significance}",
             ha='center', va='bottom', fontsize=12)

    # Plot labels
    plt.title(f"{metric_label}: TD vs ASD")
    plt.ylabel(metric_label)
    plt.xlabel("Group")

    # Adjust vertical limits to make space for annotation
    plt.ylim(bottom=df[metric_key].min() - abs(df[metric_key].min() * 0.2),
             top=y_line + height * 5)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(plot_dir, f"{metric_key}_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

    plt.show()
