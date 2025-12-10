import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =========================
# 1. CARICO IL SUMMARY
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(base_dir, "Anzalone_like_metrics_summary.xlsx")

df_summary = pd.read_excel(summary_path)
df_summary = df_summary[["Group",
                         "GazingStd_mag", "GazingStd_yaw", "GazingStd_pitch",
                         "DisplacementStd_mag", "DisplacementStd_LR", "DisplacementStd_FB",
                         "MedianHeadEnergy"]]

metrics_to_plot = [
    "GazingStd_mag",
    "GazingStd_yaw",
    "GazingStd_pitch",
    "DisplacementStd_mag",
    "DisplacementStd_LR",
    "DisplacementStd_FB",
    "MedianHeadEnergy",   # tienila se vuoi avere anche l’energia
]

groups = ["TD", "ASD"]

out_dir = os.path.join(base_dir, "figures_metrics")
os.makedirs(out_dir, exist_ok=True)

plt.rcParams["figure.figsize"] = (7, 5)

for METRIC in metrics_to_plot:
    df_plot = df_summary[["Group", METRIC]].dropna()

    # =========================
    # 2. BEESWARM / STRIPPLOT
    # =========================
    plt.figure()
    x_positions = [0, 1]

    for i, g in enumerate(groups):
        vals = df_plot[df_plot["Group"] == g][METRIC].values
        x_jitter = np.random.normal(loc=x_positions[i],
                                    scale=0.06, size=len(vals))
        plt.scatter(x_jitter, vals, alpha=0.8, label=f"{g} (n={len(vals)})")

    plt.xticks(x_positions, groups)
    plt.xlabel("Group")
    plt.ylabel(METRIC)
    plt.title(f"Beeswarm / Stripplot – {METRIC}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{METRIC}_beeswarm.png"), dpi=300)
    plt.close()

    # =========================
    # 3. ECDF
    # =========================
    plt.figure()

    for g in groups:
        vals = np.sort(df_plot[df_plot["Group"] == g][METRIC].values)
        n = len(vals)
        if n == 0:
            continue
        y = np.arange(1, n + 1) / n
        plt.step(vals, y, where="post", label=f"{g} (n={n})")

    plt.xlabel(METRIC)
    plt.ylabel("Empirical CDF")
    plt.title(f"ECDF – {METRIC} (ASD vs TD)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{METRIC}_ECDF.png"), dpi=300)
    plt.close()

    # =========================
    # 4. RAINCLOUD-STYLE
    # =========================
    plt.figure()
    offsets = [0, 1]

    for i, g in enumerate(groups):
        vals = df_plot[df_plot["Group"] == g][METRIC].values
        if len(vals) == 0:
            continue

        # densità (KDE)
        if np.std(vals) > 0:
            kde = gaussian_kde(vals)
            x_grid = np.linspace(vals.min(), vals.max(), 200)
            density = kde(x_grid)
            density = density / density.max() * 0.4  # larghezza nuvola
            plt.fill_between(
                x_grid,
                offsets[i],
                offsets[i] + density,
                alpha=0.4,
                linewidth=0,
            )

        # punti (rain)
        y_jitter = np.random.normal(loc=offsets[i],
                                    scale=0.03, size=len(vals))
        plt.scatter(vals, y_jitter, alpha=0.8)

    plt.yticks(offsets, groups)
    plt.xlabel(METRIC)
    plt.title(f"Raincloud-style – {METRIC}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{METRIC}_raincloud.png"), dpi=300)
    plt.close()

print("Figure salvate in:", out_dir)
