# ============================================================
# GAZING STANDARD DEVIATION – PER SOGGETTO (SEABORN VERSION)
# ============================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------
# 1. PATH
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")

# ------------------------------------------------------------
# 2. CALCOLO GAZING STD PER SOGGETTO
# ------------------------------------------------------------
def calculate_gazing_std(df_subject):
    df = df_subject.copy()

    if "yaw" not in df.columns or "pitch" not in df.columns:
        return np.nan

    yaw = df["yaw"].dropna()
    pitch = df["pitch"].dropna()
    if len(yaw) == 0:
        return np.nan

    yaw_centered = yaw - yaw.mean()
    pitch_centered = pitch - pitch.mean()

    magnitude = np.sqrt(yaw_centered**2 + pitch_centered**2)
    return magnitude.std()

# ------------------------------------------------------------
# 3. PROCESSA UN GRUPPO
# ------------------------------------------------------------
def process_group(file_path, group_label):
    if not os.path.exists(file_path):
        print(f"ERRORE: file non trovato -> {file_path}")
        return None

    df_all = pd.read_excel(file_path)
    summary = []

    for subj in df_all["id_soggetto"].unique():
        df_subj = df_all[df_all["id_soggetto"] == subj]
        std_val = calculate_gazing_std(df_subj)

        summary.append({
            "Subject_ID": subj,
            "Group": group_label,
            "Gazing_STD": std_val
        })

    return pd.DataFrame(summary)

# ------------------------------------------------------------
# 4. RUN METRICA
# ------------------------------------------------------------
df_TD = process_group(path_TD_clean, "TD")
df_ASD = process_group(path_ASD_clean, "ASD")
df_final = pd.concat([df_TD, df_ASD], ignore_index=True)

out_path = os.path.join(base_dir, "Final_Gazing_STD_Statistics.xlsx")
df_final.to_excel(out_path, index=False)
print("Saved:", out_path)

# ------------------------------------------------------------
# 5. FUNZIONE PLOT SEABORN
# ------------------------------------------------------------
def plot_metric(df, metric_key, metric_label):

    sns.set(style="whitegrid")

    td_vals = df[df["Group"]=="TD"][metric_key]
    asd_vals = df[df["Group"]=="ASD"][metric_key]

    stat, p_value = mannwhitneyu(td_vals, asd_vals)

    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="Group", y=metric_key,
        palette=["#86bff2", "#f4b183"],
        showfliers=False
    )
    sns.swarmplot(
        data=df,
        x="Group", y=metric_key,
        size=6, color="black"
    )

    # bracket p-value
    x1, x2 = 0, 1
    y_max = df[metric_key].max()
    h = (y_max - df[metric_key].min()) * 0.12
    y = y_max + h

    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')

    if p_value < 0.001: stars = "***"
    elif p_value < 0.01: stars = "**"
    elif p_value < 0.05: stars = "*"
    else: stars = "ns"

    plt.text((x1+x2)/2, y+h, f"p={p_value:.4f}\n{stars}",
             ha='center', fontsize=12)

    plt.title(metric_label)
    plt.ylabel(metric_label)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 6. PLOT GAZING STD
# ------------------------------------------------------------
plot_metric(df_final, "Gazing_STD", "Gazing Standard Deviation (Magnitude)")

