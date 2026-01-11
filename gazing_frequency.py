# ============================================================
# GAZING FREQUENCY LEFT / FRONT / RIGHT (SEABORN VERSION)
# ============================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")

# ------------------------------------------------------------
# 2. TRAIN KMEANS ON TD
# ------------------------------------------------------------
def train_kmeans(df_TD):
    gaze = df_TD[["yaw", "pitch"]].dropna().to_numpy()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(gaze)
    return kmeans

def get_cluster_map(centers):
    yaw_vals = centers[:,0]
    order = np.argsort(yaw_vals)
    return {
        "left":  int(order[0]),
        "front": int(order[1]),
        "right": int(order[2])
    }

# ------------------------------------------------------------
# 3. CALCOLO FREQUENZE
# ------------------------------------------------------------
def calculate_frequencies(df_subject, kmeans, cluster_map):
    gaze = df_subject[["yaw","pitch"]].dropna().to_numpy()
    if len(gaze) == 0:
        return np.nan, np.nan, np.nan

    labels = kmeans.predict(gaze)
    total = len(labels)

    return (
        np.sum(labels==cluster_map["left"])/total*100,
        np.sum(labels==cluster_map["front"])/total*100,
        np.sum(labels==cluster_map["right"])/total*100
    )

# ------------------------------------------------------------
# 4. PROCESS GROUP
# ------------------------------------------------------------
def process_group(file_path, group_label, kmeans, cluster_map):

    df_all = pd.read_excel(file_path)
    summary = []

    for subj in df_all["id_soggetto"].unique():
        df_subj = df_all[df_all["id_soggetto"] == subj]

        f_left, f_front, f_right = calculate_frequencies(
            df_subj, kmeans, cluster_map
        )

        summary.append({
            "Subject_ID": subj,
            "Group": group_label,
            "freq_left": f_left,
            "freq_front": f_front,
            "freq_right": f_right
        })

    return pd.DataFrame(summary)

# ------------------------------------------------------------
# 5. RUN
# ------------------------------------------------------------
df_TD_clean_all = pd.read_excel(path_TD_clean)
kmeans = train_kmeans(df_TD_clean_all)
cluster_map = get_cluster_map(kmeans.cluster_centers_)

df_TD = process_group(path_TD_clean, "TD", kmeans, cluster_map)
df_ASD = process_group(path_ASD_clean, "ASD", kmeans, cluster_map)
df_final = pd.concat([df_TD, df_ASD], ignore_index=True)

out_path = os.path.join(base_dir, "Final_Gazing_Frequency_ALL.xlsx")
df_final.to_excel(out_path, index=False)
print("Saved:", out_path)

# ------------------------------------------------------------
# 6. PLOT FUNZIONE (SEABORN)
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
        color="black", size=6
    )

    # p-value bracket
    x1, x2 = 0, 1
    y_max = df[metric_key].max()
    h = (y_max - df[metric_key].min()) * 0.12
    y = y_max + h

    plt.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw=1.5, c="black")

    if p_value < 0.001: stars="***"
    elif p_value < 0.01: stars="**"
    elif p_value < 0.05: stars="*"
    else: stars="ns"

    plt.text((x1+x2)/2, y+h, f"p={p_value:.4f}\n{stars}",
             ha="center", fontsize=12)

    plt.title(metric_label)
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7. PLOTS
# ------------------------------------------------------------
plot_metric(df_final, "freq_left", "Gazing Frequency – LEFT")
plot_metric(df_final, "freq_front", "Gazing Frequency – FRONT")
plot_metric(df_final, "freq_right", "Gazing Frequency – RIGHT")
