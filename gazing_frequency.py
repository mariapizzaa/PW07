# ============================================================
# GAZING FREQUENCY – LEFT / FRONT / RIGHT
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu

# -------------------------------
# 1. Percorsi file
# -------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")

# -------------------------------
# 2. KMeans su TUTTI i TD
# -------------------------------
def train_kmeans_on_TD(df_TD):
    gaze = df_TD[['yaw', 'pitch']].dropna().to_numpy()
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(gaze)
    return kmeans

def get_cluster_map(centers):
    yaw_vals = centers[:, 0]
    sort_idx = np.argsort(yaw_vals)
    return {
        "left":  int(sort_idx[0]),
        "front": int(sort_idx[1]),
        "right": int(sort_idx[2])
    }

# -------------------------------
# 3. Frequenze LEFT/FRONT/RIGHT
# -------------------------------
def calculate_frequencies(df_subject, kmeans, cluster_map):

    gaze = df_subject[['yaw', 'pitch']].dropna().to_numpy()
    if len(gaze) == 0:
        return np.nan, np.nan, np.nan

    labels = kmeans.predict(gaze)
    total = len(labels)

    freq_left  = np.sum(labels == cluster_map['left'])  / total * 100
    freq_front = np.sum(labels == cluster_map['front']) / total * 100
    freq_right = np.sum(labels == cluster_map['right']) / total * 100

    return freq_left, freq_front, freq_right

# -------------------------------
# 4. Processo per gruppo
# -------------------------------
def process_group(file_path, group_label, kmeans, cluster_map):

    df_all = pd.read_excel(file_path)
    summary = []

    for subj in df_all['id_soggetto'].unique():

        df_subj = df_all[df_all['id_soggetto'] == subj]
        f_left, f_front, f_right = calculate_frequencies(df_subj, kmeans, cluster_map)

        summary.append({
            "Subject_ID": subj,
            "Group": group_label,
            "freq_left": f_left,
            "freq_front": f_front,
            "freq_right": f_right
        })

    return pd.DataFrame(summary)

# -------------------------------
# 5. Esecuzione
# -------------------------------
df_TD_clean = pd.read_excel(path_TD_clean)
kmeans = train_kmeans_on_TD(df_TD_clean)
cluster_map = get_cluster_map(kmeans.cluster_centers_)

df_TD = process_group(path_TD_clean, "TD", kmeans, cluster_map)
df_ASD = process_group(path_ASD_clean, "ASD", kmeans, cluster_map)

df_final = pd.concat([df_TD, df_ASD], ignore_index=True)

# Salvataggio Excel
out_path = os.path.join(base_dir, "Final_Gazing_Frequency_ALL.xlsx")
df_final.to_excel(out_path, index=False)
print("Salvato:", out_path)

# -------------------------------
# 6. Funzione per creare boxplot
# -------------------------------
def plot_metric(metric_name):
    td_vals = df_final[df_final["Group"]=="TD"][metric_name].dropna().tolist()
    asd_vals = df_final[df_final["Group"]=="ASD"][metric_name].dropna().tolist()

    # Mann–Whitney
    u, p = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")
    print(f"\n{metric_name}: U={u:.3f}, p={p:.4f}")

    # Plot
    plt.figure(figsize=(6,5))
    bp = plt.boxplot([td_vals, asd_vals],
                     positions=[1,2],
                     tick_labels=["TD", "ASD"],
                     patch_artist=True)

    colors = ["#4c72b0", "#dd8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    jitter = 0.06
    for i, vals in enumerate([td_vals, asd_vals], start=1):
        x = np.random.normal(loc=i, scale=jitter, size=len(vals))
        plt.scatter(x, vals, color="black", s=25)

    y_max = max(max(td_vals), max(asd_vals))
    y_line = y_max + 5
    plt.plot([1,2],[y_line,y_line],color="black")
    plt.text(1.5, y_line+2, f"p={p:.4f}", ha="center")

    plt.title(f"Gazing Frequency — {metric_name.upper()}")
    plt.ylabel("Frequency (%)")
    plt.ylim(0, y_line+10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 7. Boxplot per LEFT / FRONT / RIGHT
# -------------------------------
plot_metric("freq_left")
plot_metric("freq_front")
plot_metric("freq_right")
