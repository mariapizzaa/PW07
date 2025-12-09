# ============================================================
# ORIENTATION SWITCH COUNT (OSC)
# Quanti cambi di direzione fa lo sguardo (left / front / right)
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. PATH: leggo i due file puliti
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

path_TD = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD = os.path.join(base_dir, "ASD_cleaned.xlsx")

df_TD = pd.read_excel(path_TD)
df_ASD = pd.read_excel(path_ASD)

# ------------------------------------------------------------
# 2. KMEANS SU YAW/PITCH (ALLENO SOLO SUI TD)
#    Serve per avere i 3 cluster: left / front / right
# ------------------------------------------------------------
def train_kmeans_on_TD(df_TD, n_clusters=3):
    """
    Allena un KMeans con 3 cluster sui dati TD
    usando solo yaw e pitch.
    """
    data = df_TD[["yaw", "pitch"]].dropna()

    # n_init messo esplicitamente per evitare warning
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data)

    return kmeans

kmeans = train_kmeans_on_TD(df_TD)

# ------------------------------------------------------------
# 3. ASSEGNO IL CLUSTER A OGNI FRAME (TD e ASD)
# ------------------------------------------------------------
def assign_clusters(df, kmeans):
    """
    Assegna ad ogni frame il cluster KMeans
    sulla base delle colonne yaw e pitch.
    """
    df = df.copy()

    # considero validi solo i frame dove ho sia yaw che pitch
    mask_valid = df[["yaw", "pitch"]].notna().all(axis=1)

    df["cluster"] = np.nan
    df.loc[mask_valid, "cluster"] = kmeans.predict(
        df.loc[mask_valid, ["yaw", "pitch"]]
    )

    return df

df_TD = assign_clusters(df_TD, kmeans)
df_ASD = assign_clusters(df_ASD, kmeans)

# ------------------------------------------------------------
# 4. MAPPO I CLUSTER IN LEFT / FRONT / RIGHT
#    Guardando il valore medio di yaw per ogni cluster
#    (più negativo = left, intermedio = front, più positivo = right)
# ------------------------------------------------------------
def add_orientation_labels(df, kmeans):
    """
    Prende i centroidi del KMeans e li ordina per yaw.
    Poi mappa:
        yaw più negativo  -> "left"
        yaw intermedio    -> "front"
        yaw più positivo  -> "right"
    """
    df = df.copy()

    centers = kmeans.cluster_centers_  # shape (3, 2) -> [yaw, pitch]

    # ordino i cluster in base al valore del yaw (colonna 0)
    order = np.argsort(centers[:, 0])

    cluster_to_label = {
        order[0]: "left",
        order[1]: "front",
        order[2]: "right",
    }

    df["orientation"] = df["cluster"].map(cluster_to_label)

    return df, cluster_to_label

df_TD, cluster_map = add_orientation_labels(df_TD, kmeans)
df_ASD, _ = add_orientation_labels(df_ASD, kmeans)

print("Mappa cluster → orientazione:", cluster_map)

# ------------------------------------------------------------
# 5. FUNZIONE PER CALCOLARE GLI SWITCH DI UN SINGOLO BAMBINO
# ------------------------------------------------------------
def osc_single(
    df_sub,
    id_col="id_soggetto",      # nome colonna soggetto
    time_col="timestamp",      # nome colonna tempo
    orient_col="orientation"   # colonna con 'left','front','right'
):
    """
    Calcola per un soggetto:
        - n_switch: numero di cambi di orientazione
        - switch_per_sec
        - switch_per_min
    """

    # ordino nel tempo
    df_sub = df_sub.sort_values(time_col)

    # prendo solo la sequenza di orientazioni valide
    ori = df_sub[orient_col].dropna().to_numpy()

    # se ho meno di 2 frame utili, niente da fare
    if len(ori) <= 1:
        return {
            id_col: df_sub[id_col].iloc[0],
            "n_switch": 0,
            "switch_per_sec": np.nan,
            "switch_per_min": np.nan,
        }

    # confronto ogni orientazione con la precedente
    # True = cambio, False = uguale
    changes = ori[1:] != ori[:-1]
    n_switch = int(changes.sum())

    # calcolo la durata dell’acquisizione
    # ATTENZIONE: qui assumo timestamp in millisecondi.
    # Se è già in secondi, togli "/ 1000.0".
    t0 = df_sub[time_col].iloc[0]
    t1 = df_sub[time_col].iloc[-1]
    duration_sec = (t1 - t0) / 1000.0

    if duration_sec <= 0:
        switch_per_sec = np.nan
        switch_per_min = np.nan
    else:
        switch_per_sec = n_switch / duration_sec
        switch_per_min = switch_per_sec * 60.0

    return {
        id_col: df_sub[id_col].iloc[0],
        "n_switch": n_switch,
        "switch_per_sec": switch_per_sec,
        "switch_per_min": switch_per_min,
    }

# ------------------------------------------------------------
# 6. APPLICO LA METRICA A TUTTI I SOGGETTI DEL GRUPPO
# ------------------------------------------------------------
def osc_group(
    df,
    id_col="id_soggetto",
    time_col="timestamp",
    orient_col="orientation"
):
    """
    Applica osc_single a tutti i soggetti
    e ritorna un DataFrame con una riga per soggetto.
    """
    results = []

    for subj_id, df_sub in df.groupby(id_col):
        res = osc_single(
            df_sub,
            id_col=id_col,
            time_col=time_col,
            orient_col=orient_col,
        )
        results.append(res)

    return pd.DataFrame(results)

osc_TD = osc_group(df_TD)
osc_ASD = osc_group(df_ASD)

osc_TD["group"] = "TD"
osc_ASD["group"] = "ASD"

osc_all = pd.concat([osc_TD, osc_ASD], ignore_index=True)

# ------------------------------------------------------------
# 7. TEST STATISTICO (MANN–WHITNEY SU switch_per_min)
# ------------------------------------------------------------
td_vals = osc_TD["switch_per_min"].dropna()
asd_vals = osc_ASD["switch_per_min"].dropna()

stat, p_value = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")

print("-------------------------------------------------")
print("Orientation Switch Count (switches per minute)")
print("TD  - median:", td_vals.median())
print("ASD - median:", asd_vals.median())
print("Mann–Whitney U:", stat)
print("p-value:", p_value)
print("-------------------------------------------------")

# ------------------------------------------------------------
# 8. SALVO I RISULTATI IN EXCEL
# ------------------------------------------------------------
osc_TD.to_excel(os.path.join(base_dir, "OSC_TD.xlsx"), index=False)
osc_ASD.to_excel(os.path.join(base_dir, "OSC_ASD.xlsx"), index=False)
osc_all.to_excel(os.path.join(base_dir, "OSC_all.xlsx"), index=False)

print("Salvati: OSC_TD.xlsx, OSC_ASD.xlsx, OSC_all.xlsx")

# ------------------------------------------------------------
# 9. BOXPLOT
# ------------------------------------------------------------

# Stile generale
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(7, 6))

# boxplot TD vs ASD
ax = sns.boxplot(
    data=osc_all,
    x="group",
    y="switch_per_min",
    width=0.5,
    showfliers=True
)

# aggiungo anche i singoli punti
sns.stripplot(
    data=osc_all,
    x="group",
    y="switch_per_min",
    dodge=True,
    alpha=0.6
)

ax.set_xlabel("Group")
ax.set_ylabel("Orientation switches per minute")
ax.set_title("Orientation Switch Count (OSC)")

# scrivo il p-value sopra al grafico
text_p = f"p-value = {p_value:.4f}"

y_max = osc_all["switch_per_min"].max()
y_min = osc_all["switch_per_min"].min()

# posizione verticale: 10% sopra il max del boxplot
y_pos = y_max + (y_max - y_min) * 0.10

plt.text(
    0.5,           # centro orizzontale
    y_pos,
    text_p,
    fontsize=12,
    ha="center",
    va="bottom"
)

sns.despine()
plt.tight_layout()

# salvo il grafico in PNG per la slide
plot_path = os.path.join(base_dir, "OSC_switch_per_min_boxplot.png")
plt.savefig(plot_path, dpi=300)
plt.show()
plt.close()

print(f"Boxplot salvato in: {plot_path}")
