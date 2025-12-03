# ============================================================
# Saccade-based Metric – Median Saccade Amplitude (per soggetto)
# ============================================================
# Questo script:
#  - legge TD_cleaned.xlsx e ASD_cleaned.xlsx
#  - per ogni soggetto calcola la mediana dell'ampiezza delle saccadi
#  - salva le statistiche in un Excel
#  - esegue un Mann–Whitney TD vs ASD
#  - genera un boxplot TD vs ASD con i puntini e il p-value
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------
# 1. PATH DEI FILE
# ------------------------------------------------------------

# Cartella in cui si trova questo script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Percorsi ai file
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")

# Parametri per rilevare le saccadi
SACCADE_VEL_THRESHOLD = 1.0   # soglia velocità angolare (rad/s)
MIN_DT = 1e-6                 # per evitare divisioni per zero


# ------------------------------------------------------------
# 2. METRICA PER UN SINGOLO SOGGETTO
# ------------------------------------------------------------
def compute_saccade_median_amplitude(df_subject):
    """
    Calcola la mediana dell'ampiezza delle saccadi per UN soggetto.

    - Usa yaw, pitch e timestamp
    - Stima velocità angolare tra frame consecutivi
    - Considera "saccadi" i frame con velocità > soglia
    - Restituisce la mediana delle ampiezze (in rad)
    """

    df = df_subject.copy()

    # Controllo che le colonne minime esistano
    for col in ["timestamp", "yaw", "pitch"]:
        if col not in df.columns:
            return np.nan

    # Ordino per timestamp
    df = df.sort_values("timestamp")

    # Converto le colonne in array numpy
    yaw = df["yaw"].astype(float).to_numpy()
    pitch = df["pitch"].astype(float).to_numpy()
    t = df["timestamp"].astype(float).to_numpy()

    # Differenze temporali tra frame consecutivi
    dt = np.diff(t)
    # Evito dt non validi o nulli
    dt[dt <= MIN_DT] = np.nan

    # Differenze angolari tra frame consecutivi
    dyaw = np.diff(yaw)
    dpitch = np.diff(pitch)

    # Ampiezza angolare tra 2 frame (in rad)
    amplitudes = np.sqrt(dyaw**2 + dpitch**2)

    # Velocità angolare (rad/s)
    angular_vel = amplitudes / dt

    # Maschera di validità (dt valido) e soglia di velocità
    valid = ~np.isnan(dt)
    is_saccade = (angular_vel > SACCADE_VEL_THRESHOLD) & valid

    # Se non ci sono saccadi, ritorno NaN
    if not np.any(is_saccade):
        return np.nan

    # Ampiezze delle sole saccadi
    saccade_amplitudes = amplitudes[is_saccade]

    # Mediana dell'ampiezza delle saccadi
    median_amp = np.median(saccade_amplitudes)

    return median_amp


# ------------------------------------------------------------
# 3. PROCESSA UN INTERO GRUPPO (TD / ASD)
# ------------------------------------------------------------
def process_group_dataset(file_path, group_label):
    """
    Per un gruppo (TD o ASD):
      - carica il file Excel
      - divide i dati per id_soggetto
      - calcola Saccade_Median_Amplitude per ogni soggetto
      - restituisce un DataFrame riassuntivo
    """

    # Controllo che il file esista
    if not os.path.exists(file_path):
        print(f"ERRORE: file non trovato: {file_path}")
        return None

    print(f"\nCaricamento dati gruppo {group_label} ...")
    df_all = pd.read_excel(file_path)

    summary = []

    # Tutti gli ID soggetto presenti nel file
    subjects = df_all["id_soggetto"].unique()
    print(f"Trovati {len(subjects)} soggetti nel gruppo {group_label}.")

    # Loop sui soggetti
    for subj in subjects:
        # Righe di quel soggetto
        df_subj = df_all[df_all["id_soggetto"] == subj]

        # Calcolo della metrica
        median_amp = compute_saccade_median_amplitude(df_subj)

        # Aggiungo ai risultati
        summary.append({
            "Subject_ID": subj,
            "Group": group_label,
            "Saccade_Median_Amplitude": median_amp
        })

    # Converto la lista in DataFrame
    return pd.DataFrame(summary)


# ------------------------------------------------------------
# 4. ESECUZIONE
# ------------------------------------------------------------

df_TD = process_group_dataset(path_TD_clean, "TD")
df_ASD = process_group_dataset(path_ASD_clean, "ASD")

if df_TD is not None and df_ASD is not None:

    # Unisco TD e ASD
    final_stats_dataset = pd.concat([df_TD, df_ASD], ignore_index=True)

    # Salvo l'Excel riassuntivo
    out_path = os.path.join(base_dir, "Final_Saccade_Statistics.xlsx")
    final_stats_dataset.to_excel(out_path, index=False)

    print("\n==============================================")
    print("PROCESSO SACCADE-BASED COMPLETATO")
    print("==============================================")
    print(f"File salvato in: {out_path}")
    print("\nAnteprima:")
    print(final_stats_dataset.head())

    # ------------------ STATISTICA ------------------------
    # Valori per TD e ASD
    td_vals = final_stats_dataset[
        final_stats_dataset["Group"] == "TD"
    ]["Saccade_Median_Amplitude"].dropna().tolist()

    asd_vals = final_stats_dataset[
        final_stats_dataset["Group"] == "ASD"
    ]["Saccade_Median_Amplitude"].dropna().tolist()

    # Test di Mann–Whitney
    u_stat, p_val = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")
    print(f"\nMann–Whitney (Saccade_Median_Amplitude): U = {u_stat:.3f}, p = {p_val:.4f}")

    # ------------------ BOXPLOT ---------------------------
    plt.figure(figsize=(6, 5))

    bp = plt.boxplot(
        [td_vals, asd_vals],
        positions=[1, 2],
        tick_labels=["TD", "ASD"],
        patch_artist=True
    )

    # Colori delle box
    colors = ["#4c72b0", "#dd8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Puntini con jitter
    jitter = 0.06
    for i, vals in enumerate([td_vals, asd_vals], start=1):
        x = np.random.normal(loc=i, scale=jitter, size=len(vals))
        plt.scatter(x, vals, color="black", s=25, zorder=3)

    # p-value sopra le box
    y_max = max(max(td_vals), max(asd_vals))
    if y_max <= 0:
        y_max = 1.0
    y_line = y_max + 0.05 * y_max

    plt.plot([1, 2], [y_line, y_line], color="black")
    plt.text(1.5, y_line + 0.02 * y_max,
             f"p = {p_val:.4f}", ha="center")

    # Titoli e assi
    plt.title("Saccade Median Amplitude per Soggetto")
    plt.ylabel("Amplitude (rad)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
    print("\nERRORE: impossibile completare il processo (file mancanti).")
