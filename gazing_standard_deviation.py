# GAZING STANDARD DEVIATION – PER SOGGETTO
import os               # per gestire i percorsi
import pandas as pd     # per leggere gli Excel
import numpy as np      # per fare i calcoli matematici
import matplotlib.pyplot as plt   # per generare il boxplot
from scipy.stats import mannwhitneyu   # test non parametrico TD vs ASD


# ------------------------------------------------------------
# 1. DEFINIZIONE DEI PERCORSI DEI FILE
# ------------------------------------------------------------
# Trovo la cartella in cui si trova questo script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Percorsi ai file puliti (TD e ASD)
path_TD_clean = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD_clean = os.path.join(base_dir, "ASD_cleaned.xlsx")


# ------------------------------------------------------------
# 2. FUNZIONE PER CALCOLARE LA GAZING STD PER UN SINGOLO SOGGETTO
# ------------------------------------------------------------
def calculate_gazing_std(df_subject):
    """
    Calcolo la deviatazione standard della magnitudo del movimento dello sguardo:
    magnitudo = sqrt( (yaw - mean_yaw)^2 + (pitch - mean_pitch)^2 )
    """

    df = df_subject.copy()

    # Controllo che le colonne esistano
    if 'yaw' not in df.columns or 'pitch' not in df.columns:
        return np.nan

    # Prendo yaw e pitch validi
    yaw = df['yaw'].dropna()
    pitch = df['pitch'].dropna()

    # Se non ci sono dati: NaN
    if len(yaw) == 0:
        return np.nan

    # Calcolo media dello sguardo
    yaw_mean = yaw.mean()
    pitch_mean = pitch.mean()

    # Spostamento rispetto alla media
    disp_yaw = yaw - yaw_mean
    disp_pitch = pitch - pitch_mean

    # Magnitudo dello spostamento
    magnitude = np.sqrt(disp_yaw**2 + disp_pitch**2)

    # Deviazione standard della magnitudo
    std_magnitude = magnitude.std()

    return std_magnitude


# ------------------------------------------------------------
# 3. FUNZIONE PER PROCESSARE L’INTERO GRUPPO TD O ASD
# ------------------------------------------------------------
def process_group_dataset(file_path, group_label):
    """
    Per un gruppo (TD o ASD):
    - carico il file Excel
    - divido per soggetto
    - calcolo Gazing STD per ogni soggetto
    - restituisco un DataFrame con una riga per soggetto
    """

    # Controllo che il file esista
    if not os.path.exists(file_path):
        print(f"ERRORE: file non trovato: {file_path}")
        return None

    print(f"\nCaricamento gruppo {group_label} ...")

    # Leggo tutti i dati
    df_all = pd.read_excel(file_path)

    # Lista dei risultati
    summary = []

    # Prendo tutti gli ID dei soggetti nel file
    subjects = df_all['id_soggetto'].unique()
    print(f"Trovati {len(subjects)} soggetti nel gruppo {group_label}.")

    # Per ogni soggetto
    for subj_id in subjects:
        # Filtro solo i frame del soggetto
        df_subj = df_all[df_all['id_soggetto'] == subj_id]

        # Calcolo la metrica
        std_val = calculate_gazing_std(df_subj)

        # Aggiungo una riga al riepilogo
        summary.append({
            "Subject_ID": subj_id,
            "Group": group_label,
            "Gazing_STD": std_val
        })

    # Converte in DataFrame
    df_summary = pd.DataFrame(summary)
    return df_summary


# ------------------------------------------------------------
# 4. ESECUZIONE
# ------------------------------------------------------------

# Calcolo statistiche per TD
df_TD_summary = process_group_dataset(path_TD_clean, "TD")

# Calcolo statistiche per ASD
df_ASD_summary = process_group_dataset(path_ASD_clean, "ASD")

# Se tutto è andato bene
if df_TD_summary is not None and df_ASD_summary is not None:

    # Unisco i due gruppi in un unico dataset
    final_stats_dataset = pd.concat([df_TD_summary, df_ASD_summary],
                                    ignore_index=True)

    # Salvo l’Excel riassuntivo
    out_path = os.path.join(base_dir, "Final_Gazing_STD_Statistics.xlsx")
    final_stats_dataset.to_excel(out_path, index=False)

    print("\n========================================")
    print("PROCESSO GAZING STD COMPLETATO")
    print("========================================")
    print(f"File salvato in: {out_path}")
    print("\nAnteprima:")
    print(final_stats_dataset.head())


    # --------------------------------------------------------
    # 5. BOX PLOT TD vs ASD
    # --------------------------------------------------------

    # Estraggo liste di valori
    td_vals = final_stats_dataset[final_stats_dataset["Group"] == "TD"]["Gazing_STD"].dropna().tolist()
    asd_vals = final_stats_dataset[final_stats_dataset["Group"] == "ASD"]["Gazing_STD"].dropna().tolist()

    # Test di Mann–Whitney
    u_stat, p_val = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")
    print(f"\nMann–Whitney Gazing_STD: U = {u_stat:.3f}, p = {p_val:.4f}")

    # Inizio il grafico
    plt.figure(figsize=(6, 5))

    # Posizioni delle due box
    positions = [1, 2]

    # Boxplot con colori
    bp = plt.boxplot([td_vals, asd_vals],
                     positions=positions,
                     tick_labels=['TD', 'ASD'],
                     patch_artist=True)

    # Colori (blu TD, arancione ASD)
    colors = ["#4c72b0", "#dd8452"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Aggiungo i puntini con jitter
    jitter = 0.06
    for i, vals in enumerate([td_vals, asd_vals], start=1):
        x = np.random.normal(loc=i, scale=jitter, size=len(vals))
        plt.scatter(x, vals, color="black", s=25, zorder=3)

    # P-value sopra la figura
    y_max = max(max(td_vals), max(asd_vals))
    y_line = y_max + 0.05
    plt.plot([1, 2], [y_line, y_line], color="black")
    plt.text(1.5, y_line + 0.02, f"p = {p_val:.4f}", ha="center")

    # Titoli e assi
    plt.title("Gazing Standard Deviation (Magnitude) per Soggetto")
    plt.ylabel("Standard Deviation")
    plt.ylim(0, y_line + 0.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Mostra grafico
    plt.show()

else:
    print("\nERRORE: impossibile completare il processo.")
