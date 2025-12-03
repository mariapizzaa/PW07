import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os
import sys

# ==============================================================================
# CONFIGURAZIONE CRITICA
# ==============================================================================
ASSUMED_FPS = 9.0  # FPS ricalcolato dai dati
VELOCITY_THRESHOLD = 30.0  # deg/s (Soglia standard Schmitt et al.)
MIN_SACCADE_DURATION_MS = 100.0

COL_MAP = {
    "yaw": "yaw",
    "pitch": "pitch",
    "timestamp": "timestamp",
    "frame_cutted": "frame_cutted",
    "id_soggetto": "id_soggetto"
}


# ==============================================================================
# 1. FUNZIONI DI UTILITÀ
# ==============================================================================

def reconstruct_time_axis(df, fps):
    """Ricostruisce un asse temporale preciso (in ms) basandosi sull'indice."""
    frame_indices = np.arange(len(df))
    timestamps_ms = frame_indices * (1000.0 / fps)
    return timestamps_ms


def calculate_velocity_2d(df, fps):
    """Calcola velocità angolare 2D (Yaw + Pitch)."""
    yaw_deg = np.degrees(df[COL_MAP["yaw"]])
    pitch_deg = np.degrees(df[COL_MAP["pitch"]])

    dy = yaw_deg.diff()
    dp = pitch_deg.diff()
    amplitude_deg = np.sqrt(dy ** 2 + dp ** 2)

    dt_sec = 1.0 / fps
    velocity = amplitude_deg / dt_sec

    # Gestione Tagli Video
    if COL_MAP["frame_cutted"] in df.columns:
        cut_diff = df[COL_MAP["frame_cutted"]].diff()
        mask_cuts = (cut_diff > 1) | (cut_diff < 0)
        velocity[mask_cuts] = np.nan

    return velocity


# ==============================================================================
# 2. RILEVAMENTO SACCADI (I-VT)
# ==============================================================================

def detect_saccades_ivt(velocity_series, time_series_ms, velocity_threshold=30.0):
    is_saccade = (velocity_series > velocity_threshold).fillna(False)
    change_points = np.diff(is_saccade.astype(int))
    starts = np.where(change_points == 1)[0] + 1
    ends = np.where(change_points == -1)[0] + 1

    if is_saccade.iloc[0]: starts = np.insert(starts, 0, 0)
    if is_saccade.iloc[-1]: ends = np.append(ends, len(is_saccade) - 1)

    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    elif len(ends) > len(starts):
        ends = ends[:len(starts)]

    events = []
    for s, e in zip(starts, ends):
        vel_seg = velocity_series.iloc[s:e]
        if len(vel_seg) < 1: continue

        duration = time_series_ms.iloc[e] - time_series_ms.iloc[s]
        if duration < MIN_SACCADE_DURATION_MS: continue

        peak_vel = vel_seg.max()
        mean_vel = vel_seg.mean()
        ampl = mean_vel * (duration / 1000.0)

        events.append({
            "duration_ms": duration,
            "peak_velocity": peak_vel,
            "mean_velocity": mean_vel,
            "amplitude_deg": ampl
        })
    return pd.DataFrame(events)


# ==============================================================================
# 3. ELABORAZIONE DATASET
# ==============================================================================

def process_dataframe(df, group_label):
    print(f"\n--- Elaborazione {group_label} ---")
    df.columns = [c.strip() for c in df.columns]
    subjects = df[COL_MAP["id_soggetto"]].unique()
    metrics_list = []
    raw_saccades_list = []

    for subj in subjects:
        df_subj = df[df[COL_MAP["id_soggetto"]] == subj].copy()

        # 1. Ricostruzione Tempo
        df_subj['timestamp_ms'] = reconstruct_time_axis(df_subj, fps=ASSUMED_FPS)
        # 2. Velocità
        vel = calculate_velocity_2d(df_subj, fps=ASSUMED_FPS)
        # 3. Rilevamento
        saccades = detect_saccades_ivt(vel, df_subj['timestamp_ms'], VELOCITY_THRESHOLD)

        if saccades.empty: continue

        saccades['subject_id'] = subj
        saccades['group'] = group_label
        raw_saccades_list.append(saccades)

        # 4. Statistiche
        total_time_sec = (df_subj['timestamp_ms'].iloc[-1] - df_subj['timestamp_ms'].iloc[0]) / 1000.0
        freq = len(saccades) / total_time_sec if total_time_sec > 0 else 0

        metrics_list.append({
            "Subject": subj,
            "Group": group_label,
            "Saccade_Frequency": freq,
            "Mean_Duration_ms": saccades['duration_ms'].mean(),
            "Mean_Peak_Velocity": saccades['peak_velocity'].mean(),
            "Mean_Amplitude_deg": saccades['amplitude_deg'].mean()
        })

    if not metrics_list:
        cols = ["Subject", "Group", "Saccade_Frequency", "Mean_Duration_ms", "Mean_Peak_Velocity", "Mean_Amplitude_deg"]
        return pd.DataFrame(columns=cols), pd.DataFrame()

    return pd.DataFrame(metrics_list), pd.concat(raw_saccades_list, ignore_index=True)


# ==============================================================================
# 4. ESECUZIONE E GRAFICI AVANZATI
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(base_dir, "dataset iniziali e risultati")
path_TD = os.path.join(BASE_PATH, "TD_cleaned_advanced.xlsx")
path_ASD = os.path.join(BASE_PATH, "ASD_cleaned_advanced.xlsx")

if not os.path.exists(path_TD) or not os.path.exists(path_ASD):
    print(f"ERRORE: Controlla i percorsi file in {BASE_PATH}")
    sys.exit()

print(f"Caricamento files Excel (Assumendo {ASSUMED_FPS} FPS)...")
df_TD = pd.read_excel(path_TD)
df_ASD = pd.read_excel(path_ASD)

summ_td, raw_td = process_dataframe(df_TD, "TD")
summ_asd, raw_asd = process_dataframe(df_ASD, "ASD")

if not summ_td.empty and not summ_asd.empty:
    full_summ = pd.concat([summ_td, summ_asd], ignore_index=True)
    out_file = os.path.join(BASE_PATH, "Risultati_Saccadi_Naturalistic.xlsx")
    full_summ.to_excel(out_file, index=False)
    print(f"\nSalvataggio completato: {out_file}")

    # --- BOXPLOT + TEST STATISTICI ---
    metrics_to_plot = [
        ("Mean_Duration_ms", "Duration (ms)"),
        ("Mean_Peak_Velocity", "Peak Velocity (deg/s)"),
        ("Mean_Amplitude_deg", "Amplitude (deg)"),
        ("Saccade_Frequency", "Frequency (Hz)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    print("\n--- RISULTATI STATISTICI ---")

    for i, (col, title) in enumerate(metrics_to_plot):
        ax = axes[i]

        # Dati puliti
        data_td = summ_td[col].dropna()
        data_asd = summ_asd[col].dropna()

        # Boxplot
        bp = ax.boxplot([data_td, data_asd], labels=['TD', 'ASD'], patch_artist=True, widths=0.6)

        # Colori
        colors = ['lightblue', 'orange']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Punti singoli (Jitter)
        x_td = np.random.normal(1, 0.04, size=len(data_td))
        x_asd = np.random.normal(2, 0.04, size=len(data_asd))
        ax.scatter(x_td, data_td, alpha=0.5, color='blue', s=10)
        ax.scatter(x_asd, data_asd, alpha=0.5, color='darkorange', s=10)

        # Statistica
        if len(data_td) > 0 and len(data_asd) > 0:
            u, p = mannwhitneyu(data_td, data_asd)
            print(f"{col}: p={p:.4f}")

            # Aggiungi asterisco se significativo
            if p < 0.05:
                y_max = max(data_td.max(), data_asd.max())
                h = y_max * 0.05
                ax.plot([1, 1, 2, 2], [y_max + h, y_max + h * 2, y_max + h * 2, y_max + h], lw=1.5, c='k')
                ax.text(1.5, y_max + h * 2.5, f"p={p:.3f} *", ha='center', va='bottom', color='red')

        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

else:
    print("\nERRORE: Dati insufficienti per i grafici.")