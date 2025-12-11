import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os
import sys

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

ASSUMED_FPS = 9.0  # FPS stimato dai dati

# Soglia equivalente a 30 deg/s in radianti/s ≈ 0.524
VEL_THRESHOLD_DEG = 30.0
VELOCITY_THRESHOLD = np.deg2rad(VEL_THRESHOLD_DEG)

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
    frame_indices = np.arange(len(df))
    timestamps_ms = frame_indices * (1000.0 / fps)
    return timestamps_ms


def gaze_angles_to_cartesian(df):
    yaw = df[COL_MAP["yaw"]].astype(float)
    pitch = df[COL_MAP["pitch"]].astype(float)

    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return pd.DataFrame({"x": x, "y": y, "z": z}, index=df.index)


def calculate_velocity_3d_cartesian(df, fps):
    coords = gaze_angles_to_cartesian(df)

    dx = coords["x"].diff()
    dy = coords["y"].diff()
    dz = coords["z"].diff()

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    velocity = dist / (1.0 / fps)

    # Correggo tagli video
    if COL_MAP["frame_cutted"] in df.columns:
        cut_diff = df[COL_MAP["frame_cutted"]].diff()
        mask = (cut_diff > 1) | (cut_diff < 0)
        velocity[mask] = np.nan

    return velocity, coords


# ==============================================================================
# 2. RILEVAMENTO MOVIMENTI OCULARI (I-VT)
# ==============================================================================

def detect_eye_movements_ivt(velocity_series, time_series_ms, coords,
                             velocity_threshold=VELOCITY_THRESHOLD):

    is_move = (velocity_series > velocity_threshold).fillna(False)

    change = np.diff(is_move.astype(int))
    starts = np.where(change == 1)[0] + 1
    ends = np.where(change == -1)[0] + 1

    if is_move.iloc[0]:
        starts = np.insert(starts, 0, 0)

    if is_move.iloc[-1]:
        ends = np.append(ends, len(is_move) - 1)

    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    elif len(ends) > len(starts):
        ends = ends[:len(starts)]

    events = []
    for s, e in zip(starts, ends):

        vel_seg = velocity_series.iloc[s:e+1]
        if len(vel_seg) < 1:
            continue

        t_start = time_series_ms.iloc[s]
        t_end = time_series_ms.iloc[e]
        duration = t_end - t_start  # ms

        peak_vel = vel_seg.max()
        mean_vel = vel_seg.mean()

        # Ampiezza = distanza tra inizio e fine
        idx_s = velocity_series.index[s]
        idx_e = velocity_series.index[e]
        p_start = coords.loc[idx_s]
        p_end = coords.loc[idx_e]
        amplitude = np.sqrt(((p_end - p_start)**2).sum())

        events.append({
            "duration_ms": duration,
            "peak_velocity": peak_vel,
            "mean_velocity": mean_vel,
            "amplitude_cartesian": amplitude
        })

    return pd.DataFrame(events)


# ==============================================================================
# 3. ELABORAZIONE PER SOGGETTO
# ==============================================================================

def process_dataframe(df, group_label):

    print(f"\n--- Elaborazione gruppo {group_label} ---")

    df.columns = [c.strip() for c in df.columns]
    subjects = df[COL_MAP["id_soggetto"]].unique()

    summary_list = []
    raw_list = []

    for subj in subjects:

        df_sub = df[df[COL_MAP["id_soggetto"]] == subj].copy()
        df_sub["timestamp_ms"] = reconstruct_time_axis(df_sub, ASSUMED_FPS)

        vel, coords = calculate_velocity_3d_cartesian(df_sub, ASSUMED_FPS)

        events = detect_eye_movements_ivt(
            vel, df_sub["timestamp_ms"], coords,
            velocity_threshold=VELOCITY_THRESHOLD
        )

        if events.empty:
            continue

        events["subject_id"] = subj
        events["group"] = group_label
        raw_list.append(events)

        total_time = (df_sub["timestamp_ms"].iloc[-1] -
                      df_sub["timestamp_ms"].iloc[0]) / 1000.0

        freq = len(events) / total_time if total_time > 0 else 0

        summary_list.append({
            "Subject": subj,
            "Group": group_label,
            "EyeMovement_Frequency_Hz": freq,
            "Mean_Duration_ms": events["duration_ms"].mean(),
            "Mean_Peak_Velocity": events["peak_velocity"].mean(),
            "Mean_Amplitude_Cartesian": events["amplitude_cartesian"].mean()
        })

    if not summary_list:
        return pd.DataFrame(), pd.DataFrame()

    return pd.DataFrame(summary_list), pd.concat(raw_list, ignore_index=True)


# ==============================================================================
# 4. MAIN: CARICA FILE + ANALISI + GRAFICI
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD = os.path.join(base_dir, "TD_cleaned.xlsx")
path_ASD = os.path.join(base_dir, "ASD_cleaned.xlsx")

if not os.path.exists(path_TD) or not os.path.exists(path_ASD):
    print("ERRORE: File Excel non trovati!")
    print("path_TD:", path_TD, os.path.exists(path_TD))
    print("path_ASD:", path_ASD, os.path.exists(path_ASD))
    sys.exit()

print("\nCaricamento file...")
df_TD = pd.read_excel(path_TD)
df_ASD = pd.read_excel(path_ASD)

summary_td, raw_td = process_dataframe(df_TD, "TD")
summary_asd, raw_asd = process_dataframe(df_ASD, "ASD")

if summary_td.empty or summary_asd.empty:
    print("\nERRORE: Dati insufficienti per creare grafici o statistiche.")
    sys.exit()

full_summary = pd.concat([summary_td, summary_asd], ignore_index=True)

out_path = os.path.join(base_dir, "EyeMovement_Results.xlsx")
full_summary.to_excel(out_path, index=False)
print(f"\nRisultati salvati in: {out_path}")

# ------------------------------------------------------------------------------
# BOX PLOT + TEST STATISTICO
# ------------------------------------------------------------------------------

metrics = [
    ("EyeMovement_Frequency_Hz", "Eye movement frequency (Hz)"),
    ("Mean_Duration_ms", "Mean duration (ms)"),
    ("Mean_Peak_Velocity", "Mean peak velocity (unit/s)"),
    ("Mean_Amplitude_Cartesian", "Mean amplitude (3D distance)")
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

print("\n--- TEST STATISTICO MANN–WHITNEY (TD vs ASD) ---")

for i, (col, title) in enumerate(metrics):
    ax = axes[i]

    td_vals = summary_td[col].dropna()
    asd_vals = summary_asd[col].dropna()

    # Boxplot con colori
    bp = ax.boxplot(
        [td_vals, asd_vals],
        labels=["TD", "ASD"],
        patch_artist=True,
        widths=0.6
    )

    colors = ["#6BAED6", "#FD8D3C"]  # azzurrino, arancio
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    # Punti singoli con jitter
    x_td = np.random.normal(1, 0.04, size=len(td_vals))
    x_asd = np.random.normal(2, 0.04, size=len(asd_vals))
    ax.scatter(x_td, td_vals, s=12, alpha=0.6, color="#2171B5")
    ax.scatter(x_asd, asd_vals, s=12, alpha=0.6, color="#D94801")

    # Test Mann–Whitney
    if len(td_vals) > 0 and len(asd_vals) > 0:
        u, p = mannwhitneyu(td_vals, asd_vals)
        print(f"{col}: p = {p:.4f}")

        # Barra e asterisco se significativo
        y_max = max(td_vals.max(), asd_vals.max())
        h = (y_max * 0.05) if y_max != 0 else 0.1
        ax.set_title(title)

        if p < 0.05:
            ax.plot([1, 1, 2, 2],
                    [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h],
                    lw=1.5, c="k")
            ax.text(1.5, y_max + 2.6*h,
                    f"p={p:.3f} *",
                    ha="center", va="bottom", fontsize=9, color="red")
        else:
            ax.text(1.5, y_max + 1.8*h,
                    f"p={p:.3f}",
                    ha="center", va="bottom", fontsize=9, color="black")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Salva figura per le slide
fig_path = os.path.join(base_dir, "EyeMovement_Boxplots.png")
plt.savefig(fig_path, dpi=300)
print(f"\nFigura boxplot salvata in: {fig_path}")

plt.show()
