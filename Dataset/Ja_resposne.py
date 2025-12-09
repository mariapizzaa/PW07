# Ja_response.py
# ==========================================
# Joint Attention (JA) Response Rate & Latency
# Implementazione ispirata ad Anzalone et al.
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN

# ============================================================
# 1. PHYSICAL CONSTANTS
# ============================================================
TOTAL_MASS_KG = 25.0
HEAD_MASS_KG = TOTAL_MASS_KG * 0.0668
HEAD_RADIUS_M = 0.0835
HEAD_INERTIA = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)

FPS = 9.0  # frame rate (Hz) dai dati

# Filtro per movimenti lenti (Anzalone: < ~1 Hz)
CUTOFF_HZ = 1.0

# DBSCAN: parametri per cluster temporali (secondi)
DBSCAN_EPS_SEC = 0.6      # punti entro 0.6 s fusi nello stesso gesto
DBSCAN_MIN_SAMPLES = 3    # minimo #frame sopra soglia per chiamarlo gesto

# Finestra di risposta alla JA (secondi)
RESPONSE_WINDOW_SEC = 4.0

# Percentile per threshold energetico (robusto)
ENERGY_PERCENTILE = 75.0

# ============================================================
# 2. DATA LOADING
# ============================================================
# --- PATH (ADATTA A PARTIRE DALLA TUA CARTELLA PROGETTO) ---

# cartella dove si trova questo script Ja_resposne.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# sottocartella con i dataset (come usavi negli altri script)
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

# Nomi file effettivi (versione _advanced / _after)
path_cones_asd = os.path.join(DATA_DIR, "ASD_cleaned_advanced.xlsx")
path_cones_td = os.path.join(DATA_DIR, "TD_cleaned_advanced.xlsx")
path_vis_asd  = os.path.join(DATA_DIR, "Visual_Analysis_ASD_after.xlsx")
path_vis_td   = os.path.join(DATA_DIR, "Visual_Analysis_TD_after.xlsx")

def load_data(filepath):
    print(f"Caricamento: {filepath}")
    try:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".xlsx"):
            return pd.read_excel(filepath)
        else:
            print(f"Formato non supportato: {filepath}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Errore nel caricamento di {filepath}: {e}")
        return pd.DataFrame()


# Pose/gaze (head kinematics)
df_cones_asd = load_data(path_cones_asd)
df_cones_td = load_data(path_cones_td)

# Eventi (JA inductions: Coniglio / Indica)
df_vis_asd = load_data(path_vis_asd)
df_vis_td = load_data(path_vis_td)

# Etichetta gruppi
df_cones_asd["Group"] = "ASD"
df_cones_td["Group"] = "TD"
df_vis_asd["Group"]  = "ASD"
df_vis_td["Group"]   = "TD"

# Merge
df_cones_all = pd.concat([df_cones_asd, df_cones_td], ignore_index=True)
df_vis_all   = pd.concat([df_vis_asd, df_vis_td], ignore_index=True)

# Uniforma nomi colonne
df_cones_all.rename(
    columns={
        "id_soggetto": "Subject",
        "frame": "Frame",
        "timestamp": "Timestamp",
    },
    inplace=True,
)

df_vis_all.rename(
    columns={
        "id_soggetto": "Subject",
        "frame": "Frame",
    },
    inplace=True,
)

# ============================================================
# 3. LOW-PASS FILTER (Butterworth)
# ============================================================

def butter_lowpass(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_lowpass(signal, cutoff_hz, fs):
    """
    Applica filtro low-pass a 1 Hz (default) sull'array 1D 'signal'.
    Usa filtfilt per non introdurre sfasamento.
    """
    b, a = butter_lowpass(cutoff_hz, fs, order=4)
    # Gestione edge case: tutti NaN o costante
    sig = np.asarray(signal, dtype=float)
    if np.all(np.isnan(sig)):
        return np.zeros_like(sig)
    # Sostituisci NaN con interpolazione semplice
    nans = np.isnan(sig)
    if np.any(nans):
        idx = np.arange(len(sig))
        sig[nans] = np.interp(idx[nans], idx[~nans], sig[~nans])
    return filtfilt(b, a, sig)


# ============================================================
# 4. HEAD ENERGY PER SUBJECT
# ============================================================

def compute_head_energy_for_subject(df_subject):
    """
    Per un singolo soggetto:
      1. Ordina per frame
      2. Interpola missing per child_keypoint_x/y/z, yaw, pitch
      3. Converte posizioni in m
      4. Calcola velocità traslazionale e energia cinetica
      5. Unwrap yaw/pitch, calcola velocità angolari e energia rotazionale
      6. Somma: Energy_Total
    """
    df = df_subject.sort_values("Frame").reset_index(drop=True).copy()

    cols_to_fix = ["child_keypoint_x", "child_keypoint_y", "child_keypoint_z",
                   "yaw", "pitch"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Posizioni in metri (erano mm)
    x_m = df["child_keypoint_x"] / 1000.0
    y_m = df["child_keypoint_y"] / 1000.0
    z_m = df["child_keypoint_z"] / 1000.0

    dt = 1.0 / FPS

    # Velocità traslazionale
    vx = x_m.diff() / dt
    vy = y_m.diff() / dt
    vz = z_m.diff() / dt
    v_sq = vx**2 + vy**2 + vz**2
    e_trans = 0.5 * HEAD_MASS_KG * v_sq

    # Velocità angolare (yaw/pitch in radianti)
    # np.unwrap elimina salti tipo -pi / +pi
    yaw = np.unwrap(df["yaw"].to_numpy(dtype=float))
    pitch = np.unwrap(df["pitch"].to_numpy(dtype=float))

    dyaw = np.diff(yaw, prepend=yaw[0]) / dt
    dpitch = np.diff(pitch, prepend=pitch[0]) / dt

    omega_sq = dyaw**2 + dpitch**2
    e_rot = 0.5 * HEAD_INERTIA * omega_sq

    energy_total = (e_trans + e_rot).fillna(0.0)

    df["Energy_Total"] = energy_total

    # Filtro low-pass sulla totale per isolare movimenti lenti (intenzionali)
    df["Energy_LP"] = apply_lowpass(df["Energy_Total"].values, CUTOFF_HZ, FPS)

    return df


print("Calcolo energia fisica della testa per ogni soggetto...")
df_cones_all = df_cones_all.groupby("Subject", group_keys=False).apply(
    compute_head_energy_for_subject
)

# ============================================================
# 5. DETECTION: SLOW HEAD MOVEMENTS (DBSCAN)
# ============================================================

def detect_slow_movements_subject(df_subject):
    """
    Implementa pipeline Anzalone:
      - usa Energy_LP (low-pass <1Hz)
      - soglia per frames "attivi"
      - DBSCAN sui tempi (sec) dei frame attivi
      - restituisce lista dei centri temporali (in secondi) dei cluster (gesti)
    """
    df = df_subject.sort_values("Frame")

    energy = df["Energy_LP"].to_numpy(dtype=float)

    # Escludi casi banali
    if np.all(np.isclose(energy, energy[0])):
        return []

    # Threshold robusto = percentile (es. 75°)
    thr = np.nanpercentile(energy, ENERGY_PERCENTILE)

    active = df.loc[energy >= thr, "Frame"].to_numpy(dtype=float)

    if active.size < DBSCAN_MIN_SAMPLES:
        return []

    # Converti in secondi
    times_sec = active / FPS
    X = times_sec.reshape(-1, 1)

    db = DBSCAN(
        eps=DBSCAN_EPS_SEC,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="euclidean",
    ).fit(X)

    labels = db.labels_
    unique_labels = sorted([lab for lab in set(labels) if lab != -1])

    centers_sec = []
    for lab in unique_labels:
        cluster_times = times_sec[labels == lab]
        if cluster_times.size > 0:
            centers_sec.append(cluster_times.mean())

    centers_sec = sorted(centers_sec)
    return centers_sec


print("Rilevamento movimenti lenti della testa (analisi spettrale + soglia + DBSCAN)...")

# Dizionario: Subject -> [tempi_sec dei centri cluster]
movement_events = {}
for subj, df_sub in df_cones_all.groupby("Subject"):
    centers = detect_slow_movements_subject(df_sub)
    movement_events[subj] = centers

# ============================================================
# 6. MATCHING: JA INDUCTIONS -> HEAD MOVEMENT EVENTS
# ============================================================

def compute_ja_trials(df_events, movement_events):
    """
    Per ogni evento di JA induction (Coniglio / Indica):
      - trova il cluster di movimento lento più vicino
        nel range (stim_time, stim_time + RESPONSE_WINDOW_SEC]
      - se trovato: Responded=1, Latency = t_cluster - t_stim
      - altrimenti: Responded=0, Latency = NaN
    """
    rows = []

    for idx, row in df_events.iterrows():
        action = row.get("Azione", None)
        if action not in ["Coniglio", "Indica"]:
            continue

        subj = row["Subject"]
        frame = row["Frame"]
        admin = row.get("Admin", None)
        group = row.get("Group", None)

        stim_time = frame / FPS
        centers = movement_events.get(subj, [])

        # Cerca centri successivi allo stimolo entro la finestra di 4s
        candidate_times = [
            t for t in centers if (t > stim_time) and (t <= stim_time + RESPONSE_WINDOW_SEC)
        ]

        if len(candidate_times) > 0:
            # centro più vicino temporalmente allo stimolo
            chosen = min(candidate_times, key=lambda t: abs(t - stim_time))
            responded = 1
            latency = chosen - stim_time
        else:
            responded = 0
            latency = np.nan

        rows.append(
            {
                "Subject": subj,
                "Group": group,
                "Admin": admin,
                "Induction": action,
                "Stim_Frame": frame,
                "Stim_Time": stim_time,
                "Responded": responded,
                "Latency": latency,  # in secondi
            }
        )

    return pd.DataFrame(rows)


print("Calcolo head movement response to JA induction (stile Anzalone adattato)...")
df_ja_trials = compute_ja_trials(df_vis_all, movement_events)

# ============================================================
# 7. METRICHE: RESPONSE RATE & LATENCY
# ============================================================

def compute_subject_level_metrics(df_ja_trials):
    """
    Aggrega a livello di soggetto:
      - Response_Rate: % stimoli con Responded=1
      - Mean_Latency: media Latency solo sulle risposte presenti
    """
    # mean sulla colonna boolean/binary = rate
    agg_rows = []

    for (subj, group, admin), sub_df in df_ja_trials.groupby(
        ["Subject", "Group", "Admin"]
    ):
        n_stim = len(sub_df)
        if n_stim == 0:
            continue

        resp_rate = sub_df["Responded"].mean()

        latencies = sub_df.loc[sub_df["Responded"] == 1, "Latency"]
        mean_lat = latencies.mean() if len(latencies) > 0 else np.nan

        agg_rows.append(
            {
                "Subject": subj,
                "Group": group,
                "Admin": admin,
                "N_Stimuli": n_stim,
                "Response_Rate": resp_rate,
                "Mean_Latency": mean_lat,
            }
        )

    return pd.DataFrame(agg_rows)


df_subject_metrics = compute_subject_level_metrics(df_ja_trials)

# ============================================================
# 8. STATISTICHE (ASD vs TD) PER ADMIN (ROBOT / THERAPIST)
# ============================================================

def run_mannwhitney_for_metric(df, metric, admin_label):
    asd = df[(df["Group"] == "ASD") & (df["Admin"] == admin_label)][metric].dropna()
    td = df[(df["Group"] == "TD") & (df["Admin"] == admin_label)][metric].dropna()

    if len(asd) < 2 or len(td) < 2:
        print(f"[ATTENZIONE] Dati insufficienti per test {metric} | {admin_label}")
        return

    stat, p = mannwhitneyu(asd, td, alternative="two-sided")
    signif = "SIGNIFICATIVO" if p < 0.05 else "non significativo"

    print(f"\nCondition: {admin_label} | Metric: {metric}")
    print(f"  ASD: {asd.mean():.3f} | TD: {td.mean():.3f}")
    print(f"  p-value: {p:.4f} -> {signif}")


print("\n=== HEAD MOVEMENT RESPONSE TO JA INDUCTION (ROBOT/THERAPIST) ===")
for admin_label in sorted(df_subject_metrics["Admin"].dropna().unique()):
    run_mannwhitney_for_metric(df_subject_metrics, "Response_Rate", admin_label)
    run_mannwhitney_for_metric(df_subject_metrics, "Mean_Latency", admin_label)

# ============================================================
# 9. VISUALIZZAZIONE (OPZIONALE)
# ============================================================

if not df_subject_metrics.empty:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df_subject_metrics,
        x="Admin",
        y="Response_Rate",
        hue="Group",
        showfliers=False,
    )
    plt.title("JA Response Rate")
    plt.ylabel("Response Rate (0-1)")
    plt.xlabel("Admin")

    plt.subplot(1, 2, 2)
    sns.boxplot(
        data=df_subject_metrics,
        x="Admin",
        y="Mean_Latency",
        hue="Group",
        showfliers=False,
    )
    plt.title("JA Latency (solo risposte)")
    plt.ylabel("Latency [s]")
    plt.xlabel("Admin")

    plt.tight_layout()
    plt.show()

# ============================================================
# 10. SALVATAGGIO RISULTATI (FACOLTATIVO)
# ============================================================

out_dir = os.path.join(base_dir, "results_JA")
os.makedirs(out_dir, exist_ok=True)

df_ja_trials.to_excel(os.path.join(out_dir, "JA_trials_detail.xlsx"), index=False)
df_subject_metrics.to_excel(os.path.join(out_dir, "JA_subject_metrics.xlsx"), index=False)

print("\nRisultati salvati in:", out_dir)
print("Script terminato.")
