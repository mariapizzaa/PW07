import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

# ============================================================
# 1. CONSTANTS FOR HEAD ENERGY (From your snippet)
# ============================================================
TOTAL_MASS_KG = 25.0
HEAD_MASS_KG = TOTAL_MASS_KG * 0.0668
HEAD_RADIUS_M = 0.0835
HEAD_INERTIA = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)

# ============================================================
# 2. SETUP PATHS & LOAD DATA
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

# File Paths
file_cones_asd = os.path.join(DATA_DIR, "ASD_cleaned_advanced.xlsx")
file_cones_td = os.path.join(DATA_DIR, "TD_cleaned_advanced.xlsx")
file_vis_asd = os.path.join(DATA_DIR, "Visual_Analysis_ASD_after.xlsx")
file_vis_td = os.path.join(DATA_DIR, "Visual_Analysis_TD_after.xlsx")


def load_data(filepath):
    try:
        if filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        return pd.read_csv(filepath, encoding='ISO-8859-1', sep=None, engine='python')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


print("Caricamento dati...")
df_vis_asd = load_data(file_vis_asd)
df_vis_td = load_data(file_vis_td)
df_cones_asd = load_data(file_cones_asd)
df_cones_td = load_data(file_cones_td)

# Label Groups
df_vis_asd['Group'] = 'ASD'
df_vis_td['Group'] = 'TD'
df_cones_asd['Group'] = 'ASD'
df_cones_td['Group'] = 'TD'

# Concat
df_vis_all = pd.concat([df_vis_asd, df_vis_td], ignore_index=True)
df_cones_all = pd.concat([df_cones_asd, df_cones_td], ignore_index=True)

# Standardize Columns
df_cones_all.rename(columns={'id_soggetto': 'Subject', 'frame': 'Frame', 'timestamp': 'Timestamp'}, inplace=True)
df_cones_all['Timestamp'] = pd.to_numeric(df_cones_all['Timestamp'], errors='coerce')


# ============================================================
# 3. IMPLEMENTAZIONE CALCOLO ENERGIA (Tuoi algoritmi)
# ============================================================
def add_instantaneous_energy(df):
    """
    Adds 'Energy_Total' column to the dataframe using Anzalone's physics constants.
    Calculates instantaneous energy per frame.
    """
    # Sort just in case
    df = df.sort_values("Timestamp")

    # --- A. PREPARE DATA ---
    # Convert from mm to meters
    x_m = df["child_keypoint_x"].astype(float) / 1000.0
    y_m = df["child_keypoint_y"].astype(float) / 1000.0
    z_m = df["child_keypoint_z"].astype(float) / 1000.0

    # Yaw/Pitch
    yaw = df["yaw"].astype(float)
    pitch = df["pitch"].astype(float)

    # Time difference
    dt = df["Timestamp"].diff().astype(float)
    # Filter invalid time steps (jumps > 10s or <= 0)
    valid_mask = (dt > 0) & (dt < 10.0)

    # --- B. TRANSLATIONAL ENERGY ---
    dx = x_m.diff()
    dy = y_m.diff()
    dz = z_m.diff()
    dist_sq = dx ** 2 + dy ** 2 + dz ** 2

    # Default FPS fallback (1/9s) if dt is messy, but prefer real dt
    DT_DEFAULT = 1.0 / 9.0

    vel_sq = dist_sq / (DT_DEFAULT ** 2)  # Init with default
    vel_sq[valid_mask] = dist_sq[valid_mask] / (dt[valid_mask] ** 2)  # Update with real time

    energy_trans = 0.5 * HEAD_MASS_KG * vel_sq

    # --- C. ROTATIONAL ENERGY ---
    d_yaw = yaw.diff()
    d_pitch = pitch.diff()

    # Unwrap yaw (handle -pi to pi jumps)
    d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

    ang_dist_sq = d_yaw ** 2 + d_pitch ** 2

    ang_vel_sq = ang_dist_sq / (DT_DEFAULT ** 2)
    ang_vel_sq[valid_mask] = ang_dist_sq[valid_mask] / (dt[valid_mask] ** 2)

    energy_rot = 0.5 * HEAD_INERTIA * ang_vel_sq

    # --- D. TOTAL ENERGY ---
    # Fill NA (first frame) with 0
    df['Energy_Total'] = (energy_trans + energy_rot).fillna(0)

    return df


print("Calcolo energie per ogni frame (questo potrebbe richiedere qualche secondo)...")
# Apply calculation per subject
df_cones_all = df_cones_all.groupby('Subject', group_keys=False).apply(add_instantaneous_energy)

# Index map for fast lookup
# We need to look up energy by (Subject, Frame)
energy_map = df_cones_all.drop_duplicates(subset=['Subject', 'Frame']).set_index(['Subject', 'Frame'])[
    'Energy_Total'].to_dict()
# Soglia per soggetto: mean + 1 * std (puoi cambiare z se vuoi)
z_threshold = 1.0

threshold_map = (
    df_cones_all
    .groupby('Subject')['Energy_Total']
    .agg(['mean', 'std'])
    .dropna()
)
threshold_map['threshold'] = threshold_map['mean'] + z_threshold * threshold_map['std']

# dict: Subject -> threshold
subject_threshold = threshold_map['threshold'].to_dict()


def calculate_dynamic_energy_response(df_events, energy_map, fps=9.0):
    """
    Calcola la risposta energetica RELATIVA (Delta).
    1. Baseline: 1 secondo PRIMA dello stimolo.
    2. Response: 3 secondi DOPO lo stimolo.
    3. Metrica: Max Energy nella finestra - Baseline.
    4. Latenza: Tempo dal 'bip' al picco massimo di energia.
    """
    results = []

    # Parametri finestra
    pre_window_sec = 1.0  # Tempo prima per la baseline
    post_window_sec = 3.0  # Tempo dopo per la risposta

    frames_pre = int(pre_window_sec * fps)
    frames_post = int(post_window_sec * fps)

    for subject, sub_df in df_events.groupby('Subject'):
        sub_df = sub_df.sort_values('Frame')

        for idx, row in sub_df.iterrows():
            action = row['Azione']
            start_frame = row['Frame']
            admin = row['Admin']
            group = row['Group']

            if action in ['Coniglio', 'Indica']:

                # A. Estrai Baseline (1 sec prima)
                baseline_vals = []
                for f in range(start_frame - frames_pre, start_frame):
                    val = energy_map.get((subject, f))
                    if val is not None: baseline_vals.append(val)

                # Se non abbiamo dati prima (es. inizio video), usiamo 0 o saltiamo
                baseline_energy = np.median(baseline_vals) if len(baseline_vals) > 0 else 0.0

                # B. Estrai Risposta (3 sec dopo)
                response_vals = []
                response_frames = []
                for f in range(start_frame, start_frame + frames_post):
                    val = energy_map.get((subject, f))
                    if val is not None:
                        response_vals.append(val)
                        response_frames.append(f)

                if len(response_vals) == 0:
                    continue

                response_vals = np.array(response_vals)

                # --- METRICHE MIGLIORATE ---

                # 1. Delta Energy (Picco massimo - Baseline)
                # Indica l'intensità del movimento di reazione "pulita" dal rumore di fondo
                max_energy = np.max(response_vals)
                delta_energy = max_energy - baseline_energy

                # 2. Peak Latency (Tempo per raggiungere il massimo sforzo)
                # Molto più robusto della soglia: ci dice QUANDO ha fatto il movimento principale
                idx_max = np.argmax(response_vals)
                frame_max = response_frames[idx_max]
                peak_latency = (frame_max - start_frame) / fps

                # 3. Reactivity (Rate): Consideriamo "Risposta" se il picco è almeno il 20% superiore alla baseline
                # Questo evita di contare il rumore come risposta
                has_responded = 1 if delta_energy > (baseline_energy * 0.20) else 0

                results.append({
                    'Subject': subject,
                    'Group': group,
                    'Admin': admin,
                    'Induction_Type': action,
                    'Delta_Energy': delta_energy,  # Intensità Reazione
                    'Peak_Latency': peak_latency,  # Velocità Reazione
                    'Energy_Reactivity': has_responded  # Rate (basato su incremento)
                })

    return pd.DataFrame(results)


print("Calcolo metriche dinamiche (Baseline Correction)...")
df_dynamic_res = calculate_dynamic_energy_response(df_vis_all, energy_map)

# ============================================================
# 5. STATISTICA SUI NUOVI DATI
# ============================================================

subject_metrics_dyn = df_dynamic_res.groupby(['Subject', 'Group', 'Admin']).agg(
    Avg_Delta_Energy=('Delta_Energy', 'mean'),
    Avg_Peak_Latency=('Peak_Latency', 'mean'),
    Reactivity_Rate=('Energy_Reactivity', 'mean')
).reset_index()

print("\n=== ANALISI STATISTICA (BASELINE CORRECTED) ===")


def run_test(df, metric, admin):
    asd = df[(df['Group'] == 'ASD') & (df['Admin'] == admin)][metric].dropna()
    td = df[(df['Group'] == 'TD') & (df['Admin'] == admin)][metric].dropna()

    if len(asd) == 0 or len(td) == 0: return

    stat, p = mannwhitneyu(asd, td)
    sig = "**SIGNIFICATIVO**" if p < 0.05 else "Non significativo"

    print(f"\nCondition: {admin} | Metric: {metric}")
    print(f"  > ASD: {asd.mean():.4f} | TD: {td.mean():.4f}")
    print(f"  > P-value: {p:.4f} ({sig})")


# Intensità della risposta (Sforzo netto)
run_test(subject_metrics_dyn, 'Avg_Delta_Energy', 'Robot')
run_test(subject_metrics_dyn, 'Avg_Delta_Energy', 'Therapist')

# Velocità della risposta (Tempo al picco)
run_test(subject_metrics_dyn, 'Avg_Peak_Latency', 'Robot')
run_test(subject_metrics_dyn, 'Avg_Peak_Latency', 'Therapist')

# Rateo di reazione (Quante volte incrementano l'energia)
run_test(subject_metrics_dyn, 'Reactivity_Rate', 'Robot')

# ============================================================
# 6. VISUALIZZAZIONE DELTA
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Intensità (Delta Energy)
sns.boxplot(x='Admin', y='Avg_Delta_Energy', hue='Group', data=subject_metrics_dyn, ax=axes[0], palette="Set2")
axes[0].set_title('Intensity of Response (Delta Energy)')
axes[0].set_ylabel('Joule (Max - Baseline)')

# Plot 2: Latenza (Peak Latency)
sns.boxplot(x='Admin', y='Avg_Peak_Latency', hue='Group', data=subject_metrics_dyn, ax=axes[1], palette="Set2")
axes[1].set_title('Speed of Response (Peak Latency)')
axes[1].set_ylabel('Seconds to Max Energy')

plt.tight_layout()
plt.show()