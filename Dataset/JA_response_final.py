import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN
import os

# 1. PHYSICAL CONSTANTS (as in Anzalone)
TOTAL_MASS_KG = 25.0
HEAD_MASS_KG = TOTAL_MASS_KG * 0.0668
HEAD_RADIUS_M = 0.0835
HEAD_INERTIA = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)

FPS = 9.0
CUTOFF_HZ = 1.0  # 1Hz for slow movm.

# DBSCAN
DBSCAN_EPS_SEC = 0.6
DBSCAN_MIN_SAMPLES = 3

# Finestra risposta
RESPONSE_WINDOW_SEC = 4.0

# Soglie per il gaze
YAW_THRESHOLD_RAD = 0.2   # ~11°, if it is higher it is looking at the poster
PITCH_LIMIT_LOW = -1.0    # Pitch value if it is looking to foot
PITCH_LIMIT_HIGH = 0.5    # Pitch value if it is looking ceiling

# 2. DATA LOG

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

path_cones_asd = os.path.join(DATA_DIR, "ASD_cleaned_advanced.xlsx")
path_cones_td = os.path.join(DATA_DIR, "TD_cleaned_advanced.xlsx")
path_vis_asd = os.path.join(DATA_DIR, "Visual_Analysis_ASD_after.xlsx")
path_vis_td = os.path.join(DATA_DIR, "Visual_Analysis_TD_after.xlsx")

def load_data(filepath):
    try:
        if filepath.endswith(".xlsx"):
            return pd.read_excel(filepath)
        return pd.read_csv(filepath, encoding='ISO-8859-1', sep=None, engine='python')
    except Exception as e:
        print(f"Err: {e}")
        return pd.DataFrame()


df_cones_asd = load_data(path_cones_asd)
df_cones_td = load_data(path_cones_td)
df_vis_asd = load_data(path_vis_asd)
df_vis_td = load_data(path_vis_td)

df_cones_asd["Group"] = "ASD"
df_cones_td["Group"] = "TD"
df_vis_asd["Group"] = "ASD"
df_vis_td["Group"] = "TD"

df_cones_all = pd.concat([df_cones_asd, df_cones_td], ignore_index=True)
df_vis_all = pd.concat([df_vis_asd, df_vis_td], ignore_index=True)

df_cones_all.rename(columns={"id_soggetto": "Subject",
                             "frame": "Frame",
                             "timestamp": "Timestamp"}, inplace=True)
df_vis_all.rename(columns={"id_soggetto": "Subject",
                           "frame": "Frame"}, inplace=True)

# 3. FILTER
def apply_lowpass(signal, cutoff_hz, fs):
    """
    SPECTRAL ANALYSIS < 1Hz
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    # NaN VALUE
    sig = np.asarray(signal, dtype=float)
    nans = np.isnan(sig)
    if np.any(nans):
        x = np.arange(len(sig))
        sig[nans] = np.interp(x[nans], x[~nans], sig[~nans])

    return filtfilt(b, a, sig)

# 4. ENERGY & CLUSTERING

def process_subject_physics(df):
    df = df.sort_values("Frame").reset_index(drop=True).copy()

    # Interpolation
    cols = ["child_keypoint_x", "child_keypoint_y", "child_keypoint_z",
            "yaw", "pitch"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').interpolate(method='linear')

    # new time set
    dt = 1.0 / FPS

    # Convert mm -> m
    v_sq = ((df["child_keypoint_x"] / 1000).diff() / dt) ** 2 + \
           ((df["child_keypoint_y"] / 1000).diff() / dt) ** 2 + \
           ((df["child_keypoint_z"] / 1000).diff() / dt) ** 2
    e_trans = 0.5 * HEAD_MASS_KG * v_sq

    # Rotational
    yaw_uw = np.unwrap(df["yaw"].fillna(0))
    pitch_uw = np.unwrap(df["pitch"].fillna(0))
    w_sq = (np.diff(yaw_uw, prepend=yaw_uw[0]) / dt) ** 2 + \
           (np.diff(pitch_uw, prepend=pitch_uw[0]) / dt) ** 2
    e_rot = 0.5 * HEAD_INERTIA * w_sq

    df["Energy_Total"] = (e_trans + e_rot).fillna(0)

    # 1 Hz filter
    df["Energy_LP"] = apply_lowpass(df["Energy_Total"], CUTOFF_HZ, FPS)

    return df


print("calcola energia...")
df_cones_all = df_cones_all.groupby("Subject", group_keys=False).apply(process_subject_physics)


def get_movement_clusters(df_subj):
    """
    DBSCAN to find the 'temporal slices'
    """
    energy = df_subj["Energy_LP"].values
    threshold = np.nanpercentile(energy, 75)  # Soglia di 75?

    # active frame
    active_mask = energy > threshold
    if not np.any(active_mask):
        return []

    times = df_subj.loc[active_mask, "Frame"].values / FPS
    X = times.reshape(-1, 1)

    if len(X) < DBSCAN_MIN_SAMPLES:
        return []

    # DBSCAN Clustering
    db = DBSCAN(eps=DBSCAN_EPS_SEC, min_samples=DBSCAN_MIN_SAMPLES).fit(X)

    clusters = []
    for label in set(db.labels_):
        if label == -1:
            continue
        cluster_times = times[db.labels_ == label]

        # Anzalone: "center of the cluster... considered as response event"
        clusters.append({
            'center': np.mean(cluster_times),
            'start': np.min(cluster_times),
            'end': np.max(cluster_times)
        })

    return sorted(clusters, key=lambda x: x['center'])

# 5.
def validate_cluster_direction(cluster, df_subj):
    """
    check if during the movement cluster the head is looking at something

    """
    start_f = int(cluster['start'] * FPS)
    end_f = int(cluster['end'] * FPS)

    segment = df_subj[(df_subj["Frame"] >= start_f) & (df_subj["Frame"] <= end_f)]

    if segment.empty:
        return False

    # chek yaw (look left/right)
    max_yaw_deviation = segment["yaw"].abs().max()

    # check pitch
    mean_pitch = segment["pitch"].mean()

    is_lateral = max_yaw_deviation > YAW_THRESHOLD_RAD
    is_pitch_valid = (mean_pitch > PITCH_LIMIT_LOW) and (mean_pitch < PITCH_LIMIT_HIGH)

    return is_lateral and is_pitch_valid


def compute_ja_metrics(df_events, df_cones):
    """
    robot / therapist
    """
    results = []

    for (subj, admin), subj_events in df_events.groupby(["Subject", "Admin"]):

        if pd.isna(admin):
            continue

        if subj not in df_cones["Subject"].values:
            continue

        subj_cones = df_cones[df_cones["Subject"] == subj]
        clusters = get_movement_clusters(subj_cones)

        valid_clusters = [c for c in clusters if validate_cluster_direction(c, subj_cones)]

        inductions = subj_events[subj_events["Azione"].isin(["Coniglio", "Indica"])].sort_values("Frame")

        n_stimuli = len(inductions)
        n_responses = 0
        latencies = []

        for _, stim in inductions.iterrows():
            stim_time = stim["Frame"] / FPS
            candidates = [c for c in valid_clusters
                          if stim_time < c['center'] <= (stim_time + RESPONSE_WINDOW_SEC)]

            if candidates:
                match = min(candidates, key=lambda x: x['center'] - stim_time)
                n_responses += 1
                latencies.append(match['center'] - stim_time)

        results.append({
            "Subject": subj,
            "Group": subj_events["Group"].iloc[0],
            "Admin": admin,
            "N_Stimuli": n_stimuli,
            "N_Responses": n_responses,
            "Response_Rate": (n_responses / n_stimuli) if n_stimuli > 0 else 0,
            "Mean_Latency": np.mean(latencies) if latencies else np.nan
        })

    return pd.DataFrame(results)


print("Find JA...")
df_results = compute_ja_metrics(df_vis_all, df_cones_all)


# 6. STATISTIC AND PLOT

print("\n=== JA RESPONSE (Anzalone 2019, per Admin) ===")

if not df_results.empty:
    for admin in sorted(df_results["Admin"].dropna().unique()):
        print(f"\n--- Admin: {admin} ---")
        for metric in ["Response_Rate", "Mean_Latency"]:
            asd = df_results[(df_results["Group"] == "ASD") &
                             (df_results["Admin"] == admin)][metric].dropna()
            td = df_results[(df_results["Group"] == "TD") &
                            (df_results["Admin"] == admin)][metric].dropna()

            if len(asd) > 1 and len(td) > 1:
                u, p = mannwhitneyu(asd, td)
                print(f"{metric}: ASD mean={asd.mean():.2f} | TD mean={td.mean():.2f} | p={p:.4f}")
            else:
                print(f"{metric}: Dati insuff. (Admin={admin})")
else:
    print("df_results empty")

# PLOT
if not df_results.empty:
    plt.figure(figsize=(12, 5))

    # Boxplot Response Rate per Admin & Group
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_results, x="Admin", y="Response_Rate", hue="Group", palette="Set2", showfliers=False)
    plt.title("JA Response Rate per Admin")
    plt.ylabel("Response Rate")
    plt.xlabel("Admin")

    # Boxplot Mean Latency per Admin & Group
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_results, x="Admin", y="Mean_Latency", hue="Group", palette="Set2", showfliers=False)
    plt.title("JA Latency (s) per Admin")
    plt.ylabel("Latency (s)")
    plt.xlabel("Admin")

    plt.tight_layout()
    plt.show()

out_dir = os.path.join(base_dir, "results_JA")
os.makedirs(out_dir, exist_ok=True)
df_results.to_excel(os.path.join(out_dir, "JA_metrics_by_subject_admin.xlsx"), index=False)
print("\nRisultati salvati in:", out_dir)
