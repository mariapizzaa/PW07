import pandas as pd
import numpy as np
import os

# ======================================================
# 1. PATH SETUP
# ======================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD  = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")

output_path = os.path.join(base_dir, "Final_Gaze_Variability_Statistics.xlsx")


# ======================================================
# 2. FPS DETECTION (for Unix second timestamps)
# ======================================================
def estimate_fps_from_unix_timestamps(timestamps):
    ts = pd.Series(timestamps).astype(int)
    counts = ts.value_counts()  # frames per second
    print ("Estimated FPS counts per second:\n", counts.value_counts().sort_index())
    return int(counts.median())



# ======================================================
# 3. SLIDING-WINDOW STD
# ======================================================
def sliding_window_std(signal, window):
    signal = np.asarray(signal)

    if len(signal) < window or window <= 1:
        return np.array([np.nan])

    return np.array([
        np.std(signal[i:i + window])
        for i in range(len(signal) - window + 1)
    ])


# ======================================================
# 4. COMPUTE GAZE VARIABILITY WITH CONFIDENCE FILTERING
# ======================================================
def compute_variability(df_child):

    # ---------------------------
    # CONFIDENCE FILTERING
    # ---------------------------
    df_child = df_child[
        (df_child["confidence_yaw"]   >= 0.5) &
        (df_child["confidence_pitch"] >= 0.5)
    ].copy()

    # If too few samples remain:
    if len(df_child) < 5:
        return np.nan

    # Extract angles
    yaw_raw = df_child["yaw"].astype(float).values
    pitch_raw = df_child["pitch"].astype(float).values
    ts = df_child["timestamp"].astype(int).values

    # Center signals
    yaw = yaw_raw - np.mean(yaw_raw)
    pitch = pitch_raw - np.mean(pitch_raw)

    # Estimate FPS
    fps = estimate_fps_from_unix_timestamps(ts)
    window = max(2, fps)  # ensure valid window

    # Sliding-window STD
    yaw_std = sliding_window_std(yaw, window)
    pitch_std = sliding_window_std(pitch, window)

    # Combine per-window variability
    combined = np.sqrt(yaw_std**2 + pitch_std**2)

    # Aggregate measure
    return np.nanmean(combined)


# ======================================================
# 5. PROCESS ONE GROUP
# ======================================================
def process_group(path, label):
    df = pd.read_excel(path)
    results = []

    for subject in df["id_soggetto"].unique():
        df_sub = df[df["id_soggetto"] == subject].sort_values("timestamp")
        metric = compute_variability(df_sub)

        results.append({
            "Subject_ID": subject,
            "Group": label,
            "Gaze_Variability_STD": metric
        })

    return pd.DataFrame(results)


# ======================================================
# 6. RUN PIPELINE
# ======================================================
td_results = process_group(path_TD, "TD")
asd_results = process_group(path_ASD, "ASD")

final = pd.concat([td_results, asd_results], ignore_index=True)
final.to_excel(output_path, index=False)

print("Saved:", output_path)
print(final.head())
