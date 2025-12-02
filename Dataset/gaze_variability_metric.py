import pandas as pd
import numpy as np
import os

# ======================================================
# 1. PATH SETUP
# ======================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
path_TD_clean  = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")
output_path = os.path.join(base_dir, "Final_Gaze_Variability_Statistics.xlsx")


# ======================================================
# 2. NEW FPS DETECTOR (for Unix-second timestamps)
# ======================================================
def estimate_fps_from_unix_timestamps(timestamps):
    ts = pd.Series(timestamps).astype(int)
    counts = ts.value_counts()       # rows per second
    fps = counts.median()            # most representative FPS
    return int(fps)


# ======================================================
# 3. SLIDING-WINDOW STD
# ======================================================
def sliding_window_std(signal, window_size):
    signal = np.asarray(signal)

    if len(signal) < window_size or window_size <= 1:
        return np.array([np.nan])

    stds = []
    for i in range(len(signal) - window_size + 1):
        stds.append(np.std(signal[i:i + window_size]))

    return np.array(stds)


# ======================================================
# 4. METRIC FOR ONE SUBJECT
# ======================================================
def compute_gaze_variability_for_subject(df_subject):
    df = df_subject.copy()

    yaw = df["yaw"].astype(float).values
    pitch = df["pitch"].astype(float).values
    timestamps = df["timestamp"].astype(int).values

    # FIXED FPS DETECTION
    fps = estimate_fps_from_unix_timestamps(timestamps)
    window_size = fps  # 1-second window

    # Compute sliding-window variability
    yaw_std = sliding_window_std(yaw, window_size)
    pitch_std = sliding_window_std(pitch, window_size)

    combined = np.sqrt(yaw_std**2 + pitch_std**2)
    return np.nanmean(combined)


# ======================================================
# 5. PROCESS GROUP
# ======================================================
def process_group_dataset(file_path, group_label):
    df_all = pd.read_excel(file_path)

    summary_rows = []
    subjects = df_all["id_soggetto"].unique()

    print(f"{group_label}: Found {len(subjects)} subjects.")

    for subj in subjects:
        df_sub = df_all[df_all["id_soggetto"] == subj].sort_values("timestamp")
        gaze_var = compute_gaze_variability_for_subject(df_sub)

        summary_rows.append({
            "Subject_ID": subj,
            "Group": group_label,
            "Gaze_Variability_STD": gaze_var
        })

    return pd.DataFrame(summary_rows)


# ======================================================
# 6. RUN
# ======================================================
td_summary  = process_group_dataset(path_TD_clean, "TD")
asd_summary = process_group_dataset(path_ASD_clean, "ASD")

final_df = pd.concat([td_summary, asd_summary], ignore_index=True)
final_df.to_excel(output_path, index=False)

print("\nSaved gaze variability results to:")
print(output_path)
print("\nPreview:")
print(final_df.head())
