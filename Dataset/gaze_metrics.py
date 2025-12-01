import os
import pandas as pd
import numpy as np
import os

# --- 1. SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))

path_TD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "TD_cleaned_advanced.xlsx")
path_ASD_clean = os.path.join(base_dir, "dataset iniziali e risultati", "ASD_cleaned_advanced.xlsx")


# --- 2. METRICHE GAZE (single subject) ---


def compute_gaze_metrics_for_subject(df_subject):
    """
    Computes gaze yaw std, pitch std, and magnitude std
    for a single subject according to Anzalone et al. (2019).
    """

    df = df_subject.copy()

    # Center yaw and pitch (Anzalone: displacement from mean)
    yaw = df["yaw"].astype(float)
    pitch = df["pitch"].astype(float)

    yaw_centered = yaw - yaw.mean()
    pitch_centered = pitch - pitch.mean()

    # 2.1 Gaze magnitude
    magnitude = np.sqrt(yaw_centered**2 + pitch_centered**2)

    metrics = {
        "gaze_yaw_std": np.std(yaw_centered),
        "gaze_pitch_std": np.std(pitch_centered),
        "gaze_magnitude_std": np.std(magnitude)
    }

    return metrics


# 3. PROCESSING FOR A WHOLE GROUP (ASD/TD)

def process_group_dataset(file_path, group_label):
    """
    Loads a dataset, iterates through each subject, computes gaze metrics,
    and returns two tables: (full_frame_dataset, summary_per_subject)
    """

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None, None

    print(f"\nLoading {group_label} data...")
    df_all = pd.read_excel(file_path)

    summary_list = []
    full_subject_dfs = []

    subjects = df_all["id_soggetto"].unique()
    print(f"Found {len(subjects)} subjects in {group_label} group.")

    for subj in subjects:

        df_subj = df_all[df_all["id_soggetto"] == subj].sort_values("timestamp")

        metrics = compute_gaze_metrics_for_subject(df_subj)

        # Save metrics to summary
        summary_list.append({
            "Subject_ID": subj,
            "Group": group_label,
            "Gaze_Yaw_STD": metrics["gaze_yaw_std"],
            "Gaze_Pitch_STD": metrics["gaze_pitch_std"],
            "Gaze_Magnitude_STD": metrics["gaze_magnitude_std"]
        })

        # (Optional) attach results to full dataframe
        df_subj["gaze_yaw_centered"] = df_subj["yaw"] - df_subj["yaw"].mean()
        df_subj["gaze_pitch_centered"] = df_subj["pitch"] - df_subj["pitch"].mean()
        df_subj["gaze_magnitude"] = np.sqrt(
            df_subj["gaze_yaw_centered"]**2 + df_subj["gaze_pitch_centered"]**2
        )

        full_subject_dfs.append(df_subj)

    df_full = pd.concat(full_subject_dfs, ignore_index=True)
    df_summary = pd.DataFrame(summary_list)

    return df_full, df_summary


# 4. EXECUTION — RUN FOR TD AND ASD

df_TD_full, df_TD_summary = process_group_dataset(path_TD_clean, "TD")
df_ASD_full, df_ASD_summary = process_group_dataset(path_ASD_clean, "ASD")


# 5. SAVE OUTPUTS

if df_TD_summary is not None and df_ASD_summary is not None:
    final_summary = pd.concat([df_TD_summary, df_ASD_summary], ignore_index=True)

    output_summary_path = os.path.join(base_dir, "Final_Gaze_Statistics.xlsx")
    final_summary.to_excel(output_summary_path, index=False)

    df_TD_full.to_csv(os.path.join(base_dir, "TD_time_series_gaze.csv"), index=False)
    df_ASD_full.to_csv(os.path.join(base_dir, "ASD_time_series_gaze.csv"), index=False)

    print("\n======================================")
    print("GAZE METRICS PROCESSING COMPLETE")
    print("======================================")
    print(f"Summary saved to: {output_summary_path}")
    print("\nPreview:")
    print(final_summary.head())

else:
    print("Error loading datasets.")
