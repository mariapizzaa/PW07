import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. SETTINGS AND PARAMETERS
# ---------------------------------------------------------
# Define paths to the dataset
data_folder = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset"
file_td = os.path.join(data_folder, "TD_cleaned.xlsx")
file_asd = os.path.join(data_folder, "ASD_cleaned.xlsx")

FPS = 30.0
WINDOW_SEC = 2.0  # Analyze using 2-second sliding windows
WINDOW_SIZE = int(WINDOW_SEC * FPS)

print(f"Analysis Mode: STRICT RAW YAW (No Normalization, Degrees Only)")


# ---------------------------------------------------------
# 2. DATA PROCESSING ENGINE
# ---------------------------------------------------------
def process_raw_yaw(file_path, group_name):
    """
    Reads the dataset, extracts the raw 'yaw' angle (degrees),
    and calculates Range of Motion (RoM) without normalization.
    """
    try:
        df = pd.read_excel(file_path)
    except:
        return []

    df.columns = df.columns.str.strip()

    # Data Validation: Check specifically for the 'yaw' column (Degrees)
    if 'yaw' not in df.columns:
        print(f"WARNING: 'yaw' column not found in {group_name} file!")
        return []

    # Handle Missing Values: Linear Interpolation
    df['yaw'] = df['yaw'].interpolate().bfill().ffill()

    # Video Segmentation: Detect Frame Resets
    # If the frame count drops, it indicates a new video clip started.
    split_indices = [0]
    if 'Frame' in df.columns:
        frames = df['Frame'].fillna(0).values
        resets = np.where(np.diff(frames) < 0)[0] + 1
        split_indices.extend(resets)
    split_indices.append(len(df))

    valid_windows = []

    # Process each continuous video segment
    for i in range(len(split_indices) - 1):
        segment = df.iloc[split_indices[i]:split_indices[i + 1]]

        # Skip segments shorter than the defined window size
        if len(segment) < WINDOW_SIZE: continue

        # Apply Sliding Window
        num_wins = len(segment) // WINDOW_SIZE
        for w in range(num_wins):
            # EXTRACT RAW DATA (in Degrees)
            # We use raw values to preserve physical magnitude.
            win_sig = segment.iloc[w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]['yaw'].values

            # Calculate RoM: Max Angle - Min Angle
            rom = np.max(win_sig) - np.min(win_sig)

            # Noise Filter:
            # Exclude 0.0 (sensor error or absolute stillness)
            # Exclude > 180 (physically impossible head rotation/sensor glitch)
            if 0.01 < rom < 180:
                valid_windows.append({
                    'Group': group_name,
                    'RoM_Degrees': rom
                })

    return valid_windows


# ---------------------------------------------------------
# 3. ANALYSIS AND STATISTICS
# ---------------------------------------------------------
print("Processing TD Data...")
td_data = process_raw_yaw(file_td, "TD")
print(f"TD: {len(td_data)} windows analyzed.")

print("Processing ASD Data...")
asd_data = process_raw_yaw(file_asd, "ASD")
print(f"ASD: {len(asd_data)} windows analyzed.")

if not td_data or not asd_data:
    print("Error: No valid data found.")
    exit()

# Convert results to DataFrame
df_results = pd.DataFrame(td_data + asd_data)

# Statistical Test (Mann-Whitney U)
# Comparing the distribution of RoM (Degrees) between groups
td_vals = df_results[df_results['Group'] == 'TD']['RoM_Degrees']
asd_vals = df_results[df_results['Group'] == 'ASD']['RoM_Degrees']

stat, p = mannwhitneyu(td_vals, asd_vals)

print("\n" + "=" * 60)
print(f"METRIC: RANGE OF MOTION (RAW DEGREES)")
print("=" * 60)
print(f"TD Mean        : {td_vals.mean():.2f} degrees")
print(f"ASD Mean       : {asd_vals.mean():.2f} degrees")
print(f"Diff (ASD-TD)  : {asd_vals.mean() - td_vals.mean():.2f} degrees")
print(f"P-Value        : {p:.5f}")

if p < 0.05:
    print("Result         : SIGNIFICANT DIFFERENCE FOUND (ASD movements are physically wider)")
else:
    print("Result         : NO SIGNIFICANT DIFFERENCE")
print("=" * 60)

# Visualization (Boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_results, x='Group', y='RoM_Degrees', palette='Set2', showfliers=False)
plt.title(f'Range of Motion (Raw Degrees)\np={p:.5f}')
plt.ylabel('Range of Motion (Degrees)')

save_path = os.path.join(data_folder, "RoM_Raw_Degrees.png")
plt.savefig(save_path)
print(f"Plot saved to: {save_path}")
plt.show()