import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. SETTINGS AND PARAMETERS
# ---------------------------------------------------------
data_folder = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset"
file_td = os.path.join(data_folder, "TD_cleaned.xlsx")
file_asd = os.path.join(data_folder, "ASD_cleaned.xlsx")

WINDOW_SEC = 2.0
FILTER_CUTOFF = 6.0  # 6Hz Low-pass filter

print(f"Analysis Mode: FINAL (Median & IQR Reporting)")


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Low-pass filter to remove sensor jitter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1: return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_by_subject_id(file_path, group_name):
    print(f"\n--- Loading {group_name}: {os.path.basename(file_path)} ---")

    try:
        df_raw = pd.read_excel(file_path)
        df_raw.columns = df_raw.columns.str.strip()

        req_cols = ['id_soggetto', 'timestamp', 'yaw']

        if 'yaw' not in df_raw.columns:
            for alt in ['Yaw', 'pose_Ry', 'head_yaw']:
                if alt in df_raw.columns:
                    df_raw.rename(columns={alt: 'yaw'}, inplace=True)
                    break

        if 'timestamp' not in df_raw.columns:
            # Timestamp yoksa index'ten uydur
            df_raw['timestamp'] = df_raw.index / 30.0

        df = df_raw[req_cols].copy()

    except Exception as e:
        print(f"CRITICAL ERROR reading file: {e}")
        return []

    df['yaw'] = pd.to_numeric(df['yaw'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    subjects = df.groupby('id_soggetto')

    subject_results = []

    for subj_id, subject_data in subjects:
        try:
            data = subject_data.copy()
            data = data.dropna(subset=['yaw', 'timestamp'])

            if len(data) < 30: continue

            data = data.sort_values('timestamp')
            data['time_idx'] = pd.to_timedelta(data['timestamp'], unit='s')
            data = data.drop_duplicates(subset=['time_idx'])
            data = data.set_index('time_idx')

            # Resample (33ms ~ 30fps)
            resampled = data['yaw'].resample('33ms').mean().interpolate()

            if len(resampled) < (WINDOW_SEC * 30): continue

            # Filtering
            fs = 30.0
            yaw_filtered = butter_lowpass_filter(resampled.values, FILTER_CUTOFF, fs)

            # Windowing
            win_samples = int(WINDOW_SEC * fs)
            num_wins = len(yaw_filtered) // win_samples

            spans = []
            for w in range(num_wins):
                segment = yaw_filtered[w * win_samples: (w + 1) * win_samples]
                span = np.max(segment) - np.min(segment)
                if 0.1 < span < 180:
                    spans.append(span)

            if len(spans) > 0:
                avg_span = np.mean(spans)
                subject_results.append({
                    'Group': group_name,
                    'ID': subj_id,
                    'Avg_Yaw_Span': avg_span
                })

        except Exception:
            continue

    print(f"Successfully processed {len(subject_results)} subjects in {group_name}.")
    return subject_results


# ---------------------------------------------------------
# 3. ANALYSIS AND STATISTICS (UPDATED FOR MEDIAN/IQR)
# ---------------------------------------------------------
td_data = process_by_subject_id(file_td, "TD")
asd_data = process_by_subject_id(file_asd, "ASD")

if not td_data or not asd_data:
    print("\nERROR: Not enough data points to compare.")
else:
    df_results = pd.DataFrame(td_data + asd_data)

    td_vals = df_results[df_results['Group'] == 'TD']['Avg_Yaw_Span']
    asd_vals = df_results[df_results['Group'] == 'ASD']['Avg_Yaw_Span']

    # Mann-Whitney U Test
    stat, p = mannwhitneyu(td_vals, asd_vals)


    # --- YENİ HESAPLAMALAR (Median & IQR) ---
    def get_stats(series):
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return median, q1, q3


    td_med, td_q1, td_q3 = get_stats(td_vals)
    asd_med, asd_q1, asd_q3 = get_stats(asd_vals)

    print("\n" + "=" * 60)
    print(f"METRIC: HEAD YAW SPAN (Subject Averaged)")
    print("=" * 60)
    # Raporlama Formatı: Median [Q1 - Q3]
    print(f"TD (N={len(td_vals)})  : Median={td_med:.2f} [IQR: {td_q1:.2f} - {td_q3:.2f}]")
    print(f"ASD (N={len(asd_vals)}) : Median={asd_med:.2f} [IQR: {asd_q1:.2f} - {asd_q3:.2f}]")
    print("-" * 60)
    print(f"P-Value       : {p:.5f}")

    if p < 0.05:
        print("Result        : SIGNIFICANT DIFFERENCE")
    else:
        print("Result        : NO SIGNIFICANT DIFFERENCE")
    print("=" * 60)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x='Group', y='Avg_Yaw_Span', palette='Set2', showfliers=False)
    sns.swarmplot(data=df_results, x='Group', y='Avg_Yaw_Span', color='black', alpha=0.6, size=5)

    plt.title(f'Head Yaw Span (Subject Averages)\nMedian Comparison | p={p:.5f}')
    plt.ylabel('Average Motion Span (Degrees)')

    save_path = os.path.join(data_folder, "Head_Yaw_Span_Final.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()
