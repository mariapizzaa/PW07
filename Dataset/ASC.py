import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CONFIGURATION AND PATHS
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_dir, "dataset iniziali e risultati")

# Paths to the preprocessed datasets (Cleaned & Advanced)
path_TD = os.path.join(data_folder, "TD_cleaned_advanced.xlsx")
path_ASD = os.path.join(data_folder, "ASD_cleaned_advanced.xlsx")

# --- GEOMETRIC THRESHOLDS FOR "VALID ATTENTION ZONE" ---
# We define a broad spatial area that encompasses all task-relevant targets
# (Robot, Therapist, Posters, Table). Gaze outside this area is considered "Vacancy".

# Horizontal Limit (Yaw): +/- 1.5 rad (approx. 85 degrees)
# Wide enough to include lateral posters but excludes looking behind/extreme sides.
YAW_LIMIT = 1.5

# Vertical Limits (Pitch): From -1.3 (deep table view) to +0.6 (Therapist's face)
# Excludes looking at the floor (too low) or the ceiling (too high).
PITCH_MIN = -1.3
PITCH_MAX = 0.6

# Confidence Threshold: Filters out low-quality gaze estimations
CONF_THRESH = 0.15

# Temporal Filter (Anti-Jitter)
# We ignore "Vacancy" events shorter than 5 frames (~150ms) to reduce sensor noise.
MIN_VACANCY_DURATION = 5


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def smart_load(filepath):
    """Loads Excel files safely, handling path errors."""
    if not os.path.exists(filepath): return None
    try:
        return pd.read_excel(filepath)
    except:
        return None


def calculate_avc_metric(df):
    """
    Calculates the AVC (AOI Vacancy Count) Ratio for a single subject.

    Returns:
        float: The ratio of time (0.0 - 1.0) the subject spent in 'Vacancy' (Disengagement).
    """
    # A. Filter valid data based on sensor confidence
    valid_mask = (df['confidence_yaw'] > CONF_THRESH) & (df['confidence_pitch'] > CONF_THRESH)
    df_clean = df[valid_mask].copy()

    total_valid_frames = len(df_clean)
    # Skip subjects with insufficient data (< 50 frames) to avoid statistical artifacts
    if total_valid_frames < 50: return None

    # B. Identify Valid Attention (Inside the Geometric Zone)
    # Check if gaze is within Horizontal limits
    is_inside_yaw = df_clean['yaw'].abs() <= YAW_LIMIT
    # Check if gaze is within Vertical limits
    is_inside_pitch = (df_clean['pitch'] >= PITCH_MIN) & (df_clean['pitch'] <= PITCH_MAX)

    # A frame is "Attention" if it satisfies BOTH spatial conditions
    is_attention = is_inside_yaw & is_inside_pitch

    # C. Signal Stabilization (Noise Removal)
    # We want to count Vacancy only if the gaze actually leaves the zone for a meaningful time.
    # 's_vac' is True when the child is NOT looking at the task.
    s_vac = pd.Series(~is_attention)

    # Rolling Window Filter:
    # If a window of N frames contains ANY Attention frame (0), the minimum becomes 0.
    # This effectively erodes short spikes of Vacancy (noise/blinks).
    is_stable_vacancy = s_vac.rolling(window=MIN_VACANCY_DURATION, center=True).min().fillna(0).astype(bool)

    # D. Final Calculation
    vacancy_frames = is_stable_vacancy.sum()

    # Ratio = Vacancy Frames / Total Valid Frames
    avc_ratio = vacancy_frames / total_valid_frames

    return avc_ratio


def process_group(path, group_name):
    """Iterates through all subjects in a group dataset."""
    df = smart_load(path)
    if df is None: return []

    print(f"Processing Group: {group_name}...")
    results = []

    for subj in df['id_soggetto'].unique():
        df_subj = df[df['id_soggetto'] == subj]

        # Compute metric for this subject
        avc = calculate_avc_metric(df_subj)

        if avc is not None:
            results.append({
                'Subject_ID': subj,
                'Group': group_name,
                'AVC_Ratio': avc
            })

    return results


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Compute Metrics for both groups
    res_td = process_group(path_TD, "TD")
    res_asd = process_group(path_ASD, "ASD")

    if res_td and res_asd:
        # Combine results into a single DataFrame
        df_final = pd.concat([pd.DataFrame(res_td), pd.DataFrame(res_asd)], ignore_index=True)

        # Save Final Results to Excel
        out_file = os.path.join(base_dir, "Final_AVC_Optimized.xlsx")
        df_final.to_excel(out_file, index=False)

        # 2. Statistical Analysis (Mann-Whitney U Test)
        td_vals = df_final[df_final['Group'] == 'TD']['AVC_Ratio']
        asd_vals = df_final[df_final['Group'] == 'ASD']['AVC_Ratio']

        # Hypothesis: ASD > TD (One-sided test 'greater')
        stat, p_value = mannwhitneyu(asd_vals, td_vals, alternative='greater')

        # 3. Print Report
        print("\n" + "=" * 60)
        print("AVC RESULTS (OPTIMIZED DISTRACTION RATIO)")
        print("=" * 60)
        print(f"TD Mean:  {td_vals.mean():.4f} (±{td_vals.std():.3f})")
        print(f"ASD Mean: {asd_vals.mean():.4f} (±{asd_vals.std():.3f})")
        print("-" * 60)
        print(f"Mann-Whitney U Test (Hypothesis: ASD > TD):")
        print(f"U-stat: {stat}")

        # Significance stars
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"P-value: {p_value:.5f}  [{sig}]")

        if p_value < 0.05:
            print("\n✅ SIGNIFICANT RESULT! Hypothesis confirmed.")
        else:
            print("\n❌ No significant difference found.")

        # 4. Generate Visualization (Boxplot)
        plt.figure(figsize=(6, 6))
        sns.set_style("whitegrid")

        # Draw Boxplot
        ax = sns.boxplot(data=df_final, x='Group', y='AVC_Ratio', palette='Set2', showfliers=False)
        # Overlay individual data points (Swarmplot)
        sns.swarmplot(data=df_final, x='Group', y='AVC_Ratio', color='.25', size=6)

        # Add statistical annotation bracket
        y_max = df_final['AVC_Ratio'].max()
        h = 0.02
        plt.plot([0, 0, 1, 1], [y_max + h, y_max + 2 * h, y_max + 2 * h, y_max + h], lw=1.5, c='k')
        plt.text(0.5, y_max + 2.5 * h, f"p = {p_value:.4f} [{sig}]", ha='center', va='bottom')

        plt.title("AOI Vacancy Ratio (Visual Disengagement)")
        plt.ylabel("AVC Ratio (0-1)")
        plt.ylim(top=y_max + 0.15)

        # Save Plot
        out_img = os.path.join(base_dir, "AVC_Boxplot_Optimized.png")
        plt.savefig(out_img, dpi=300)
        print(f"Chart saved to: {out_img}")

    else:
        print("Error: Missing data.")