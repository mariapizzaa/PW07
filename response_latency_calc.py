import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
import os

# ---------------------------------------------------------
# 1. SETTINGS AND FILE PATHS
# ---------------------------------------------------------
# Using raw strings (r"...") to handle Windows backslashes correctly
file_td = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\Visual_Analysis_TD_after.xlsx"
file_asd = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\Visual_Analysis_ASD_after.xlsx"

print("Loading files...")

# ---------------------------------------------------------
# 2. DATA LOADING AND CLEANING
# ---------------------------------------------------------
try:
    # Reading Excel files (requires 'openpyxl' engine)
    df_td = pd.read_excel(file_td, engine='openpyxl')
    df_asd = pd.read_excel(file_asd, engine='openpyxl')
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not read files. Details: {e}")
    print("Please ensure 'openpyxl' is installed (pip install openpyxl).")
    exit()


# Function to calculate Response Latency
def calculate_latency(df, group_name):
    latencies = []

    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()

    # Handle 'Dietro' column: Create if missing, fill NaNs with 'no'
    if 'Dietro' not in df.columns: df['Dietro'] = 'no'
    df['Dietro'] = df['Dietro'].fillna('no')

    # Sort by Subject and Frame to ensure chronological order
    if 'Frame' in df.columns and 'Subject' in df.columns:
        df = df.sort_values(by=['Subject', 'Frame'])

    # Iterate through each subject
    for subject, group in df.groupby('Subject'):
        last_stimulus_frame = None
        is_trial_clean = True

        for idx, row in group.iterrows():
            action = row['Azione']
            frame = row['Frame']
            dietro = str(row['Dietro']).lower()

            # 1. Stimulus Event (Coniglio)
            if action == 'Coniglio':
                last_stimulus_frame = frame
                is_trial_clean = True
                # If child is looking back (Dietro) during stimulus, mark trial as dirty
                if 'yes' in dietro: is_trial_clean = False

            # 2. Waiting Period
            elif last_stimulus_frame is not None:
                # If child looks back while waiting, invalidate the trial
                if 'yes' in dietro: is_trial_clean = False

                # 3. Response Event (Risponde)
                if action == 'Risponde_Coniglio':
                    if is_trial_clean:
                        latency = frame - last_stimulus_frame
                        # Sanity check: Latency must be positive
                        if latency > 0:
                            latencies.append({
                                'Group': group_name,
                                'Subject': subject,
                                'Latency_Frames': latency
                            })
                    # Reset for the next stimulus
                    last_stimulus_frame = None

    return pd.DataFrame(latencies)


# Perform Calculations
print("Calculating latencies...")
latency_td_df = calculate_latency(df_td, 'TD')
latency_asd_df = calculate_latency(df_asd, 'ASD')

if latency_td_df.empty or latency_asd_df.empty:
    print("WARNING: No valid latency data found for one or both groups!")
    exit()

# Combine Dataframes
all_latencies = pd.concat([latency_td_df, latency_asd_df], ignore_index=True)

# ---------------------------------------------------------
# 3. STATISTICAL ANALYSIS
# ---------------------------------------------------------
group_td = latency_td_df['Latency_Frames']
group_asd = latency_asd_df['Latency_Frames']

# Normality Test (Shapiro-Wilk)
_, p_norm_td = shapiro(group_td)
_, p_norm_asd = shapiro(group_asd)

# Choose Test based on Normality
if p_norm_td < 0.05 or p_norm_asd < 0.05:
    test_type = "Mann-Whitney U Test"
    # Non-parametric test (median-based)
    stat, p_val = mannwhitneyu(group_td, group_asd, alternative='two-sided')
else:
    test_type = "Independent T-Test"
    # Parametric test (mean-based)
    stat, p_val = ttest_ind(group_td, group_asd, equal_var=False)

print("\n" + "=" * 40)
print(f"STATISTICAL RESULTS ({test_type})")
print(f"P-Value: {p_val:.5f}")
print("=" * 40)

# ---------------------------------------------------------
# 4. VISUALIZATION AND SAVING
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))

# Draw Boxplot
ax = sns.boxplot(x='Group', y='Latency_Frames', data=all_latencies,
                 palette="Set2", order=['TD', 'ASD'], width=0.5)

# Add Swarmplot (individual dots) to show distribution density
sns.swarmplot(x='Group', y='Latency_Frames', data=all_latencies,
              color=".25", alpha=0.5, size=4)

# Add Significance Bar and P-Value
y_max = all_latencies['Latency_Frames'].max()
y_line = y_max + 5  # Height of the bar
y_text = y_line + 5  # Height of the text

# Draw the bracket line between groups
plt.plot([0, 0, 1, 1], [y_line, y_line + 2, y_line + 2, y_line], lw=1.5, c='k')

# Format P-value text
if p_val < 0.001:
    p_text = "p < 0.001"
else:
    p_text = f"p = {p_val:.3f}"

# Add text above the bracket
plt.text(0.5, y_text, f"{test_type}\n{p_text}\n(ns)", ha='center', va='bottom', color='black', fontsize=12)
# Note: 'ns' stands for 'not significant' (commonly used if p > 0.05)

# Plot Labels (English)
plt.title('Comparison of Response Latency: ASD vs TD', fontsize=14, fontweight='bold')
plt.ylabel('Latency (Frames)', fontsize=12)
plt.xlabel('Group', fontsize=12)
plt.ylim(0, y_text + 20)  # Adjust top margin

# Add grid
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Save Outputs
save_path_img = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\latency_pvalue_plot.png"
save_path_csv = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\latency_final_results.csv"

plt.savefig(save_path_img, dpi=300)
all_latencies.to_csv(save_path_csv, index=False)

print(f"\nPlot saved to: {save_path_img}")
print(f"Data saved to: {save_path_csv}")

plt.show()