import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu

# --- CORRECTION 1: Configure Graphics Backend ---
# These lines force the window to open.
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    pass  # Uses default if TkAgg is not available
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------------------
path_td = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\Visual_Analysis_TD_after.xlsx"
path_asd = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\Visual_Analysis_ASD_after.xlsx"
# Path where the plot will be saved
save_plot_path = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset\Result_Graph.png"


# ---------------------------------------------------------
# 2. DATA LOADING AND CLEANING
# ---------------------------------------------------------
def load_and_clean_data(file_path, group_label):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df.columns = df.columns.str.strip()

        required_cols = ['Subject', 'Frame', 'Azione', 'Admin']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return pd.DataFrame()

        df['Group'] = group_label
        if 'Dietro' not in df.columns:
            df['Dietro'] = 'no'
        df['Dietro'] = df['Dietro'].fillna('no')
        return df
    except Exception as e:
        print(f"Error loading {group_label}: {e}")
        return pd.DataFrame()


print("Loading files...")
df_td = load_and_clean_data(path_td, 'TD')
df_asd = load_and_clean_data(path_asd, 'ASD')

if df_td.empty or df_asd.empty:
    print("Data load failed.")
    exit()

full_raw_data = pd.concat([df_td, df_asd], ignore_index=True)


# ---------------------------------------------------------
# 3. RESPONSE LATENCY CALCULATION
# ---------------------------------------------------------
def calculate_latencies(df):
    results = []
    if 'Subject' in df.columns and 'Frame' in df.columns:
        df = df.sort_values(by=['Subject', 'Frame'])

    for subject, person_data in df.groupby('Subject'):
        last_stimulus_frame = None
        current_condition = None
        is_clean_trial = True

        for idx, row in person_data.iterrows():
            action = str(row['Azione'])
            frame = row['Frame']
            dietro = str(row['Dietro']).lower()
            admin_raw = str(row['Admin']).lower()

            condition = "Other"
            if 'robot' in admin_raw or 'nao' in admin_raw:
                condition = 'Robot'
            elif 'therapist' in admin_raw or 'terapista' in admin_raw or 'human' in admin_raw:
                condition = 'Therapist'

            if action == 'Coniglio':
                last_stimulus_frame = frame
                current_condition = condition
                is_clean_trial = True
                if 'yes' in dietro: is_clean_trial = False

            elif last_stimulus_frame is not None:
                if 'yes' in dietro: is_clean_trial = False

                if action == 'Risponde_Coniglio':
                    if is_clean_trial and current_condition in ['Robot', 'Therapist']:
                        latency = frame - last_stimulus_frame
                        if 0 < latency < 400:
                            results.append({
                                'Subject': subject,
                                'Group': row['Group'],
                                'Condition': current_condition,
                                'Latency': latency
                            })
                    last_stimulus_frame = None
                    current_condition = None
    return pd.DataFrame(results)


print("Calculating latencies...")
latency_df = calculate_latencies(full_raw_data)

if latency_df.empty:
    print("No valid data found.")
    exit()

print(f"\nTotal Valid Response Count: {len(latency_df)}")
print(latency_df.groupby(['Group', 'Condition'])['Latency'].count())

# ---------------------------------------------------------
# 4. STATISTICS AND VISUALIZATION
# ---------------------------------------------------------
sns.set(style="whitegrid", context="talk")

# Plotting
g = sns.catplot(
    data=latency_df,
    x="Group",
    y="Latency",
    col="Condition",
    kind="box",
    palette={"TD": "#a1c9f4", "ASD": "#ff9f9b"},
    height=6,
    aspect=0.8,
    showfliers=False
)

g.map_dataframe(sns.swarmplot, x="Group", y="Latency", color=".2", alpha=0.5, size=4)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Joint Attention Response Latency: Robot vs Therapist', fontsize=16, fontweight='bold')
g.set_axis_labels("Group", "Latency (Frame)")
g.set_titles("{col_name} Condition")

print("\n--- STATISTICAL RESULTS (Mann-Whitney U) ---")
axes = g.axes.flatten()

for ax_idx, condition in enumerate(g.col_names):
    subset = latency_df[latency_df['Condition'] == condition]
    td_vals = subset[subset['Group'] == 'TD']['Latency']
    asd_vals = subset[subset['Group'] == 'ASD']['Latency']

    if len(td_vals) > 0 and len(asd_vals) > 0:
        stat, p = mannwhitneyu(td_vals, asd_vals)
        print(f"\nCONDITION: {condition}")
        print(f"P-Value: {p:.5f}")

        y_max = subset['Latency'].max()
        ax = axes[ax_idx]
        ax.plot([0, 0, 1, 1], [y_max + 10, y_max + 20, y_max + 20, y_max + 10], lw=1.5, c='k')

        sig_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        if p > 0.05: sig_text += " (ns)"
        ax.text(0.5, y_max + 25, sig_text, ha='center', va='bottom', fontsize=12, color='black')

# --- CORRECTION 2: SAVING AND SHOWING ---
print(f"\nSaving plot to file: {save_plot_path}")
plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
print("Plot saved. Now showing...")

# Even if show() doesn't work, you will have a .png file in the folder
plt.show(block=True)
