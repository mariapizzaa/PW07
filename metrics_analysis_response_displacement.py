import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

# =============================================================================
# 1. SETUP AND FILE PATHS
# =============================================================================
# We define the main directory where our project data is stored.
BASE_PATH = r"C:\Users\user\Desktop\PW2025\Students\Students\Dataset"

# 1. RESPONSE METRICS FILES (Event-based Data)
# These files contain labels like "Robot looked at Dog" or "Child responded".
# We use these to calculate the 'Response Rate'.
file_vis_td = os.path.join(BASE_PATH, 'Visual_Analysis_TD_after.xlsx')
file_vis_asd = os.path.join(BASE_PATH, 'Visual_Analysis_ASD_after.xlsx')

# 2. DISPLACEMENT METRICS FILES (Cleaned Kinematic Data)
# These are the files we cleaned in the previous step (data_cleaning.py).
# They contain the X, Z coordinates (movement) of each child.
file_cones_td = os.path.join(BASE_PATH, 'TD_cleaned_final.xlsx')
file_cones_asd = os.path.join(BASE_PATH, 'ASD_cleaned_final.xlsx')

# OUTPUT DIRECTORY
# We create a specific folder to save the resulting graphs.
OUTPUT_PATH = r"C:\Users\user\Desktop\PW2025"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# =============================================================================
# HELPER FUNCTION: PLOTTING SIGNIFICANCE
# =============================================================================
def add_significance_bar(ax, data, x1, x2, p_value):
    """
    This function adds a professional significance bar (bracket) and stars
    to the boxplots, making them publication-ready.

    Logic:
    1. Find the maximum value in the plot to know where to draw the bar.
    2. Determine the number of stars based on the P-value.
    3. Draw the bracket and write the P-value.
    """
    y_max = data.max()
    y_h = y_max * 0.1  # Height of the bracket legs
    y_pos = y_max + y_h  # Vertical position of the bar

    # Assign stars based on statistical significance level
    if p_value < 0.001:
        sig_symbol = "***"  # Highly Significant
    elif p_value < 0.01:
        sig_symbol = "**"  # Very Significant
    elif p_value < 0.05:
        sig_symbol = "*"  # Significant
    else:
        sig_symbol = "ns"  # Not Significant

    # Draw the bracket lines
    line_color = 'black'
    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + y_h, y_pos + y_h, y_pos], lw=1.5, c=line_color)

    # Add the text (Stars and exact p-value)
    text = f"{sig_symbol}\n(p={p_value:.4f})"
    ax.text((x1 + x2) * 0.5, y_pos + y_h, text, ha='center', va='bottom', color=line_color, fontsize=10)


# =============================================================================
# PART 1: RESPONSE METRICS ANALYSIS
# Objective: Replicate Anzalone et al. (2019) Response Metrics.
# We measure how often the child responds to the robot's Joint Attention bids.
# =============================================================================
print("--- 1. STARTING RESPONSE METRICS ANALYSIS ---")


def analyze_response(filename, group_name):
    """
    Calculates the percentage of successful responses.
    Formula: Response Rate = (Total Responses / Total Stimuli) * 100
    """
    try:
        # Load data (supporting both Excel and CSV formats)
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename)
        else:
            df = pd.read_csv(filename)

        # Identify the column containing event labels (usually 'Azione')
        if 'Azione' in df.columns:
            act_col = 'Azione'
        elif len(df.columns) > 2:
            act_col = df.columns[2]
        else:
            return None

        # Count occurrences of stimuli and responses
        counts = df[act_col].value_counts()

        # Stimulus: Robot calls attention (Rabbit/Dog)
        stimulus = counts.get('Coniglio', 0) + counts.get('Cane', 0)

        # Response: Child actually looks/responds
        response = 0
        for index, val in counts.items():
            if isinstance(index, str) and 'Risponde' in index:
                response += val

        # Fallback for data consistency
        if stimulus == 0:
            stimulus = df[act_col].str.contains('Coniglio|Cane', case=False, na=False).sum()
            response = df[act_col].str.contains('Risponde', case=False, na=False).sum()

        # Calculate Rate
        rate = (response / stimulus * 100) if stimulus > 0 else 0
        return {'Group': group_name, 'Stimulus': stimulus, 'Response': response, 'Rate': rate}

    except Exception as e:
        print(f"Error in {group_name}: {e}")
        return None


# Execute analysis for both groups
res_td = analyze_response(file_vis_td, 'TD')
res_asd = analyze_response(file_vis_asd, 'ASD')

if res_td and res_asd:
    df_response = pd.DataFrame([res_td, res_asd])
    print("\n✅ Response Results Summary:")
    print(df_response)

    # VISUALIZATION: RESPONSE RATE
    plt.figure(figsize=(6, 6))
    sns.barplot(x='Group', y='Rate', data=df_response, palette=['blue', 'red'])
    plt.title('Head Movement Response to JA Induction (%)\n(Anzalone Response Metrics)', fontsize=12, fontweight='bold')
    plt.ylabel('Response Rate (%)')
    plt.ylim(0, 115)  # Extra space for text

    # Add percentage labels on top of bars
    for index, row in df_response.iterrows():
        plt.text(index, row.Rate + 2, f"{row.Rate:.1f}%", color='black', ha="center", fontsize=12, fontweight='bold')

    save_path = os.path.join(OUTPUT_PATH, "Response_Metrics_Final.png")
    plt.savefig(save_path)
    print(f"📊 Graph saved: {save_path}")
else:
    print("\n⚠️ Skipping Response Analysis (Files missing).")

# =============================================================================
# PART 2: DISPLACEMENT METRICS ANALYSIS
# Objective: Measure Postural Instability (Micro-movements).
# According to Anzalone (2019), ASD children show higher displacement/instability.
# =============================================================================
print("\n--- 2. STARTING DISPLACEMENT METRICS ANALYSIS ---")


def analyze_displacement(filename, group_name):
    """
    Calculates spatial stability metrics for each subject:
    1. Centroid: The average center position.
    2. Displacement Magnitude: Mean distance from center (Radius).
    3. Axial Displacement: Standard deviation along X (Lateral) and Z (Longitudinal) axes.
    """
    try:
        if not os.path.exists(filename): return pd.DataFrame()
        df = pd.read_excel(filename)

        # Remove artifacts (0,0 coordinates)
        if 'child_keypoint_x' in df.columns:
            df = df[df['child_keypoint_x'] != 0]
        else:
            return pd.DataFrame()

        metrics = []
        subjects = df['id_soggetto'].unique()

        # Process each subject individually
        for sub in subjects:
            sub_data = df[df['id_soggetto'] == sub]
            if len(sub_data) < 10: continue

            x = sub_data['child_keypoint_x']
            z = sub_data['child_keypoint_z']

            # Calculate Metrics
            centroid_x = x.mean()
            centroid_z = z.mean()

            # Magnitude (Overall Instability)
            magnitude = np.sqrt((x - centroid_x) ** 2 + (z - centroid_z) ** 2).mean()

            # Axial Sway (Left-Right and Front-Back)
            std_x = x.std()
            std_z = z.std()

            metrics.append({
                'Group': group_name,
                'Subject': sub,
                'Disp_Mag': magnitude,
                'Disp_Axis_X': std_x,
                'Disp_Axis_Z': std_z
            })
        return pd.DataFrame(metrics)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


# Run Displacement Analysis
disp_td = analyze_displacement(file_cones_td, 'TD')
disp_asd = analyze_displacement(file_cones_asd, 'ASD')

if not disp_td.empty and not disp_asd.empty:
    df_disp = pd.concat([disp_td, disp_asd], ignore_index=True)

    print("\n✅ Displacement Results (Group Averages):")
    print(df_disp.groupby('Group')[['Disp_Mag', 'Disp_Axis_X', 'Disp_Axis_Z']].mean())

    # ---------------------------------------------------------
    # STATISTICAL ANALYSIS & VISUALIZATION
    # We generate Boxplots with P-value annotations.
    # ---------------------------------------------------------
    print("\n✅ Generating Plots with P-Values...")

    plt.figure(figsize=(18, 7))

    # List of metrics to plot
    plot_metrics = [
        ('Disp_Mag', 'Displacement Magnitude\n(Overall Instability)', 'Mean Radius (mm)'),
        ('Disp_Axis_X', 'Displacement Left-Right\n(Lateral Sway)', 'Std Dev (mm)'),
        ('Disp_Axis_Z', 'Displacement Front-Back\n(Longitudinal Sway)', 'Std Dev (mm)')
    ]

    for i, (metric, title, ylabel) in enumerate(plot_metrics):
        ax = plt.subplot(1, 3, i + 1)

        # 1. Draw the Boxplot
        sns.boxplot(x='Group', y=metric, data=df_disp, palette=['blue', 'red'], ax=ax)
        # Add individual data points (Stripplot) for transparency
        sns.stripplot(x='Group', y=metric, data=df_disp, color='black', alpha=0.5, ax=ax)

        # 2. Perform Mann-Whitney U Test
        # We compare TD vs ASD distributions for statistical significance.
        data_td = disp_td[metric].dropna()
        data_asd = disp_asd[metric].dropna()
        stat, p = mannwhitneyu(data_td, data_asd, alternative='two-sided')

        print(f"{metric}: p={p:.5f}")

        # 3. Add Significance Bar (The bracket with stars)
        add_significance_bar(ax, df_disp[metric], 0, 1, p)

        # Formatting
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')

        # Adjust Y-axis to fit the annotation
        y_max = df_disp[metric].max()
        ax.set_ylim(bottom=0, top=y_max * 1.3)

    plt.tight_layout()
    save_path_disp = os.path.join(OUTPUT_PATH, "Displacement_Metrics_Final_Pvalues.png")
    plt.savefig(save_path_disp)
    print(f"📊 Graph saved with p-values: {save_path_disp}")

    print("\n--- ANALYSIS COMPLETED SUCCESSFULLY ---")
    plt.show()
else:
    print("\n⚠️ Displacement analysis skipped.")