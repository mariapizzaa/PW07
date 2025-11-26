import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

# --- SETTINGS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Silva_Analysis_Results.xlsx")

if os.path.exists(file_path):
    # Load data
    df = pd.read_excel(file_path)
    print("Data loaded successfully. Starting analysis...\n")

    # Separate groups
    td_robot = df[df['Group'] == 'TD']['TFD_NAO_Robot_%']
    asd_robot = df[df['Group'] == 'ASD']['TFD_NAO_Robot_%']

    # --- 1. STATISTICAL TEST (Mann-Whitney U) ---
    # Hypothesis: ASD and TD children have different attention to the Robot.
    stat, p_value = mannwhitneyu(td_robot, asd_robot, alternative='two-sided')

    print("=" * 50)
    print("RESULT REPORT (Silva et al. 2024 Model)")
    print("=" * 50)
    print(f"TD Group (n={len(td_robot)})  Median Attention: %{td_robot.median():.2f}")
    print(f"ASD Group (n={len(asd_robot)}) Median Attention: %{asd_robot.median():.2f}")
    print("-" * 50)
    print(f"P-Value: {p_value:.5f}")

    if p_value < 0.05:
        print("✅ RESULT: Statistically SIGNIFICANT DIFFERENCE found.")
        if td_robot.median() > asd_robot.median():
            print("   -> TD children looked at the robot more.")
        else:
            print("   -> ASD children looked at the robot more.")
    else:
        print("❌ RESULT: No statistically significant difference found.")
    print("=" * 50)

    # --- 2. PLOTTING (Boxplot) ---
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    # Color palette
    my_pal = {"TD": "skyblue", "ASD": "salmon"}

    # Boxplot
    ax = sns.boxplot(x='Group', y='TFD_NAO_Robot_%', data=df, palette=my_pal, showfliers=False)

    # Swarmplot (Actual data points)
    sns.swarmplot(x='Group', y='TFD_NAO_Robot_%', data=df, color=".25", size=6, alpha=0.7)

    # Plot decorations
    plt.title('Attention to Robot: TD vs ASD (Total Fixation Duration)', fontsize=14, fontweight='bold')
    plt.ylabel('Time Spent Looking at Robot (%)', fontsize=12)
    plt.xlabel('Group', fontsize=12)

    # Add P-Value to plot
    y_max = df['TFD_NAO_Robot_%'].max()
    # Peak point of the line
    y_line = y_max + 1

    # Significance Stars
    if p_value < 0.001:
        sig_symbol = "***"
    elif p_value < 0.01:
        sig_symbol = "**"
    elif p_value < 0.05:
        sig_symbol = "*"
    else:
        sig_symbol = "ns (not significant)"

    # Line and Text
    plt.text(0.5, y_line, f"p = {p_value:.4f}\n{sig_symbol}",
             ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    plt.ylim(bottom=-1, top=y_line + 5)  # Expand Y-axis slightly
    plt.tight_layout()

    # Save
    save_path = os.path.join(base_dir, "Result_Graph_Silva.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[STEP 3 COMPLETED] Graph saved: {save_path}")
    plt.show()

else:
    print(f"ERROR: File '{file_path}' not found. Please run the Step 2 code first.")
