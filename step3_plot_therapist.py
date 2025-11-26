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
    print("Data loaded successfully. 'THERAPIST' analysis starting...\n")

    # --- CHANGE HERE: Selecting 'TFD_Therapist_%' column ---
    td_therapist = df[df['Group'] == 'TD']['TFD_Therapist_%']
    asd_therapist = df[df['Group'] == 'ASD']['TFD_Therapist_%']

    # --- 1. STATISTICAL TEST (Mann-Whitney U) ---
    stat, p_value = mannwhitneyu(td_therapist, asd_therapist, alternative='two-sided')

    print("=" * 50)
    print("RESULT REPORT (Attention to Therapist)")
    print("=" * 50)
    print(f"TD Group (n={len(td_therapist)})  Median Attention: %{td_therapist.median():.2f}")
    print(f"ASD Group (n={len(asd_therapist)}) Median Attention: %{asd_therapist.median():.2f}")
    print("-" * 50)
    print(f"P-Value: {p_value:.5f}")

    if p_value < 0.05:
        print("✅ RESULT: SIGNIFICANT DIFFERENCE in attention to therapist.")
        if td_therapist.median() > asd_therapist.median():
            print("   -> TD children looked at the therapist more.")
        else:
            print("   -> ASD children looked at the therapist more.")
    else:
        print("❌ RESULT: No statistically significant difference found in attention to therapist.")
    print("=" * 50)

    # --- 2. PLOTTING (Boxplot) ---
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    # Color palette (Green/Purple tones for distinction)
    my_pal = {"TD": "lightgreen", "ASD": "mediumpurple"}

    # Boxplot
    # Y-axis is now 'TFD_Therapist_%'
    ax = sns.boxplot(x='Group', y='TFD_Therapist_%', data=df, palette=my_pal, showfliers=False)

    # Swarmplot
    sns.swarmplot(x='Group', y='TFD_Therapist_%', data=df, color=".25", size=6, alpha=0.7)

    # Titles
    plt.title('Attention to Therapist: TD vs ASD (Silva et al. 2024)', fontsize=14, fontweight='bold')
    plt.ylabel('Time Spent Looking at Therapist (%)', fontsize=12)
    plt.xlabel('Group', fontsize=12)

    # Add P-Value to graph
    y_max = df['TFD_Therapist_%'].max()
    y_line = y_max + 2

    if p_value < 0.001:
        sig_symbol = "***"
    elif p_value < 0.01:
        sig_symbol = "**"
    elif p_value < 0.05:
        sig_symbol = "*"
    else:
        sig_symbol = "ns"

    plt.text(0.5, y_line, f"p = {p_value:.4f}\n{sig_symbol}",
             ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    plt.ylim(bottom=-1, top=y_line + 10)
    plt.tight_layout()

    # Save
    save_path = os.path.join(base_dir, "Result_Graph_Therapist.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[GRAPH SAVED]: {save_path}")
    plt.show()

else:
    print(f"ERROR: File '{file_path}' not found. Ensure you ran Step 2 first.")
