import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURAZIONE ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_dir, "dataset iniziali e risultati")  # Modifica se hai salvato altrove

# Nomi dei file (assicurati che siano quelli generati dagli script precedenti)
file_energy = "Final_Energy_Statistics.xlsx"
file_avc = "AVC_Comparison_Final.xlsx"

path_energy = os.path.join(base_dir, file_energy)
path_avc = os.path.join(base_dir, file_avc)

# Se non li trova nella root, prova nella cartella dati
if not os.path.exists(path_energy): path_energy = os.path.join(data_folder, file_energy)
if not os.path.exists(path_avc): path_avc = os.path.join(data_folder, file_avc)


# --- 2. FUNZIONI STATISTICHE ---

def perform_test(df, metric_col, metric_name):
    """
    Esegue il test di Mann-Whitney U e stampa i risultati formattati.
    """
    print(f"\n🔷 ANALISI METRICA: {metric_name}")

    # Separa i gruppi
    group_td = df[df['Group'] == 'TD'][metric_col].dropna()
    group_asd = df[df['Group'] == 'ASD'][metric_col].dropna()

    n_td = len(group_td)
    n_asd = len(group_asd)

    if n_td == 0 or n_asd == 0:
        print("⚠️ Dati insufficienti per uno dei gruppi.")
        return

    # Statistiche Descrittive (Mediana e IQR)
    med_td = group_td.median()
    iqr_td = group_td.quantile(0.75) - group_td.quantile(0.25)

    med_asd = group_asd.median()
    iqr_asd = group_asd.quantile(0.75) - group_asd.quantile(0.25)

    print(f"   TD  (n={n_td}): Mediana={med_td:.4f} (IQR={iqr_td:.4f})")
    print(f"   ASD (n={n_asd}): Mediana={med_asd:.4f} (IQR={iqr_asd:.4f})")

    # Test Statistico (Mann-Whitney U)
    # alternative='two-sided' controlla se sono DIVERSI (senza assumere chi è maggiore)
    stat, p_value = mannwhitneyu(group_td, group_asd, alternative='two-sided')

    print(f"   📊 Risultato Test:")
    print(f"      U-stat = {stat}")
    print(f"      P-value = {p_value:.5f}")

    # Interpretazione
    alpha = 0.05
    if p_value < alpha:
        print("   ✅ DIFFERENZA SIGNIFICATIVA (p < 0.05)")
        if med_asd > med_td:
            print("      -> Il gruppo ASD ha valori MAGGIORI.")
        else:
            print("      -> Il gruppo ASD ha valori MINORI.")
    else:
        print("   ❌ NESSUNA differenza significativa (p >= 0.05)")

    return p_value


def plot_boxplot(df, metric_col, metric_name, p_val, output_name):
    """Crea un boxplot con annotazione del p-value."""
    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")

    # Boxplot
    ax = sns.boxplot(x='Group', y=metric_col, data=df, palette="Set2", showfliers=False)
    # Swarmplot (punti singoli)
    sns.swarmplot(x='Group', y=metric_col, data=df, color=".25", size=5)

    # Titolo e Label
    plt.title(f'{metric_name}\n(TD vs ASD)', fontsize=14)
    plt.ylabel(metric_name)

    # Annotazione P-Value
    if p_val is not None:
        # Coordinate per la barra sopra il grafico
        y_max = df[metric_col].max()
        y_h = y_max * 0.05
        y_line = y_max + y_h

        plt.plot([0, 0, 1, 1], [y_line, y_line + y_h, y_line + y_h, y_line], lw=1.5, c='k')

        sig_symbol = "ns"
        if p_val < 0.001:
            sig_symbol = "***"
        elif p_val < 0.01:
            sig_symbol = "**"
        elif p_val < 0.05:
            sig_symbol = "*"

        plt.text(0.5, y_line + y_h, f"p = {p_val:.4f}\n{sig_symbol}",
                 ha='center', va='bottom', color='k', fontsize=11)

        plt.ylim(top=y_line + y_h * 4)  # Spazio extra in alto

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, output_name), dpi=300)
    print(f"   🖼️ Grafico salvato: {output_name}")
    plt.close()


# --- 3. ESECUZIONE ---
if __name__ == "__main__":
    print("--- INIZIO ANALISI STATISTICA ---")

    # 1. ANALISI ENERGIA
    if os.path.exists(path_energy):
        df_energy = pd.read_excel(path_energy)
        p_en = perform_test(df_energy, 'Median_Energy_J', 'Energia Cinetica (J)')
        plot_boxplot(df_energy, 'Median_Energy_J', 'Kinetic Energy (J)', p_en, "Boxplot_Energy.png")
    else:
        print(f"❌ File Energia non trovato: {file_energy}")

    # 2. ANALISI AVC (Disimpegno)
    if os.path.exists(path_avc):
        df_avc = pd.read_excel(path_avc)

        # A. AVC LABELS (Ground Truth) - Il più importante!
        p_avc_lbl = perform_test(df_avc, 'AVC_Labels', 'AVC (Labels - Ground Truth)')
        plot_boxplot(df_avc, 'AVC_Labels', 'AVC Ratio (Labels)', p_avc_lbl, "Boxplot_AVC_Labels.png")

        # B. AVC GEOMETRIC (Il tuo algoritmo)
        p_avc_geo = perform_test(df_avc, 'AVC_Geometric', 'AVC (Geometric - Algorithm)')
        plot_boxplot(df_avc, 'AVC_Geometric', 'AVC Ratio (Geometric)', p_avc_geo, "Boxplot_AVC_Geometric.png")

    else:
        print(f"❌ File AVC non trovato: {file_avc}")

    print("\n--- FINE ANALISI ---")