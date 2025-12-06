import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. SETUP ---
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Final_Energy_Statistics.xlsx")

# --- 2. LOAD DATA ---
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Dati caricati correttamente.")
else:
    print(f"Errore: File non trovato in {file_path}")
    # Se vuoi testare al volo senza file, scommenta queste righe e usa i dati che hai incollato
    # from io import StringIO
    # data_str = """Subject_ID Group Median_Energy_J
    # C01 TD 0.000012
    # ... (incolla qui i tuoi dati) ...
    # DG13 ASD 0.000218"""
    # df = pd.read_csv(StringIO(data_str), sep='\s+')
    exit()

# --- 3. STATISTICAL TEST (Mann-Whitney U) ---
# Separiamo i gruppi
td_energy = df[df['Group'] == 'TD']['Median_Energy_J']
asd_energy = df[df['Group'] == 'ASD']['Median_Energy_J']

# Eseguiamo il test (alternative='two-sided' è standard, 'less' o 'greater' se hai ipotesi direzionali forti)
stat, p_value = mannwhitneyu(td_energy, asd_energy)

print("\n" + "="*40)
print("RISULTATI STATISTICI (Mann-Whitney U)")
print("="*40)
print(f"TD (n={len(td_energy)}): Mediana = {td_energy.median():.6f} J")
print(f"ASD (n={len(asd_energy)}): Mediana = {asd_energy.median():.6f} J")
print("-" * 40)
print(f"U-statistic: {stat}")
print(f"P-value: {p_value:.5f}")

if p_value < 0.05:
    print("✅ RISULTATO SIGNIFICATIVO (p < 0.05)")
else:
    print("❌ RISULTATO NON SIGNIFICATIVO (p >= 0.05)")

# --- 4. PLOTTING ---
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Boxplot
ax = sns.boxplot(x='Group', y='Median_Energy_J', data=df, palette="Set2", showfliers=False)

# Swarmplot (punti singoli sopra il boxplot per vedere la distribuzione reale)
sns.swarmplot(x='Group', y='Median_Energy_J', data=df, color=".25", size=6)

# --- ANNOTAZIONE SIGNIFICATIVITÀ ---
# Coordinate per la barra
x1, x2 = 0, 1   # Indici delle categorie TD e ASD
y_max = df['Median_Energy_J'].max()
y_h = y_max * 0.05 # Altezza della stanghetta
y_line = y_max + y_h # Posizione Y della linea orizzontale

# Disegna la barra
plt.plot([x1, x1, x2, x2], [y_line, y_line + y_h, y_line + y_h, y_line], lw=1.5, c='k')

# Testo del p-value
significance = "ns"
if p_value < 0.001: significance = "***"
elif p_value < 0.01: significance = "**"
elif p_value < 0.05: significance = "*"

plt.text((x1+x2)*.5, y_line + y_h, f"p = {p_value:.4f}\n{significance}",
         ha='center', va='bottom', color='k', fontsize=12)

# Titoli e Label
plt.title('Kinetic Energy of Head Movements: TD vs ASD', fontsize=14)
plt.ylabel('Median Energy (J)', fontsize=12)
plt.xlabel('Group', fontsize=12)

# Aggiusta i limiti Y per far stare l'annotazione
plt.ylim(bottom=-0.0001, top=y_line + y_h * 5)

plt.tight_layout()

# Salva
output_img = os.path.join(base_dir, 'boxplot_energy_comparison.png')
plt.savefig(output_img, dpi=300)
print(f"\nGrafico salvato in: {output_img}")
plt.show()
