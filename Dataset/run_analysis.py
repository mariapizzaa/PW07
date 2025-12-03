import pandas as pd
import os
# Assicurati che Saccade_detection sia accessibile nel percorso o installato
from Saccade_detection import (
    extract_saccades_all_subjects,
    summarize_saccades_by_subject,
    compare_groups_on_metric,
    boxplot_metric_by_group
)

# Gestione robusta del percorso base (funziona sia come script che potenzialmente in IDE)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback se __file__ non è definito (es. Jupyter Notebook)
    base_dir = os.getcwd()

BASE_PATH = os.path.join(base_dir, "dataset iniziali e risultati")
filename_TD = "TD_cleaned_advanced.xlsx"
filename_ASD = "ASD_cleaned_advanced.xlsx"

# Definizione percorsi completi
path_TD = os.path.join(BASE_PATH, filename_TD)
path_ASD = os.path.join(BASE_PATH, filename_ASD)

print(f"Path TD: {path_TD}")
print(f"Path ASD: {path_ASD}")

# ---------------------------------------------------------
# 1) LOAD DATA (CORREZIONE QUI)
# ---------------------------------------------------------

# Controllo se i file esistono prima di caricare
if not os.path.exists(path_TD):
    raise FileNotFoundError(f"File non trovato: {path_TD}")
if not os.path.exists(path_ASD):
    raise FileNotFoundError(f"File non trovato: {path_ASD}")

# Caricamento effettivo dei DataFrame
# Nota: Se i file fossero CSV, usare pd.read_csv(path_TD)
print("Caricamento dati in corso...")
df_td = pd.read_excel(path_TD)
df_asd = pd.read_excel(path_ASD)

# Add group labels at frame-level
df_td["group"] = "TD"
df_asd["group"] = "ASD"

# Concatenate all data
df_all = pd.concat([df_td, df_asd], ignore_index=True)

print("TD shape:", df_td.shape)
print("ASD shape:", df_asd.shape)
print("ALL shape:", df_all.shape)

# Optional: check column names to confirm id_soggetto & timestamp & yaw
print("Columns:", df_all.columns.tolist())

import numpy as np

# dopo aver caricato df_all
dt_ms = df_all["timestamp"].sort_values().diff()
print("dt_ms (descrizione):")
print(dt_ms.describe())

# ---------------------------------------------------------
# 2) EXTRACT SACCADES FOR ALL SUBJECTS
# ---------------------------------------------------------
df_sacc_all = extract_saccades_all_subjects(
    df_all,
    subject_col="id_soggetto",
    time_col="timestamp",        # Unix time in ms
    yaw_col="yaw",               # radians
    vel_threshold=30.0,
    min_samples=2
)

print("\nSaccades extracted:", df_sacc_all.shape)
# print(df_sacc_all.head()) # Decommenta se vuoi vedere l'anteprima

# ---------------------------------------------------------
# 3) SUMMARY BY SUBJECT
# ---------------------------------------------------------
df_summary = summarize_saccades_by_subject(df_sacc_all)

print("\nSummary by subject (before merging group):")
# print(df_summary.head())

# ---------------------------------------------------------
# 4) PROPAGATE GROUP (TD/ASD) FROM ORIGINAL DATA TO SUMMARY
# ---------------------------------------------------------

# Build a mapping: subject_id -> group (TD/ASD)
# Take the first non-null group for each subject from the original df_all
subject_group_map = (
    df_all.groupby("id_soggetto")["group"]
          .first()
          .to_dict()
)

# Map onto df_summary.subject_id
# Assicurati che la colonna in df_summary si chiami 'subject_id' (output standard della tua funzione)
if 'subject_id' in df_summary.columns:
    df_summary["group"] = df_summary["subject_id"].map(subject_group_map)
else:
    # Fallback se la funzione summarize usa 'id_soggetto' come indice o colonna
    print("Attenzione: colonna 'subject_id' non trovata in df_summary. Controllo indice o nomi alternativi.")
    # Se 'id_soggetto' è l'indice:
    if df_summary.index.name == 'id_soggetto' or 'id_soggetto' in df_summary.columns:
         # Logica adattiva se necessario, altrimenti assumiamo che la funzione ritorni subject_id
         pass

print("\nSummary by subject (with group):")
print(df_summary.head())

# ---------------------------------------------------------
# 5) STATS: MANN-WHITNEY ON PEAK VELOCITY
# ---------------------------------------------------------
res_peak = compare_groups_on_metric(
    df_summary,
    metric_col="peak_vel_mean_deg_s",
    group_col="group",
    group1_label="TD",
    group2_label="ASD"
)

print("\nPeak velocity TD vs ASD:")
print(res_peak)

# ---------------------------------------------------------
# 6) STATS: MANN-WHITNEY ON MEAN DURATION
# ---------------------------------------------------------
res_dur = compare_groups_on_metric(
    df_summary,
    metric_col="duration_mean_ms",
    group_col="group",
    group1_label="TD",
    group2_label="ASD"
)

print("\nSaccade duration TD vs ASD:")
print(res_dur)

# ---------------------------------------------------------
# 7) BOXPLOTS
# ---------------------------------------------------------

# Peak velocity boxplot
boxplot_metric_by_group(
    df_summary,
    metric_col="peak_vel_mean_deg_s",
    group_col="group",
    group_order=("TD", "ASD"),
    title="Peak saccade velocity (TD vs ASD)",
    ylabel="Peak velocity (deg/s)"
)

# Duration boxplot
boxplot_metric_by_group(
    df_summary,
    metric_col="duration_mean_ms",
    group_col="group",
    group_order=("TD", "ASD"),
    title="Saccade duration (TD vs ASD)",
    ylabel="Duration (ms)"
)