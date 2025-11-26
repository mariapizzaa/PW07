import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURAZIONE PERCORSI ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_dir, "dataset iniziali e risultati")

# Nomi dei file (modifica se necessario)
# File GEOMETRICI (Dati grezzi sensore)
file_geo_td = "TD_cleaned_advanced.xlsx"
file_geo_asd = "ASD_cleaned_advanced.xlsx"

# File LABELS (Annotazioni manuali - Ground Truth)
# Assicurati che siano .xlsx o .csv
file_lbl_td = "Visual_Analysis_TD_after.xlsx"
file_lbl_asd = "Visual_Analysis_ASD_after.xlsx"

path_TD_geo = os.path.join(data_folder, file_geo_td)
path_ASD_geo = os.path.join(data_folder, file_geo_asd)
path_TD_lbl = os.path.join(data_folder, file_lbl_td)
path_ASD_lbl = os.path.join(data_folder, file_lbl_asd)


# --- 2. FUNZIONI UTILI (Caricamento e Pulizia) ---

def smart_load(filepath):
    """Carica file Excel o CSV gestendo errori comuni."""
    if not os.path.exists(filepath):
        print(f"❌ FILE MANCANTE: {filepath}")
        return None
    try:
        return pd.read_excel(filepath)
    except:
        try:
            return pd.read_csv(filepath, sep=None, engine='python')
        except Exception as e:
            print(f"❌ Errore caricamento {os.path.basename(filepath)}: {e}")
            return None


def standardize_columns(df, file_type):
    """
    Rinomina le colonne per avere standard: 'id_soggetto', 'frame', 'label'.
    """
    # Rimuove spazi dai nomi colonne
    df.columns = [str(c).strip() for c in df.columns]

    # Mappa dei nomi possibili -> Nome standard
    column_map = {
        'id_soggetto': ['Subject', 'ID', 'Soggetto', 'id', 'subject'],
        'frame': ['Frame', 'frame_number', 'Time', 'timestamp'],  # Frame
        'label': ['Azione', 'Label', 'Action', 'azione']
    }

    for target, candidates in column_map.items():
        if target in df.columns: continue  # Se c'è già, ok
        for cand in candidates:
            # Cerca corrispondenza case-insensitive
            matches = [c for c in df.columns if c.lower() == cand.lower()]
            if matches:
                df.rename(columns={matches[0]: target}, inplace=True)
                break

    return df


# --- 3. CALCOLI GEOMETRICI ---

def get_gaze_vector(yaw, pitch):
    """Converte angoli in vettore direzione 3D (x,y,z)."""
    x = -np.sin(yaw) * np.cos(pitch)
    y = np.sin(pitch)
    z = np.cos(yaw) * np.cos(pitch)
    return np.column_stack((x, y, z))


def calculate_angle(v1, v2):
    """Calcola angolo in radianti tra due vettori."""
    # Normalizza i vettori
    v1_n = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2_n = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    # Prodotto scalare con clip per sicurezza numerica
    dot = np.clip(np.einsum('ij,ij->i', v1_n, v2_n), -1.0, 1.0)
    return np.arccos(dot)


def compute_geometric_attention(df):
    """
    Determina per ogni frame se è ATTENZIONE (True) o VACANCY (False)
    usando la logica geometrica (Terapista + Robot + Poster).
    """
    # Rimuovi frame senza sguardo
    mask_valid_gaze = ~df['yaw'].isna() & ~df['pitch'].isna()

    # Inizializza tutto a False (Vacancy)
    is_attention = np.zeros(len(df), dtype=bool)

    # Se non ci sono dati validi, ritorna array di False
    if not mask_valid_gaze.any():
        return is_attention

    # Subset dati validi per calcoli
    df_val = df[mask_valid_gaze]

    # 1. TERAPISTA (Dinamico 3D)
    looking_therapist = np.zeros(len(df_val), dtype=bool)
    if 'ther_keypoint_x' in df.columns:
        # Prendi coordinate
        cp = df_val[['child_keypoint_x', 'child_keypoint_y', 'child_keypoint_z']].values
        tp = df_val[['ther_keypoint_x', 'ther_keypoint_y', 'ther_keypoint_z']].values

        # Maschera coordinate valide
        valid_coords = ~np.isnan(cp).any(axis=1) & ~np.isnan(tp).any(axis=1)

        if np.any(valid_coords):
            gaze_vec = get_gaze_vector(df_val['yaw'].values, df_val['pitch'].values)
            vec_to_ther = tp[valid_coords] - cp[valid_coords]
            angles = calculate_angle(gaze_vec[valid_coords], vec_to_ther)

            # Soglia angolo: < 0.6 rad (~35 gradi)
            looking_therapist[valid_coords] = angles < 0.6

    # 2. ROBOT / TAVOLO (Centrale)
    # Pitch: guarda giù (-1.2) o dritto (0.4)
    # Yaw: centrale (+/- 0.4)
    looking_robot = (df_val['pitch'] > -1.2) & (df_val['pitch'] < 0.4) & \
                    (np.abs(df_val['yaw']) < 0.4)

    # 3. POSTER (Laterali)
    # Yaw: guarda ai lati (> 0.4 e < 1.4)
    looking_posters = (np.abs(df_val['yaw']) >= 0.4) & (np.abs(df_val['yaw']) < 1.4) & \
                      (df_val['pitch'] > -1.2) & (df_val['pitch'] < 0.4)

    # Unione logica
    is_att_val = looking_therapist | looking_robot | looking_posters

    # Riporta i risultati nell'array originale (gestendo i NaN iniziali)
    is_attention[mask_valid_gaze] = is_att_val

    return is_attention


# --- 4. MOTORE DI ANALISI (Confronto) ---

def process_group(path_geo, path_lbl, group_name):
    print(f"\n--- Analisi Gruppo {group_name} ---")

    # 1. Caricamento
    df_geo = smart_load(path_geo)
    df_lbl = smart_load(path_lbl)

    if df_geo is None or df_lbl is None: return []

    # 2. Standardizzazione Colonne
    df_geo = standardize_columns(df_geo, "Geometrico")
    df_lbl = standardize_columns(df_lbl, "Labels")

    # Controllo Frame (fondamentale)
    if 'frame' not in df_geo.columns:
        # Fallback: se manca frame, prova frame_cutted o usa l'indice
        if 'frame_cutted' in df_geo.columns:
            df_geo.rename(columns={'frame_cutted': 'frame'}, inplace=True)
        else:
            print("⚠️ Colonna 'frame' non trovata nel geometrico. Uso indice progressivo.")
            df_geo['frame'] = range(len(df_geo))

    results = []
    # Trova soggetti in comune
    subjects = set(df_geo['id_soggetto'].unique()) & set(df_lbl['id_soggetto'].unique())
    print(f"Soggetti analizzati: {len(subjects)}")

    for subj in subjects:
        # Filtra dati per soggetto
        d_g = df_geo[df_geo['id_soggetto'] == subj].sort_values('frame')
        d_l = df_lbl[df_lbl['id_soggetto'] == subj].sort_values('frame')

        if len(d_l) == 0: continue  # Salta se non ha etichette

        # --- DEFINIZIONE FINESTRA TEMPORALE ---
        # L'analisi vale solo dal primo all'ultimo frame annotato dall'umano
        start_frame = d_l['frame'].min()
        end_frame = d_l['frame'].max()
        total_window_frames = end_frame - start_frame + 1

        # Taglia i dati geometrici sulla stessa finestra
        d_g_window = d_g[(d_g['frame'] >= start_frame) & (d_g['frame'] <= end_frame)].copy()

        if len(d_g_window) == 0:
            print(f"   ⚠️ {subj}: Nessun dato geometrico nel range {start_frame}-{end_frame}")
            continue

        # --- CALCOLO A: GEOMETRICO (Sulla finestra) ---
        attention_mask = compute_geometric_attention(d_g_window)
        frames_att_geo = attention_mask.sum()

        # AVC Geometrico = (Totale - Attenzione) / Totale
        # Usiamo len(d_g_window) come denominatore reale dei dati disponibili
        valid_frames_geo = len(d_g_window)
        if valid_frames_geo > 0:
            avc_geo = (valid_frames_geo - frames_att_geo) / valid_frames_geo
        else:
            avc_geo = np.nan

        # --- CALCOLO B: LABELS (Sulla finestra) ---
        # Il file Labels contiene SOLO i frame di attenzione.
        # Quindi: Righe nel file = Attenzione.
        # Vacancy = Finestra Totale - Righe nel file.
        frames_att_lbl = len(d_l)

        # Calcolo Vacancy frames
        frames_vac_lbl = total_window_frames - frames_att_lbl
        # Gestione casi limite (es. duplicati che fanno sembrare più attenzioni che frame)
        if frames_vac_lbl < 0: frames_vac_lbl = 0

        avc_lbl = frames_vac_lbl / total_window_frames

        # --- SALVATAGGIO ---
        results.append({
            'Subject_ID': subj,
            'Group': group_name,
            'Window_Frames': total_window_frames,
            'Att_Frames_Geo': frames_att_geo,
            'Att_Frames_Lbl': frames_att_lbl,
            'AVC_Geometric': avc_geo,
            'AVC_Labels': avc_lbl,
            'Error': avc_geo - avc_lbl
        })

    return results


# --- 5. MAIN ---
if __name__ == "__main__":
    res_td = process_group(path_TD_geo, path_TD_lbl, "TD")
    res_asd = process_group(path_ASD_geo, path_ASD_lbl, "ASD")

    if res_td or res_asd:
        df_final = pd.DataFrame(res_td + res_asd)

        # Ordina e Salva
        df_final.sort_values(by=['Group', 'Subject_ID'], inplace=True)
        output_path = os.path.join(base_dir, "AVC_Comparison_Final.xlsx")
        df_final.to_excel(output_path, index=False)

        print("\n" + "=" * 50)
        print("✅ ANALISI AVC COMPLETATA")
        print(f"File salvato: {output_path}")
        print("=" * 50)

        # Statistiche rapide
        mae = df_final['Error'].abs().mean()
        print(f"Errore Medio Assoluto (Geo vs Label): {mae:.3f}")

        print("\nMedie AVC (Disimpegno):")
        print(df_final.groupby('Group')[['AVC_Geometric', 'AVC_Labels']].mean())
    else:
        print("⚠️ Nessun dato elaborato.")
