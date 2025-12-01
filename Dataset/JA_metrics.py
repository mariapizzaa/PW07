import os
import numpy as np
import pandas as pd

# -----------------------------------------
# PATHS
# -----------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

path_TD_pose  = os.path.join(DATA_DIR, "TD_cleaned_advanced.xlsx")
path_ASD_pose = os.path.join(DATA_DIR, "ASD_cleaned_advanced.xlsx")
path_TD_ev    = os.path.join(DATA_DIR, "Visual_Analysis_TD_after.xlsx")
path_ASD_ev   = os.path.join(DATA_DIR, "Visual_Analysis_ASD_after.xlsx")

# -----------------------------------------
# PARAMETERS
# -----------------------------------------

INVITE_ACTIONS = {"Coniglio", "Indica", "Guarda"}

PRE_WIN = 2.0    # seconds before invitation (baseline)
POST_WIN = 3.0   # seconds after invitation (response)

# -----------------------------------------
# LOAD DATA
# -----------------------------------------

def load_pose(path, group_label):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    df["Group"] = group_label
    df = df.sort_values(["id_soggetto", "timestamp"])
    return df

def load_events(path, group_label):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    df = df.rename(columns={"Subject": "id_soggetto", "Frame": "frame"})
    df["Group"] = group_label
    return df

df_TD_pose  = load_pose(path_TD_pose,  "TD")
df_ASD_pose = load_pose(path_ASD_pose, "ASD")
df_pose = pd.concat([df_TD_pose, df_ASD_pose], ignore_index=True)

df_TD_ev  = load_events(path_TD_ev,  "TD")
df_ASD_ev = load_events(path_ASD_ev, "ASD")
df_ev = pd.concat([df_TD_ev, df_ASD_ev], ignore_index=True)


# -----------------------------------------
# HEAD ENERGY (same as before)
# -----------------------------------------

TOTAL_MASS_KG = 25.0
HEAD_MASS_KG  = TOTAL_MASS_KG * 0.0668
HEAD_RADIUS_M = 0.0835
HEAD_INERTIA  = 0.4 * HEAD_MASS_KG * (HEAD_RADIUS_M ** 2)

def add_head_energy(df_subj):
    df = df_subj.sort_values("timestamp").copy()

    for col in ["child_keypoint_x", "child_keypoint_y", "child_keypoint_z"]:
        if col not in df.columns:
            df[col] = np.nan

    x_m = df["child_keypoint_x"].astype(float) / 1000.0
    y_m = df["child_keypoint_y"].astype(float) / 1000.0
    z_m = df["child_keypoint_z"].astype(float) / 1000.0

    dt = df["timestamp"].diff().astype(float)
    valid = (dt > 0) & (dt < 1.0)

    dx = x_m.diff()
    dy = y_m.diff()
    dz = z_m.diff()

    dist_sq = dx**2 + dy**2 + dz**2

    vel_sq = np.full(len(df), np.nan)
    vel_sq[valid] = dist_sq[valid] / (dt[valid] ** 2)

    yaw   = df["yaw"].astype(float)
    pitch = df["pitch"].astype(float)

    d_yaw   = yaw.diff()
    d_pitch = pitch.diff()
    d_yaw   = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

    ang_dist_sq = d_yaw**2 + d_pitch**2

    ang_vel_sq = np.full(len(df), np.nan)
    ang_vel_sq[valid] = ang_dist_sq[valid] / (dt[valid] ** 2)

    energy_trans = 0.5 * HEAD_MASS_KG * vel_sq
    energy_rot   = 0.5 * HEAD_INERTIA * ang_vel_sq

    df["energy_total"] = energy_trans + energy_rot

    return df


# -----------------------------------------
# LEVEL 1 "REACTIVITY" METRIC
# -----------------------------------------

def compute_reactivity_for_subject(df_pose_subj, df_ev_subj):
    if df_pose_subj.empty:
        return {
            "n_invites": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "prop_positive_delta": np.nan
        }

    df_pose_subj = add_head_energy(df_pose_subj)

    # attach timestamps to events
    df_ev_subj = pd.merge(
        df_ev_subj,
        df_pose_subj[["frame", "timestamp"]],
        on="frame",
        how="left"
    ).sort_values("timestamp")

    invites = df_ev_subj[
        (df_ev_subj["Admin"] == "Robot") &
        (df_ev_subj["Azione"].isin(INVITE_ACTIONS))
    ]

    if invites.empty:
        return {
            "n_invites": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "prop_positive_delta": np.nan
        }

    deltas = []

    for _, inv in invites.iterrows():
        t0 = inv["timestamp"]
        if pd.isna(t0):
            continue

        # baseline: PRE_WIN seconds before t0
        mask_pre = (df_pose_subj["timestamp"] >= t0 - PRE_WIN) & \
                   (df_pose_subj["timestamp"] <  t0)
        # response: POST_WIN seconds after t0
        mask_post = (df_pose_subj["timestamp"] >= t0) & \
                    (df_pose_subj["timestamp"] <= t0 + POST_WIN)

        E_pre  = df_pose_subj.loc[mask_pre,  "energy_total"].dropna()
        E_post = df_pose_subj.loc[mask_post, "energy_total"].dropna()

        if E_pre.empty or E_post.empty:
            continue

        baseline = float(np.median(E_pre))
        response = float(np.median(E_post))

        deltas.append(response - baseline)

    if len(deltas) == 0:
        return {
            "n_invites": int(len(invites)),
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "prop_positive_delta": np.nan
        }

    deltas = np.array(deltas)
    mean_delta   = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    prop_pos     = float(np.mean(deltas > 0))

    return {
        "n_invites": int(len(invites)),
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "prop_positive_delta": prop_pos
    }


# -----------------------------------------
# LOOP SU TUTTI I SOGGETTI
# -----------------------------------------

rows = []

for subj_id in df_pose["id_soggetto"].unique():
    df_pose_subj = df_pose[df_pose["id_soggetto"] == subj_id].sort_values("timestamp")
    df_ev_subj   = df_ev[df_ev["id_soggetto"] == subj_id]

    metrics = compute_reactivity_for_subject(df_pose_subj, df_ev_subj)
    metrics["Subject_ID"] = subj_id
    metrics["Group"]      = df_pose_subj["Group"].iloc[0]

    rows.append(metrics)

df_reac = pd.DataFrame(rows)

out_path = os.path.join(base_dir, "JA_Level1_reactivity_summary.xlsx")
df_reac.to_excel(out_path, index=False)

print("\n=== JA LEVEL 1 (reactivity) COMPLETE ===")
print("Saved to:", out_path)
print(df_reac)
# dentro il tuo JA_metrics.py, dopo aver caricato df_pose e df_ev

subj = "C01"  # o un altro
dfp = df_pose[df_pose["id_soggetto"] == subj].sort_values("timestamp")
dfe = df_ev[df_ev["id_soggetto"] == subj].copy()

dfp = add_head_energy(dfp)

# attach timestamps
dfe = pd.merge(
    dfe,
    dfp[["frame", "timestamp"]],
    on="frame",
    how="left"
).sort_values("timestamp")

print("Inviti Robot per", subj)
inv = dfe[(dfe["Admin"] == "Robot") & (dfe["Azione"].isin(INVITE_ACTIONS))]
print(inv[["frame", "timestamp", "Azione"]].head(10))

print("Quanti inviti hanno timestamp NaN?", inv["timestamp"].isna().sum())

