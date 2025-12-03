import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
FPS = 30.0  # assumed frame rate (30 frames per second)

# ============================================================
# Saccade detection and basic metrics from gaze yaw
#
# Inspired by:
# Schmitt et al. (2014), Molecular Autism, "Saccadic eye
# movement abnormalities in autism spectrum disorder..."
#
# Key ideas replicated here:
# - Use a velocity threshold (~30 deg/s) to define saccade
#   onset and offset (Methods section, velocity-based criteria).
# - For each saccade, compute:
#     * duration (ms and s)
#     * peak velocity (deg/s)
#     * amplitude (difference in gaze angle)
#
# Assumptions:
# - timestamp is Unix time in milliseconds
# - yaw is in radians (horizontal gaze angle)
# - data may contain multiple subjects
# ============================================================


# -----------------------------
# 1. Helper: angular velocity
# -----------------------------

def compute_angular_velocity_deg(df_subj,
                                 yaw_col="yaw",
                                 fps=FPS):
    """
    Compute horizontal gaze angular velocity (deg/s)
    assuming a constant frame rate (fps), because timestamps
    are not reliable in this dataset.

    Parameters
    ----------
    df_subj : pd.DataFrame
        Data for one subject, sorted by time.
    yaw_col : str
        Name of the yaw column (in radians).
    fps : float
        Frame rate (frames per second), default = FPS (30 Hz).

    Returns
    -------
    vel_deg : pd.Series
        Angular velocity in deg/s aligned with df_subj index.
    """
    df_subj = df_subj.copy().sort_values("timestamp")

    # Convert yaw from radians to degrees
    yaw_deg = np.degrees(df_subj[yaw_col])

    # Constant delta time (s) between frames
    dt_sec = 1.0 / fps

    # Angular velocity in deg/s
    vel_deg = yaw_deg.diff() / dt_sec

    return vel_deg



# -----------------------------
# 2. Helper: detect saccades
# -----------------------------

def detect_saccades(df_subj,
                    vel_deg,
                    vel_threshold=30.0,
                    min_samples=2,
                    time_col="timestamp",
                    yaw_col="yaw"):
    """
    Detect saccades in a single subject based on gaze velocity.

    Saccades are defined as segments where |velocity| > vel_threshold
    for at least `min_samples` consecutive frames.
    """
    df_subj = df_subj.sort_values(time_col)
    vel_abs = vel_deg.abs()

    high = vel_abs > vel_threshold

    saccades = []
    in_saccade = False
    start_pos = None

    idx_array = df_subj.index.to_list()

    for pos in range(len(idx_array)):
        idx = idx_array[pos]
        is_high = bool(high.loc[idx]) if idx in high.index else False

        if not in_saccade and is_high:
            in_saccade = True
            start_pos = pos

        elif in_saccade and not is_high:
            end_pos = pos - 1
            if end_pos - start_pos + 1 >= min_samples:
                saccades.append({"start_pos": start_pos, "end_pos": end_pos})
            in_saccade = False
            start_pos = None

    # Edge case: saccade continues until last sample
    if in_saccade and start_pos is not None:
        end_pos = len(idx_array) - 1
        if end_pos - start_pos + 1 >= min_samples:
            saccades.append({"start_pos": start_pos, "end_pos": end_pos})

    # Convert positional indices to actual DataFrame indices
    for s in saccades:
        s["onset_idx"] = idx_array[s["start_pos"]]
        s["offset_idx"] = idx_array[s["end_pos"]]
        del s["start_pos"]
        del s["end_pos"]

    return saccades


# -----------------------------
# 3. Helper: metrics per saccade
# -----------------------------

def compute_saccade_metrics_for_subject(df_subj,
                                        vel_deg,
                                        saccades,
                                        subj_id,
                                        time_col="timestamp",
                                        yaw_col="yaw",
                                        fps=FPS):

    """
    Given detected saccades, compute amplitude, duration,
    and peak velocity for each saccade of a subject.
    """
    rows = []
    df_subj = df_subj.sort_values(time_col)

    for s_id, s in enumerate(saccades):
        onset_idx = s["onset_idx"]
        offset_idx = s["offset_idx"]

        seg = df_subj.loc[onset_idx:offset_idx]
        seg_vel = vel_deg.loc[onset_idx:offset_idx]

        # Approximate duration from number of frames and fps
        n_frames = len(seg)
        duration_ms = n_frames * (1000.0 / fps)
        duration_s = duration_ms / 1000.0

        # We still store onset/offset timestamps just for reference
        t_on_ms = seg[time_col].iloc[0]
        t_off_ms = seg[time_col].iloc[-1]

        yaw_on = seg[yaw_col].iloc[0]
        yaw_off = seg[yaw_col].iloc[-1]

        amplitude = yaw_off - yaw_on      # rad, signed
        amplitude_abs = abs(amplitude)

        peak_vel = seg_vel.abs().max()    # deg/s

        if amplitude > 0:
            direction = "right"
        elif amplitude < 0:
            direction = "left"
        else:
            direction = None

        rows.append({
            "subject_id": subj_id,
            "saccade_id": s_id,
            "onset_idx": onset_idx,
            "offset_idx": offset_idx,
            "onset_time_ms": t_on_ms,
            "offset_time_ms": t_off_ms,
            "duration_ms": duration_ms,
            "duration_s": duration_s,
            "amplitude_rad": amplitude,
            "amplitude_abs_rad": amplitude_abs,
            "peak_velocity_deg_s": peak_vel,
            "direction": direction
        })

    return pd.DataFrame(rows)


# -----------------------------
# 4. High-level pipeline
# -----------------------------

def extract_saccades_all_subjects(df,
                                  subject_col="id_soggetto",
                                  time_col="timestamp",
                                  yaw_col="yaw",
                                  vel_threshold=30.0,
                                  min_samples=3):
    """
    Full pipeline over all subjects:
    - Group by subject
    - Compute angular velocity (deg/s) from Unix ms timestamps
    - Detect saccades based on velocity threshold
    - Compute per-saccade metrics
    """
    saccade_dfs = []

    for subj_id, df_subj in df.groupby(subject_col):
        df_subj = df_subj.sort_values(time_col).copy()

        vel_deg = compute_angular_velocity_deg(
            df_subj,
            yaw_col=yaw_col,
            fps=FPS
        )

        saccades = detect_saccades(
            df_subj,
            vel_deg,
            vel_threshold=vel_threshold,
            min_samples=min_samples,
            time_col=time_col,
            yaw_col=yaw_col
        )

        if not saccades:
            continue

        df_sacc = compute_saccade_metrics_for_subject(
            df_subj,
            vel_deg,
            saccades,
            subj_id=subj_id,
            time_col=time_col,
            yaw_col=yaw_col,
            fps=FPS
        )

        saccade_dfs.append(df_sacc)

    if not saccade_dfs:
        return pd.DataFrame()

    return pd.concat(saccade_dfs, ignore_index=True)


# -----------------------------
# 5. Summary by subject
# -----------------------------

def summarize_saccades_by_subject(df_sacc,
                                  subject_col="subject_id"):
    """
    Compute subject-level summary metrics:
    - number of saccades
    - mean / std of duration
    - mean / std of amplitude (absolute)
    - mean / std of peak velocity
    """
    if df_sacc.empty:
        return pd.DataFrame()

    summary = df_sacc.groupby(subject_col).agg(
        n_saccades=("saccade_id", "count"),
        duration_mean_ms=("duration_ms", "mean"),
        duration_std_ms=("duration_ms", "std"),
        amplitude_mean_abs_rad=("amplitude_abs_rad", "mean"),
        amplitude_std_abs_rad=("amplitude_abs_rad", "std"),
        peak_vel_mean_deg_s=("peak_velocity_deg_s", "mean"),
        peak_vel_std_deg_s=("peak_velocity_deg_s", "std")
    ).reset_index()

    return summary


# ============================================================
# 6. Group-level statistical analysis (TD vs ASD)
#    Using ONLY Mann-Whitney U (non-parametric)
# ============================================================

def compare_groups_on_metric(df_summary,
                             metric_col,
                             group_col="group",
                             group1_label="TD",
                             group2_label="ASD"):
    """
    Compare two groups (e.g., TD vs ASD) on a given summary metric
    using Mann-Whitney U test (non-parametric).

    Parameters
    ----------
    df_summary : pd.DataFrame
        Output of summarize_saccades_by_subject(), plus a group column.
        Each row = one subject.
    metric_col : str
        Name of the metric column to compare, e.g.:
        - 'duration_mean_ms'
        - 'peak_vel_mean_deg_s'
    group_col : str
        Column indicating group membership, e.g. 'TD' or 'ASD'.
    group1_label : str
        Name of the first group (e.g. 'TD').
    group2_label : str
        Name of the second group (e.g. 'ASD').

    Returns
    -------
    result : dict
        Contains:
        - metric
        - group1, group2
        - n_group1, n_group2
        - mean_group1, mean_group2
        - test: 'Mann-Whitney U'
        - statistic, p_value
    """
    # Remove NaNs in metric or group
    df_valid = df_summary.dropna(subset=[metric_col, group_col])

    g1 = df_valid[df_valid[group_col] == group1_label][metric_col].values
    g2 = df_valid[df_valid[group_col] == group2_label][metric_col].values

    # If too few data, avoid crashing
    if len(g1) < 1 or len(g2) < 1:
        return {
            "metric": metric_col,
            "group1": group1_label,
            "group2": group2_label,
            "n_group1": len(g1),
            "n_group2": len(g2),
            "mean_group1": float(np.nan) if len(g1) == 0 else float(np.nanmean(g1)),
            "mean_group2": float(np.nan) if len(g2) == 0 else float(np.nanmean(g2)),
            "test": "Mann-Whitney U",
            "statistic": np.nan,
            "p_value": np.nan
        }

    mw_res = mannwhitneyu(g1, g2, alternative="two-sided")

    result = {
        "metric": metric_col,
        "group1": group1_label,
        "group2": group2_label,
        "n_group1": len(g1),
        "n_group2": len(g2),
        "mean_group1": float(np.nanmean(g1)),
        "mean_group2": float(np.nanmean(g2)),
        "test": "Mann-Whitney U",
        "statistic": mw_res.statistic,
        "p_value": mw_res.pvalue
    }

    return result


def boxplot_metric_by_group(df_summary,
                            metric_col,
                            group_col="group",
                            group_order=("TD", "ASD"),
                            title=None,
                            ylabel=None,
                            show=True,
                            save_path=None):
    """
    Draw a boxplot (TD vs ASD) for a given summary metric.

    Parameters
    ----------
    df_summary : pd.DataFrame
        Output of summarize_saccades_by_subject(), plus a group column.
        Each row = one subject.
    metric_col : str
        Metric to plot, e.g.:
        - 'duration_mean_ms'
        - 'peak_vel_mean_deg_s'
    group_col : str
        Column name containing group labels (e.g. 'group').
    group_order : tuple
        Order of groups on the x-axis, e.g. ('TD', 'ASD').
    title : str or None
        Plot title. If None, a default will be used.
    ylabel : str or None
        Y-axis label. If None, metric_col will be used.
    show : bool
        If True, calls plt.show().
    save_path : str or None
        If not None, saves the figure to this path.
    """
    df_valid = df_summary.dropna(subset=[metric_col, group_col])

    data = []
    labels = []
    for g in group_order:
        vals = df_valid[df_valid[group_col] == g][metric_col].values
        if len(vals) == 0:
            # Avoid empty lists that break boxplot
            continue
        data.append(vals)
        labels.append(g)

    if not data:
        print(f"No valid data to plot for metric '{metric_col}'.")
        return

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels)

    if title is None:
        title = f"Boxplot of {metric_col} by group"
    if ylabel is None:
        ylabel = metric_col

    plt.title(title)
    plt.ylabel(ylabel)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

# -----------------------------
# 7. Example usage (commented)
# -----------------------------
if __name__ == "__main__":
    # Example:
    # df = pd.read_excel("Pose_and_Gaze_dataset.xlsx")

    # df_sacc_all = extract_saccades_all_subjects(
    #     df,
    #     subject_col="id_soggetto",
    #     time_col="timestamp",
    #     yaw_col="yaw",
    #     vel_threshold=30.0,
    #     min_samples=2
    # )
    #
    # df_summary = summarize_saccades_by_subject(df_sacc_all)
    #
    # # Map each subject to its group (TD / ASD)
    # group_map = {
    #     "TD_01": "TD",
    #     "TD_02": "TD",
    #     "ASD_01": "ASD",
    #     # ...
    # }
    # df_summary["group"] = df_summary["subject_id"].map(group_map)
    #
    # # Compare TD vs ASD on peak velocity
    # res_peak = compare_groups_on_metric(
    #     df_summary,
    #     metric_col="peak_vel_mean_deg_s",
    #     group_col="group",
    #     group1_label="TD",
    #     group2_label="ASD"
    # )
    # print("Peak velocity comparison:", res_peak)
    #
    # # Compare TD vs ASD on mean duration
    # res_dur = compare_groups_on_metric(
    #     df_summary,
    #     metric_col="duration_mean_ms",
    #     group_col="group",
    #     group1_label="TD",
    #     group2_label="ASD"
    # )
    # print("Duration comparison:", res_dur)
    pass
