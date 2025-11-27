"""
ASC_rate computation script

This script computes ASC_rate (Area-of-interest Switch Count *per AOI frame*)
for TD and ASD groups, following the logic of the ASC metric described in the
eye-tracking article you are using.

Main idea (from the article):
--------------------------------
- ASC measures how often the gaze switches from one AOI to another.
- Here we normalize ASC by the number of AOI frames, obtaining ASC_rate:
      ASC_rate = (number of AOI-to-AOI switches) / (number of AOI frames)

Author: you :)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# ============================================================
# 1. PATHS AND FILE NAMES
# ============================================================

# We assume this script is saved in the same folder as "cleaning dataset.py".
# The cleaned Excel files are stored in the subfolder:
#   "dataset iniziali e risultati"
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_dir, "dataset iniziali e risultati")

# File names produced by the cleaning script
TD_FILENAME = "TD_cleaned_advanced.xlsx"
ASD_FILENAME = "ASD_cleaned_advanced.xlsx"

td_path = os.path.join(DATA_DIR, TD_FILENAME)
asd_path = os.path.join(DATA_DIR, ASD_FILENAME)

if not os.path.exists(td_path) or not os.path.exists(asd_path):
    raise FileNotFoundError(
        "Could not find the cleaned datasets.\n"
        f"  TD:  {td_path}\n"
        f"  ASD: {asd_path}\n"
        "Please check that your cleaning script saved them with these names."
    )

print("Loading cleaned datasets...")
df_td = pd.read_excel(td_path)
df_asd = pd.read_excel(asd_path)
print("   TD shape:", df_td.shape)
print("   ASD shape:", df_asd.shape, "\n")

# ============================================================
# 2. COLUMN NAMES AND AOI LABELS
# ============================================================

# These column names reflect your cleaned files.
# If your actual column names differ, update them here.
ID_COL = "id_soggetto"      # subject identifier
FRAME_COL = "frame_cutted"  # temporal order (frame index)
LABEL_COL = "label"         # AOI label (robot, giocattolo, poster_dx, ...)

# Basic checks to ensure the required columns exist
for name, df in [("TD", df_td), ("ASD", df_asd)]:
    for col in (ID_COL, FRAME_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in {name} dataset.")

# We build the set of AOI labels directly from the data.
# This is safer than hard-coding names, and will adapt if you change labels.
all_labels = pd.concat(
    [df_td[LABEL_COL], df_asd[LABEL_COL]],
    axis=0
)

# Convert to strings, strip spaces, and lowercase.
all_labels = (
    all_labels
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

unique_labels = sorted(all_labels.unique())

# Labels that are NOT AOIs (e.g., gaze outside any region of interest).
# These labels will be excluded when computing ASC_rate.
NON_AOI = {"nowhere", "none", "background"}

# AOI labels are all labels that are not in NON_AOI.
AOI_LABELS = [lab for lab in unique_labels if lab not in NON_AOI]

print("AOI labels found in the datasets (used for ASC_rate):")
for lab in AOI_LABELS:
    print("  -", lab)
print()

if not AOI_LABELS:
    raise ValueError(
        "No valid AOI labels were found (AOI_LABELS is empty). "
        "Check the content of the 'label' column."
    )

# For consistency, also normalize labels in the dataframes (lowercase, stripped)
df_td[LABEL_COL] = df_td[LABEL_COL].astype(str).str.strip().str.lower()
df_asd[LABEL_COL] = df_asd[LABEL_COL].astype(str).str.strip().str.lower()

# ============================================================
# 3. FUNCTION TO COMPUTE ASC_rate
# ============================================================

def compute_asc_rate_per_subject(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """
    Compute ASC_rate for each subject in a given group (TD or ASD).

    According to the article:
    -------------------------
    - ASC = number of times the gaze switches from one AOI to another AOI.
    - We only consider AOI frames (frames where the label is one of the AOI_LABELS).
    - Consecutive AOI frames with the same label do NOT count as a switch.
    - Here we compute ASC_rate = ASC / N_AOI_FRAMES.

    Parameters
    ----------
    df : DataFrame
        The cleaned eye-tracking data for one group (TD or ASD).
    group_name : str
        Just for printing/logging ("TD" or "ASD").

    Returns
    -------
    result_df : DataFrame
        One row per subject with:
        - id_soggetto
        - ASC (raw number of switches)
        - n_AOI_frames (number of frames with a valid AOI label)
        - ASC_rate (switches per AOI frame)
    """

    results = []

    # Group by subject
    for subject_id, df_sub in df.groupby(ID_COL):

        # 1) Sort by frame index to respect temporal order
        df_sub = df_sub.sort_values(FRAME_COL)

        # 2) Keep only rows where label is one of the AOI labels
        df_aoi = df_sub[df_sub[LABEL_COL].isin(AOI_LABELS)].copy()

        # Total number of AOI frames for this subject
        n_aoi_frames = len(df_aoi)

        # If there are fewer than 2 AOI frames, no switches can occur
        if n_aoi_frames <= 1:
            asc = 0
            asc_rate = 0.0

        else:
            # 3) Extract the AOI sequence as a numpy array
            aoi_seq = df_aoi[LABEL_COL].to_numpy()

            # 4) Compute where the label changes between consecutive AOI frames:
            #    changes[i] is True when aoi_seq[i+1] != aoi_seq[i]
            changes = aoi_seq[1:] != aoi_seq[:-1]

            # 5) ASC is the number of such changes (AOI-to-AOI switches)
            asc = int(changes.sum())

            # 6) ASC_rate = ASC / number of AOI frames
            #    (this normalizes for how long the subject actually spent on AOIs)
            asc_rate = asc / float(n_aoi_frames)

        results.append(
            {
                ID_COL: subject_id,
                "ASC": asc,
                "n_AOI_frames": n_aoi_frames,
                "ASC_rate": asc_rate,
            }
        )

    result_df = pd.DataFrame(results).sort_values(ID_COL).reset_index(drop=True)
    print(f"{group_name}: computed ASC_rate for {len(result_df)} subjects.")
    return result_df

# ============================================================
# 4. COMPUTE ASC_rate FOR TD AND ASD
# ============================================================

asc_td = compute_asc_rate_per_subject(df_td, "TD")
asc_asd = compute_asc_rate_per_subject(df_asd, "ASD")

print("\n--- Descriptive statistics for ASC_rate ---")
print("TD  - N subjects:", len(asc_td),
      "  mean:", asc_td["ASC_rate"].mean(),
      "  sd:", asc_td["ASC_rate"].std(),
      "  median:", asc_td["ASC_rate"].median())
print("ASD - N subjects:", len(asc_asd),
      "  mean:", asc_asd["ASC_rate"].mean(),
      "  sd:", asc_asd["ASC_rate"].std(),
      "  median:", asc_asd["ASC_rate"].median())

# ============================================================
# 5. STATISTICAL TESTS ON ASC_rate (TD vs ASD)
# ============================================================

def compare_groups_on_asc_rate(td_df: pd.DataFrame, asd_df: pd.DataFrame) -> None:
    """
    Compare TD and ASD groups on ASC_rate using:

    - Welch's t-test (does not assume equal variances)
    - Mann–Whitney U test (non-parametric)

    This is consistent with the article's approach of comparing
    eye-tracking metrics between groups.
    """

    td_vals = td_df["ASC_rate"].to_numpy(dtype=float)
    asd_vals = asd_df["ASC_rate"].to_numpy(dtype=float)

    # Remove any possible NaNs (should not be present, but just in case)
    td_vals = td_vals[~np.isnan(td_vals)]
    asd_vals = asd_vals[~np.isnan(asd_vals)]

    print("\n--- Group comparison on ASC_rate ---")
    print("Sample sizes: TD =", len(td_vals), " ASD =", len(asd_vals))

    if len(td_vals) < 2 or len(asd_vals) < 2:
        print("Not enough data in one of the groups to run statistical tests.")
        return

    # Check for degenerate case where all values are identical in both groups
    if np.all(td_vals == td_vals[0]) and np.all(asd_vals == asd_vals[0]):
        print("All ASC_rate values are identical in both groups; no variance.")
        print("Statistical tests would not be meaningful in this case.")
        return

    # Welch's t-test (robust to unequal variances)
    t_stat, p_t = ttest_ind(td_vals, asd_vals, equal_var=False)
    print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_t:.5f}")

    # Mann–Whitney U test (non-parametric)
    u_stat, p_u = mannwhitneyu(td_vals, asd_vals, alternative="two-sided")
    print(f"Mann–Whitney U: U = {u_stat:.3f}, p = {p_u:.5f}")

compare_groups_on_asc_rate(asc_td, asc_asd)

# ============================================================
# 6. SAVE RESULTS TO EXCEL
# ============================================================

td_out = os.path.join(DATA_DIR, "ASC_rate_TD_per_subject.xlsx")
asd_out = os.path.join(DATA_DIR, "ASC_rate_ASD_per_subject.xlsx")

asc_td.to_excel(td_out, index=False)
asc_asd.to_excel(asd_out, index=False)

print("\nASC_rate results saved to:")
print("  TD: ", td_out)
print("  ASD:", asd_out)
print("\nDone ✅")
