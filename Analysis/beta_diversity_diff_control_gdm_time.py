import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os
import warnings

warnings.filterwarnings('ignore')

# metric_to_use = 'braycurtis' #jaccard
metric_to_use = 'jaccard'

cohort = 'Israel'
root_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(root_dir, 'Data', cohort)

t1 = pd.read_csv(os.path.join(data_path, "T1.csv"), index_col=0) if cohort == 'Israel' else pd.read_csv(
    os.path.join(data_path, "T2.csv"), index_col=0)
t2 = pd.read_csv(os.path.join(data_path, "T2.csv"), index_col=0)
t3 = pd.read_csv(os.path.join(data_path, "T3.csv"), index_col=0)
tag = pd.read_csv(os.path.join(data_path, 'gdm_control_tag.csv'), index_col=0, delimiter='\t')
# remove duplicated indices
# tag = tag[~tag.index.duplicated(keep='first')]


group_column = "Status"


def extract_data(df, tag, timepoint):
    """Extract metadata and OTU table for a given timepoint with aligned indices."""
    mutual = df.index.intersection(tag.index)
    if len(mutual) == 0 or df.shape[1] == 0:
        # Return empty if nothing to analyze for this tp
        return pd.DataFrame(columns=['Sample_ID', 'Group', 'Time', 'UniqueID']).set_index('UniqueID'), df.iloc[0:0]
    df = df.loc[mutual]
    tag = tag.loc[mutual]
    metadata = pd.DataFrame({
        'Sample_ID': df.index,
        'Group': tag[group_column],
        'Time': timepoint,
    })
    # Make IDs unique by appending timepoint
    metadata['UniqueID'] = metadata['Sample_ID'].astype(str) + "_" + timepoint
    metadata = metadata.set_index('UniqueID')
    # Set the OTU table index to match UniqueID
    df = df.copy()
    df.index = metadata.index
    return metadata, df


# Extract per timepoint
if cohort == "Israel":
    t1_metadata, t1_otu = extract_data(t1, tag, "T1")
else:
    t1_metadata, t1_otu = '', ''
t2_metadata, t2_otu = extract_data(t2, tag, "T2")
t3_metadata, t3_otu = extract_data(t3, tag, "T3")

# Keep only non-empty timepoints
metas = [m for m in [t1_metadata, t2_metadata, t3_metadata] if len(m) > 0] if cohort == 'Israel' else [m for m in
                                                                                                       [t2_metadata,
                                                                                                        t3_metadata] if
                                                                                                       len(m) > 0]
otus = [o for o in [t1_otu, t2_otu, t3_otu] if len(o) > 0] if cohort == 'Israel' else [o for o in [t2_otu, t3_otu] if
                                                                                       len(o) > 0]

if len(metas) == 0:
    raise ValueError("No overlapping samples between OTU tables and metadata. Check indices/filenames.")

# Combine all timepoints
metadata_all = pd.concat(metas, axis=0)
otu_all = pd.concat(otus, axis=0)

# Standardize group names (GDM vs Control)
metadata_all['Group'] = metadata_all['Group'].apply(
    lambda x: 'GDM' if 'gdm' in str(x).lower() else 'Control'
)

# Ensure ordering consistency between metadata and OTU (index is UniqueID)
otu_all = otu_all.loc[metadata_all.index]

print(f"Total samples (rows): {len(metadata_all)}")
print(f"GDM samples: {int((metadata_all['Group'] == 'GDM').sum())}")
print(f"Control samples: {int((metadata_all['Group'] == 'Control').sum())}")
print(f"Number of microbial features (columns): {otu_all.shape[1]}")

# ============================================================================
# 2. CALCULATE BRAY–CURTIS DISTANCES
# ============================================================================

print("\nCalculating Bray–Curtis distances with scipy...")

# Replace negatives/NaNs if any (defensive)
otu_mat = otu_all.fillna(0).clip(lower=0).astype(float).values
bc_condensed = pdist(otu_mat, metric=metric_to_use)
bray_dist_matrix = squareform(bc_condensed)  # square matrix (n x n)

# Construct DistanceMatrix with UNIQUE ids that match metadata index
ids = metadata_all.index.astype(str).tolist()
dm = DistanceMatrix(bray_dist_matrix, ids=ids)

# ============================================================================
# 3. TWO-WAY PERMANOVA (Group, Time, and Group×Time interaction)
# ============================================================================

print("\n" + "=" * 50)
print("TWO-WAY PERMANOVA RESULTS")
print("=" * 50 + "\n")

# Important: PERMANOVA expects grouping frame indexed by the same IDs as dm
grouping_df = metadata_all.copy()
grouping_df.index = ids  # ensure exact match

# Check if we have multiple timepoints
has_multiple_timepoints = grouping_df['Time'].nunique() > 1


def two_way_permanova(dm, metadata_df, factor1='Group', factor2='Time', permutations=9999):
    """
    Perform a two-way PERMANOVA to test main effects and interaction.

    This function performs sequential PERMANOVA tests to estimate:
    1. Main effect of factor1 (e.g., Group)
    2. Main effect of factor2 (e.g., Time) after accounting for factor1
    3. Interaction effect (factor1 × factor2) after accounting for both main effects

    Parameters:
    -----------
    dm : DistanceMatrix
        Distance matrix for samples
    metadata_df : pd.DataFrame
        Metadata with factors as columns
    factor1 : str
        Name of first factor (default: 'Group')
    factor2 : str
        Name of second factor (default: 'Time')
    permutations : int
        Number of permutations (default: 9999)

    Returns:
    --------
    dict with results for each effect
    """
    results = {}

    # Create interaction term
    metadata_df['Interaction'] = (metadata_df[factor1].astype(str) + "_" +
                                  metadata_df[factor2].astype(str))

    # Main effect of factor1
    perm_factor1 = permanova(dm, metadata_df, column=factor1, permutations=permutations)
    results['factor1'] = {
        'name': factor1,
        'result': perm_factor1,
        'F': float(perm_factor1.get('test statistic', np.nan)),
        'p': float(perm_factor1.get('p-value', np.nan)),
        'R2': float(perm_factor1.get('number of groups', np.nan))
    }

    # Main effect of factor2
    perm_factor2 = permanova(dm, metadata_df, column=factor2, permutations=permutations)
    results['factor2'] = {
        'name': factor2,
        'result': perm_factor2,
        'F': float(perm_factor2.get('test statistic', np.nan)),
        'p': float(perm_factor2.get('p-value', np.nan))
    }

    # Interaction effect (using combined factor)
    perm_interaction = permanova(dm, metadata_df, column='Interaction', permutations=permutations)
    results['interaction'] = {
        'name': f"{factor1} × {factor2}",
        'result': perm_interaction,
        'F': float(perm_interaction.get('test statistic', np.nan)),
        'p': float(perm_interaction.get('p-value', np.nan))
    }

    return results


# Run two-way PERMANOVA if we have multiple timepoints
if has_multiple_timepoints:
    print("Running Two-Way PERMANOVA (Group × Time)...\n")
    two_way_results = two_way_permanova(dm, grouping_df, factor1='Group', factor2='Time', permutations=9999)

    # Display results
    print(f"Main Effect - {two_way_results['factor1']['name']}:")
    print(two_way_results['factor1']['result'])
    print(f"\nF-statistic: {two_way_results['factor1']['F']:.4f}")
    print(f"P-value: {two_way_results['factor1']['p']:.4f}\n")
    print("=" * 50 + "\n")

    print(f"Main Effect - {two_way_results['factor2']['name']}:")
    print(two_way_results['factor2']['result'])
    print(f"\nF-statistic: {two_way_results['factor2']['F']:.4f}")
    print(f"P-value: {two_way_results['factor2']['p']:.4f}\n")
    print("=" * 50 + "\n")

    print(f"Interaction Effect - {two_way_results['interaction']['name']}:")
    print(two_way_results['interaction']['result'])
    print(f"\nF-statistic: {two_way_results['interaction']['F']:.4f}")
    print(f"P-value: {two_way_results['interaction']['p']:.4f}\n")
    print("=" * 50 + "\n")

    # Store key values for later use
    group_F = two_way_results['factor1']['F']
    group_p = two_way_results['factor1']['p']
    time_F = two_way_results['factor2']['F']
    time_p = two_way_results['factor2']['p']
    interaction_F = two_way_results['interaction']['F']
    interaction_p = two_way_results['interaction']['p']

else:
    # If only one timepoint, just run simple PERMANOVA for Group
    print("Only one timepoint present. Running one-way PERMANOVA for Group effect...\n")
    perm_group = permanova(dm, grouping_df, column='Group', permutations=9999)
    print(perm_group)
    group_F = float(perm_group.get('test statistic', np.nan))
    group_p = float(perm_group.get('p-value', np.nan))
    time_F, time_p = np.nan, np.nan
    interaction_F, interaction_p = np.nan, np.nan


# ============================================================================
# 3b. R²-like effect size from PCoA coordinates (between/total SS)
# ============================================================================

def eta_squared_from_coords(coords: pd.DataFrame, groups: pd.Series) -> float:
    """
    Compute between/total SS using all PCoA axes as an R²-like effect size.
    Uses label-based selection to avoid NumPy integer-only indexing errors.
    """
    # Ensure the same order and membership
    coords = coords.loc[groups.index]
    X = coords.values  # (n, k)
    mu = X.mean(axis=0, keepdims=True)
    ss_total = float(((X - mu) ** 2).sum())

    ss_between = 0.0
    # groups.groupby(groups).groups returns a dict: {label -> Index of row labels}
    for g, idx_labels in groups.groupby(groups).groups.items():
        Xg = coords.loc[idx_labels].values
        mug = Xg.mean(axis=0, keepdims=True)
        ss_between += len(Xg) * float(((mug - mu) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else np.nan


# ============================================================================
# 4. PERMDISP - Homogeneity of dispersion
# ============================================================================

print("\n" + "=" * 50)
print("DISPERSION ANALYSIS (PERMDISP)")
print("=" * 50 + "\n")


def permdisp(distance_matrix: np.ndarray, grouping: pd.Series, permutations: int = 999) -> tuple[float, float]:
    """
    Test homogeneity of multivariate dispersions (PERMDISP) on a square distance matrix.
    Returns (F-stat, p-value).
    """
    groups = grouping.astype(str).values
    unique_groups = np.unique(groups)
    n = len(groups)

    # Distance to group centroid proxies: mean distance to others in the same group
    distances_to_centroid = []
    group_labels = []

    for g in unique_groups:
        mask = (groups == g)
        if mask.sum() < 2:
            # skip groups with 1 sample (no dispersion)
            continue
        sub = distance_matrix[np.ix_(mask, mask)]
        # Mean distance to others in the same group for each sample
        # subtract diagonal (zeros) by averaging row over (k-1) members
        k = sub.shape[0]
        mean_to_centroid = sub.sum(axis=1) / (k - 1)
        distances_to_centroid.extend(mean_to_centroid.tolist())
        group_labels.extend([g] * k)

    distances_df = pd.DataFrame({'distance': distances_to_centroid, 'group': group_labels})
    if len(distances_df) == 0:
        return np.nan, np.nan

    observed_F = _compute_f_stat(distances_df)

    # Permutation test
    perm_F_stats = []
    np.random.seed(42)
    for _ in range(permutations):
        distances_df_perm = distances_df.copy()
        distances_df_perm['group'] = np.random.permutation(distances_df_perm['group'].values)
        perm_F_stats.append(_compute_f_stat(distances_df_perm))

    p_value = (np.sum(np.array(perm_F_stats) >= observed_F) + 1) / (permutations + 1)
    return observed_F, p_value


def _compute_f_stat(df: pd.DataFrame) -> float:
    groups = df['group'].unique()
    if len(groups) < 2:
        return 0.0
    k = len(groups)
    n = len(df)
    grand_mean = df['distance'].mean()
    ss_between = sum(
        len(df[df['group'] == g]) * ((df[df['group'] == g]['distance'].mean() - grand_mean) ** 2)
        for g in groups
    )
    ss_within = sum(
        ((df[df['group'] == g]['distance'] - df[df['group'] == g]['distance'].mean()) ** 2).sum()
        for g in groups
    )
    if ss_within == 0 or (n - k) == 0:
        return 0.0
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    return ms_between / ms_within if ms_within != 0 else 0.0


# PERMDISP for Group
disp_F_group, disp_p_group = permdisp(bray_dist_matrix, grouping_df['Group'], permutations=999)
print(f"PERMDISP (Group - GDM vs Control):")
print(f"  F = {disp_F_group:.4f}, p = {disp_p_group:.4f}")

if has_multiple_timepoints:
    # PERMDISP for Time
    disp_F_time, disp_p_time = permdisp(bray_dist_matrix, grouping_df['Time'], permutations=999)
    print(f"\nPERMDISP (Time):")
    print(f"  F = {disp_F_time:.4f}, p = {disp_p_time:.4f}")
else:
    disp_F_time, disp_p_time = np.nan, np.nan

print()

# ============================================================================
# 5. PCoA ORDINATION
# ============================================================================

print("\n" + "=" * 50)
print("PCoA ORDINATION")
print("=" * 50 + "\n")

pcoa_result = pcoa(dm)
print(f"PCoA computed. First 3 eigenvalues: {pcoa_result.eigvals[:3].values}")
print(f"Variance explained by first 3 axes: {(pcoa_result.proportion_explained[:3] * 100).values}")

# ============================================================================
# 5a. PCoA colored by Time (if multiple timepoints)
# ============================================================================

if has_multiple_timepoints:
    coords_all = pcoa_result.samples
    coords_2d = coords_all.iloc[:, :2]
    variance_explained = (pcoa_result.proportion_explained[:2] * 100).values

    plot_df_time = pd.DataFrame({
        'PC1': coords_2d.iloc[:, 0].values,
        'PC2': coords_2d.iloc[:, 1].values,
        'Time': grouping_df['Time'].values
    }, index=grouping_df.index)

    fig, ax = plt.subplots(figsize=(10, 7))
    time_colors = {'T1': '#2ca02c', 'T2': '#ff7f0e', 'T3': '#d62728'}
    present_times = sorted(plot_df_time['Time'].unique().tolist())

    for tp in present_times:
        mask = plot_df_time['Time'] == tp
        ax.scatter(
            plot_df_time.loc[mask, 'PC1'],
            plot_df_time.loc[mask, 'PC2'],
            c=time_colors.get(tp, '#1f77b4'),
            label=tp,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)', fontsize=12)

    # Title with PERMANOVA result
    ax.set_title(
        f"{cohort} - PCoA of Oral Microbiome Composition by Pregnancy Trimester\n"
        f'Metric: {metric_to_use.capitalize()}\n'
        f"PERMANOVA (Time): F = {time_F:.3f}, p = {time_p:.3f}",
        fontsize=13,
        fontweight='bold',
        pad=15
    )

    # Style and layout
    ax.legend(title="Trimester", loc='best', frameon=False)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    path_to_save = os.path.join(root_dir, 'Appendix_plots')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # plt.savefig(os.path.join(path_to_save, f'PCoA_Time_differ_{metric_to_use}_{cohort}.png'), dpi=300,
    #             bbox_inches='tight')
    plt.show()

# Coordinates (use first 2 for plotting; all for effect size)
coords_all = pcoa_result.samples  # all axes
coords_2d = coords_all.iloc[:, :2]
variance_explained = (pcoa_result.proportion_explained[:2] * 100).values

# R²-like from coords (Group)
eta2_group = eta_squared_from_coords(coords_all, grouping_df['Group'])
eta2_time = eta_squared_from_coords(coords_all, grouping_df['Time']) if has_multiple_timepoints else np.nan

# Prepare plotting dataframe
plot_df = pd.DataFrame({
    'PC1': coords_2d.iloc[:, 0].values,
    'PC2': coords_2d.iloc[:, 1].values,
    'Group': grouping_df['Group'].values,
    'Time': grouping_df['Time'].values
}, index=grouping_df.index)

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = {'Control': '#1f77b4', 'GDM': '#ff7f0e'}

present_times = sorted(plot_df['Time'].unique().tolist())
# Distinct markers for up to three timepoints
marker_map = {'T1': 'o', 'T2': 's', 'T3': '^'}
default_markers = ['o', 's', '^', 'D', 'P', 'X']
for i, tp in enumerate(present_times):
    if tp not in marker_map:
        marker_map[tp] = default_markers[i % len(default_markers)]

for group in ['Control', 'GDM']:
    for time in present_times:
        mask = (plot_df['Group'] == group) & (plot_df['Time'] == time)
        if mask.sum() == 0:
            continue
        ax.scatter(
            plot_df.loc[mask, 'PC1'],
            plot_df.loc[mask, 'PC2'],
            c=colors[group],
            marker=marker_map[time],
            s=100,
            alpha=0.7,
            label=f"{group} - {time}",
            edgecolors='black',
            linewidth=0.5
        )


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    if np.any(~np.isfinite(cov)) or cov.shape != (2, 2):
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    ell_radius_x = np.sqrt(max(1e-12, 1 + pearson))
    ell_radius_y = np.sqrt(max(1e-12, 1 - pearson))
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(max(1e-12, cov[0, 0])) * n_std
    scale_y = np.sqrt(max(1e-12, cov[1, 1])) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    transf = transforms.Affine2D().scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


# Add 95% ellipses per group
for group in ['Control', 'GDM']:
    mask = (plot_df['Group'] == group)
    if mask.sum() >= 3:
        confidence_ellipse(plot_df.loc[mask, 'PC1'], plot_df.loc[mask, 'PC2'],
                           ax, n_std=2, edgecolor=colors[group],
                           linestyle='--', linewidth=2, alpha=0.5)

# Updated title to include interaction effect if available
title_text = (
    'PCoA: GDM vs Control Across Pregnancy\n'
    f'Metric: {metric_to_use.capitalize()}\n'
    f'PERMANOVA (Group): F = {group_F:.3f}, p = {group_p:.3f}  |  '
    f'PCoA η² (Group) ≈ {eta2_group * 100:.1f}%'
)

if has_multiple_timepoints:
    title_text += f'\nInteraction (Group × Time): F = {interaction_F:.3f}, p = {interaction_p:.3f}'

ax.set_xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)', fontsize=12)
ax.set_title(title_text, fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
plt.tight_layout()
path_to_save = os.path.join(root_dir, 'Appendix_plots')
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
# plt.savefig(os.path.join(path_to_save, f'PCoA_GDM_vs_Control_{metric_to_use}_{cohort}.png'), dpi=300,
#             bbox_inches='tight')
print("Saved: PCoA_GDM_vs_Control.png")
plt.show()

# ============================================================================
# 6. TIMEPOINT-SPECIFIC PERMANOVA
# ============================================================================

print("\n" + "=" * 50)
print("TIMEPOINT-SPECIFIC PERMANOVA")
print("=" * 50 + "\n")

timepoint_results = []
for tp in present_times:
    tp_mask = (grouping_df['Time'] == tp)
    ids_tp = grouping_df.index[tp_mask].tolist()
    if len(ids_tp) < 3 or grouping_df.loc[tp_mask, 'Group'].nunique() < 2:
        print(f"{tp}: Not enough samples or only one group — skipping.")
        continue
    # Submatrix for this timepoint
    idx = [ids.index(i) for i in ids_tp]
    tp_dist = bray_dist_matrix[np.ix_(idx, idx)]
    tp_dm = DistanceMatrix(tp_dist, ids=ids_tp)

    tp_perm = permanova(tp_dm, grouping_df.loc[ids_tp], column='Group', permutations=999)
    tp_F = float(tp_perm.get('test statistic', np.nan))
    tp_p = float(tp_perm.get('p-value', np.nan))
    timepoint_results.append({'Timepoint': tp, 'F': tp_F, 'P_value': tp_p})
    print(f"{tp}: F = {tp_F:.4f}, p = {tp_p:.4f}")

# ============================================================================
# 6b. POST-HOC PAIRWISE COMPARISONS FOR TIME EFFECT
# ============================================================================

print("\n" + "=" * 50)
print("POST-HOC PAIRWISE COMPARISONS (TIME)")
print("=" * 50 + "\n")

from itertools import combinations
from statsmodels.stats.multitest import multipletests

# Get all pairwise combinations of timepoints
pairwise_results = []
present_times_sorted = sorted(grouping_df['Time'].unique().tolist())

print(f"Performing pairwise comparisons between timepoints...\n")

for time1, time2 in combinations(present_times_sorted, 2):
    # Get samples for both timepoints
    mask = grouping_df['Time'].isin([time1, time2])
    ids_pair = grouping_df.index[mask].tolist()

    if len(ids_pair) < 4:  # Need at least 4 samples (2 per group minimum)
        print(f"{time1} vs {time2}: Not enough samples — skipping.")
        continue

    # Submatrix for this pair
    idx = [ids.index(i) for i in ids_pair]
    pair_dist = bray_dist_matrix[np.ix_(idx, idx)]
    pair_dm = DistanceMatrix(pair_dist, ids=ids_pair)

    # PERMANOVA for this pair
    pair_perm = permanova(pair_dm, grouping_df.loc[ids_pair], column='Time', permutations=999)
    pair_F = float(pair_perm.get('test statistic', np.nan))
    pair_p = float(pair_perm.get('p-value', np.nan))

    pairwise_results.append({
        'Comparison': f"{time1} vs {time2}",
        'F': pair_F,
        'P_value': pair_p
    })
    print(f"{time1} vs {time2}: F = {pair_F:.4f}, p = {pair_p:.4f}")

# FDR correction
if len(pairwise_results) > 0:
    p_values = [r['P_value'] for r in pairwise_results]
    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    print("\n" + "-" * 50)
    print("FDR-Corrected Results:")
    print("-" * 50)

    for i, result in enumerate(pairwise_results):
        result['P_adj_FDR'] = p_adj[i]
        result['Significant'] = reject[i]
        sig_marker = "***" if reject[i] else ""
        print(f"{result['Comparison']}: p = {result['P_value']:.4f}, "
              f"p_adj = {result['P_adj_FDR']:.4f} {sig_marker}")

    # Create summary dataframe
    pairwise_df = pd.DataFrame(pairwise_results)
    print("\nPairwise Comparison Table:")
    print(pairwise_df.to_string(index=False))

# ============================================================================
# 7. SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 50)
print("SUMMARY TABLE - TWO-WAY PERMANOVA")
print("=" * 50 + "\n")

summary_rows = [
    {
        'Analysis': 'Main Effect: Group',
        'Pseudo-F': group_F,
        'P_value': group_p,
        'PCoA_eta2': eta2_group
    }
]

if has_multiple_timepoints:
    summary_rows.append({
        'Analysis': 'Main Effect: Time',
        'Pseudo-F': time_F,
        'P_value': time_p,
        'PCoA_eta2': eta2_time
    })
    summary_rows.append({
        'Analysis': 'Interaction: Group × Time',
        'Pseudo-F': interaction_F,
        'P_value': interaction_p,
        'PCoA_eta2': np.nan
    })

    # Add timepoint-specific results
    for r in timepoint_results:
        summary_rows.append({
            'Analysis': f"{r['Timepoint']} only (Group effect)",
            'Pseudo-F': r['F'],
            'P_value': r['P_value'],
            'PCoA_eta2': np.nan
        })

summary_table = pd.DataFrame(summary_rows)
print(summary_table.to_string(index=False))

# Save summary table
# summary_table.to_csv(os.path.join(path_to_save, 'PERMANOVA_2way_summary_table.csv'), index=False)
print("\nSaved: PERMANOVA_2way_summary_table.csv")

# ============================================================================
# 8. CONCLUSION (PRINT-READY)
# ============================================================================

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
print("\nFiles saved:")
print("- PCoA_GDM_vs_Control.png")
if has_multiple_timepoints:
    print("- PCoA_Time_differ.png")
print("- PERMANOVA_2way_summary_table.csv")

print("\n" + "=" * 50)
print("TWO-WAY PERMANOVA INTERPRETATION")
print("=" * 50)

conclusion = f"\nTwo-way PERMANOVA Results:\n\n"
conclusion += f"1. Main Effect of Group (GDM vs Control):\n"
conclusion += f"   F = {group_F:.3f}, p = {group_p:.3f}\n"
conclusion += f"   Effect size (η²) ≈ {eta2_group * 100:.1f}% of variance\n"
conclusion += f"   Interpretation: {'Significant' if group_p < 0.05 else 'Non-significant'} difference in microbiome composition between GDM and Control groups.\n\n"

if has_multiple_timepoints:
    conclusion += f"2. Main Effect of Time (Pregnancy Trimesters):\n"
    conclusion += f"   F = {time_F:.3f}, p = {time_p:.3f}\n"
    conclusion += f"   Effect size (η²) ≈ {eta2_time * 100:.1f}% of variance\n"
    conclusion += f"   Interpretation: {'Significant' if time_p < 0.05 else 'Non-significant'} changes in microbiome composition across pregnancy timepoints.\n\n"

    conclusion += f"3. Interaction Effect (Group × Time):\n"
    conclusion += f"   F = {interaction_F:.3f}, p = {interaction_p:.3f}\n"
    conclusion += f"   Interpretation: The interaction term tests whether the effect of GDM status on microbiome composition\n"
    conclusion += f"   differs across pregnancy timepoints. A {'significant' if interaction_p < 0.05 else 'non-significant'} interaction was observed,\n"
    if interaction_p < 0.05:
        conclusion += f"   suggesting that the difference between GDM and Control groups varies across trimesters.\n"
    else:
        conclusion += f"   suggesting that the difference between GDM and Control groups is consistent across trimesters.\n"

print(conclusion)

# Additional dispersion analysis interpretation
print("\n" + "=" * 50)
print("DISPERSION ANALYSIS")
print("=" * 50)
print(f"\nPERMDISP tests whether the variance (dispersion) differs between groups:")
print(f"Group dispersion: F = {disp_F_group:.3f}, p = {disp_p_group:.3f}")
if disp_p_group < 0.05:
    print("  → Significant difference in dispersion between GDM and Control groups.")
    print("  → PERMANOVA results should be interpreted with caution.")
else:
    print("  → No significant difference in dispersion between groups.")
    print("  → PERMANOVA assumption of homogeneous dispersion is met.")

if has_multiple_timepoints:
    print(f"\nTime dispersion: F = {disp_F_time:.3f}, p = {disp_p_time:.3f}")
    if disp_p_time < 0.05:
        print("  → Significant difference in dispersion across timepoints.")
    else:
        print("  → No significant difference in dispersion across timepoints.")

print("\n" + "=" * 50)
print("END OF ANALYSIS")
print("=" * 50)