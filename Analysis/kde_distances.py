import os

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import braycurtis, jaccard


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

root_dir = os.path.dirname(os.getcwd())


# ----------------------------- P-VALUE ADJUSTMENT -----------------------------

def adjust_pvalues(pvals, method="fdr_bh"):
    """
    Adjust a list/array of p-values for multiple testing.

    Parameters
    ----------
    pvals : list or np.ndarray
        Raw p-values.
    method : {"fdr_bh", "bonferroni"}
        Multiple testing correction method.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the same order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals

    if method == "bonferroni":
        adj = np.minimum(pvals * m, 1.0)
        return adj

    if method == "fdr_bh":
        # Benjamini-Hochberg (non-negative, monotone)
        order = np.argsort(pvals)
        ranked = np.empty_like(order)
        ranked[order] = np.arange(1, m + 1)

        sorted_p = pvals[order]
        bh = sorted_p * m / np.arange(1, m + 1)
        # enforce monotonicity from the end
        bh_monotone = np.minimum.accumulate(bh[::-1])[::-1]
        # clip at 1 and map back to original order
        adj_sorted = np.minimum(bh_monotone, 1.0)
        adj = np.empty_like(adj_sorted)
        adj[order] = adj_sorted
        return adj

    raise ValueError(f"Unknown method: {method}")


def p_to_symbol(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# ----------------------------- PLOTTING UTILS --------------------------------

def add_significance_brackets1(ax, x1, x2, y, p_value, height=0.02):
    """
    Add significance brackets to the plot using the provided p_value
    (assumed already adjusted if you performed correction).

    Parameters:
    -----------
    ax : matplotlib axis
    x1, x2 : int
    y : float
    p_value : float
    height : float
    """
    sig_symbol = p_to_symbol(p_value)

    # Only draw bracket if significant after correction
    if sig_symbol != 'ns':
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=2, c='black')
        ax.text((x1 + x2) * 0.5, y + height * 1.5 - 0.05, sig_symbol,
                ha='center', va='bottom', fontsize=20, fontweight='bold')


# --------------------------- MAIN FIGURE (ALL METRICS) -----------------------

def create_combined_distance_plot_with_within(all_distance_results,
                                              plot_title="Microbiome Distance Analysis - All Metrics",
                                              p_adjust_method="fdr_bh"):
    """
    Create a combined plot showing all distance metrics in subplots including within-timepoint distances.
    Uses adjusted p-values for significance display.
    """
    if cohort == 'Israel':
        categories_order = [
            ('SAME_WOMEN_T1_T2', 'Same Women\nT1 vs T2'),
            ('SAME_WOMEN_T2_T3', 'Same Women\nT2 vs T3'),
            ('SAME_WOMEN_T1_T3', 'Same Women\nT1 vs T3'),
            ('WITHIN_T1', 'Within T1'),
            ('WITHIN_T2', 'Within T2'),
            ('WITHIN_T3', 'Within T3'),
        ]
    elif cohort == 'Russia':
        categories_order = [
            ('SAME_WOMEN_T2_T3', 'Same Women\nT2 vs T3'),
            ('WITHIN_T2', 'Within T2'),
            ('WITHIN_T3', 'Within T3'),
        ]
    else:
        raise ValueError("cohort must be 'Israel' or 'Russia'")

    # Define colors - up to 6 categories
    colors = ['#FFE5B4', '#E5D4FF', '#B4E5FF', '#9FAFFF', '#FF9F9F', '#ff810a']

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric_name, data_dict) in enumerate(reversed(all_distance_results.items())):
        ax = axes[idx]

        # Prepare data for plotting
        plot_data = []
        category_labels = []
        for key, label in categories_order:
            if key in data_dict and data_dict[key].size > 0:
                plot_data.extend(data_dict[key])
                category_labels.extend([label] * len(data_dict[key]))

        plot_df = pd.DataFrame({'Distance': plot_data, 'Category': category_labels})

        # Get unique categories in order
        unique_categories = [label for key, label in categories_order
                             if any(plot_df['Category'] == label)]

        # Create box plot
        box_parts = ax.boxplot([plot_df[plot_df['Category'] == cat]['Distance'].values
                                for cat in unique_categories],
                               positions=range(len(unique_categories)),
                               patch_artist=True,
                               showfliers=False,
                               medianprops=dict(color='black', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))

        # Color the boxes
        for patch, color in zip(box_parts['boxes'], colors[:len(unique_categories)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add individual points with jitter
        for i, cat in enumerate(unique_categories):
            y_data = plot_df[plot_df['Category'] == cat]['Distance'].values
            x_data = np.random.normal(i, 0.08, size=len(y_data))
            ax.scatter(x_data, y_data, alpha=0.8, s=20,
                       color=colors[i], edgecolors='black', linewidth=0.3)

        # Key comparisons (we'll adjust p-values across these within each metric)
        key_comparisons = [
            # Within vs across (directional)
            ('Within T1', 'Same Women\nT1 vs T2', 'greater', 'Within T1 < Same Women T1-T2'),
            ('Within T2', 'Same Women\nT1 vs T2', 'greater', 'Within T2 < Same Women T1-T2'),
            ('Within T2', 'Same Women\nT2 vs T3', 'greater', 'Within T2 < Same Women T2-T3'),
            ('Within T3', 'Same Women\nT2 vs T3', 'greater', 'Within T3 < Same Women T2-T3'),

            # Across-timepoint originals (directional)
            ('Same Women\nT1 vs T2', 'Same Women\nT2 vs T3', 'greater', 'T1-T2 > T2-T3'),
            ('Same Women\nT2 vs T3', 'Same Women\nT1 vs T2', 'less', 'T2-T3 < T1-T2'),
            ('Same Women\nT1 vs T2', 'Same Women\nT1 vs T3', 'less', 'T1-T2 < T1-T3'),
            ('Same Women\nT2 vs T3', 'Same Women\nT1 vs T3', 'less', 'T2-T3 < T1-T3'),

            # Within vs within (two-sided)
            ('Within T1', 'Within T2', 'two-sided', 'Within T1 vs Within T2'),
            ('Within T2', 'Within T3', 'two-sided', 'Within T2 vs Within T3'),
            ('Within T1', 'Within T3', 'two-sided', 'Within T1 vs Within T3'),
        ]

        # Compute p-values for available comparisons
        raw_comps, raw_pvals = [], []
        for cat1, cat2, alternative, desc in key_comparisons:
            if cat1 in unique_categories and cat2 in unique_categories:
                data1 = plot_df[plot_df['Category'] == cat1]['Distance'].values
                data2 = plot_df[plot_df['Category'] == cat2]['Distance'].values
                if len(data1) > 0 and len(data2) > 0:
                    statistic, p = stats.mannwhitneyu(data1, data2, alternative=alternative)
                    x1 = unique_categories.index(cat1)
                    x2 = unique_categories.index(cat2)
                    raw_comps.append({'x1': x1, 'x2': x2, 'p_raw': p, 'desc': desc})
                    raw_pvals.append(p)

        # Adjust p-values and attach
        if raw_pvals:
            p_adj = adjust_pvalues(raw_pvals, method=p_adjust_method)
            for comp, p_corr in zip(raw_comps, p_adj):
                comp['p_adj'] = float(p_corr)

        # Use adjusted p-values for significance
        y_max = plot_df['Distance'].max() if len(plot_df) else 1.0
        y_min = plot_df['Distance'].min() if len(plot_df) else 0.0
        y_range = max(y_max - y_min, 1e-6)
        bracket_height = y_range * 0.04

        significant_results = [c for c in raw_comps if c.get('p_adj', 1.0) < 0.05]
        significant_results.sort(key=lambda x: x['p_adj'])

        # Add brackets with adjusted p-values
        for level, comp in enumerate(significant_results):
            x1, x2 = comp['x1'], comp['x2']
            y_pos = y_max + bracket_height * 2 + (bracket_height * 1.5 * level)
            add_significance_brackets1(ax, x1, x2, y_pos, comp['p_adj'], bracket_height)

        # Customize subplot
        ax.set_xticks(range(len(unique_categories)))
        ax.set_xticklabels(unique_categories, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel(f'{metric_name} Distance', fontsize=12)
        ax.set_title(f'{metric_name} Distance', fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)

        # Adjust y-axis limits to accommodate brackets
        max_bracket_level = len(significant_results)
        y_top = y_max + bracket_height * 2 + (bracket_height * 1.5 * max_bracket_level) + bracket_height
        ax.set_ylim(y_min - y_range * 0.05, y_top)

        # Customize grid
        ax.yaxis.grid(True, linestyle='-', alpha=0.2)
        ax.xaxis.grid(False)

    axes = axes.flatten()
    if cohort == 'Israel':
        fig.delaxes(axes[-1])  # deletes the last subplot (bottom-right)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    path_to_save = os.path.join(root_dir, 'Appendix_plots')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(os.path.join(path_to_save, f'boxplots_microbiome_all_distance_metrics_{cohort}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------- SINGLE-METRIC FIGURE -----------------------------

def create_clean_boxplot_with_within_timepoints1(data_dict, plot_title="Microbiome Distance Analysis",
                                                 distance_metric="Cosine",
                                                 p_adjust_method="fdr_bh"):
    """
    Create clean box plots with individual points and significance annotations including within-timepoint distances.
    Uses adjusted p-values (default FDR-BH) for reporting and brackets.
    """
    print("\n" + "=" * 100)
    print(f"MEDIAN VALUES ({distance_metric} Distance)")
    print("=" * 100)

    if cohort == 'Israel':
        categories_order = [
            ('SAME_WOMEN_T1_T2', 'Same Women\nT1 vs T2'),
            ('SAME_WOMEN_T2_T3', 'Same Women\nT2 vs T3'),
            ('SAME_WOMEN_T1_T3', 'Same Women\nT1 vs T3'),
            ('WITHIN_T1', 'Within T1'),
            ('WITHIN_T2', 'Within T2'),
            ('WITHIN_T3', 'Within T3'),
        ]
    elif cohort == 'Russia':
        categories_order = [
            ('SAME_WOMEN_T2_T3', 'Same Women\nT2 vs T3'),
            ('WITHIN_T2', 'Within T2'),
            ('WITHIN_T3', 'Within T3'),
        ]
    else:
        raise ValueError("cohort must be 'Israel' or 'Russia'")

    for key, label in categories_order:
        if key in data_dict and data_dict[key].size > 0:
            median_val = np.median(data_dict[key])
            n_samples = len(data_dict[key])
            print(f"{label.replace(chr(10), ' '):<25}: {median_val:.4f} (n={n_samples})")

    # Prepare data
    plot_data, category_labels = [], []
    for key, label in categories_order:
        if key in data_dict and data_dict[key].size > 0:
            plot_data.extend(data_dict[key])
            category_labels.extend([label] * len(data_dict[key]))

    plot_df = pd.DataFrame({'Distance': plot_data, 'Category': category_labels})

    # Colors
    colors = ['#FFE5B4', '#E5D4FF', '#B4E5FF', '#9FAFFF', '#FF9F9F', '#ff810a']

    # Ordered categories present
    unique_categories = [label for key, label in categories_order
                         if any(plot_df['Category'] == label)]

    # Statistical tests (Mann-Whitney U) with correction
    print("\n" + "=" * 100)
    print(f"STATISTICAL TESTS (Mann-Whitney U) - {distance_metric} Distance")
    print(f"Multiple-testing correction: {p_adjust_method}")
    print("=" * 100)

    key_comparisons = [
        # Within vs across (directional)
        ('Within T1', 'Same Women\nT1 vs T2', 'greater', 'Within T1 < Same Women T1-T2'),
        ('Within T2', 'Same Women\nT1 vs T2', 'greater', 'Within T2 < Same Women T1-T2'),
        ('Within T2', 'Same Women\nT2 vs T3', 'greater', 'Within T2 < Same Women T2-T3'),
        ('Within T3', 'Same Women\nT2 vs T3', 'greater', 'Within T3 < Same Women T2-T3'),

        # Across-timepoint originals (directional)
        ('Same Women\nT1 vs T2', 'Same Women\nT2 vs T3', 'greater', 'T1-T2 > T2-T3'),
        ('Same Women\nT2 vs T3', 'Same Women\nT1 vs T2', 'less', 'T2-T3 < T1-T2'),
        ('Same Women\nT1 vs T2', 'Same Women\nT1 vs T3', 'less', 'T1-T2 < T1-T3'),
        ('Same Women\nT2 vs T3', 'Same Women\nT1 vs T3', 'less', 'T2-T3 < T1-T3'),

        # Within vs within (two-sided)
        ('Within T1', 'Within T2', 'two-sided', 'Within T1 vs Within T2'),
        ('Within T2', 'Within T3', 'two-sided', 'Within T2 vs Within T3'),
        ('Within T1', 'Within T3', 'two-sided', 'Within T1 vs Within T3'),
    ]

    raw_comps, raw_pvals = [], []
    for cat1, cat2, alternative, description in key_comparisons:
        if cat1 in unique_categories and cat2 in unique_categories:
            data1 = plot_df[plot_df['Category'] == cat1]['Distance'].values
            data2 = plot_df[plot_df['Category'] == cat2]['Distance'].values
            if len(data1) > 0 and len(data2) > 0:
                statistic, p = stats.mannwhitneyu(data1, data2, alternative=alternative)
                comp_name = f"{cat1.replace(chr(10), ' ')} vs {cat2.replace(chr(10), ' ')}"
                raw_comps.append({
                    'name': comp_name,
                    'alt': alternative,
                    'desc': description,
                    'stat': float(statistic),
                    'p_raw': float(p),
                    'cat1': cat1,
                    'cat2': cat2
                })
                raw_pvals.append(p)

    # Adjust p-values
    if raw_pvals:
        p_adj = adjust_pvalues(raw_pvals, method=p_adjust_method)
        for comp, p_corr in zip(raw_comps, p_adj):
            comp['p_adj'] = float(p_corr)
            # Print detailed line
            print(f"\n{comp['name']} ({comp['desc']}):")
            print(f"  Alternative hypothesis: {comp['alt']}")
            print(f"  U-statistic: {comp['stat']:.2f}")
            print(f"  P-value (raw): {comp['p_raw']:.6f}")
            print(f"  P-value (adj, {p_adjust_method}): {comp['p_adj']:.6f}")
            print(f"  Significance (adj): {p_to_symbol(comp['p_adj'])}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 9))
    box_parts = ax.boxplot([plot_df[plot_df['Category'] == cat]['Distance'].values
                            for cat in unique_categories],
                           positions=range(len(unique_categories)),
                           patch_artist=True,
                           showfliers=False,
                           medianprops=dict(color='black', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))

    for patch, color in zip(box_parts['boxes'], colors[:len(unique_categories)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, cat in enumerate(unique_categories):
        y_data = plot_df[plot_df['Category'] == cat]['Distance'].values
        x_data = np.random.normal(i, 0.08, size=len(y_data))
        ax.scatter(x_data, y_data, alpha=0.6, s=25,
                   color=colors[i], edgecolors='black', linewidth=0.3)

    # Brackets using adjusted p-values
    y_max = plot_df['Distance'].max() if len(plot_df) else 1.0
    y_min = plot_df['Distance'].min() if len(plot_df) else 0.0
    y_range = max(y_max - y_min, 1e-6)
    bracket_height = y_range * 0.03

    # Compute spans & sort (shorter span first, then p_adj)
    comparison_results = []
    for comp in raw_comps:
        x1 = unique_categories.index(comp['cat1'])
        x2 = unique_categories.index(comp['cat2'])
        span = abs(x2 - x1)
        comparison_results.append({
            'x1': x1,
            'x2': x2,
            'p_adj': comp.get('p_adj', 1.0),
            'span': span
        })

    significant_comps = [c for c in comparison_results if c['p_adj'] < 0.05]
    significant_comps.sort(key=lambda x: (x['span'], x['p_adj']))

    bracket_positions = []
    for comp in significant_comps:
        x1, x2 = sorted([comp['x1'], comp['x2']])
        level = 0
        while True:
            y_pos = y_max + bracket_height * (level * 2.5 + 1)
            conflict = False
            for existing_x1, existing_x2, existing_level in bracket_positions:
                if level == existing_level:
                    if not (x2 < existing_x1 - 0.3 or x1 > existing_x2 + 0.3):
                        conflict = True
                        break
            if not conflict or level > 10:
                break
            level += 1

        bracket_positions.append((x1, x2, level))
        add_significance_brackets1(ax, x1, x2, y_pos, comp['p_adj'], bracket_height)

    ax.set_xticks(range(len(unique_categories)))
    ax.set_xticklabels(unique_categories, fontsize=18, rotation=45, ha='right')
    ax.set_ylabel(f'{distance_metric} Distance', fontsize=23)
    ax.tick_params(axis='y', labelsize=16)
    max_level = max([pos[2] for pos in bracket_positions]) if bracket_positions else -1
    y_top = y_max + bracket_height * (max_level * 2.5 + 4)
    ax.set_ylim(y_min - y_range * 0.05, y_top)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    plt.tight_layout()

    # Save
    if distance_metric != 'Cosine':
        metric_name = distance_metric.lower().replace(' ', '_').replace('-', '_')
        path_to_save = os.path.join(root_dir, 'Appendix_plots')
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(os.path.join(path_to_save,
                                 f"boxplots_microbiome_{metric_name}_distance_scenarios_{cohort}.png"),
                    dpi=300, bbox_inches='tight')
    else:
        metric_name = distance_metric.lower().replace(' ', '_').replace('-', '_')
        path_to_save = os.path.join(root_dir, 'Paper_plots')
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        # Example if you want to save Cosine too:
        # plt.savefig(os.path.join(path_to_save,
        #                          f"boxplots_microbiome_{metric_name}_distance_scenarios_{cohort}.png"),
        #             dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 100)


# ------------------------------- ANALYSIS CORE --------------------------------

def calculate_within_timepoint_distances(data, metric_type):
    """
    Calculate all pairwise distances within a single timepoint.
    """
    distances = []
    n_samples = data.shape[0]
    if n_samples < 2:
        return distances

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric_type in ['cosine', 'euclidean']:
                dist = pairwise_distances(
                    data.iloc[[i]].values,
                    data.iloc[[j]].values,
                    metric=metric_type
                )[0, 0]
            elif metric_type == 'custom_braycurtis':
                dist = braycurtis(data.iloc[i].values, data.iloc[j].values)
            elif metric_type == 'custom_jaccard':
                x_bin = (data.iloc[i].values > 0).astype(int)
                y_bin = (data.iloc[j].values > 0).astype(int)
                dist = jaccard(x_bin, y_bin)
            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")
            distances.append(dist)

    return distances


def analyze_microbiome_distances_with_within_timepoints(
        T1: pd.DataFrame,
        T2: pd.DataFrame,
        T3: pd.DataFrame,
        project_name: str = 'Microbiome'
):
    """
    Analyze microbiome distances using multiple distance metrics including within-timepoint distances.
    """
    print(f"\nAnalyzing microbiome distances with within-timepoint comparisons for project: {project_name}")
    print("=" * 100)

    # Harmonize feature space
    mutual_cols = T1.columns.intersection(T2.columns).intersection(T3.columns)
    T1_rel = T1.loc[:, mutual_cols].copy()
    T2_rel = T2.loc[:, mutual_cols].copy()
    T3_rel = T3.loc[:, mutual_cols].copy()

    print(f"Number of samples at T1: {T1_rel.shape[0]}")
    print(f"Number of samples at T2: {T2_rel.shape[0]}")
    print(f"Number of samples at T3: {T3_rel.shape[0]}")
    print(f"Number of common features: {len(mutual_cols)}")

    # Mutual samples
    mutual_t1_t2 = T1_rel.index.intersection(T2_rel.index)
    mutual_t1_t3 = T1_rel.index.intersection(T3_rel.index)
    mutual_t2_t3 = T2_rel.index.intersection(T3_rel.index)

    print(f"\nMutual samples:")
    print(f"T1 ∩ T2: {len(mutual_t1_t2)}")
    print(f"T1 ∩ T3: {len(mutual_t1_t3)}")
    print(f"T2 ∩ T3: {len(mutual_t2_t3)}")

    # Distance metrics
    distance_metrics = {
        'Cosine': 'cosine',
        'Euclidean': 'euclidean',
        'Bray-Curtis': 'custom_braycurtis',
        'Jaccard': 'custom_jaccard'
    }

    all_results = {}

    for metric_name, metric_type in distance_metrics.items():
        print(f"\nCalculating {metric_name} distances...")

        # Within-timepoint distances
        within_t1 = calculate_within_timepoint_distances(T1_rel, metric_type)
        within_t2 = calculate_within_timepoint_distances(T2_rel, metric_type)
        within_t3 = calculate_within_timepoint_distances(T3_rel, metric_type)

        # Same women across timepoints
        same_women_t1_t2 = []
        same_women_t1_t3 = []
        same_women_t2_t3 = []

        # T1 vs T2
        if len(mutual_t1_t2) > 0:
            for sample in mutual_t1_t2:
                if metric_type in ['cosine', 'euclidean']:
                    dist = pairwise_distances(
                        T1_rel.loc[[sample]].values,
                        T2_rel.loc[[sample]].values,
                        metric=metric_type
                    )[0, 0]
                elif metric_type == 'custom_braycurtis':
                    dist = braycurtis(T1_rel.loc[sample].values, T2_rel.loc[sample].values)
                elif metric_type == 'custom_jaccard':
                    x_bin = (T1_rel.loc[sample].values > 0).astype(int)
                    y_bin = (T2_rel.loc[sample].values > 0).astype(int)
                    dist = jaccard(x_bin, y_bin)
                same_women_t1_t2.append(dist)

        # T1 vs T3
        if len(mutual_t1_t3) > 0:
            for sample in mutual_t1_t3:
                if metric_type in ['cosine', 'euclidean']:
                    dist = pairwise_distances(
                        T1_rel.loc[[sample]].values,
                        T3_rel.loc[[sample]].values,
                        metric=metric_type
                    )[0, 0]
                elif metric_type == 'custom_braycurtis':
                    dist = braycurtis(T1_rel.loc[sample].values, T3_rel.loc[sample].values)
                elif metric_type == 'custom_jaccard':
                    x_bin = (T1_rel.loc[sample].values > 0).astype(int)
                    y_bin = (T3_rel.loc[sample].values > 0).astype(int)
                    dist = jaccard(x_bin, y_bin)
                same_women_t1_t3.append(dist)

        # T2 vs T3
        if len(mutual_t2_t3) > 0:
            for sample in mutual_t2_t3:
                if metric_type in ['cosine', 'euclidean']:
                    dist = pairwise_distances(
                        T2_rel.loc[[sample]].values,
                        T3_rel.loc[[sample]].values,
                        metric=metric_type
                    )[0, 0]
                elif metric_type == 'custom_braycurtis':
                    dist = braycurtis(T2_rel.loc[sample].values, T3_rel.loc[sample].values)
                elif metric_type == 'custom_jaccard':
                    x_bin = (T2_rel.loc[sample].values > 0).astype(int)
                    y_bin = (T3_rel.loc[sample].values > 0).astype(int)
                    dist = jaccard(x_bin, y_bin)
                same_women_t2_t3.append(dist)

        # Store results
        if cohort == 'Israel':
            data_dict = {
                'WITHIN_T1': np.array(within_t1),
                'WITHIN_T2': np.array(within_t2),
                'WITHIN_T3': np.array(within_t3),
                'SAME_WOMEN_T1_T2': np.array(same_women_t1_t2),
                'SAME_WOMEN_T2_T3': np.array(same_women_t2_t3),
                'SAME_WOMEN_T1_T3': np.array(same_women_t1_t3)
            }
        elif cohort == 'Russia':
            data_dict = {
                'WITHIN_T2': np.array(within_t2),
                'WITHIN_T3': np.array(within_t3),
                'SAME_WOMEN_T2_T3': np.array(same_women_t2_t3),
            }
        else:
            raise ValueError("cohort must be 'Israel' or 'Russia'")

        all_results[metric_name] = data_dict

        # Individual plot for this metric (with adjusted p-values)
        create_clean_boxplot_with_within_timepoints1(
            data_dict,
            f"{project_name} {metric_name} Distance Analysis",
            metric_name,
            p_adjust_method="fdr_bh"  # change to "bonferroni" if you prefer
        )

    # Combined plot across metrics (with adjusted p-values per metric panel)
    create_combined_distance_plot_with_within(
        all_results,
        f"{project_name} Distance Analysis - All Metrics",
        p_adjust_method="fdr_bh"  # change to "bonferroni" if you prefer
    )

    return all_results


# ------------------------------- MAIN (EXAMPLE) -------------------------------

# Updated main execution
if __name__ == "__main__":
    # Load data
    cohort = 'Israel'
    root_dir = os.path.dirname(os.getcwd())
    data_path = os.path.join(root_dir, 'Data', cohort)
    T1 = pd.read_csv(os.path.join(root_dir, 'Data', cohort, 'T1.csv'), index_col=0) if cohort == 'Israel' else pd.read_csv(os.path.join(root_dir, 'Data', cohort, 'T2.csv'), index_col=0)
    T2 = pd.read_csv(os.path.join(root_dir, 'Data', cohort, 'T2.csv'), index_col=0)
    T3 = pd.read_csv(os.path.join(root_dir, 'Data', cohort, 'T3.csv'), index_col=0)

    # Ensure same microbes across all timepoints
    mutual_cols = T1.columns.intersection(T2.columns).intersection(T3.columns)
    T1 = T1.loc[:, mutual_cols]
    T2 = T2.loc[:, mutual_cols]
    T3 = T3.loc[:, mutual_cols]

    # Apply transformations
    T1 = T1.map(lambda x: x + 0.1).apply(np.log10)
    T2 = T2.map(lambda x: x + 0.1).apply(np.log10)
    T3 = T3.map(lambda x: x + 0.1).apply(np.log10)

    T1 = (T1 - T1.mean()) / T1.std()
    T2 = (T2 - T2.mean()) / T2.std()
    T3 = (T3 - T3.mean()) / T3.std()

    # Run analysis with multiple distance metrics including within-timepoint distances
    all_results = analyze_microbiome_distances_with_within_timepoints(
        T1, T2, T3,
        project_name='Microbiome_T1_T2_T3'
    )
