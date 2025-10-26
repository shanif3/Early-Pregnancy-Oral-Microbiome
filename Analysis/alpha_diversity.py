import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, kruskal, f_oneway
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os

cohort = 'Israel'
root_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(root_dir, 'Data', cohort)

processed1 = pd.read_csv(os.path.join(data_path, "T1.csv"), index_col=0) if cohort == 'Israel' else pd.read_csv(
    os.path.join(data_path, "T2.csv"), index_col=0)
processed2 = pd.read_csv(os.path.join(data_path, "T2.csv"), index_col=0)
processed3 = pd.read_csv(os.path.join(data_path, "T3.csv"), index_col=0)


def enhanced_statistical_comparison_three_groups(alpha_t1, alpha_t2, alpha_t3, alpha=0.05):
    """
    Perform comprehensive statistical comparison between T1, T2, and T3
    WITH FDR CORRECTION for multiple testing
    """
    metrics = ['Richness', 'Shannon']  # , 'Simpson', 'Pielou']
    results = {}

    # Store all p-values for FDR correction
    all_pairwise_pvalues = []
    all_pairwise_keys = []

    print("=" * 60)
    print("STATISTICAL COMPARISON: T1 vs T2 vs T3")
    print("WITH FDR (BENJAMINI-HOCHBERG) CORRECTION")
    print("=" * 60)

    # FIRST PASS: Compute all statistics
    for metric in metrics:
        t1_values = alpha_t1[metric].dropna()
        t2_values = alpha_t2[metric].dropna()
        t3_values = alpha_t3[metric].dropna()

        print(f"\n{metric.upper()}:")
        print("-" * 40)

        # Descriptive statistics
        print(
            f"T1: n={len(t1_values)}, mean={t1_values.mean():.4f} ± {t1_values.std():.4f}, median={t1_values.median():.4f}")
        print(
            f"T2: n={len(t2_values)}, mean={t2_values.mean():.4f} ± {t2_values.std():.4f}, median={t2_values.median():.4f}")
        print(
            f"T3: n={len(t3_values)}, mean={t3_values.mean():.4f} ± {t3_values.std():.4f}, median={t3_values.median():.4f}")

        # Test for normality
        _, p_norm_t1 = shapiro(t1_values)
        _, p_norm_t2 = shapiro(t2_values)
        _, p_norm_t3 = shapiro(t3_values)

        print(f"Normality tests - T1: p={p_norm_t1:.4f}, T2: p={p_norm_t2:.4f}, T3: p={p_norm_t3:.4f}")

        # Overall comparison (all three groups)
        if p_norm_t1 > alpha and p_norm_t2 > alpha and p_norm_t3 > alpha:
            # All normal - use ANOVA
            overall_stat, overall_p = f_oneway(t1_values, t2_values, t3_values)
            test_used = "One-way ANOVA"
        else:
            # Non-normal - use Kruskal-Wallis test
            overall_stat, overall_p = kruskal(t1_values, t2_values, t3_values)
            test_used = "Kruskal-Wallis test"

        print(f"\nOVERALL COMPARISON:")
        print(f"Test used: {test_used}")
        print(f"Test statistic: {overall_stat:.4f}")
        print(f"p-value (uncorrected): {overall_p:.4f}")
        print(f"Significant: {'YES' if overall_p < alpha else 'NO'} (α = {alpha})")

        # Pairwise comparisons
        pairs = [('T1', 'T2', t1_values, t2_values),
                 ('T1', 'T3', t1_values, t3_values),
                 ('T2', 'T3', t2_values, t3_values)]

        pairwise_results = {}

        print(f"\nPAIRWISE COMPARISONS (uncorrected p-values):")
        for group1, group2, values1, values2 in pairs:
            # Choose appropriate test based on normality
            if shapiro(values1)[1] > alpha and shapiro(values2)[1] > alpha:
                # Both normal - use t-test
                stat, p_value = stats.ttest_ind(values1, values2)
                pair_test = "Independent t-test"
            else:
                # Non-normal - use Mann-Whitney U test
                stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
                pair_test = "Mann-Whitney U test"

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(values1) - 1) * values1.var() + (len(values2) - 1) * values2.var()) /
                                 (len(values1) + len(values2) - 2))
            cohens_d = (values1.mean() - values2.mean()) / pooled_std

            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            print(
                f"{pair_test}:  {group1} vs {group2}: p={p_value:.4f}, Cohen's d={cohens_d:.4f} ({effect_interpretation})")

            # Store for FDR correction
            all_pairwise_pvalues.append(p_value)
            all_pairwise_keys.append((metric, f"{group1}_vs_{group2}"))

            pairwise_results[f"{group1}_vs_{group2}"] = {
                'test_used': pair_test,
                'statistic': stat,
                'p_value': p_value,
                'p_value_corrected': None,  # Will be filled after FDR correction
                'significant_uncorrected': p_value < alpha,
                'significant_corrected': None,  # Will be filled after FDR correction
                'cohens_d': cohens_d,
                'effect_size': effect_interpretation
            }

        # Store results
        results[metric] = {
            'overall_test': test_used,
            'overall_statistic': overall_stat,
            'overall_p_value': overall_p,
            'overall_significant': overall_p < alpha,
            'pairwise': pairwise_results,
            't1_mean': t1_values.mean(),
            't2_mean': t2_values.mean(),
            't3_mean': t3_values.mean(),
            't1_std': t1_values.std(),
            't2_std': t2_values.std(),
            't3_std': t3_values.std(),
            't1_n': len(t1_values),
            't2_n': len(t2_values),
            't3_n': len(t3_values)
        }

    # APPLY FDR CORRECTION to all pairwise comparisons
    print("\n" + "=" * 60)
    print("APPLYING FDR (BENJAMINI-HOCHBERG) CORRECTION")
    print("=" * 60)

    if len(all_pairwise_pvalues) > 0:
        rejected, p_corrected, alphacSidak, alphacBonf = multipletests(
            all_pairwise_pvalues,
            alpha=alpha,
            method='fdr_bh'
        )

        # Update results with corrected p-values
        for idx, (metric, pair_key) in enumerate(all_pairwise_keys):
            results[metric]['pairwise'][pair_key]['p_value_corrected'] = p_corrected[idx]
            results[metric]['pairwise'][pair_key]['significant_corrected'] = rejected[idx]

        # Print corrected results
        print(f"\nNumber of pairwise tests: {len(all_pairwise_pvalues)}")
        print(f"FDR threshold (alpha): {alpha}")
        print(f"\nCORRECTED RESULTS:")

        for metric in metrics:
            print(f"\n{metric.upper()} - PAIRWISE COMPARISONS (FDR-corrected):")
            pairwise = results[metric]['pairwise']

            for pair_key in ['T1_vs_T2', 'T1_vs_T3', 'T2_vs_T3']:
                if pair_key in pairwise:
                    pair_results = pairwise[pair_key]
                    print(f"  {pair_key.replace('_vs_', ' vs ')}: "
                          f"p_uncorrected={pair_results['p_value']:.4f}, "
                          f"p_corrected={pair_results['p_value_corrected']:.4f}, "
                          f"Significant (FDR): {'YES' if pair_results['significant_corrected'] else 'NO'}, "
                          f"Cohen's d={pair_results['cohens_d']:.4f}")

    return results


def create_enhanced_visualizations_three_groups(alpha_t1, alpha_t2, alpha_t3, results):
    """
    Create enhanced visualizations with statistical annotations for three groups
    UPDATED: Uses FDR-corrected p-values for significance annotations
    """
    metrics = ['Richness']  # 'Shannon', 'Simpson', 'Pielou']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        if len(metrics) == 1:
            fig, ax = plt.subplots(figsize=(6, 8))

        # Prepare data for plotting
        t1_values = alpha_t1[metric].dropna()
        t2_values = alpha_t2[metric].dropna()
        t3_values = alpha_t3[metric].dropna()

        # Create box plot with individual points
        data_to_plot = [t1_values, t2_values, t3_values]
        bp = ax.boxplot(data_to_plot, labels=['T1', 'T2', 'T3'], patch_artist=True)

        # Color boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Make median lines black and thicker
        for median in bp['medians']:
            median.set(color='black', linewidth=2)

        # Add individual points with jitter
        point_colors = ['darkblue', 'darkred', 'darkgreen']
        for j, data in enumerate(data_to_plot):
            y = data
            x = np.random.normal(j + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=30, color=point_colors[j])

        # Get y-axis range for positioning annotations
        y_max = max(max(t1_values), max(t2_values), max(t3_values))
        y_min = min(min(t1_values), min(t2_values), min(t3_values))
        y_range = y_max - y_min

        # Add overall significance annotation (PINK asterisks for Kruskal-Wallis/ANOVA)
        overall_p = results[metric]['overall_p_value']
        overall_significance = "***" if overall_p < 0.001 else "**" if overall_p < 0.01 else "*" if overall_p < 0.05 else "ns"

        if results[metric]['overall_significant']:
            # Significant overall - add significance bar across all groups
            ax.plot([1, 3], [y_max + 0.12 * y_range, y_max + 0.12 * y_range], 'k-', lw=2)
            ax.plot([1, 1], [y_max + 0.10 * y_range, y_max + 0.12 * y_range], 'k-', lw=2)
            ax.plot([3, 3], [y_max + 0.10 * y_range, y_max + 0.12 * y_range], 'k-', lw=2)
            ax.text(2, y_max + 0.12 * y_range, overall_significance, ha='center', va='bottom',
                    fontweight='bold', fontsize=14, color='hotpink')

        # Add pairwise comparisons with significance annotations (USING FDR-CORRECTED P-VALUES)
        pairwise = results[metric]['pairwise']
        pairwise_comparisons = [
            ('T1_vs_T2', 1, 2, 0.04),  # (pair_key, pos1, pos2, y_offset)
            ('T1_vs_T3', 1, 3, 0.08),
            ('T2_vs_T3', 2, 3, 0.04)
        ]

        for pair_key, pos1, pos2, y_offset in pairwise_comparisons:
            if pair_key in pairwise:
                # USE CORRECTED P-VALUE
                p_val = pairwise[pair_key]['p_value_corrected']
                test_used = pairwise[pair_key]['test_used']
                is_significant = pairwise[pair_key]['significant_corrected']

                # Determine significance symbol based on CORRECTED p-value
                if p_val < 0.001:
                    sig_symbol = "***"
                elif p_val < 0.01:
                    sig_symbol = "**"
                elif p_val < 0.05:
                    sig_symbol = "*"
                else:
                    sig_symbol = "ns"

                # Set color based on test type
                if "Mann-Whitney" in test_used or "Kruskal" in test_used:
                    line_color = 'black'
                    text_color = 'black'
                else:  # t-test or ANOVA
                    line_color = 'blue'
                    text_color = 'blue'

                # Only draw if significant (or always draw for ns if you prefer)
                if is_significant or sig_symbol == "ns":
                    y_line_pos = y_max + y_offset * y_range
                    ax.plot([pos1, pos2], [y_line_pos, y_line_pos], color=line_color, lw=1.5)
                    ax.plot([pos1, pos1], [y_line_pos - 0.01 * y_range, y_line_pos], color=line_color, lw=1.5)
                    ax.plot([pos2, pos2], [y_line_pos - 0.01 * y_range, y_line_pos], color=line_color, lw=1.5)

                    # Add text
                    fontsize = 12 if is_significant else 10
                    text_weight = 'bold' if is_significant else 'normal'
                    ax.text((pos1 + pos2) / 2, y_line_pos + 0.005 * y_range, sig_symbol,
                            ha='center', va='bottom', fontsize=fontsize, color=text_color, weight=text_weight)

        ax.set_ylabel(f'{metric} Index', fontsize=12)
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.22 * y_range)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path_to_save = os.path.join(root_dir, 'Paper_plots')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(os.path.join(path_to_save, f'alpha_diversity_three_groups_{cohort}_FDR_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY WITH FDR CORRECTION")
    print("=" * 60)

    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  Overall test: {results[metric]['overall_test']}")
        print(f"  Overall p-value: {results[metric]['overall_p_value']:.4f}")
        print(f"  Overall significant: {'YES' if results[metric]['overall_significant'] else 'NO'}")

        print(f"\n  Pairwise comparisons (FDR-corrected):")
        pairwise = results[metric]['pairwise']
        for pair_key in ['T1_vs_T2', 'T1_vs_T3', 'T2_vs_T3']:
            if pair_key in pairwise:
                pair_results = pairwise[pair_key]
                print(f"    {pair_key.replace('_vs_', ' vs ')}: "
                      f"Test={pair_results['test_used']}, "
                      f"p_uncorrected={pair_results['p_value']:.4f}, "
                      f"p_corrected={pair_results['p_value_corrected']:.4f}, "
                      f"Cohen's d={pair_results['cohens_d']:.4f} ({pair_results['effect_size']}), "
                      f"Significant (FDR): {'YES' if pair_results['significant_corrected'] else 'NO'}")

    # Count significant differences
    overall_sig_count = sum([results[m]['overall_significant'] for m in metrics])
    print(f"\nOVERALL SIGNIFICANT DIFFERENCES: {overall_sig_count} out of {len(metrics)} metrics")

    # Count pairwise significant differences (FDR-corrected)
    pairwise_sig_counts = {}
    for pair in ['T1_vs_T2', 'T1_vs_T3', 'T2_vs_T3']:
        count = sum([results[m]['pairwise'].get(pair, {}).get('significant_corrected', False) for m in metrics])
        pairwise_sig_counts[pair] = count

    print("PAIRWISE SIGNIFICANT DIFFERENCES (FDR-corrected):")
    for pair, count in pairwise_sig_counts.items():
        groups = pair.replace('_vs_', ' vs ')
        print(f"  {groups}: {count} out of {len(metrics)} metrics")

    # Create summary DataFrame
    overall_summary = []
    for metric in metrics:
        row = {
            'Metric': metric,
            'Overall_Test': results[metric]['overall_test'],
            'Overall_P': results[metric]['overall_p_value'],
            'Overall_Sig': results[metric]['overall_significant']
        }

        pairwise = results[metric]['pairwise']
        for pair_key in ['T1_vs_T2', 'T1_vs_T3', 'T2_vs_T3']:
            if pair_key in pairwise:
                row[f'{pair_key}_P_uncorrected'] = pairwise[pair_key]['p_value']
                row[f'{pair_key}_P_corrected'] = pairwise[pair_key]['p_value_corrected']
                row[f'{pair_key}_Sig_FDR'] = pairwise[pair_key]['significant_corrected']
                row[f'{pair_key}_CohenD'] = pairwise[pair_key]['cohens_d']

        overall_summary.append(row)

    overall_summary = pd.DataFrame(overall_summary)

    return overall_summary


def create_visualizations_specific_metric_three_groups1(alpha_t1, alpha_t2, alpha_t3, results, metric='Shannon'):
    """
    Create enhanced visualization for a specific metric with three groups
    UPDATED: Uses FDR-corrected p-values for significance annotations
    """
    # Create a single plot for the specified metric
    fig, ax = plt.subplots(figsize=(6, 8))

    # Prepare data for plotting
    t1_values = alpha_t1[metric].dropna()
    t2_values = alpha_t2[metric].dropna()
    t3_values = alpha_t3[metric].dropna()

    # Create box plot with individual points
    data_to_plot = [t1_values, t2_values, t3_values]
    bp = ax.boxplot(data_to_plot, labels=['T1', 'T2', 'T3'], patch_artist=True)

    # Make median lines black and thicker
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    # Color boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points with jitter
    point_colors = ['darkblue', 'darkred', 'darkgreen']
    for j, data in enumerate(data_to_plot):
        y = data
        x = np.random.normal(j + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=30, color=point_colors[j])

    # Set font sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Add statistical annotation
    overall_p = results[metric]['overall_p_value']
    overall_significance = "***" if overall_p < 0.001 else "**" if overall_p < 0.01 else "*" if overall_p < 0.05 else "ns"

    # Get y-axis range for positioning annotations
    y_max = max(max(t1_values), max(t2_values), max(t3_values))
    y_min = min(min(t1_values), min(t2_values), min(t3_values))
    y_range = y_max - y_min

    # Overall significance bar (hotpink for Kruskal-Wallis)
    if results[metric]['overall_significant']:
        ax.plot([1, 3], [y_max + 0.12 * y_range, y_max + 0.12 * y_range], 'k-', lw=2)
        ax.plot([1, 1], [y_max + 0.10 * y_range, y_max + 0.12 * y_range], 'k-', lw=2)
        ax.plot([3, 3], [y_max + 0.10 * y_range, y_max + 0.12 * y_range], 'k-', lw=2)
        ax.text(2, y_max + 0.12 * y_range, overall_significance, ha='center', va='bottom',
                fontweight='bold', fontsize=18, color='hotpink')

    # Add pairwise comparisons with BLACK asterisks (USING FDR-CORRECTED P-VALUES)
    pairwise = results[metric]['pairwise']
    pairwise_comparisons = [
        ('T1_vs_T2', 1, 2, 0.03),  # Reduced spacing for single metric plot
        ('T1_vs_T3', 1, 3, 0.07),
        ('T2_vs_T3', 2, 3, 0.03)
    ]

    for pair_key, pos1, pos2, y_offset in pairwise_comparisons:
        if pair_key in pairwise:
            # USE CORRECTED P-VALUE
            p_val = pairwise[pair_key]['p_value_corrected']
            test_used = pairwise[pair_key]['test_used']
            is_significant = pairwise[pair_key]['significant_corrected']

            # Determine significance symbol based on CORRECTED p-value
            if p_val < 0.001:
                sig_symbol = "***"
            elif p_val < 0.01:
                sig_symbol = "**"
            elif p_val < 0.05:
                sig_symbol = "*"
            else:
                sig_symbol = "ns"

            # Set color based on significance
            if is_significant:
                line_color = 'black'
                text_color = 'black'
                text_weight = 'bold'
                fontsize = 16
            else:
                continue  # Skip non-significant comparisons for cleaner plot

            # Draw significance line
            y_line_pos = y_max + y_offset * y_range
            ax.plot([pos1, pos2], [y_line_pos, y_line_pos], color=line_color, lw=2)
            ax.plot([pos1, pos1], [y_line_pos - 0.005 * y_range, y_line_pos], color=line_color, lw=2)
            ax.plot([pos2, pos2], [y_line_pos - 0.005 * y_range, y_line_pos], color=line_color, lw=2)

            # Add significance text
            if sig_symbol != 'ns':
                ax.text((pos1 + pos2) / 2, y_line_pos + 0.0001 * y_range, sig_symbol,
                        ha='center', va='bottom', fontsize=fontsize, color=text_color, weight=text_weight)

    ax.set_ylabel(f'{metric} Index', fontsize=18)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.22 * y_range)  # Increased top margin
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # check if the folder exists
    path_to_save = os.path.join(root_dir, 'Paper_plots')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(os.path.join(path_to_save, f'alpha_diversity_{metric.lower()}_three_groups_{cohort}_FDR_corrected.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


# Calculate alpha diversity for both trimesters
def calculate_alpha_diversity(df):
    """
    Calculate various alpha diversity metrics
    df: DataFrame where rows are samples and columns are microbes/taxa
    """
    results = {}

    for sample_id in df.index:  # Iterate over samples (rows)
        counts = df.loc[sample_id].values  # Get microbial counts for this sample
        counts = counts[counts > 0]  # Remove zeros (absent microbes)

        if len(counts) == 0:
            # Handle samples with no microbes
            results[sample_id] = {
                'Richness': 0,
                'Shannon': 0,
                'Simpson': 0,
                'Pielou': 0
            }
            continue

        total_counts = sum(counts)
        proportions = counts / total_counts

        # Species richness (number of observed microbes)
        richness = len(counts)

        # Shannon diversity
        shannon = -sum([p * np.log(p) for p in proportions if p > 0])

        # Simpson diversity (1 - Simpson's dominance index)
        simpson = 1 - sum([p ** 2 for p in proportions])

        # Pielou's evenness (Shannon evenness)
        pielou = shannon / np.log(richness) if richness > 1 else 0

        results[sample_id] = {
            'Richness': richness,
            'Shannon': shannon,
            'Simpson': simpson,
            'Pielou': pielou
        }

    return pd.DataFrame(results).T


# Calculate alpha diversity for all three trimesters
alpha_t1 = calculate_alpha_diversity(processed1)
alpha_t2 = calculate_alpha_diversity(processed2)
alpha_t3 = calculate_alpha_diversity(processed3)

# Add trimester labels
alpha_t1['Trimester'] = 'T1'
alpha_t2['Trimester'] = 'T2'
alpha_t3['Trimester'] = 'T3'

# Run the complete analysis
print("Computing alpha diversity metrics for T1, T2, and T3...")
print("WITH FDR (BENJAMINI-HOCHBERG) CORRECTION FOR MULTIPLE TESTING")
results = enhanced_statistical_comparison_three_groups(alpha_t1, alpha_t2, alpha_t3)
summary_df = create_enhanced_visualizations_three_groups(alpha_t1, alpha_t2, alpha_t3, results)

# Save summary to CSV
path_to_save = os.path.join(root_dir, 'Paper_plots')
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
summary_df.to_csv(os.path.join(path_to_save, f'statistical_summary_FDR_corrected_{cohort}.csv'), index=False)
print(f"\nSummary table saved to: {os.path.join(path_to_save, f'statistical_summary_FDR_corrected_{cohort}.csv')}")

# Create specific metric visualization (Shannon as example)
create_visualizations_specific_metric_three_groups1(alpha_t1, alpha_t2, alpha_t3, results, 'Shannon')