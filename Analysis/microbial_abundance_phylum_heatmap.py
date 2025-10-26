import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
import MIPMLP
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os
warnings.filterwarnings('ignore')


def load_and_process_data(cohort,t1_file, t2_file, t3_file, abundance_threshold=1.0):
    """
    Load and process microbiome data from three CSV files

    Parameters:
    t1_file: path to T1 group CSV file
    t2_file: path to T2 group CSV file
    t3_file: path to T3 group CSV file
    abundance_threshold: minimum relative abundance threshold (default 1%)

    Expected CSV format:
    - First column: Sample IDs
    - Remaining columns: Taxa names with abundance values
    """
    # Load data
    t1_data = pd.read_csv(t1_file, index_col=0)
    t2_data = pd.read_csv(t2_file, index_col=0)
    t3_data = pd.read_csv(t3_file, index_col=0)

    t1_data.loc['taxonomy'] = t1_data.columns
    t2_data.loc['taxonomy'] = t2_data.columns
    t3_data.loc['taxonomy'] = t3_data.columns

    t1_data = t1_data.reset_index()
    t2_data = t2_data.reset_index()
    t3_data = t3_data.reset_index()

    # add ID column
    t1_data = t1_data.rename(columns={t1_data.columns[0]: 'ID'})
    t2_data = t2_data.rename(columns={t2_data.columns[0]: 'ID'})
    t3_data = t3_data.rename(columns={t3_data.columns[0]: 'ID'})

    t1_data = MIPMLP.preprocess(t1_data, taxnomy_group='mean', normalization='none', taxonomy_level=2)
    t2_data = MIPMLP.preprocess(t2_data, taxnomy_group='mean', normalization='none', taxonomy_level=2)
    t3_data = MIPMLP.preprocess(t3_data, taxnomy_group='mean', normalization='none', taxonomy_level=2)

    t1_data.index = [t1_data.index.map(lambda x: f"T1_{x}")]  # Prefix T1 to sample IDs
    t2_data.index = [t2_data.index.map(lambda x: f"T2_{x}")]  # Prefix T2 to sample IDs
    t3_data.index = [t3_data.index.map(lambda x: f"T3_{x}")]  # Prefix T3 to sample IDs

    # Combine data and add group labels
    t1_data['Group'] = 'T1'
    t2_data['Group'] = 'T2'
    t3_data['Group'] = 'T3'
    if cohort=='Israel':
        combined_data = pd.concat([t1_data, t2_data, t3_data])
    else:
        combined_data = pd.concat([t2_data, t3_data])


    # Separate group column
    groups = combined_data['Group']
    abundance_data = combined_data.drop('Group', axis=1)

    # Convert to relative abundance (percentages)
    abundance_data = abundance_data.div(abundance_data.sum(axis=1), axis=0) * 100

    # Filter taxa by abundance threshold
    mean_abundance = abundance_data.mean(axis=0)
    high_abundance_taxa = mean_abundance[mean_abundance >= 0].index

    # Keep high abundance taxa and group remaining as "No_Rank"
    filtered_data = abundance_data[high_abundance_taxa].copy()

    # make sure each sample sum to 100
    filtered_data = filtered_data.div(filtered_data.sum(axis=1), axis=0) * 100

    return filtered_data, groups


def perform_statistical_tests(abundance_data, groups, group_names=['T1', 'T2', 'T3'],
                              correction_method='fdr_bh', alpha=0.05):
    """
    Perform Mann-Whitney U tests between consecutive trimesters for all taxa
    with separate multiple comparison correction for each comparison

    Parameters:
    abundance_data: DataFrame with taxa abundance data
    groups: Series with group labels
    group_names: list of group names to compare
    correction_method: method for multiple comparison correction
                      ('bonferroni', 'fdr_bh', 'fdr_by', etc.)
    alpha: significance level

    Returns:
    results_df: DataFrame with test results for all taxa and group comparisons
    """
    # Get taxa with taxonomy level 2 (containing exactly one semicolon)
    taxa_list = [taxa for taxa in abundance_data.columns if len(taxa.split(';')) == 2]
    abundance_data= np.log10(abundance_data + 0.1)
    # Define consecutive comparisons
    consecutive_pairs = [('T1', 'T2'), ('T2', 'T3')]

    results = []

    print(f"\nPerforming statistical tests for {len(taxa_list)} taxa across consecutive trimesters...")
    print(f"Comparisons: {consecutive_pairs}")
    print(f"Each comparison will have separate multiple comparison correction")

    # Process each consecutive comparison separately
    for group1, group2 in consecutive_pairs:
        print(f"\nProcessing {group1} vs {group2}...")

        comparison_results = []
        comparison_pvalues = []

        # Collect all p-values for this comparison
        for taxa in taxa_list:
            # Get data for each group
            group1_data = abundance_data.loc[groups == group1, taxa]
            group2_data = abundance_data.loc[groups == group2, taxa]

            # Perform Mann-Whitney U test (two-sided)
            try:
                stat, pval = mannwhitneyu(group1_data, group2_data, alternative='two-sided')

                # Calculate medians and effect size (rank-biserial correlation)
                median1 = group1_data.median()
                median2 = group2_data.median()
                n1, n2 = len(group1_data), len(group2_data)

                # Effect size (rank-biserial correlation approximation)
                effect_size = 1 - (2 * stat) / (n1 * n2)

                result_dict = {
                    'Taxa': taxa,
                    'Group1': group1,
                    'Group2': group2,
                    'Comparison': f'{group1}_vs_{group2}',
                    'Median_Group1': median1,
                    'Median_Group2': median2,
                    'Difference': median2 - median1,
                    'Mean_Group1': group1_data.mean(),
                    'Mean_Group2': group2_data.mean(),
                    'Difference_mean': group2_data.mean() - group1_data.mean(),
                    'U_statistic': stat,
                    'p_value': pval,
                    'Effect_size': effect_size,
                    'n_Group1': n1,
                    'n_Group2': n2
                }

                comparison_results.append(result_dict)
                comparison_pvalues.append(pval)

            except Exception as e:
                print(f"Error testing {taxa} between {group1} and {group2}: {e}")
                result_dict = {
                    'Taxa': taxa,
                    'Group1': group1,
                    'Group2': group2,
                    'Comparison': f'{group1}_vs_{group2}',
                    'Median_Group1': np.nan,
                    'Median_Group2': np.nan,
                    'Difference': np.nan,
                    'U_statistic': np.nan,
                    'p_value': np.nan,
                    'Effect_size': np.nan,
                    'n_Group1': len(abundance_data.loc[groups == group1, taxa]),
                    'n_Group2': len(abundance_data.loc[groups == group2, taxa])
                }
                comparison_results.append(result_dict)
                comparison_pvalues.append(np.nan)

        # Apply multiple comparison correction for this specific comparison
        valid_pvals = [p for p in comparison_pvalues if not np.isnan(p)]
        if valid_pvals:
            rejected, corrected_pvals, _, _ = multipletests(valid_pvals,
                                                            alpha=alpha,
                                                            method=correction_method)

            # Map corrected p-values back to results
            valid_idx = 0
            for i, result_dict in enumerate(comparison_results):
                if not np.isnan(comparison_pvalues[i]):
                    result_dict['p_corrected'] = corrected_pvals[valid_idx]
                    result_dict['significant'] = rejected[valid_idx]
                    valid_idx += 1
                else:
                    result_dict['p_corrected'] = np.nan
                    result_dict['significant'] = False
        else:
            for result_dict in comparison_results:
                result_dict['p_corrected'] = np.nan
                result_dict['significant'] = False

        # Add results for this comparison
        results.extend(comparison_results)

        # Print summary for this comparison
        sig_count = sum([r['significant'] for r in comparison_results])
        print(f"  {group1} vs {group2}: {sig_count}/{len(comparison_results)} significant after correction")

    # Create final results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by comparison, then by significance and effect size
    results_df = results_df.sort_values(['Comparison', 'significant', 'p_corrected'],
                                        ascending=[True, False, True])

    total_significant = results_df['significant'].sum()
    print(f"\nStatistical testing completed:")
    print(f"- Total tests performed: {len(results_df)}")
    print(f"- Multiple comparison correction: {correction_method} (applied separately to each comparison)")
    print(f"- Total significant results (α = {alpha}): {total_significant}")
    # results_df.to_csv('phylum_all_statistical_results.csv', index=False)
    return results_df


def print_statistical_summary(results_df, alpha=0.05):
    """Print a summary of statistical test results"""

    print("\n" + "=" * 80)
    print("STATISTICAL TEST SUMMARY - CONSECUTIVE TRIMESTERS")
    print("=" * 80)

    # Overall summary
    total_tests = len(results_df)
    significant_tests = results_df['significant'].sum()

    print(f"Total statistical tests performed: {total_tests}")
    print(f"Significant results after correction: {significant_tests} ({significant_tests / total_tests * 100:.1f}%)")
    print(f"Significance threshold (α): {alpha}")
    print(f"NOTE: Multiple comparison correction applied separately to each consecutive comparison")

    # Summary by comparison with detailed breakdown
    print(f"\nDetailed results by consecutive comparison:")
    print("-" * 80)
    for comparison in ['T1_vs_T2', 'T2_vs_T3']:
        comp_data = results_df[results_df['Comparison'] == comparison]
        sig_count = comp_data['significant'].sum()
        total_count = len(comp_data)

        print(f"\n{comparison} ({total_count} taxa tested):")
        print(f"  Significant after correction: {sig_count} ({sig_count / total_count * 100:.1f}%)")

        # Show significant taxa for this comparison
        sig_taxa = comp_data[comp_data['significant']].sort_values('p_corrected')
        if len(sig_taxa) > 0:
            print(f"  Significant taxa:")
            for _, row in sig_taxa.iterrows():
                direction = "↑" if row['Difference'] > 0 else "↓" if row['Difference'] < 0 else "="
                print(f"    {row['Taxa']:<30} p_corrected: {row['p_corrected']:.2e} "
                      f"({row['Median_Group1']:.2f} → {row['Median_Group2']:.2f} {direction})")
        else:
            print(f"    No significant taxa found")

    # Taxa of interest analysis
    taxa_of_interest = ['Bacteria;Verrucomicrobia', 'Bacteria;Synergistetes', 'Archaea;Euryarchaeota','Bacteria;Cyanobacteriota']

    print(f"\n" + "=" * 80)
    print("TAXA OF INTEREST - DETAILED ANALYSIS")
    print("=" * 80)

    for taxa in taxa_of_interest:
        taxa_results = results_df[results_df['Taxa'] == taxa]
        if len(taxa_results) > 0:
            print(f"\n{taxa}:")
            print("-" * 60)
            for _, row in taxa_results.iterrows():
                sig_marker = " ***SIGNIFICANT***" if row['significant'] else ""
                direction = "increase" if row['Difference'] > 0 else "decrease" if row[
                                                                                       'Difference'] < 0 else "no change"

                print(f"  {row['Comparison']}:")
                print(f"    Raw p-value: {row['p_value']:.2e}")
                print(f"    Corrected p-value: {row['p_corrected']:.2e}{sig_marker}")
                print(f"    Medians: {row['Median_Group1']:.3f} → {row['Median_Group2']:.3f}")
                print(f"    Change: {row['Difference']:.3f}% ({direction})")
                print(f"    Effect size: {row['Effect_size']:.3f}")
                print()

    # Summary of trends across pregnancy
    print("=" * 80)
    print("PREGNANCY PROGRESSION SUMMARY")
    print("=" * 80)

    # Find taxa that show significant changes in either comparison
    all_significant_taxa = set(results_df[results_df['significant']]['Taxa'].unique())

    if all_significant_taxa:
        print(f"\nTaxa showing significant changes during pregnancy ({len(all_significant_taxa)} total):")
        print("-" * 60)

        for taxa in sorted(all_significant_taxa):
            taxa_data = results_df[results_df['Taxa'] == taxa]

            # Get T1->T2 and T2->T3 results
            t1_t2 = taxa_data[taxa_data['Comparison'] == 'T1_vs_T2'].iloc[0] if len(
                taxa_data[taxa_data['Comparison'] == 'T1_vs_T2']) > 0 else None
            t2_t3 = taxa_data[taxa_data['Comparison'] == 'T2_vs_T3'].iloc[0] if len(
                taxa_data[taxa_data['Comparison'] == 'T2_vs_T3']) > 0 else None

            print(f"\n{taxa}:")

            if t1_t2 is not None:
                sig1 = "***" if t1_t2['significant'] else ""
                trend1 = "↑" if t1_t2['Difference'] > 0 else "↓" if t1_t2['Difference'] < 0 else "="
                print(
                    f"  T1→T2: {t1_t2['Median_Group1']:.6f} → {t1_t2['Median_Group2']:.6f} {trend1} (p={t1_t2['p_corrected']:.2e}) {sig1}")

            if t2_t3 is not None:
                sig2 = "***" if t2_t3['significant'] else ""
                trend2 = "↑" if t2_t3['Difference'] > 0 else "↓" if t2_t3['Difference'] < 0 else "="
                print(
                    f"  T2→T3: {t2_t3['Median_Group1']:.6f} → {t2_t3['Median_Group2']:.6f} {trend2} (p={t2_t3['p_corrected']:.2e}) {sig2}")
    else:
        print("\nNo taxa showed significant changes after multiple comparison correction.")
        print("Consider examining raw p-values or using less stringent correction methods.")

    print("\n" + "=" * 80)


def save_statistical_results(results_df, save_path='statistical_results.csv'):
    """Save statistical results to CSV file"""
    # results_df.to_csv(save_path, index=False)
    print(f"\nStatistical results saved to: {save_path}")


def create_color_palette(taxa_list):
    """Create a color palette for taxa"""
    color_map = {
        'Archaea;Euryarchaeota': '#535473',
        'Bacteria;Actinobacteria': '#BE95DC',
        'Bacteria;Bacteroidetes': '#EFA2D2',
        'Bacteria;Chloroflexi': '#FFCBE1',
        'Bacteria;Cyanobacteria': '#F9C6AB',
        'Bacteria;Deferribacteres': '#faa007',
        'Bacteria;Firmicutes': '#C8E6C9',
        'Bacteria;Fusobacteria': '#A8D8EA',
        'Bacteria;Proteobacteria': '#90EE90',
        'Bacteria;SR1': '#8b7da8',
        'Bacteria;Spirochaetes': '#db9ec1',
        'Bacteria;Synergistetes': '#fc5d83',
        'Bacteria;TM7': '#EFA2D2',
        'Bacteria;Tenericutes': '#BE95DC',
        'Bacteria;Verrucomicrobia': '#6B8AD7'
    }
    return color_map


def sort_samples_by_verrucomicrobia(abundance_data, groups, group_names=['T1', 'T2', 'T3']):
    """
    Sort samples within each group by Bacteria;Verrucomicrobia abundance (high to low)

    Returns:
    sorted_indices: list of sample indices sorted by group and Verrucomicrobia abundance
    """
    target_taxa = 'Bacteria;Verrucomicrobia'

    # Check if the target taxa exists in the data
    if target_taxa not in abundance_data.columns:
        print(f"Warning: '{target_taxa}' not found in data. Available taxa:")
        print(abundance_data.columns.tolist())
        # Fallback to original sorting if target taxa not found
        sorted_indices = []
        for group_name in group_names:
            group_indices = groups[groups == group_name].index
            sorted_indices.extend(list(group_indices))
        return sorted_indices

    sorted_indices = []

    # Sort each group by Verrucomicrobia (high to low)
    for group_name in group_names:
        group_indices = groups[groups == group_name].index
        group_verrucomicrobia = abundance_data.loc[group_indices, target_taxa]
        group_sorted = group_verrucomicrobia.sort_values(ascending=False).index
        sorted_indices.extend(group_sorted)
        print(
            f"{group_name} group - Verrucomicrobia range: {group_verrucomicrobia.min():.2f}% to {group_verrucomicrobia.max():.2f}%")

    print(f"Samples sorted by '{target_taxa}' abundance within each group")

    return sorted_indices


def plot_microbiome_composition(abundance_data, groups, group_names=['T1', 'T2', 'T3'],
                                figsize=(25, 13), save_path=None):
    """
    Create microbiome composition plot with individual samples and group comparison
    Samples are sorted by Bacteria;Verrucomicrobia abundance within each group
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': [5, 1]})

    # Get taxa list and create color palette
    taxa_list = abundance_data.columns.tolist()
    # keep the taxa that has len 2 when split by ';'
    taxa_list = [taxa for taxa in taxa_list if len(taxa.split(';')) == 2]
    # drop from abundance_data
    abundance_data = abundance_data[taxa_list]
    color_map = create_color_palette(taxa_list)

    # Panel A: Individual samples
    # Sort samples by Verrucomicrobia abundance within each group
    sorted_indices = sort_samples_by_verrucomicrobia(abundance_data, groups, group_names)

    # Create stacked bar plot for individual samples
    bottom = np.zeros(len(sorted_indices))
    x_pos = range(len(sorted_indices))

    for taxa in taxa_list:
        values = abundance_data.loc[sorted_indices, taxa].values
        ax1.bar(x_pos, values, bottom=bottom, label=taxa,
                color=color_map[taxa], width=0.8, edgecolor='none')
        bottom += values

    # Customize Panel A
    ax1.set_ylabel('Relative abundance (%)', fontsize=36)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.5, len(sorted_indices) - 0.5)
    # font size yaxis
    ax1.tick_params(axis='y', labelsize=20)

    # Add group labels and separators
    group_counts = [len(groups[groups == group]) for group in group_names]

    # Calculate positions for group labels
    current_pos = 0
    group_centers = []
    separator_positions = []

    for i, count in enumerate(group_counts):
        center = current_pos + count / 2 - 0.5
        group_centers.append(center)

        ax1.text(center, -2, f'{group_names[i]}',
                 ha='center', va='top', fontsize=28, fontweight='bold')

        # Add separator line (except after the last group)
        if i < len(group_counts) - 1:
            separator_pos = current_pos + count - 0.5
            separator_positions.append(separator_pos)
            ax1.axvline(x=separator_pos, color='black', linewidth=5, alpha=0.8)

        current_pos += count

    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # Panel B: Group comparison
    # Calculate mean abundance for each group
    group_data = pd.DataFrame()
    for group_name in group_names:
        group_indices = groups[groups == group_name].index
        group_mean = abundance_data.loc[group_indices].mean()
        group_data[group_name] = group_mean

    # Create stacked bar plot for groups
    bottom_group = np.zeros(len(group_names))
    x_pos_group = list(range(len(group_names)))

    for taxa in taxa_list:
        values_group = [group_data.loc[taxa, group] for group in group_names]
        ax2.bar(x_pos_group, values_group, bottom=bottom_group,
                color=color_map[taxa], width=0.6, edgecolor='none')
        bottom_group += values_group

    # Customize Panel B
    ax2.set_ylim(0, 100)
    ax2.set_xlim(-0.5, len(group_names) - 0.5)
    ax2.set_xticks([])
    ax2.tick_params(axis='y', labelsize=20)

    # Add group labels for Panel B
    for i, group_name in enumerate(group_names):
        ax2.text(i, -2, group_name, ha='center', va='top', fontsize=28, fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Create legend as a rectangle under the plots
    handles = [Rectangle((0, 0), 1, 1, color=color_map[taxa]) for taxa in taxa_list]

    fig.legend(handles, taxa_list,
               bbox_to_anchor=(0.5, -0.008),  # Position at bottom center
               loc='lower center',
               ncol=4,  # 5 columns
               fontsize=28,
               frameon=True,
               columnspacing=1.0,
               handletextpad=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make space for legend at bottom

    # Save figure if path provided
    root_dir = os.path.dirname(os.getcwd())
    path_to_save= os.path.join(root_dir,'Paper_plots')
    os.makedirs(path_to_save, exist_ok=True)


    if save_path:
        plt.savefig(os.path.join(path_to_save,save_path), dpi=300, bbox_inches='tight')

    plt.show()

    return fig, (ax1, ax2)


def main():
    cohort='Russia'
    root_dir = os.path.dirname(os.getcwd())
    data_path = os.path.join(root_dir, 'Data', cohort)
    t1_file = os.path.join(data_path, 'T1.csv') if cohort=='Israel' else os.path.join(data_path, 'T2.csv')
    t2_file =  os.path.join(data_path, 'T2.csv')
    t3_file =  os.path.join(data_path, 'T3.csv')

    try:
        # Load and process data
        print("Loading and processing data...")
        abundance_data, groups = load_and_process_data(cohort,t1_file, t2_file, t3_file,
                                                       abundance_threshold=1.0)

        # Perform statistical tests
        print("Performing statistical analysis...")
        results_df = perform_statistical_tests(abundance_data, groups,
                                               group_names=['T1', 'T2', 'T3'],
                                               correction_method='fdr_bh',  # Benjamini-Hochberg FDR
                                               alpha=0.05)

        # Print statistical summary
        print_statistical_summary(results_df, alpha=0.05)

        # Save statistical results
        # save_statistical_results(results_df, '../statistical_results_trimesters.csv')

        # Create the plot
        print("Creating microbiome composition plot...")
        fig, axes = plot_microbiome_composition(abundance_data, groups,
                                                group_names=['T1', 'T2', 'T3'] if cohort=='Israel' else ['','T2', 'T3'],
                                                save_path=f'microbiome_composition_sorted_t1_t2_t3_{cohort}.png')

        print("Analysis completed successfully!")
        print(f"Data summary:")
        print(f"- T1 group: {sum(groups == 'T1')} samples")
        print(f"- T2 group: {sum(groups == 'T2')} samples")
        print(f"- T3 group: {sum(groups == 'T3')} samples")
        print(f"- Taxa included: {len(abundance_data.columns)}")

    except FileNotFoundError:
        print("Data files not found. Please ensure T1, T2, and T3 CSV files are available.")
        print("Expected format: First column = Sample IDs, Remaining columns = Taxa abundances")


if __name__ == "__main__":
    main()