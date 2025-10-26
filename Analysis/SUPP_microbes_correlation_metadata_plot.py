import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
import os


warnings.filterwarnings('ignore')
root_dir = os.path.dirname(os.getcwd())


def load_and_process_data(csv_file):
    """Load correlation data and process for plotting"""
    print("Loading correlation data...")
    df = pd.read_csv(csv_file)

    # todo : remove all the rows with 'food_remarks_Gluten-Free' for paper figure- otherwise for supp keep all
    # df = df[~df['metadata_column'].str.contains('food_remarks_Gluten_Free')]
    # df = df[~df['metadata_column'].str.contains('Smoking_Past')]
    print(f"Data shape: {df.shape}")
    print(f"Unique bacteria: {df['bacteria_name'].nunique()}")
    print(f"Unique metadata: {df['metadata_column'].nunique()}")
    print(f"Timepoints: {df['timepoint'].unique()}")

    return df


def filter_significant_correlations(df, p_threshold=0.05, use_fdr=True):
    """Filter for significant correlations"""
    p_col = 'p_value_fdr' if use_fdr else 'p_value'

    print(f"\nFiltering significant correlations (p < {p_threshold})...")
    print(f"Using {'FDR-corrected' if use_fdr else 'raw'} p-values")

    significant_df = df[df[p_col] < p_threshold].copy()
    print(f"Significant correlations: {len(significant_df)}/{len(df)}")

    return significant_df


def create_significance_matrix(df, significance_levels=None):
    """Create matrices for plotting based on significance levels"""
    if significance_levels is None:
        significance_levels = [0.001, 0.01, 0.05]

    # Create pivot table for correlation coefficients
    corr_matrix = df.pivot_table(
        index='bacteria_name',
        columns=['timepoint', 'metadata_column'],
        values='correlation_coefficient',
        fill_value=0
    )

    # Create pivot table for p-values
    pval_matrix = df.pivot_table(
        index='bacteria_name',
        columns=['timepoint', 'metadata_column'],
        values='p_value_fdr',
        fill_value=1.0
    )

    # Create significance level matrix
    sig_matrix = np.ones_like(pval_matrix.values) * 4  # Default: not significant

    for i, threshold in enumerate(reversed(significance_levels)):
        sig_matrix[pval_matrix.values < threshold] = i

    # reorder th columns to be T1, T2, T3, T2-T1, T3-T2
    corr_matrix = corr_matrix.reindex(columns=sorted(corr_matrix.columns, key=lambda x: (['T1', 'T2', 'T3', 'T2_minus_T1', 'T3_minus_T2'].index(x[0]), x[1])))
    pval_matrix = pval_matrix.reindex(columns=corr_matrix.columns)

    return corr_matrix, pval_matrix, sig_matrix, corr_matrix.columns, corr_matrix.index


def parse_taxonomic_name(tax_name):
    """Parse taxonomic name into components"""
    levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    prefixes = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']

    taxonomy = {}
    parts = str(tax_name).split(';')

    for i, part in enumerate(parts):
        if i < len(levels):
            # Remove prefix and get clean name
            clean_name = part
            if i < len(prefixes) and prefixes[i] in part:
                clean_name = part.replace(prefixes[i], '')
            taxonomy[levels[i]] = clean_name if clean_name else 'unknown'

    return taxonomy


def find_shared_taxonomic_level(tax1, tax2):
    """Find the deepest taxonomic level that two organisms share"""
    levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    shared_level = -1  # No shared level

    for level_idx, level in enumerate(levels):
        tax1_val = tax1.get(level, 'unknown')
        tax2_val = tax2.get(level, 'unknown')

        # Check if both values are valid and equal
        if (tax1_val != 'unknown' and tax2_val != 'unknown' and
                tax1_val != '' and tax2_val != '' and tax1_val == tax2_val):
            shared_level = level_idx
        else:
            break  # Once they differ, stop checking deeper levels

    return shared_level


def create_taxonomic_distance_matrix(bacteria_names):
    """Create a distance matrix based on shared taxonomic levels"""
    print("\nCreating taxonomic distance matrix based on hierarchy levels...")

    # Parse all taxonomic names
    taxonomies = []
    valid_names = []

    for name in bacteria_names:
        try:
            tax = parse_taxonomic_name(name)
            taxonomies.append(tax)
            valid_names.append(name)
        except Exception as e:
            print(f"Warning: Could not parse {name}: {e}")
            continue

    if len(taxonomies) < 2:
        print("Not enough valid taxonomic names for clustering")
        return None, list(bacteria_names)

    print(f"Successfully parsed {len(taxonomies)} taxonomic names")

    # Create distance matrix based on shared taxonomic levels
    n_bacteria = len(taxonomies)
    distance_matrix = np.zeros((n_bacteria, n_bacteria))

    levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    max_level = len(levels)

    for i in range(n_bacteria):
        for j in range(i + 1, n_bacteria):
            # Find the deepest shared taxonomic level
            shared_level = find_shared_taxonomic_level(taxonomies[i], taxonomies[j])

            if shared_level == -1:
                # No shared levels - maximum distance
                distance = max_level + 1
            else:
                # Distance is inversely related to shared level
                # Species level (6) gives distance 1, Kingdom level (0) gives distance 7
                distance = max_level - shared_level

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: {np.min(distance_matrix)} to {np.max(distance_matrix)}")

    # Print some examples of distances for debugging
    if len(taxonomies) >= 2:
        print("\nExample distance calculations:")
        for i in range(min(3, len(taxonomies))):
            for j in range(i + 1, min(3, len(taxonomies))):
                shared_level = find_shared_taxonomic_level(taxonomies[i], taxonomies[j])
                level_name = levels[shared_level] if shared_level >= 0 else "none"
                print(f"  {str(valid_names[i])[:50]}... vs {str(valid_names[j])[:50]}...")
                print(f"    Shared level: {level_name} ({shared_level}), Distance: {distance_matrix[i, j]}")

    return distance_matrix, valid_names


def build_taxonomic_linkage(bacteria_names, method='average'):
    """
    Build a valid SciPy linkage matrix from the taxonomic distance matrix.
    Ensures Z has exactly (n-1) merges and labels length == n.
    """
    distance_matrix, valid_names = create_taxonomic_distance_matrix(bacteria_names)

    if distance_matrix is None or len(valid_names) < 2:
        print("Insufficient taxa for dendrogram.")
        return None, valid_names

    # Convert to condensed form expected by scipy.linkage
    condensed = squareform(distance_matrix, checks=False)
    # Use a standard hierarchical algorithm over the taxonomic distance
    Z = linkage(condensed, method=method)
    # Safety check
    expected_n = Z.shape[0] + 1
    if expected_n != len(valid_names):
        raise RuntimeError(
            f"Internal error: linkage implies n={expected_n} but we have {len(valid_names)} labels."
        )
    print(f"Created linkage with shape {Z.shape} (n={expected_n} leaves).")
    return Z, valid_names


def create_shortened_labels(bacteria_names):
    """Create shortened labels for bacteria names for better readability"""
    shortened_labels = []
    for name in bacteria_names:
        try:
            parts = str(name).split(';')
            # loop in reverse to find the first non-empty taxonomic level with "__"
            picked = None
            for part in reversed(parts):
                if '__' in part:
                    prefix, val = part.split('__', 1)
                    if val.strip() != "":
                        picked = part
                        break
            if picked is None:
                # fallback: last part or the whole
                picked = parts[-1] if parts else str(name)
            idx = parts.index(picked)
            prev_and_current = (parts[idx - 1] + ";" + parts[idx]) if idx > 0 else parts[idx]
            shortened_labels.append(prev_and_current)
        except Exception as e:
            print(f"Warning: Could not process {name}: {e}")
            shortened_labels.append("Unknown")
    return shortened_labels


# Dot scaling functions
def linear_dot_scaling(corr_sig, min_size=100, max_size=400):
    """Scale dots linearly based on absolute correlation coefficient"""
    min_size = 1000
    max_size = 4000
    abs_corr = np.abs(corr_sig)
    dot_sizes = min_size + (max_size - min_size) * abs_corr
    return dot_sizes


def sqrt_dot_scaling(corr_sig, min_size=20, max_size=300):
    """Scale dots using square root of absolute correlation"""
    abs_corr = np.abs(corr_sig)
    dot_sizes = min_size + (max_size - min_size) * np.sqrt(abs_corr)
    return dot_sizes


def log_dot_scaling(corr_sig, min_size=20, max_size=300):
    """Scale dots logarithmically based on absolute correlation"""
    abs_corr = np.abs(corr_sig)
    min_corr = np.maximum(abs_corr, 0.01)
    log_corr = np.log10(min_corr / 0.01) / np.log10(1.0 / 0.01)
    dot_sizes = min_size + (max_size - min_size) * log_corr
    return dot_sizes


def categorical_dot_scaling(corr_sig):
    """Create discrete dot sizes based on correlation strength categories"""
    abs_corr = np.abs(corr_sig)
    dot_sizes = np.zeros_like(abs_corr)

    dot_sizes[abs_corr < 0.2] = 30
    dot_sizes[(abs_corr >= 0.2) & (abs_corr < 0.4)] = 80
    dot_sizes[(abs_corr >= 0.4) & (abs_corr < 0.6)] = 150
    dot_sizes[abs_corr >= 0.6] = 250

    return dot_sizes


def plot_phylogenetic_heatmap(corr_matrix, pval_matrix, sig_matrix, columns, bacteria_names,
                              linkage_matrix=None, bacteria_labels=None, figsize=(36, 25),
                              scaling_method='linear', row_spacing=2.0):
    """Create the main phylogenetic heatmap plot with taxonomic dendrogram"""
    print("\nCreating phylogenetic heatmap with taxonomic dendrogram...")

    # Set up the figure with three panels: labels, heatmap, dendrogram
    fig = plt.figure(figsize=figsize)

    if linkage_matrix is not None:
        # Three panels: bacteria names (left), heatmap (center), dendrogram (right)
        gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 3, 1.5], wspace=0.02)

        # First, create the dendrogram to get the ordering
        ax_dend = fig.add_subplot(gs[2])

        display_labels = bacteria_labels if bacteria_labels else list(bacteria_names)

        # Final guard: Z must imply len(labels) == Z.shape[0] + 1
        n_from_Z = linkage_matrix.shape[0] + 1
        if len(display_labels) != n_from_Z:
            print(f"[Guard] Adjusting labels to match linkage: labels={len(display_labels)} vs expected={n_from_Z}")
            # Try to align by intersecting names with corr_matrix index order
            # If still mismatched, truncate or pad with placeholders
            if len(display_labels) > n_from_Z:
                display_labels = display_labels[:n_from_Z]
            else:
                display_labels = display_labels + [f"unk_{i}" for i in range(n_from_Z - len(display_labels))]

        # Create dendrogram
        try:
            dend = dendrogram(linkage_matrix, orientation='right', ax=ax_dend,
                              leaf_font_size=0, labels=display_labels,
                              color_threshold=0.7 * max(linkage_matrix[:, 2]) if len(linkage_matrix) > 0 else 0,
                              above_threshold_color='gray')

            for coll in ax_dend.collections:
                coll.set_linewidth(10.0)
                coll.set_capstyle('round')

            for line in ax_dend.lines:
                line.set_linewidth(10.0)
                line.set_solid_capstyle('round')

            ax_dend.set_yticks([])
            ax_dend.set_xticks([])
            ax_dend.spines['top'].set_visible(False)
            ax_dend.spines['left'].set_visible(False)
            ax_dend.spines['right'].set_visible(False)
            ax_dend.spines['bottom'].set_visible(False)

            dend_order = dend['leaves']
            ordered_bacteria = [bacteria_names[i] for i in dend_order]
            ordered_labels = [display_labels[i] for i in dend_order]

            print(f"Dendrogram created successfully with {len(dend_order)} leaves")

        except Exception as e:
            print(f"Error creating dendrogram: {e}")
            # Fallback to original order
            ordered_bacteria = list(bacteria_names)
            ordered_labels = display_labels
            ax_dend.text(0.5, 0.5, 'Dendrogram\nError', ha='center', va='center',
                         transform=ax_dend.transAxes, fontsize=12)
            ax_dend.set_xticks([])
            ax_dend.set_yticks([])

        # Create bacteria names panel (left)
        ax_names = fig.add_subplot(gs[0])
        ax_names.set_xlim(0, 1)
        ax_names.set_ylim(0, len(ordered_bacteria) * row_spacing)

        for i, label in enumerate(ordered_labels):
            y_pos = i * row_spacing + row_spacing / 2  # Center text in each row
            ax_names.text(0.98, y_pos, label, ha='right', va='center',
                          fontsize=30)

        ax_names.set_xticks([])
        ax_names.set_yticks([])
        for spine in ax_names.spines.values():
            spine.set_visible(False)

        # Reorder correlation data according to dendrogram
        try:
            plot_corr_data = corr_matrix.loc[ordered_bacteria]
            plot_pval_data = pval_matrix.loc[ordered_bacteria]
        except KeyError as e:
            print(f"Warning: Some bacteria not found in correlation matrix: {e}")
            available_bacteria = [b for b in ordered_bacteria if b in corr_matrix.index]
            plot_corr_data = corr_matrix.loc[available_bacteria]
            plot_pval_data = pval_matrix.loc[available_bacteria]
            ordered_bacteria = available_bacteria

        # Plot heatmap (center)
        ax_heat = fig.add_subplot(gs[1])

    else:
        # Fallback: just heatmap without dendrogram
        ax_heat = fig.add_subplot(111)
        plot_corr_data = corr_matrix
        plot_pval_data = pval_matrix
        ordered_bacteria = list(bacteria_names)
        ordered_labels = bacterial_labels if bacteria_labels else list(bacteria_names)

    # Create white background for the plot
    ax_heat.set_facecolor('white')

    # Create the dot plot based on correlations and significance
    col_spacing = 1.0
    y_positions = [i * row_spacing for i in range(len(ordered_bacteria))]
    x_positions = [i * col_spacing for i in range(len(columns))]

    # Create meshgrid for positions
    X, Y = np.meshgrid(x_positions, y_positions)

    # Flatten arrays for scatter plot
    x_flat = X.flatten()
    y_flat = Y.flatten()
    corr_flat = plot_corr_data.values.flatten()
    pval_flat = plot_pval_data.values.flatten()

    # Only plot significant correlations (FDR < 0.05)
    significant_mask = pval_flat < 0.05

    if np.any(significant_mask):
        x_sig = x_flat[significant_mask]
        y_sig = y_flat[significant_mask]
        corr_sig = corr_flat[significant_mask]

        # Calculate dot sizes based on chosen scaling method
        if scaling_method == 'linear':
            dot_sizes = linear_dot_scaling(corr_sig, min_size=20, max_size=300)
        elif scaling_method == 'sqrt':
            dot_sizes = sqrt_dot_scaling(corr_sig, min_size=20, max_size=300)
        elif scaling_method == 'log':
            dot_sizes = log_dot_scaling(corr_sig, min_size=20, max_size=300)
        elif scaling_method == 'categorical':
            dot_sizes = categorical_dot_scaling(corr_sig)
        else:
            dot_sizes = linear_dot_scaling(corr_sig, min_size=20, max_size=300)

        # Colors: blue for positive, red for negative
        colors = ['#1f77b4' if corr > 0 else '#d62728' for corr in corr_sig]

        # Create scatter plot
        ax_heat.scatter(x_sig, y_sig, s=dot_sizes, c=colors,
                        alpha=0.8, edgecolors='black', linewidths=0.8)

    # Set up the plot appearance with adjusted spacing
    ax_heat.set_xlim(-col_spacing * 0.5, (len(columns) - 1) * col_spacing + col_spacing * 0.5)
    ax_heat.set_ylim(-row_spacing * 0.5, (len(ordered_bacteria) - 1) * row_spacing + row_spacing * 0.5)
    ax_heat.invert_yaxis()

    # Set x-axis ticks and labels
    ax_heat.set_xticks(x_positions)
    timepoints = [col[0] for col in columns]
    timepoints = [tp.replace('T2_minus_T1', 'T2-T1').replace('T3_minus_T2', 'T3-T2') for tp in timepoints]
    metadata_vars = [col[1] for col in columns]
    ax_heat.set_xticklabels([f"{mv}" for tp, mv in zip(timepoints, metadata_vars)],
                            rotation=45, ha='right', fontsize=32)

    # Remove y-axis ticks and labels from heatmap (they're on the left panel)
    ax_heat.set_yticks([])
    ax_heat.set_yticklabels([])

    # Add grid for better readability with adjusted spacing
    for i, y_pos in enumerate(y_positions):
        if i > 0:
            ax_heat.axhline(y=y_pos - row_spacing / 2, color='gray', alpha=0.2, linestyle='-', linewidth=0.5)

    for i, x_pos in enumerate(x_positions):
        if i > 0:
            ax_heat.axvline(x=x_pos - col_spacing / 2, color='gray', alpha=0.2, linestyle='-', linewidth=0.5)

    ax_heat.set_axisbelow(True)

    # Add vertical lines to separate timepoints
    unique_timepoints = list(dict.fromkeys(timepoints))
    tp_positions = []
    current_pos = 0

    for tp in unique_timepoints:
        tp_count = timepoints.count(tp)
        tp_center = current_pos + tp_count / 2 - 0.5
        tp_positions.append(tp_center * col_spacing)
        if current_pos + tp_count < len(columns):
            separator_pos = (current_pos + tp_count - 0.5) * col_spacing
            ax_heat.axvline(x=separator_pos, color='black', linewidth=3, alpha=0.8)
        current_pos += tp_count

    # Add timepoint labels at the top
    ax_top = ax_heat.twiny()
    ax_top.set_xlim(ax_heat.get_xlim())
    ax_top.set_xticks(tp_positions)
    ax_top.set_xticklabels(unique_timepoints, fontsize=40, fontweight='bold', alpha=0.9)
    ax_top.tick_params(length=0)

    return fig


    return sorted_cols
def main():
    """Main function to generate the plot"""
    # File paths
    cohort= 'Israel'
    csv_file = 'ALL_metadata_microbe_metadata_correlations_t1_t2_t3.csv'

    # Choose scaling method: 'linear', 'sqrt', 'log', or 'categorical'
    scaling_method = 'linear'

    # Adjust this value to control spacing between bacteria rows
    row_spacing = 3.5

    try:
        # Load and process data
        df = load_and_process_data(csv_file)

        # Filter for significant correlations
        # todo: for supp- p_threshold= 0.05
        # for paper figure- p_threshold=0.01
        sig_df = filter_significant_correlations(df, p_threshold=0.05, use_fdr=True)

        if len(sig_df) == 0:
            print("No significant correlations found!")
            return

        # Create matrices
        corr_matrix, pval_matrix, sig_matrix, columns, bacteria_names = create_significance_matrix(sig_df)


        print(f"\nMatrix dimensions: {corr_matrix.shape}")
        print(f"Bacteria: {len(bacteria_names)}")
        print(f"Metadata combinations: {len(columns)}")

        # ---- Build linkage from taxonomic distances (robust) ----
        Z, valid_bacteria_names = build_taxonomic_linkage(bacteria_names, method='average')

        # Filter matrices to only include bacteria with valid taxonomic names
        if Z is not None and len(valid_bacteria_names) != len(corr_matrix.index):
            print(f"Filtering data to {len(valid_bacteria_names)} bacteria with valid taxonomic names")
            # Keep order of valid_bacteria_names
            valid_set = [b for b in valid_bacteria_names if b in corr_matrix.index]
            corr_matrix = corr_matrix.loc[valid_set]
            pval_matrix = pval_matrix.loc[valid_set]
            bacteria_names = pd.Index(valid_set)
            # Safety check: Z must have n-1 rows where n == len(bacteria_names)
            expected_rows = len(bacteria_names) - 1
            if Z.shape[0] != expected_rows:
                raise RuntimeError(
                    f"Linkage rows ({Z.shape[0]}) != n-1 ({expected_rows})."
                )

        # Create shortened, readable labels (aligned to current bacteria_names)
        bacteria_labels = create_shortened_labels(bacteria_names)

        # Calculate appropriate figure height based on number of bacteria and spacing
        n_bacteria = len(bacteria_names)
        fig_height = max(25, n_bacteria * row_spacing * 0.3)

        # Create the plot
        fig = plot_phylogenetic_heatmap(
            corr_matrix, pval_matrix, sig_matrix, columns, bacteria_names,
            linkage_matrix=Z, bacteria_labels=bacteria_labels,
            figsize=(110, fig_height + 3),
            scaling_method=scaling_method,
            row_spacing=row_spacing
        )

        # Save the plot with higher DPI for better quality
        plt.tight_layout()
        output_filename = f'SUPP_{cohort}_phylogenetic_metadata_heatmap_with_improved_dendrogram_{scaling_method}.pdf'
        path_to_save= os.path.join(root_dir, 'Appendix_plots')
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(os.path.join(os.path.join(path_to_save,output_filename)), bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"\nPlot saved as '{output_filename}.pdf'")
        print(f"Row spacing used: {row_spacing}")
        # plt.show()

        # Print summary statistics
        print("\nSummary Statistics:")
        total_significant = np.sum(pval_matrix.values < 0.05)
        total_positive = np.sum((pval_matrix.values < 0.05) & (corr_matrix.values > 0))
        total_negative = np.sum((pval_matrix.values < 0.05) & (corr_matrix.values < 0))

        print(f"Total significant correlations (FDR < 0.05): {total_significant}")
        print(f"Positive correlations: {total_positive}")
        print(f"Negative correlations: {total_negative}")
        print(f"Dot scaling method used: {scaling_method}")

        if total_significant > 0:
            significant_corrs = corr_matrix.values[pval_matrix.values < 0.05]
            print(f"Correlation range: {np.min(significant_corrs):.3f} to {np.max(significant_corrs):.3f}")
            print(f"Mean absolute correlation: {np.mean(np.abs(significant_corrs)):.3f}")

    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the file is in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
