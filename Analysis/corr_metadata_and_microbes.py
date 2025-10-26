import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pointbiserialr
from statsmodels.stats.multitest import multipletests
from utils import *

cohort= 'Russia'#'Israel'
root_dir= os.path.dirname(os.getcwd())
data_path= os.path.join(root_dir, 'Data',cohort)

t1 = pd.read_csv(os.path.join(data_path,"T1.csv"), index_col=0) if cohort=='Israel' else pd.read_csv(os.path.join(data_path,"T2.csv"), index_col=0)
t2 = pd.read_csv(os.path.join(data_path,"T2.csv"), index_col=0)
t3 = pd.read_csv(os.path.join(data_path,"T3.csv"), index_col=0)

metadata= pd.read_csv(os.path.join(data_path,'Women_metadata.csv'), index_col=0)
if cohort=='Israel':
    metadata = preprocess_metadata(metadata)

mutual= t1.index.union(t2.index.union(t3.index))
metadata= metadata.loc[metadata.index.intersection(mutual)]
# drop duplicates index
metadata= metadata[~metadata.index.duplicated(keep='first')]


if cohort=='Russia':
    metadata = metadata.loc[metadata['Smoking_before'] != '#NULL!', ['Smoking_before']].astype(float).astype(int)
    significant_micrbos_israel= pd.read_csv('ALL_metadata_microbe_metadata_correlations_t1_t2_t3.csv',index_col=0)
    significant_micrbos_israel= significant_micrbos_israel[significant_micrbos_israel['metadata_column']=='Smoking_Past']
    significant_micrbos_israel_cols= significant_micrbos_israel.loc[significant_micrbos_israel['p_value_fdr']<0.05].index.tolist()


    mutual_t2= t2.columns.intersection(significant_micrbos_israel_cols)
    mutual_t3= t3.columns.intersection(significant_micrbos_israel_cols)
    t1= t1.loc[:, mutual_t2]
    t2= t2.loc[:, mutual_t2]
    t3= t3.loc[:, mutual_t3]
    # drop duplicates columns
    t1= t1.loc[:, ~t1.columns.duplicated(keep='first')]
    t2= t2.loc[:, ~t2.columns.duplicated(keep='first')]
    t3= t3.loc[:, ~t3.columns.duplicated(keep='first')]

# apply log in df
t1= t1.apply(lambda x: x + 0.1).apply(np.log10)
t2 = t2.apply(lambda x: x + 0.1).apply(np.log10)
t3 = t3.apply(lambda x: x + 0.1).apply(np.log10)
#
# # apply z score
t1= t1.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
t2 = t2.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
t3 = t3.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


def identify_column_types(df, binary_threshold=2):
    """
    Identify binary and continuous columns in a dataframe.

    Parameters:
    df: pandas DataFrame
    binary_threshold: int, maximum number of unique values to consider a column binary

    Returns:
    dict with 'binary' and 'continuous' keys containing lists of column names
    """
    column_types = {'binary': [], 'continuous': []}

    for col in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        unique_vals = df[col].dropna().nunique()
        unique_values = set(df[col].dropna().unique())

        # Check if binary (0,1 or True,False or similar)
        if (unique_vals == 2 and
                (unique_values <= {0, 1} or
                 unique_values <= {True, False} or
                 unique_vals <= binary_threshold)):
            column_types['binary'].append(col)
        elif unique_vals > 2:
            column_types['continuous'].append(col)

    return column_types


def compute_all_correlations(microbe_data, metadata, timepoint_name=""):
    """
    Compute all correlations and return a single comprehensive dataframe.

    Parameters:
    microbe_data: pandas DataFrame with microbe abundance data (samples x microbes)
    metadata: pandas DataFrame with metadata (samples x metadata features)
    timepoint_name: string to identify the timepoint

    Returns:
    pandas DataFrame with all correlation results
    """

    # Ensure we have matching samples
    common_samples = microbe_data.index.intersection(metadata.index)
    if len(common_samples) == 0:
        print(f"Warning: No common samples found between microbe data and metadata")
        return pd.DataFrame()

    microbe_subset = microbe_data.loc[common_samples]
    metadata_subset = metadata.loc[common_samples]

    print(f"Computing correlations for {len(common_samples)} samples at {timepoint_name}")

    # Identify column types
    col_types = identify_column_types(metadata_subset)
    print(f"Binary columns: {col_types['binary']}")
    print(f"Continuous columns: {col_types['continuous']}")

    # Store all results
    all_results = []

    # Process binary metadata columns
    for meta_col in col_types['binary']:
        print(f"Processing binary metadata: {meta_col}")
        meta_values = metadata_subset[meta_col].dropna()

        # Find samples with both microbe and metadata data
        valid_samples = microbe_subset.index.intersection(meta_values.index)
        if len(valid_samples) < 10:  # Minimum sample size
            print(f"  Skipping {meta_col}: insufficient samples ({len(valid_samples)})")
            continue

        microbe_valid = microbe_subset.loc[valid_samples]
        meta_valid = meta_values.loc[valid_samples]

        # Collect results for this metadata column
        meta_results = []

        for microbe_col in microbe_valid.columns:
            microbe_vals = microbe_valid[microbe_col]

            # Remove samples where either microbe or metadata is NaN
            mask = ~(microbe_vals.isna() | meta_valid.isna())
            if mask.sum() < 10:  # Need at least 10 valid pairs
                continue

            try:
                # Point-biserial correlation
                corr, p_val = pointbiserialr(meta_valid[mask], microbe_vals[mask])

                meta_results.append({
                    'bacteria_name': microbe_col,
                    'metadata_column': meta_col,
                    'correlation_type': 'point_biserial',
                    'correlation_coefficient': corr,
                    'p_value': p_val,
                    'timepoint': timepoint_name,
                    'n_samples': mask.sum()
                })

            except Exception as e:
                print(f"  Error computing correlation for {meta_col} vs {microbe_col}: {e}")
                continue

        # Apply FDR correction for this metadata column
        if meta_results:
            p_values = [r['p_value'] for r in meta_results]
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')

            for i, result in enumerate(meta_results):
                result['p_value_fdr'] = p_adj[i]

            all_results.extend(meta_results)
            print(f"  Computed {len(meta_results)} correlations for {meta_col}")

    # Process continuous metadata columns
    for meta_col in col_types['continuous']:
        print(f"Processing continuous metadata: {meta_col}")
        meta_values = metadata_subset[meta_col].dropna()

        # Find samples with both microbe and metadata data
        valid_samples = microbe_subset.index.intersection(meta_values.index)
        if len(valid_samples) < 10:  # Minimum sample size
            print(f"  Skipping {meta_col}: insufficient samples ({len(valid_samples)})")
            continue

        microbe_valid = microbe_subset.loc[valid_samples]
        meta_valid = meta_values.loc[valid_samples]

        # Collect results for this metadata column
        meta_results = []

        for microbe_col in microbe_valid.columns:
            microbe_vals = microbe_valid[microbe_col]

            # Remove samples where either microbe or metadata is NaN
            mask = ~(microbe_vals.isna() | meta_valid.isna())
            if mask.sum() < 10:  # Need at least 10 valid pairs
                continue

            try:
                # Spearman correlation
                corr, p_val = spearmanr(meta_valid[mask], microbe_vals[mask])

                meta_results.append({
                    'bacteria_name': microbe_col,
                    'metadata_column': meta_col,
                    'correlation_type': 'spearman',
                    'correlation_coefficient': corr,
                    'p_value': p_val,
                    'timepoint': timepoint_name,
                    'n_samples': mask.sum()
                })

            except Exception as e:
                print(f"  Error computing correlation for {meta_col} vs {microbe_col}: {e}")
                continue

        # Apply FDR correction for this metadata column
        if meta_results:
            p_values = [r['p_value'] for r in meta_results]
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')

            for i, result in enumerate(meta_results):
                result['p_value_fdr'] = p_adj[i]

            all_results.extend(meta_results)
            print(f"  Computed {len(meta_results)} correlations for {meta_col}")

    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Add absolute correlation for easy sorting
        results_df['abs_correlation'] = np.abs(results_df['correlation_coefficient'])

        # Reorder columns
        column_order = [
            'bacteria_name', 'metadata_column', 'correlation_type',
            'correlation_coefficient', 'p_value', 'p_value_fdr',
            'abs_correlation', 'timepoint', 'n_samples'
        ]
        results_df = results_df[column_order]

        # Sort by absolute correlation (strongest first)
        results_df = results_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

        print(f"Total correlations computed: {len(results_df)}")
        return results_df
    else:
        print("No correlations computed")
        return pd.DataFrame()


# Create T2 - T1 difference dataframe
print("Creating T2 - T1 difference dataframe...")
common_samples = t1.index.intersection(t2.index)
common_bacteria = t1.columns.intersection(t2.columns)

if len(common_samples) > 0 and len(common_bacteria) > 0:
    print(f"Found {len(common_samples)} common samples and {len(common_bacteria)} common bacteria")

    # Create difference dataframe (T2 - T1)
    t1_common = t1.loc[common_samples, common_bacteria]
    t2_common = t2.loc[common_samples, common_bacteria]
    t2_minus_t1 = t2_common - t1_common

    print(f"T2-T1 difference dataframe shape: {t2_minus_t1.shape}")
    print(f"Sample of differences (first bacteria, first 5 samples):")
    print(t2_minus_t1.iloc[:5, 0])
else:
    print("Warning: No common samples or bacteria found between T1 and T2")
    t2_minus_t1 = pd.DataFrame()

print("Creating T3 - T2 difference dataframe...")
common_samples = t3.index.intersection(t2.index)
common_bacteria = t3.columns.intersection(t2.columns)

if len(common_samples) > 0 and len(common_bacteria) > 0:
    print(f"Found {len(common_samples)} common samples and {len(common_bacteria)} common bacteria")

    # Create difference dataframe (T2 - T1)
    t2_common = t2.loc[common_samples, common_bacteria]
    t3_common = t3.loc[common_samples, common_bacteria]
    t3_minus_t2 = t3_common - t2_common

    print(f"T3-T2 difference dataframe shape: {t3_minus_t2.shape}")
    print(f"Sample of differences (first bacteria, first 5 samples):")
    print(t3_minus_t2.iloc[:5, 0])
else:
    print("Warning: No common samples or bacteria found between T1 and T2")
    t3_minus_t2 = pd.DataFrame()

# Compute correlations for individual timepoints
print("\nComputing correlations for T1...")
t1_results = compute_all_correlations(t1, metadata, "T1")

print("\nComputing correlations for T2...")
t2_results = compute_all_correlations(t2, metadata, "T2")

t3_results= compute_all_correlations(t3, metadata, "T3")

# Compute correlations for T2 - T1 differences
if not t2_minus_t1.empty:
    print("\nComputing correlations for T2 - T1 differences...")
    diff_results_t2_t1 = compute_all_correlations(t2_minus_t1, metadata, "T2_minus_T1")
else:
    print("Skipping T2 - T1 correlation analysis due to empty difference dataframe")
    diff_results_t2_t1 = pd.DataFrame()

# Compute correlations for T3 - T2 differences
if not t3_minus_t2.empty:
    print("\nComputing correlations for T3 - T2 differences...")
    diff_results_t3_t2 = compute_all_correlations(t3_minus_t2, metadata, "T3_minus_T2")
else:
    print("Skipping T3 - T2 correlation analysis due to empty difference dataframe")
    diff_results_t3_t2 = pd.DataFrame()

# Combine results from all analyses
result_dataframes = []
if not t1_results.empty:
    result_dataframes.append(t1_results)
if not t2_results.empty:
    result_dataframes.append(t2_results)
if not t3_results.empty:
    result_dataframes.append(t3_results)
if not diff_results_t2_t1.empty:
    result_dataframes.append(diff_results_t2_t1)
if not diff_results_t3_t2.empty:
    result_dataframes.append(diff_results_t3_t2)

if result_dataframes:
    combined_results = pd.concat(result_dataframes, ignore_index=True)
else:
    combined_results = pd.DataFrame()

if not combined_results.empty:
    print(f"\n=== CORRELATION ANALYSIS SUMMARY ===")
    print(f"Total correlations: {len(combined_results)}")
    print(f"Timepoints: {combined_results['timepoint'].unique()}")
    print(f"Metadata columns: {combined_results['metadata_column'].nunique()}")
    print(f"Unique bacteria: {combined_results['bacteria_name'].nunique()}")
    print(f"Correlation types: {combined_results['correlation_type'].value_counts().to_dict()}")

    # Summary by timepoint
    print(f"\n=== CORRELATIONS BY TIMEPOINT ===")
    for tp in combined_results['timepoint'].unique():
        tp_count = (combined_results['timepoint'] == tp).sum()
        print(f"{tp}: {tp_count} correlations")

    # Display top 20 correlations overall
    print(f"\n=== TOP 20 STRONGEST CORRELATIONS (ALL TIMEPOINTS) ===")
    top_correlations = combined_results.head(20)
    for _, row in top_correlations.iterrows():
        print(f"{row['bacteria_name']} ~ {row['metadata_column']} ({row['timepoint']}): "
              f"{row['correlation_type']} r={row['correlation_coefficient']:.3f}, "
              f"p={row['p_value']:.2e}, p_fdr={row['p_value_fdr']:.2e}")

    # Save combined results
    output_filename = f"ALL_metadata_microbe_metadata_correlations_t1_t2_t3_{cohort}.csv"
    combined_results.to_csv(output_filename, index=False)
    print(f"\nAll results saved to: {output_filename}")

    # Save separate files for each timepoint
    for timepoint in combined_results['timepoint'].unique():
        tp_data = combined_results[combined_results['timepoint'] == timepoint]
        tp_filename = f"microbe_metadata_correlations_{timepoint}.csv"
        # tp_data.to_csv(tp_filename, index=False)
        print(f"{timepoint} results saved to: {tp_filename}")

    # Summary statistics
    print(f"\n=== SIGNIFICANCE SUMMARY ===")
    sig_raw = (combined_results['p_value'] < 0.05).sum()
    sig_fdr = (combined_results['p_value_fdr'] < 0.05).sum()
    print(
        f"Significant correlations (p < 0.05): {sig_raw}/{len(combined_results)} ({100 * sig_raw / len(combined_results):.1f}%)")
    print(
        f"Significant correlations (FDR < 0.05): {sig_fdr}/{len(combined_results)} ({100 * sig_fdr / len(combined_results):.1f}%)")

    # Specific analysis for T2 - T1 differences
    if not diff_results_t2_t1.empty:
        print(f"\n=== T2 - T1 DIFFERENCE CORRELATION SUMMARY ===")
        diff_sig_raw = (diff_results_t2_t1['p_value'] < 0.05).sum()
        diff_sig_fdr = (diff_results_t2_t1['p_value_fdr'] < 0.05).sum()
        print(f"T2-T1 difference correlations: {len(diff_results_t2_t1)}")
        print(
            f"Significant (p < 0.05): {diff_sig_raw}/{len(diff_results_t2_t1)} ({100 * diff_sig_raw / len(diff_results_t2_t1):.1f}%)")
        print(
            f"Significant (FDR < 0.05): {diff_sig_fdr}/{len(diff_results_t2_t1)} ({100 * diff_sig_fdr / len(diff_results_t2_t1):.1f}%)")

        print(f"\n=== TOP 10 STRONGEST T2-T1 DIFFERENCE CORRELATIONS ===")
        top_diff_correlations = diff_results_t2_t1.head(10)
        for _, row in top_diff_correlations.iterrows():
            print(f"{row['bacteria_name']} change ~ {row['metadata_column']}: "
                  f"{row['correlation_type']} r={row['correlation_coefficient']:.3f}, "
                  f"p={row['p_value']:.2e}, p_fdr={row['p_value_fdr']:.2e}")

else:
    print("No correlations computed - check your data!")
    print(f"\n=== CORRELATION ANALYSIS SUMMARY ===")
    print(f"Total correlations: {len(combined_results)}")
    print(f"Timepoints: {combined_results['timepoint'].unique()}")
    print(f"Metadata columns: {combined_results['metadata_column'].nunique()}")
    print(f"Unique bacteria: {combined_results['bacteria_name'].nunique()}")
    print(f"Correlation types: {combined_results['correlation_type'].value_counts().to_dict()}")

    # Display top 20 correlations
    print(f"\n=== TOP 20 STRONGEST CORRELATIONS ===")
    top_correlations = combined_results.head(20)
    for _, row in top_correlations.iterrows():
        print(f"{row['bacteria_name']} ~ {row['metadata_column']} ({row['timepoint']}): "
              f"{row['correlation_type']} r={row['correlation_coefficient']:.3f}, "
              f"p={row['p_value']:.2e}, p_fdr={row['p_value_fdr']:.2e}")

    # Save combined results
    output_filename = "microbe_metadata_correlations_combined.csv"
    # combined_results.to_csv(output_filename, index=False)
    print(f"\nAll results saved to: {output_filename}")

    # Save separate files for each timepoint if desired
    for timepoint in combined_results['timepoint'].unique():
        tp_data = combined_results[combined_results['timepoint'] == timepoint]
        tp_filename = f"microbe_metadata_correlations_{timepoint}.csv"
        # tp_data.to_csv(tp_filename, index=False)
        print(f"{timepoint} results saved to: {tp_filename}")

    # Summary statistics
    print(f"\n=== SIGNIFICANCE SUMMARY ===")
    sig_raw = (combined_results['p_value'] < 0.05).sum()
    sig_fdr = (combined_results['p_value_fdr'] < 0.05).sum()
    print(
        f"Significant correlations (p < 0.05): {sig_raw}/{len(combined_results)} ({100 * sig_raw / len(combined_results):.1f}%)")
    print(
        f"Significant correlations (FDR < 0.05): {sig_fdr}/{len(combined_results)} ({100 * sig_fdr / len(combined_results):.1f}%)")

