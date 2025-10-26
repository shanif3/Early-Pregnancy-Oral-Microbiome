import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
trimester='ALL_timepoints'
df_original= pd.read_csv('ALL_metadata_microbe_metadata_correlations_t1_t2_t3.csv', index_col=0)
if trimester=='ALL_timepoints':
    df_original = df_original[
        (df_original['timepoint'] == 'T1') |
        (df_original['timepoint'] == 'T2') |
        (df_original['timepoint'] == 'T3')
    ]
else:
    df_original = df_original[df_original['timepoint'] == trimester]

df = df_original[df_original['p_value_fdr'] < 0.05]
c=0

matrix_dict = {}
for microbe in df.index.unique():
    matrix_dict[microbe] = {}
    for metadata in df['metadata_column'].unique():
        coeffs = df[(df.index == microbe) & (df['metadata_column'] == metadata)]['correlation_coefficient'].tolist()
        matrix_dict[microbe][metadata] = coeffs[0] if coeffs else df_original[(df_original.index == microbe) & (df_original['metadata_column'] == metadata)]['correlation_coefficient'].iloc[0]

matrix_df = pd.DataFrame(matrix_dict).T
metadata_correlation = matrix_df.corr(method='pearson')

# Compute p-values
def compute_correlation_pvalues(df):
    pvals = np.zeros((df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            if i != j:
                x = df.iloc[:, i]
                y = df.iloc[:, j]
                corr, pval = pearsonr(x, y)
                pvals[i, j] = pval
            else:
                pvals[i, j] = 0  # diagonal
    return pd.DataFrame(pvals, index=df.columns, columns=df.columns)

pvalues = compute_correlation_pvalues(matrix_df)
significant = pvalues < 0.05

# Create the plot
plt.figure(figsize=(30, 25))
ax = plt.gca()

n = len(metadata_correlation)

# Create the heatmap using correlations for coloring
sns.heatmap(metadata_correlation,
           annot=False,  # We'll add custom annotations
           cmap='RdBu',
           center=0,
           square=True,
           linewidths=0.5,
           linecolor='white',
           cbar_kws={'label': 'Correlation Coefficient'},
           ax=ax)

# Add custom text annotations
for i in range(n):
    for j in range(n):
        if i == j:  # Diagonal - correlation coefficient
            ax.text(j + 0.5, i + 0.5, '1.00',
                   ha='center', va='center', fontweight='bold', fontsize=10)
        elif i > j:  # Lower triangle - correlation coefficients
            corr_val = metadata_correlation.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f'{corr_val:.2f}',
                   ha='center', va='center', fontweight='bold', fontsize=10)
        else:  # Upper triangle - p-values
            pval = pvalues.iloc[i, j]
            if pval < 0.001:
                ax.text(j + 0.5, i + 0.5, 'p=0.0',
                       ha='center', va='center', fontsize=9)
            else:
                ax.text(j + 0.5, i + 0.5, f'p={pval:.3f}',
                       ha='center', va='center', fontsize=9)

# Add black edges for significant correlations
for i in range(n):
    for j in range(n):
        if significant.iloc[i, j] and i != j:  # exclude diagonal
            # Create rectangle with black border

            if i > j:  # Lower triangle - correlation coefficients
                rect = plt.Rectangle((j, i), 1, 1, fill=False,
                                         edgecolor='black', linewidth=5)
                ax.add_patch(rect)

# Customize the plot
# plt.title('Metadata Correlations\nLower: Correlations, Upper: P-values, Black edges: p < 0.05',
#           fontsize=14, pad=20)
plt.xlabel('')
plt.ylabel('')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)

# Adjust layout
plt.tight_layout()
plt.savefig(os.path.join('../Appendix_plots', f'corr_matrix_metadata_vs_metadata_{trimester}.png'), dpi=300)
plt.show()

# Print some statistics
print(f"Significant correlations: {(significant & ~np.eye(n, dtype=bool)).sum().sum()}")
print(f"Range of correlations: {metadata_correlation.min().min():.3f} to {metadata_correlation.max().max():.3f}")
print(f"Range of p-values: {pvalues[pvalues > 0].min().min():.6f} to {pvalues.max().max():.3f}")