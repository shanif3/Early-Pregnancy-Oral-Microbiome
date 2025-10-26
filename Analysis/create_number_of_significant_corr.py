import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
# Set the style for better aesthetics
plt.style.use('default')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('ALL_metadata_microbe_metadata_correlations_t1_t2_t3.csv', index_col=0)

# Filter for significant correlations (assuming p_value_fdr < 0.05 indicates significance)
significant_df = df[df['p_value_fdr'] < 0.05]
to_fix = [c for c in significant_df['metadata_column'] if c.startswith('Delivery_vategory')]
renamer = {c: c.replace('Delivery_vategory', 'Delivery_Category', 1) for c in to_fix}
significant_df['metadata_column'] = significant_df['metadata_column'].replace(renamer)
# Count significant correlations by metadata_column and timepoint
correlation_counts = significant_df.groupby(['metadata_column', 'timepoint']).size().reset_index(name='count')

# Pivot to get timepoints as columns
pivot_df = correlation_counts.pivot(index='metadata_column', columns='timepoint', values='count').fillna(0)

# Ensure we have all expected timepoints, create them if missing
expected_timepoints = ['T1', 'T2', 'T3', 'T2_minus_T1', 'T3_minus_T2']
for tp in expected_timepoints:
    if tp not in pivot_df.columns:
        pivot_df[tp] = 0

# Reorder columns to match the legend order
pivot_df = pivot_df[expected_timepoints]

# Sort by total correlations (descending) to match the original chart order
pivot_df['total'] = pivot_df.sum(axis=1)
pivot_df = pivot_df.sort_values('total', ascending=True)  # ascending=True for bottom-to-top order
pivot_df = pivot_df.drop('total', axis=1)

# Create the figure with better size and DPI
fig, ax = plt.subplots(figsize=(26, 9), dpi=100)
fig.patch.set_facecolor('white')


# KEEP TOP 10 SIGNIFICANT CORRELATIONS features
pivot_df = pivot_df.tail(20)
# Define modern, accessible colors for all 5 timepoints
colors = ['#2E86AB', '#A23B72', '#F18F01', '#00A896', '#E63946']  # Blue, purple, orange, teal, red
labels = ['T1', 'T2', 'T3', 'T2-T1', 'T3-T2']

# Get the variable names and clean them up for better display
variables = pivot_df.index
clean_variables = []
for var in variables:
    # Clean up variable names for better readability
    clean_var = var.replace('_', ' ').title()
    if len(clean_var) > 30:  # Truncate very long names
        clean_var = clean_var[:27] + '...'
    clean_variables.append(clean_var)

# Get values for each timepoint
t1_values = pivot_df['T1'].values
t2_values = pivot_df['T2'].values
t3_values = pivot_df['T3'].values
t2_minus_t1_values = pivot_df['T2_minus_T1'].values
t3_minus_t2_values = pivot_df['T3_minus_T2'].values

# Create the horizontal stacked bar chart with enhanced styling
y_pos = np.arange(len(variables))
bar_height = 0.7

# Create all 5 bars for the stacked chart
bar1 = ax.barh(y_pos, t1_values, height=bar_height, color=colors[0],
               label=labels[0], alpha=0.9, edgecolor='white', linewidth=0.5)

bar2 = ax.barh(y_pos, t2_values, left=t1_values, height=bar_height,
               color=colors[1], label=labels[1], alpha=0.9, edgecolor='white', linewidth=0.5)

bar3 = ax.barh(y_pos, t3_values, left=t1_values + t2_values, height=bar_height,
               color=colors[2], label=labels[2], alpha=0.9, edgecolor='white', linewidth=0.5)

bar4 = ax.barh(y_pos, t2_minus_t1_values,
               left=t1_values + t2_values + t3_values, height=bar_height,
               color=colors[3], label=labels[3], alpha=0.9, edgecolor='white', linewidth=0.5)

bar5 = ax.barh(y_pos, t3_minus_t2_values,
               left=t1_values + t2_values + t3_values + t2_minus_t1_values, height=bar_height,
               color=colors[4], label=labels[4], alpha=0.9, edgecolor='white', linewidth=0.5)

# Customize the plot with better typography and spacing
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_variables, fontsize=27, fontweight='500')
# x axis bigger ticks

ax.set_xlabel('Number of Significant Correlations', fontsize=30)

# Enhanced legend
legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=False,
                  fontsize=30, title='Timepoint', title_fontsize=32)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_edgecolor('#cccccc')

# Enhanced grid
# ax.grid(axis='x', alpha=0.4, linestyle='-', linewidth=0.5, color='#e0e0e0')
ax.set_axisbelow(True)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# Set better tick parameters
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=24)

# Add subtle background color
# ax.set_facecolor('#fafafa')

# Set x-axis limits with some padding
max_correlations = (t1_values + t2_values + t3_values + t2_minus_t1_values + t3_minus_t2_values).max()
ax.set_xlim(0, max_correlations * 1.08)

# Add value labels on bars for better readability
for i, (t1, t2, t3, t2_diff, t3_diff) in enumerate(zip(t1_values, t2_values, t3_values,
                                                        t2_minus_t1_values, t3_minus_t2_values)):
    total = t1 + t2 + t3 + t2_diff + t3_diff
    if total > 0:  # Only add label if there are correlations
        ax.text(total + max_correlations * 0.01, i, f'{int(total)}',
                va='center', ha='left', fontsize=15, fontweight='500', color='#555555')

# Adjust layout with better margins
plt.tight_layout()
root_dir= os.path.dirname(os.getcwd())
if not os.path.exists(os.path.join(root_dir,'Paper_plots')):
    os.mkdir(os.path.join(root_dir,'Paper_plots'))
# Optional: Save the plot with high quality
plt.savefig(os.path.join(root_dir,'Paper_plots','microbe_number_correlations_barh_t1_t3_enhanced.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Show the plot
plt.show()

# Print enhanced summary statistics
print("\n" + "="*60)
print("📊 SUMMARY: Significant Correlations by Metadata Column (T1-T3)")
print("="*60)
summary_stats = pivot_df.sum(axis=1).sort_values(ascending=False)
for idx, (metadata, count) in enumerate(summary_stats.head(10).items(), 1):
    print(f"{idx:2d}. {metadata:<30} {int(count):>4} correlations")

if len(summary_stats) > 10:
    print(f"    ... and {len(summary_stats) - 10} more metadata columns")

print(f"\nTotal significant correlations: {int(summary_stats.sum())}")
print(f"Average per metadata column: {summary_stats.mean():.1f}")

# Print breakdown by timepoint
print(f"\n📊 BREAKDOWN BY TIMEPOINT:")
print("="*40)
timepoint_totals = pivot_df.sum(axis=0)
for tp, total in timepoint_totals.items():
    print(f"{tp:<15} {int(total):>4} correlations")