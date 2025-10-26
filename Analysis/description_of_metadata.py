import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_categorical_features_plot(df, save_plot=False, figsize_multiplier=5):
    """
    Create a comprehensive grid plot showing all categorical features

    Parameters:
    df: pandas DataFrame
    save_plot: bool, whether to save the plot
    figsize_multiplier: int, multiplier for figure size (larger = bigger plots)
    """

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        print("No categorical features found.")
        return None

    print(f"Categorical Features Plot:")
    print(f"Total categorical features: {len(categorical_cols)}")
    print("-" * 50)

    # Calculate grid dimensions
    total_features = len(categorical_cols)
    cols_grid = min(5, total_features)
    rows_grid = (total_features + cols_grid - 1) // cols_grid

    # Create the figure
    fig, axes = plt.subplots(rows_grid, cols_grid,
                             figsize=(figsize_multiplier * cols_grid, 4 * rows_grid))

    # Handle different subplot configurations
    if total_features == 1:
        axes = [axes]
    elif rows_grid == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Plot each categorical feature
    for i, col in enumerate(categorical_cols):
        ax = axes[i]

        # Categorical feature - bar plot
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        if len(value_counts) > 0:
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                          alpha=0.7, color=plt.cm.tab10(np.linspace(0, 1, len(value_counts))),
                          edgecolor='black', linewidth=0.5)

            # Set x-axis labels
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels([str(x) for x in value_counts.index],
                               rotation=45, ha='right', fontsize=12)

            # Add count labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=11)

        # Formatting for each subplot
        ax.set_title(col, fontsize=20, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)

        # Set y-axis labels
        if i % cols_grid == 0:
            ax.set_ylabel('Frequency', fontsize=20)
        else:
            ax.set_ylabel('')

    # Hide empty subplots
    for i in range(total_features, len(axes)):
        axes[i].set_visible(False)

    # Overall formatting
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save if requested
    if save_plot:
        cohort = 'Israel'
        root_dir = os.path.dirname(os.getcwd())
        path_to_save = os.path.join(root_dir, 'Appendix_plots')
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(os.path.join(path_to_save, f'categorical_features_{cohort}.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()

    return fig


def create_numerical_features_plot(df, save_plot=False, figsize_multiplier=5):
    """
    Create a comprehensive grid plot showing all numerical features with histograms and KDE

    Parameters:
    df: pandas DataFrame
    save_plot: bool, whether to save the plot
    figsize_multiplier: int, multiplier for figure size (larger = bigger plots)
    """

    # Convert Carb_T1 to numeric
    df['Carb_T1'] = pd.to_numeric(df['Carb_T1'], errors='coerce')

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        print("No numerical features found.")
        return None

    print(f"Numerical Features Plot:")
    print(f"Total numerical features: {len(numerical_cols)}")
    print("-" * 50)

    # Calculate grid dimensions
    total_features = len(numerical_cols)
    cols_grid = min(5, total_features)
    rows_grid = (total_features + cols_grid - 1) // cols_grid

    # Create the figure
    fig, axes = plt.subplots(rows_grid, cols_grid,
                             figsize=(figsize_multiplier * cols_grid, 4 * rows_grid))

    # Handle different subplot configurations
    if total_features == 1:
        axes = [axes]
    elif rows_grid == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Plot each numerical feature
    for i, col in enumerate(numerical_cols):
        ax = axes[i]

        # Numerical feature - histogram with KDE
        clean_data = df[col].dropna()
        if len(clean_data) > 0:
            # Create histogram
            n, bins, patches = ax.hist(clean_data, bins=20, alpha=0.6, color='skyblue',
                                       edgecolor='black', linewidth=0.5, density=True)

            # Add KDE curve
            try:
                kde = stats.gaussian_kde(clean_data)
                x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                kde_values = kde(x_range)
                ax.plot(x_range, kde_values, color='darkblue', linewidth=2,
                        alpha=0.8, label='KDE')
            except Exception as e:
                print(f"Could not compute KDE for {col}: {e}")

            # Add mean line
            mean_val = clean_data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Mean: {mean_val:.2f}')

            # Add median line
            median_val = clean_data.median()
            ax.axvline(median_val, color='orange', linestyle='-.', linewidth=2,
                       alpha=0.8, label=f'Median: {median_val:.2f}')

            # Add legend for numerical features
            ax.legend(loc='upper right', fontsize=9, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.8)

        # Formatting for each subplot
        display_col = 'Stress_T2' if col == 'Stress_2' else col
        ax.set_title(display_col, fontsize=20, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)

        # Set y-axis labels
        if i % cols_grid == 0:
            ax.set_ylabel('Density', fontsize=20)
        else:
            ax.set_ylabel('')

    # Hide empty subplots
    for i in range(total_features, len(axes)):
        axes[i].set_visible(False)

    # Overall formatting
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save if requested
    if save_plot:
        cohort = 'Israel'
        root_dir = os.path.dirname(os.getcwd())
        path_to_save = os.path.join(root_dir, 'Appendix_plots')
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(os.path.join(path_to_save, f'numerical_features_histogram_with_kde_{cohort}.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()

    return fig


def create_summary_stats_table(df):
    """
    Create a summary statistics table for numerical features
    """
    df['Delivery_week'] = df['Delivery_week'].apply(
        lambda x: np.nan if x == ' Termination' or x == 'termination' else x
    )
    numeric_col = ['Delivery_week','Carb_T1', 'PAPP - A[mU / L]', 'PAPP - A[MoM]', 'free_hCG_[ng / mL]',
                   'free_hCG_[MoM]', 'AFP[ng / mL]', 'AFP[MoM]', 'NT[MoM]', 'NT[mm]']
    for col in numeric_col:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cat= ['parity','Gestation','OGTT_category','Extreme180','Extreme120','Extreme60','Extreme0']
    # convert cat to category dtype
    for col in cat:
        if col in df.columns:
            df[col] = df[col].astype('category')

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numerical_cols:
        print("\nSummary Statistics for Numerical Features:")
        print("=" * 60)
        summary_stats = df[numerical_cols].describe()
        print(summary_stats.round(3))

        # Additional statistics
        additional_stats = pd.DataFrame({
            'skewness': df[numerical_cols].skew(),
            'kurtosis': df[numerical_cols].kurtosis()
        })
        print("\nSkewness and Kurtosis:")
        print("-" * 30)
        print(additional_stats.round(3))


def load_and_plot_data(file_path):
    """
    Load data and create separate plots for categorical and numerical features
    """
    # Load your data (adjust based on your file format)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Please use CSV or Excel.")
        return None

    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head(3))
    print("\n")

    # Create summary statistics table
    create_summary_stats_table(df)

    # Create the two separate plots
    print("\n" + "=" * 60)
    fig_categorical = create_categorical_features_plot(df, save_plot=True)

    print("\n" + "=" * 60)
    fig_numerical = create_numerical_features_plot(df, save_plot=True)

    return fig_categorical, fig_numerical


import os

# Run the analysis
cohort = 'Israel'
root_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(root_dir, 'Data', cohort)
metadata = pd.read_csv(os.path.join(data_path, 'Women_metadata.csv'), index_col=0)

result = load_and_plot_data(os.path.join(data_path, 'Women_metadata.csv'))