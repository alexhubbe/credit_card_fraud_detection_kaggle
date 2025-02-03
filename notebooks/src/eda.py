import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from matplotlib.ticker import PercentFormatter


def plot_violin_box_qq(df, cat_variable=None, n_cols_per_row=1):
    """
    This function creates violin and boxplots for each numerical variable in the DataFrame,
    along with a QQ-plot of the same variable. If a categorical variable is provided,
    it separates the violin and boxplots by category and generates QQ-plots for each category.
    
    Args:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - cat_variable (str, optional): The column name of a categorical variable. If None, plots are made for the whole dataset.
    - n_cols_per_row (int, optional): The number of variable pairs (violin+boxplot and QQ-plot) per row. Default is 2.
    
    Returns:
    - None: Displays the plots.
    """

    
    if cat_variable is not None and not isinstance(cat_variable, str):  # Check if it's not None or a string
        cat_variable = ''.join(cat_variable)  # Convert list to string if needed
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include='number')

    # Calculate number of rows needed (two columns per variable: one for violin+boxplot, one for QQ-plot)
    n_cols = len(df_numeric.columns) * 2
    n_rows = (n_cols // (2 * n_cols_per_row)) + int(n_cols % (2 * n_cols_per_row) > 0)

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(n_rows, n_cols_per_row * 2, figsize=(7 * n_cols_per_row, 5 * n_rows))
    axes = axes.flatten()

    # Loop over each numeric variable
    for i, column in enumerate(df_numeric.columns):
        # Index positions for the pair of plots
        ax_violin_box = axes[i * 2]
        ax_qq = axes[i * 2 + 1]

        # Plot violin+boxplot
        if cat_variable:
            sns.violinplot(x=cat_variable, y=column, data=df, inner=None, ax=ax_violin_box, width=0.6)
            sns.boxplot(x=cat_variable, y=column, data=df, width=0.1, color='black', showcaps=True,
                        showmeans=False, showfliers=True, medianprops={"color": "red", "linewidth": 2}, ax=ax_violin_box,
                        flierprops=dict(marker='o', color='red', markerfacecolor= 'red', markersize=6))  # Solid red dots for outliers
            ax_violin_box.set_xlabel(cat_variable)
        else:
            sns.violinplot(x=[column] * len(df), y=df[column], inner=None, ax=ax_violin_box, width=0.6)
            sns.boxplot(x=[column] * len(df), y=df[column], width=0.1, color='black', showcaps=True,
                        showmeans=False, showfliers=True, medianprops={"color": "red", "linewidth": 2}, ax=ax_violin_box,
                        flierprops=dict(marker='o', color='red', markerfacecolor= 'red', markersize=6))  # Solid red dots for outliers
            ax_violin_box.set_xlabel(column)
        
        ax_violin_box.set_title(f'Violin+Boxplot: {column}')
        ax_violin_box.set_ylabel('')

        # Plot QQ-plot for each category and for the whole variable
        if cat_variable:
            categories = df[cat_variable].unique()
            colors = sns.color_palette("Set2", len(categories))  # Define a color palette

            for j, cat in enumerate(categories):
                sample_data = df[df[cat_variable] == cat][column].dropna()
                (osm, osr), (slope, intercept, r) = stats.probplot(sample_data, dist="norm")
                
                # Plot the QQ points with reduced size
                ax_qq.plot(osm, osr, 'o', label=f'{cat_variable}={cat}', color=colors[j], markersize=4.5)  # Reduced size by 10%
                
                # Plot the reference line
                ax_qq.plot(osm, slope * np.array(osm) + intercept, color=colors[j], linestyle='--')

                # Calculate and display kurtosis and skewness for each category
                kurtosis = stats.kurtosis(sample_data)
                skewness = stats.skew(sample_data)
                ax_qq.text(0.05, 0.95 - j * 0.1, f'{cat} | Kurt: {kurtosis:.2f}, Skew: {skewness:.2f}',
                           transform=ax_qq.transAxes, fontsize=9, color=colors[j], ha='left', va='top')

            ax_qq.legend(title=cat_variable, loc="lower right")  # Move legend to the lower right corner
            ax_qq.set_title(f'QQ-plot by {cat_variable}: {column}')
        else:
            # QQ-plot without categories
            (osm, osr), (slope, intercept, r) = stats.probplot(df[column].dropna(), dist="norm", plot=ax_qq)
            ax_qq.get_lines()[1].set_color('black')  # Set reference line color to black
            ax_qq.get_lines()[1].set_linestyle('--')  # Set reference line style to dashed

            # Calculate and display kurtosis and skewness for the whole variable
            kurtosis = stats.kurtosis(df[column].dropna())
            skewness = stats.skew(df[column].dropna())
            ax_qq.text(0.05, 0.95, f'Kurt: {kurtosis:.2f}, Skew: {skewness:.2f}',
                       transform=ax_qq.transAxes, fontsize=10, color='black', ha='left', va='top')

            ax_qq.set_title(f'QQ-plot: {column}')

    # Hide any unused subplots if there are fewer variables than the grid
    for j in range(i * 2 + 2, len(axes)):
        axes[j].axis('off')

    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, figsize=(12, 12), cmap="cividis", method="pearson", fontsize=10):
    """
    Generates and displays a heatmap of the correlation matrix for numeric columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        figsize (tuple): The size of the figure. Default is (12, 12).
        cmap (str): The color map for the heatmap. Default is "cividis".
        method (str): The correlation method to use. Options are 'pearson', 'spearman', 'kendall'.
                      Default is 'pearson'.
        fontsize (int): The font size for the correlation values displayed on the heatmap. Default is 10.
    
    Returns:
        None
    """
    # Calculate the correlation matrix for numeric columns
    cor_results = df.select_dtypes(include='number').corr(method=method)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw the heatmap with annotations, including the fontsize parameter for the annotations
    sns.heatmap(cor_results, annot=True, fmt=".2f", ax=ax, cmap=cmap, annot_kws={"size": fontsize})
    
    # Set the title
    ax.set_title(f'Correlation Heatmap: {method}', fontsize=16)
    
    # Show the plot
    plt.show()

# Example usage:
# plot_correlation_heatmap(df, figsize=(10, 10), cmap="coolwarm", method="spearman", fontsize=12)


def plot_boxplot_by_focal_variable(df, focal_column, focal_variable=None, n_cols=3):
    """
    This function generates box plots for numeric columns of a DataFrame, grouped by the focal_variable or NaN values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        focal_column (str): The column name to check for NaN or string matching.
        focal_variable (str or None): The string to check for in focal_column or None to check for NaN values.
        n_cols (int): Number of columns in the grid of subplots (default is 3).
    """
    # Debug information about focal_column dtype and data
    print(f"Processing column: {focal_column} (dtype: {df[focal_column].dtype})")

    # Ensure proper handling of NaN and string matches
    if focal_variable:
        bin_var = df[focal_column].astype(str).apply(lambda x: 1 if focal_variable in x else 0)
    else:
        # Handle NaN explicitly for all dtypes, including category
        bin_var = df[focal_column].isna().astype(int)
        focal_variable = "NaN"

    # Debug output: Check bin_var values
    print(f"Binary variable (bin_var):\n{bin_var.value_counts()}")

    # Exclude focal_column from numeric columns if it exists
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col != focal_column]
    print(f"Numeric columns for plotting: {numeric_columns}")

    # Determine the number of rows needed
    n_rows = math.ceil(len(numeric_columns) / n_cols)

    # Create a figure to hold the subplots
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Flatten the axs array to easily iterate, even if it's multi-dimensional
    axs = axs.flatten()

    # Loop through each numeric column and plot it on the respective axis
    for i, column in enumerate(numeric_columns):
        sns.boxplot(data=df, x=bin_var, y=column, ax=axs[i])
        axs[i].set_title(f"{column} by {focal_variable}")

    # Hide any remaining empty subplots if there are fewer columns than subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def cross_table_NaN(df, focal_column, decimals=3):
    """
    Generates and prints crosstables for categorical columns grouped by the presence of NaN values in a focal column.

    Parameters:
    - df: DataFrame containing the data to analyze.
    - focal_column: The column name to check for NaN values.
    - decimals: The number of decimal places to round the results (default is 3).
    """
    # Identify categorical columns in the DataFrame
    categorical_columns = df.select_dtypes(exclude='number').columns

    # Create a binary variable where 1 represents NaN and 0 represents non-NaN in the focal column
    bin_var = df[focal_column].apply(lambda x: 1 if pd.isna(x) else 0)
    
    # Loop through each categorical column and generate the crosstab
    for cat_col in categorical_columns:
        crosstable = pd.crosstab(df[cat_col], bin_var, margins=True, normalize=True)
        
        # Round the crosstable to the specified number of decimals
        crosstable = crosstable.round(decimals)

        # Print the crosstable
        print(f"\nCrosstable for {cat_col}:")
        print(crosstable)


def plot_boxplot_cat(df, cat_variable, focal_variables, ncols=1):
    """
    Plots boxplots for focal variables grouped by a categorical variable.
    
    Parameters:
    df (DataFrame): The dataset containing the data.
    cat_variable (str): The categorical variable to group by.
    focal_variables (list or str): The variable(s) for which to plot the boxplots.
    ncols (int): The number of columns in the subplot grid (default is 1).
    """

    if not isinstance(cat_variable, str):  # Check if it's not a string
        cat_variable = ''.join(cat_variable)  # Convert list to string if needed
    
    # Ensure focal_variables is a list (if only a single variable is provided)
    if isinstance(focal_variables, str):
        focal_variables = [focal_variables]
    
    n_focal = len(focal_variables)
    
    # Calculate the number of rows needed for the subplot grid
    nrows = (n_focal // ncols) + (n_focal % ncols > 0)  # Calculate number of rows needed for subplots
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6 * nrows))
    
    # Flatten axes array to easily loop through it
    axes = axes.flatten() if n_focal > 1 else [axes]
    
    # Loop over the focal variables and create a plot for each
    for idx, focal_variable in enumerate(focal_variables):
        # Boxplot plotting
        ax = axes[idx]  # Select the appropriate axis for this plot
        
        # Use seaborn to create a boxplot
        sns.boxplot(x=cat_variable, y=focal_variable, data=df, ax=ax, color='skyblue', fliersize=6, width=0.6)
        
        # Labels and title
        ax.set_xlabel(cat_variable.replace('_', ' ').title())
        ax.set_ylabel(focal_variable.replace('_', ' ').title())
        ax.set_title(f'{focal_variable.replace("_", " ").title()} by {cat_variable.replace("_", " ").title()}')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Remove any unused axes (in case the number of focal variables is less than the grid size)
    for i in range(n_focal, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_stripplot_cat(df, cat_variable, focal_variables, ncols=1):
    """
    Plots boxplots for focal variables grouped by a categorical variable.
    
    Parameters:
    df (DataFrame): The dataset containing the data.
    cat_variable (str): The categorical variable to group by.
    focal_variables (list or str): The variable(s) for which to plot the boxplots.
    ncols (int): The number of columns in the subplot grid (default is 1).
    """

    if not isinstance(cat_variable, str):  # Check if it's not a string
        cat_variable = ''.join(cat_variable)  # Convert list to string if needed
    
    # Ensure focal_variables is a list (if only a single variable is provided)
    if isinstance(focal_variables, str):
        focal_variables = [focal_variables]
    
    n_focal = len(focal_variables)
    
    # Calculate the number of rows needed for the subplot grid
    nrows = (n_focal // ncols) + (n_focal % ncols > 0)  # Calculate number of rows needed for subplots
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6 * nrows))
    
    # Flatten axes array to easily loop through it
    axes = axes.flatten() if n_focal > 1 else [axes]
    
    # Loop over the focal variables and create a plot for each
    for idx, focal_variable in enumerate(focal_variables):
        # Boxplot plotting
        ax = axes[idx]  # Select the appropriate axis for this plot
        
        # Use seaborn to create a boxplot
        sns.stripplot(x=cat_variable, y=focal_variable, data=df, ax=ax, color='skyblue', jitter=True, size=6, alpha=0.8)
        
        # Labels and title
        ax.set_xlabel(cat_variable.replace('_', ' ').title())
        ax.set_ylabel(focal_variable.replace('_', ' ').title())
        ax.set_title(f'{focal_variable.replace("_", " ").title()} by {cat_variable.replace("_", " ").title()}')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Remove any unused axes (in case the number of focal variables is less than the grid size)
    for i in range(n_focal, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()


def plot_categories_comparison(df, cat_variable, focal_variables, max_columns=4, figsize=(10, 10), title_distance=0.925):
    """
    Plot comparison graphs of a target variable across focal variables.
    
    Parameters:
    - df: DataFrame containing the data to plot.
    - focal_variables: List of focal variables (columns) to compare.
    - cat_variable: List containing the categorical variable(s) for comparison.
    - max_columns: Maximum number of columns for the plot layout (default is 4).
    - figsize: Size of the entire figure (default is (10, 10)).
    - title_distance: Distance between the figure title and the subplots (default is 0.925).
    """
    
    if not isinstance(cat_variable, list):
        cat_variable = [cat_variable]

    
    data_length = len(focal_variables)  # Length of your focal variables list
    
    # Calculate the number of rows and columns for the subplots
    ncols = min(data_length, max_columns)  # Ensure we don't exceed the max columns
    nrows = math.ceil(data_length / ncols)  # Calculate rows based on the number of columns
    
    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    
    for i, coluna in enumerate(focal_variables):
        h = sns.histplot(x=coluna, hue=cat_variable[0], data=df, multiple='fill', ax=axs.flat[i], stat='percent',
                         shrink=0.8)
        h.tick_params(axis='x', labelrotation=45)
        h.grid(False)
    
        h.yaxis.set_major_formatter(PercentFormatter(1))  # Format y-axis as percentage
        h.set_ylabel('')  # Remove y-label for cleaner layout
    
        # Add percentage labels on bars
        for bar in h.containers:
            h.bar_label(bar, label_type='center', labels=[f'{b.get_height():.1%}' for b in bar], color='white', weight='bold', fontsize=11)
    
        # Remove legend from each subplot
        legend = h.get_legend()
        legend.remove()
    
    # Collect labels from the legend to display it outside the subplots
    labels = [text.get_text() for text in legend.get_texts()]
    
    # Create a single legend for the entire figure
    fig.legend(handles=legend.legend_handles, labels=labels, loc='upper center', ncols=2, title=cat_variable[0], bbox_to_anchor=(0.5, 0.965))
    
    # Title for the whole figure
    fig.suptitle(f'{cat_variable[0]} por variável categórica', fontsize=16)
    
    # Adjust labels and layout spacing, including distance from title
    fig.align_labels()
    plt.subplots_adjust(wspace=0.4, hspace=0.4, top=title_distance)
    
    # Show the plot
    plt.show()


def check_columns_in_feature_dict(df, feature_dict):
    """
    Check if all columns were included in the feature_dict and compare with df columns.
    
    Args:
        df (DataFrame): The pandas DataFrame containing the data.
        feature_dict (dict): A dictionary where values contain lists of features.
    """
    # Flatten the features list from feature_dict
    all_features = [item for sublist in feature_dict.values() for item in sublist]

    # Check for duplicates in the feature_dict
    duplicates = [feature for feature in set(all_features) if all_features.count(feature) > 1]

    if duplicates:
        print("Warning: Duplicate features found in feature_dict")
        print("Duplicated features:", duplicates)

    elif sorted(df.columns) == sorted(all_features):
        print('All good in feature_dict')

    else:
        columns_set = set(df.columns)
        features_set = set(all_features)
        missing_in_df = features_set - columns_set  # Features in feature_dict but not in df.columns
        extra_in_df = columns_set - features_set   # Columns in df.columns but not in feature_dict
        print("Missing in DataFrame:", missing_in_df)
        print("Extra in DataFrame:", extra_in_df)

