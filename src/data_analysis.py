import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    missing_percentage = df.isnull().mean()
    
    # Identify columns to drop based on missing percentage
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    cols_to_drop = cols_to_drop.append(missing_percentage[missing_percentage == 1.0].index)
    
    # Drop the identified columns
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns with >{threshold*100}% or 100% missing values: {list(cols_to_drop)}")
    
    # Fill missing values for categorical and numerical columns
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:  # Categorical Columns
            df[col] = df[col].fillna(df[col].mode()[0]) if df[col].isna().sum() > 0 else df[col]
        elif df[col].dtype in ['int64', 'float64']:  # Numerical Columns
            if df[col].isna().sum() > 0:
                fill_value = df[col].median() if abs(df[col].skew()) > 1 else df[col].mean()
                df[col] = df[col].fillna(fill_value)
    
    print("Filled missing values: Mode for categorical, Mean/Median for numerical columns.")
    return df
def handle_outliers(df: pd.DataFrame, preserve_fraud: bool = True) -> pd.DataFrame:
    """
    Handles outliers by capping them using the IQR method.
    Preserves fraud cases if `preserve_fraud` is True.
    
    :param df: Input DataFrame
    :param preserve_fraud: If True, preserves all fraud cases (FraudResult = 1)
    :return: DataFrame with outliers handled
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = df.copy()
    
    # Separate fraud and non-fraud data
    if preserve_fraud and 'FraudResult' in df_cleaned.columns:
        fraud_data = df_cleaned[df_cleaned['FraudResult'] == 1].copy()
        non_fraud_data = df_cleaned[df_cleaned['FraudResult'] == 0].copy()
    else:
        non_fraud_data = df_cleaned.copy()
    
    # Handle outliers in non-fraud data
    numeric_cols = non_fraud_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = non_fraud_data[col].quantile(0.25)
        Q3 = non_fraud_data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap values outside the IQR bounds
        non_fraud_data[col] = non_fraud_data[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Combine fraud and non-fraud data back together
    if preserve_fraud and 'FraudResult' in df_cleaned.columns:
        df_cleaned = pd.concat([fraud_data, non_fraud_data], axis=0)
    else:
        df_cleaned = non_fraud_data
    
    # Reset the index to maintain continuity
    df_cleaned.reset_index(drop=True, inplace=True)
    
    print("Outliers handled using IQR method while preserving fraud cases.")
    return df_cleaned


def summary_statistics(df: pd.DataFrame):
    print("Summary Statistics:")
    summary = df.describe(include='all')
    print(summary)
    return summary

def distribution_numerical(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    # Define a color palette to use for the plots
    color_palette = sns.color_palette("viridis", len(num_cols))

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=color_palette[i], bins=20, edgecolor='black')
        axes[i].set_title(f"Distribution of {col}", fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    plt.suptitle("Distribution of Numerical Features", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.show()

def distribution_categorical(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    print("Distribution of Categorical Features:")
    for col in cat_cols:
        print(f"\n{col} - Frequency Distribution:")
        print(df[col].value_counts())
        print("-" * 40)
def correlation_analysis(df: pd.DataFrame):
    # Select only numeric columns for correlation analysis
    num_cols = df.select_dtypes(include=[np.number])
    
    # Check if there are numeric columns to analyze
    if num_cols.empty:
        print("No numeric columns found for correlation analysis.")
        return

    correlation_matrix = num_cols.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
    plt.show()



def missing_value_detection(df: pd.DataFrame):
    """
    Detects and returns a summary of missing values in the dataset.
    Also prints a table summarizing missing values for each column.
    """
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage Missing': missing_percentage
    })
    
    print("Missing Values Summary:")
    print(missing_df[missing_df['Missing Values'] > 0])
    
    return missing_df[missing_df['Missing Values'] > 0]

def outlier_detection(df: pd.DataFrame, threshold: float = 1.5):
    """
    Detects and visualizes outliers using the IQR method with box plots in subplots.
    
    :param df: DataFrame with numerical columns
    :param threshold: Multiplier for the IQR to detect outliers (default 1.5)
    :return: A DataFrame summarizing the outlier information for each column
    """
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    num_plots = len(numeric_cols)
    rows = (num_plots // 3) + (num_plots % 3 > 0)
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {'Total Outliers': len(outliers), 'Lower Bound': lower_bound, 'Upper Bound': upper_bound}
        
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color='lightblue')
        axes[i].set_title(f'Outlier Detection for {col}')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame.from_dict(outlier_info, orient='index')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

