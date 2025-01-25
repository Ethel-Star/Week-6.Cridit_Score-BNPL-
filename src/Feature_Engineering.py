import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import scorecardpy as sc
import matplotlib.pyplot as plt
import warnings

# Suppress FutureWarnings and UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def feature_engineering(input_file, output_file):
    """
    Perform feature engineering on the input dataset and save the transformed dataset.
    
    Parameters:
        input_file (str): Path to the input CSV file (cleaned dataset).
        output_file (str): Path to save the feature-engineered dataset.
    """
    # Load the cleaned dataset
    df = pd.read_csv(input_file)
    print("Initial Dataset Sample:")
    print(df.head())  # Print first 5 rows of the initial dataset

    # Convert TransactionStartTime to datetime with explicit format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%dT%H:%M:%SZ')
    print("\nDataset after converting TransactionStartTime to datetime:")
    print(df.head())

    # Drop CurrencyCode if it has only one unique value
    if 'CurrencyCode' in df.columns and df['CurrencyCode'].nunique() == 1:
        df = df.drop(columns=['CurrencyCode'])
        print("\nDataset after dropping CurrencyCode:")
        print(df.head())

    # 1. Extract Time-Based Features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    df.drop(columns=['TransactionStartTime'], inplace=True)  # Drop redundant column
    print("\nDataset after extracting temporal features and dropping TransactionStartTime:")
    print(df.head())

    # 2. Encode Categorical Variables
    # One-Hot Encoding for CountryCode, ChannelId, and ProductCategory
    df = pd.get_dummies(df, columns=['CountryCode', 'ChannelId', 'ProductCategory'], drop_first=True)
    print("\nDataset after One-Hot Encoding:")
    print(df.head())

    # Label Encoding for ProviderId
    label_encoder = LabelEncoder()
    df['ProviderId_encoded'] = label_encoder.fit_transform(df['ProviderId'])
    print("\nDataset after Label Encoding ProviderId:")
    print(df.head())

    # 3. Normalize/Standardize Numerical Features
    # Standardization
    scaler = StandardScaler()
    df['Amount_standardized'] = scaler.fit_transform(df[['Amount']])

    # Normalization
    scaler = MinMaxScaler()
    df['Amount_normalized'] = scaler.fit_transform(df[['Amount']])
    print("\nDataset after Standardization and Normalization:")
    print(df.head())

    # 4. Feature Engineering Using Weight of Evidence (WoE) and Information Value (IV)
    # Check if the target column is binary
    if 'FraudResult' in df.columns:
        print("\nUnique values in FraudResult:", df['FraudResult'].unique())

        if df['FraudResult'].nunique() == 2:
            # Ensure the target column is numeric
            df['FraudResult'] = df['FraudResult'].astype(int)

            # Exclude high-cardinality or irrelevant columns
            exclude_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
            df_for_woe = df.drop(columns=exclude_cols)

            # Ensure no datetime columns are present in the dataset for WOE transformation
            datetime_cols = df_for_woe.select_dtypes(include=['datetime64']).columns.tolist()
            if datetime_cols:
                df_for_woe = df_for_woe.drop(columns=datetime_cols)

            # Calculate WOE bins
            bins = sc.woebin(df_for_woe, y='FraudResult')

            # Apply WOE transformation
            df_woe = sc.woebin_ply(df, bins)

            # Calculate Information Value (IV)
            iv_values = sc.iv(df_for_woe, y='FraudResult')
            print("\nInformation Values (IV):")
            print(iv_values)

            # Visualize IV values
            plt.figure(figsize=(10, 6))
            iv_values.sort_values(by='info_value', ascending=False).plot(kind='bar', color='skyblue')
            plt.title('Information Value (IV) of Features')
            plt.xlabel('Features')
            plt.ylabel('Information Value (IV)')
            plt.axhline(y=0.02, color='red', linestyle='--', label='Weak Predictive Power')
            plt.axhline(y=0.1, color='orange', linestyle='--', label='Moderate Predictive Power')
            plt.axhline(y=0.3, color='green', linestyle='--', label='Strong Predictive Power')
            plt.legend()
            plt.show()

            print("\nDataset after WOE Transformation:")
            print(df_woe.head())
        else:
            print("\nSkipping WoE transformation: Target column is not binary.")
            df_woe = df
    else:
        print("\nSkipping WoE transformation: Target column 'FraudResult' not found.")
        df_woe = df

    # 5. Save the Feature-Engineered Dataset
    df_woe.to_csv(output_file, index=False)
    print(f"\nFeature-engineered dataset saved to {output_file}")
