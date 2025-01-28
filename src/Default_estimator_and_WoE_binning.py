# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Function to load data from a CSV file
def load_data(input_file_path):
    data = pd.read_csv(input_file_path)
    print("Loaded Dataset:")
    print(data.head())
    return data

# Function to calculate Recency, Frequency, Monetary, and Stability (RFMS)
def calculate_rfm(data):
    rfm = data.groupby('CustomerId').agg(
        Recency=('TransactionDay_woe', 'min'),  # Most recent day (lower = more recent)
        Frequency=('TransactionId', 'count'),  # Number of transactions
        Monetary=('Value_woe', 'sum'),         # Total value woe
        Stability=('Value_woe', 'std')         # Standard deviation of transaction values
    ).reset_index()

    # Handle missing values in Stability (customers with only one transaction)
    rfm['Stability'] = rfm['Stability'].fillna(0)
    print("Calculated RFMS:")
    print(rfm.head())

    # Normalize RFMS features
    scaler = MinMaxScaler()
    rfm[['Recency', 'Frequency', 'Monetary', 'Stability']] = scaler.fit_transform(
        rfm[['Recency', 'Frequency', 'Monetary', 'Stability']]
    )
    print("Normalized RFMS:")
    print(rfm.head())

    # Calculate RFMS Score
    weights = {'Recency': 0.3, 'Frequency': 0.2, 'Monetary': 0.3, 'Stability': 0.2}
    rfm['RFMS_Score'] = sum(rfm[feature] * weight for feature, weight in weights.items())

    # Assign Good/Bad labels based on RFMS Score
    threshold = rfm['RFMS_Score'].quantile(0.5)  # Use median as threshold
    rfm['Default'] = rfm['RFMS_Score'].apply(lambda x: 0 if x > threshold else 1)
    print("Good/Bad Labels Distribution:")
    print(rfm['Default'].value_counts())

    return rfm

# Function to calculate WoE and IV for a feature
def calculate_woe_iv(df, feature, target):
    # Bin the feature into 5 quantiles
    df['bins'] = pd.qcut(df[feature], q=5, duplicates='drop')

    # Calculate Goods (0) and Bads (1)
    woe_iv = df.groupby('bins', observed=True).agg(
        Goods=pd.NamedAgg(column=target, aggfunc=lambda x: (x == 0).sum()),
        Bads=pd.NamedAgg(column=target, aggfunc=lambda x: (x == 1).sum())
    ).reset_index()

    # Add small constant to avoid division by zero
    woe_iv['Goods'] = woe_iv['Goods'] + 1e-6
    woe_iv['Bads'] = woe_iv['Bads'] + 1e-6

    # Calculate percentages and WoE
    woe_iv['Percentage_Goods'] = woe_iv['Goods'] / woe_iv['Goods'].sum()
    woe_iv['Percentage_Bads'] = woe_iv['Bads'] / woe_iv['Bads'].sum()

    # Handle division by zero or log(0)
    woe_iv['WoE'] = np.where(
        (woe_iv['Percentage_Goods'] == 0) | (woe_iv['Percentage_Bads'] == 0),
        0,  # Replace infinite WoE with 0
        np.log(woe_iv['Percentage_Goods'] / woe_iv['Percentage_Bads'])
    )

    # Calculate IV
    woe_iv['IV'] = (woe_iv['Percentage_Goods'] - woe_iv['Percentage_Bads']) * woe_iv['WoE']

    return woe_iv

# Main execution
def Default_estimator_and_woe_binning(input_file_path, output_file_path):
    # Load data
    data = load_data(input_file_path)

    # Calculate RFM
    rfm = calculate_rfm(data)

    # Apply WoE transformation for RFMS features
    numerical_features = ['Recency', 'Frequency', 'Monetary', 'Stability']
    woe_mappings = {}

    for feature in numerical_features:
        # Calculate WoE and IV
        woe_iv = calculate_woe_iv(rfm, feature, 'Default')
        print(f"WoE and IV for {feature}:")
        print(woe_iv)

        # Map bins to WoE values
        woe_mapping = dict(zip(woe_iv['bins'], woe_iv['WoE']))
        woe_mappings[feature] = woe_mapping

        # Apply WoE mapping to the feature
        rfm[f"{feature}_WoE"] = rfm[feature].map(
            lambda x: next((woe_mapping[bin] for bin in woe_mapping if bin.left <= x <= bin.right), np.nan)
        )

    # Display the RFM table with WoE-transformed features
    print("RFM Table with WoE Features:")
    print(rfm[[f"{feature}_WoE" for feature in numerical_features]].head())

    # Save the final dataset with WoE-transformed features
    rfm.to_csv(output_file_path, index=False)
    print(f"Dataset saved to {output_file_path}")

    # Visualization: RFMS Score distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(rfm['RFMS_Score'], bins=30, kde=True, color='purple', edgecolor='black')
    plt.title('Distribution of RFMS Scores', fontsize=16)
    plt.xlabel('RFMS Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Visualization: WoE for Monetary (Subplots)
    woe_iv_monetary = calculate_woe_iv(rfm, 'Monetary', 'Default')
    woe_iv_monetary.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    # WoE barplot
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.barplot(x='bins', y='WoE', data=woe_iv_monetary, ax=axes[0], palette='viridis', hue=None)
    warnings.filterwarnings("ignore", category=FutureWarning)
    axes[0].set_title('Weight of Evidence (WoE) for Monetary', fontsize=16)
    axes[0].set_xlabel('Monetary Bins', fontsize=14)
    axes[0].set_ylabel('WoE', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)

    # IV pie chart
    axes[1].pie([woe_iv_monetary['IV'].sum(), 1 - woe_iv_monetary['IV'].sum()],
                labels=['Information Value', 'Remaining'],
                autopct='%1.1f%%', colors=['#4caf50', '#ff9800'], startangle=90, explode=[0.1, 0])
    axes[1].set_title('Information Value Contribution', fontsize=16)

    plt.tight_layout()
    plt.show()

