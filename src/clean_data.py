# src/clean_data.py

import pandas as pd
import os

def load_and_clean_data(
    input_path='../data/raw/MachineLearningRating_v3.txt',
    output_path='../data/processed/insurance_data_cleaned.csv'
):
    print(f"Checking if input file exists at: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read data with pipe separator
    df = pd.read_csv(input_path, sep='|')
    print("Data loaded successfully.")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"Data shape after dropping duplicates: {df.shape}")

    # Convert TransactionMonth to datetime if exists
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        print("Converted 'TransactionMonth' to datetime.")

    # Convert specified columns to categorical if present
    categorical_cols = ['Gender', 'Province', 'VehicleType', 'VehicleMake', 'CoverType']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to categorical.")

    # Drop rows missing TotalClaims or TotalPremium
    df = df.dropna(subset=['TotalClaims', 'TotalPremium'])
    print(f"Data shape after dropping rows with missing 'TotalClaims' or 'TotalPremium': {df.shape}")

    # Fill missing CustomValueEstimate with median
    if 'CustomValueEstimate' in df.columns:
        median_val = df['CustomValueEstimate'].median()
        df['CustomValueEstimate'] = df['CustomValueEstimate'].fillna(median_val)
        print(f"Filled missing 'CustomValueEstimate' with median value {median_val}.")

    # Create ClaimFrequency: 1 if TotalClaims > 0 else 0
    df['ClaimFrequency'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)
    print("Created 'ClaimFrequency' column.")

    # Create ClaimCount if missing, same as ClaimFrequency
    if 'ClaimCount' not in df.columns:
        df['ClaimCount'] = df['ClaimFrequency']
        print("Created 'ClaimCount' column from 'ClaimFrequency'.")

    # Create Severity = TotalClaims / ClaimCount if ClaimCount > 0 else 0
    df['Severity'] = df.apply(lambda row: row['TotalClaims'] / row['ClaimCount'] if row['ClaimCount'] > 0 else 0, axis=1)
    print("Created 'Severity' column.")

    # Create Margin = TotalPremium - TotalClaims
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    print("Created 'Margin' column.")

    # Save cleaned data to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    return df

if __name__ == "__main__":
    load_and_clean_data()
