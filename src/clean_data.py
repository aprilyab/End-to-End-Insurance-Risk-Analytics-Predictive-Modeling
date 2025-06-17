# src/clean_data.py

import pandas as pd

def load_and_clean_data(input_path='data/raw/MachineLearningRating_v3.txt',
                        output_path='data/processed/insurance_data_cleaned.csv'):
    df = pd.read_csv(input_path)

    df.drop_duplicates(inplace=True)

    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    for col in ['Gender', 'Province', 'VehicleType', 'VehicleMake', 'CoverType']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    df = df.dropna(subset=['TotalClaims', 'TotalPremium'])

    df['CustomValueEstimate'] = df['CustomValueEstimate'].fillna(df['CustomValueEstimate'].median())

    df['ClaimFrequency'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)

    if 'ClaimCount' not in df.columns:
        df['ClaimCount'] = df['ClaimFrequency']

    df['Severity'] = df.apply(lambda row: row['TotalClaims']/row['ClaimCount']
                              if row['ClaimCount'] > 0 else 0, axis=1)

    df['Margin'] = df['TotalPremium'] - df['TotalClaims']

    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to {output_path}")

