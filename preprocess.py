import pandas as pd
import numpy as np
from datetime import datetime


def preprocess_data(df):

    date_cols = [
        'Date_birth',
        'Date_driving_licence',
        'Date_start_contract'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            dayfirst=True,
            errors='coerce'
        )

    current_year = datetime.now().year

    df['Driver_Age'] = (
        current_year -
        df['Date_birth'].dt.year
    )

    df['Driving_Experience'] = (
        current_year -
        df['Date_driving_licence'].dt.year
    )

    df['Vehicle_Age'] = (
        current_year -
        df['Year_matriculation']
    )

    df['Length'] = df['Length'].replace(
        'NA',
        np.nan
    )

    numeric_cols = [
        'Length',
        'Premium',
        'Value_vehicle',
        'Cost_claims_year'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # Fill numeric columns with 0
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Fill text columns with empty string
    text_cols = df.select_dtypes(exclude=np.number).columns
    df[text_cols] = df[text_cols].fillna('')

    df['Type_fuel'] = df['Type_fuel'].map({
        'P': 0,
        'D': 1
    })

    # Use only non-zero claims
    claims_df = df[
        df['Cost_claims_year'] > 0
    ]

    cost_threshold = claims_df[
        'Cost_claims_year'
    ].quantile(0.75)

    # Auto approval rule without R_Claims_history and Lapse
    df['Auto_Approve'] = np.where(
        (
            df['Cost_claims_year']
            <= cost_threshold
        ) &
        (
            df['N_claims_year']
            <= 2
        ),
        1,
        0
    )

    print("\nThresholds Used:")
    print(f"Cost Threshold: {cost_threshold:.2f}")

    print("\nTarget Distribution:")
    print(
        df['Auto_Approve']
        .value_counts(normalize=True) * 100
    )

    return df