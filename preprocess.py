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
        'Cost_claims_year',
        'R_Claims_history'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df.fillna(0, inplace=True)


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

    claim_ratio_threshold = claims_df[
        'R_Claims_history'
    ].quantile(0.75)


    df['Auto_Approve'] = np.where(
        (
            df['Cost_claims_year']
            <= cost_threshold
        ) &
        (
            df['N_claims_year']
            <= 2
        ) &
        (
            df['R_Claims_history']
            <= claim_ratio_threshold
        ) &
        (
            df['Lapse'] == 0
        ),
        1,
        0
    )


    print("\nThresholds Used:")
    print(f"Cost Threshold: {cost_threshold:.2f}")
    print(f"Claim Ratio Threshold: {claim_ratio_threshold:.2f}")

    print("\nTarget Distribution:")
    print(
        df['Auto_Approve']
        .value_counts(normalize=True) * 100
    )

    return df