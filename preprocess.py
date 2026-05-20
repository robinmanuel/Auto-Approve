import pandas as pd
import numpy as np
from datetime import datetime


def preprocess_data(df):

    # Convert dates
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

    # Feature Engineering
    df['Driver_Age'] = (
        current_year - df['Date_birth'].dt.year
    )

    df['Driving_Experience'] = (
        current_year -
        df['Date_driving_licence'].dt.year
    )

    df['Vehicle_Age'] = (
        current_year - df['Year_matriculation']
    )

    # Replace missing values
    df['Length'] = df['Length'].replace('NA', np.nan)

    numeric_cols = [
        'Length',
        'Premium',
        'Value_vehicle'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df.fillna(0, inplace=True)

    # Encode fuel type
    df['Type_fuel'] = df['Type_fuel'].map({
        'P': 0,
        'D': 1
    })

    # Target Variable
    df['Auto_Approve'] = np.where(
        (df['N_claims_year'] <= 1) &
        (df['Cost_claims_year'] < 5000) &
        (df['Lapse'] == 0),
        1,
        0
    )

    return df