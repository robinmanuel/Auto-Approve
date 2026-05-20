import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

from preprocess import preprocess_data

# Load dataset
df = pd.read_csv(
    'insurance_data.csv',
    sep=';'
)

# Preprocess
df = preprocess_data(df)

# Features
features = [
    'Seniority',
    'Policies_in_force',
    'Premium',
    'N_claims_history',
    'R_Claims_history',
    'Lapse',
    'Value_vehicle',
    'Power',
    'Driver_Age',
    'Driving_Experience',
    'Vehicle_Age'
]

X = df[features]

y = df['Auto_Approve']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier()

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

print(classification_report(y_test, preds))

# Save model
joblib.dump(model, 'model.pkl')

print("Model saved successfully.")