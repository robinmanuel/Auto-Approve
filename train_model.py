import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
    random_state=42,
    stratify=y
)

# Train model
model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Save model
joblib.dump(model, 'model.pkl')

print("\nModel saved successfully.")