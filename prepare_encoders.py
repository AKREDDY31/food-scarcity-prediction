import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset (Ensure dataset.csv is available in the project directory)
df = pd.read_csv("dataset.csv")

# Define categorical and numerical columns
categorical_cols = ['region', 'crop_type', 'irrigation', 'soil_quality', 'harvest_period']

# Ensure categorical columns are strings and convert to lowercase (to avoid mismatches)
df[categorical_cols] = df[categorical_cols].astype(str).apply(lambda x: x.str.lower().str.strip())

# Identify numerical columns dynamically
excluded_cols = categorical_cols + ["category", "forecast_yield"]
numerical_cols = [col for col in df.columns if col not in excluded_cols]

# Apply Label Encoding to categorical columns and save encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

    # Save LabelEncoder for later use
    with open(f"{col}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

# Standardize numerical columns and save scaler
scaler = StandardScaler()
df_numerical_scaled = scaler.fit_transform(df[numerical_cols])

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Encoders and scaler saved successfully!")
