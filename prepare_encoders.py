import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert categorical values to lowercase before encoding
categorical_cols = ['region', 'crop_type', 'irrigation', 'soil_quality', 'harvest_period']
df[categorical_cols] = df[categorical_cols].astype(str).apply(lambda x: x.str.lower().str.strip())

# Identify numerical columns dynamically
excluded_cols = categorical_cols + ["category", "forecast_yield"]
numerical_cols = [col for col in df.columns if col not in excluded_cols]

# Apply Label Encoding and save encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

    with open(f"{col}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

# Standardize numerical columns and save scaler
scaler = StandardScaler()
df_numerical_scaled = scaler.fit_transform(df[numerical_cols])

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Encoders and scaler saved successfully!")
