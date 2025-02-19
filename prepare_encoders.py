import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")  # Ensure this file is in your project

# Define categorical and numerical columns
categorical_cols = ['region', 'crop_type', 'irrigation', 'soil_quality', 'harvest_period']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["category", "forecast_yield"]]

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

    # Save LabelEncoder for later use
    with open(f"{col}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

# Standardize numerical columns
scaler = StandardScaler()
df_numerical = df[numerical_cols]
df_numerical_scaled = scaler.fit_transform(df_numerical)

# Save StandardScaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Encoders and scaler saved successfully!")
