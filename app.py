from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load trained models
with open("classifier_model.pkl", "rb") as clf_file:
    classification_model = pickle.load(clf_file)

with open("regressor_model.pkl", "rb") as reg_file:
    regression_model = pickle.load(reg_file)

# Load encoders for categorical variables
categorical_cols = ['region', 'crop_type', 'irrigation', 'soil_quality', 'harvest_period']
label_encoders = {}

for col in categorical_cols:
    with open(f"{col}_encoder.pkl", "rb") as f:
        label_encoders[col] = pickle.load(f)

# Load StandardScaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Food Scarcity Prediction API is running!"})

@app.route("/categories", methods=["GET"])
def get_categories():
    """Returns available categories for debugging."""
    categories = {col: label_encoders[col].classes_.tolist() for col in categorical_cols}
    return jsonify(categories)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input

        # Convert categorical inputs to lowercase before encoding
        input_features = []
        for col in categorical_cols:
            user_input = data[col].strip().lower()  # Convert input to lowercase

            if user_input in label_encoders[col].classes_:
                input_features.append(label_encoders[col].transform([user_input])[0])
            else:
                available_values = label_encoders[col].classes_.tolist()
                return jsonify({
                    "error": f"Invalid value for {col}: {data[col]}",
                    "valid_options": available_values
                })

        # Convert and scale numerical inputs
        numerical_inputs = np.array(data["numerical_features"]).reshape(1, -1)
        numerical_scaled = scaler.transform(numerical_inputs)

        # Combine categorical + numerical features
        final_features = np.hstack([input_features, numerical_scaled])

        # Predict classification (Surplus/Shortage)
        category_pred = classification_model.predict([final_features])[0]
        category_result = "Surplus" if category_pred == 1 else "Shortage"

        # Predict shortage/surplus amount
        amount_pred = regression_model.predict([final_features])[0]

        return jsonify({
            "Scarcity Prediction": category_result,
            "Predicted Surplus/Shortage Amount (tons)": round(amount_pred, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
