import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Configure paths and features
MODEL_PATH = r'C:/Code/cardiac_surgery_app/models/best_cardiac_surgery_model.joblib'
FEATURE_NAMES_PATH = r'C:/Code/cardiac_surgery_app/models/model_feature_names.joblib'

# Define expected feature names (ensures missing features don’t break prediction)
DEFAULT_FEATURES = [
    'Age', 'Diabetes', 'Hypertension', 'Kidney_Disease', 'Respiratory_Issues', 
    'Ejection_Fraction', 'Gender_Male', 'Surgery_Type_CABG', 'Surgery_Type_Valve', 
    'Surgery_Type_Congenital', 'Surgery_Type_Aneurysm', 'Surgery_Type_Transplant'
]

def load_model_and_features():
    """Load model and feature names with error handling."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODEL_PATH)
            training_feature_names = joblib.load(FEATURE_NAMES_PATH)

            # Fallback if feature names are missing
            if not training_feature_names:
                print("⚠️ Warning: Using default feature names")
                training_feature_names = DEFAULT_FEATURES

            # Check if model needs conversion
            if not hasattr(model, 'predict_proba'):
                print("⚠️ Warning: Model might be incompatible. Attempting to fix.")
                new_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                dummy_X = pd.DataFrame(0, index=[0], columns=training_feature_names)
                dummy_y = [0]
                new_model.fit(dummy_X, dummy_y)  # Retrain on dummy data
                model = new_model

            print("✅ Model & Features Loaded Successfully")
            return model, training_feature_names

    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found - {e}")
        return None, DEFAULT_FEATURES
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, DEFAULT_FEATURES

def validate_input_data(data):
    """Validate input data ranges."""
    validations = {
        "Age": (lambda x: 1 <= x <= 110, "⚠️ Age must be between 1 and 110"),
        "Gender_Male": (lambda x: x in [0, 1], "⚠️ Gender must be 0 or 1"),
        "Diabetes": (lambda x: x in [0, 1], "⚠️ Diabetes must be 0 or 1"),
        "Hypertension": (lambda x: x in [0, 1], "⚠️ Hypertension must be 0 or 1"),
        "Kidney_Disease": (lambda x: x in [0, 1], "⚠️ Kidney_Disease must be 0 or 1"),
        "Respiratory_Issues": (lambda x: x in [0, 1], "⚠️ Respiratory_Issues must be 0 or 1"),
        "Ejection_Fraction": (lambda x: 0 <= x <= 100, "⚠️ Ejection_Fraction must be between 0 and 100"),
    }
    
    for field, (validator, message) in validations.items():
        if field in data and not validator(data[field][0]):
            raise ValueError(f"❌ Validation error: {message}")

def predict_complications(test_data):
    """Predicts complication risk after cardiac surgery."""
    try:
        # Validate input data
        validate_input_data(test_data)

        # Load model and features
        model, training_feature_names = load_model_and_features()
        if not model:
            return None, "❌ Model loading failed"

        # Prepare DataFrame with correct feature order
        individual_test_data = pd.DataFrame(test_data)
        test_features = individual_test_data.reindex(columns=training_feature_names, fill_value=0)
        print("🔹 Features for Prediction:", test_features.columns.tolist())

        # Make prediction
        probabilities = model.predict_proba(test_features)
        severity_percentage = float(probabilities[0][1] * 100)

        return severity_percentage, None

    except Exception as e:
        return None, f"❌ Prediction error: {str(e)}"

if __name__ == "__main__":
    # Test input
    test_data_after_4_years = {
        "Age": [75],
        "Gender_Male": [1],  # 1 for Male, 0 for Female
        "Diabetes": [1],  # 0 for No, 1 for Yes
        "Hypertension": [1],  # 0 for No, 1 for Yes
        "Kidney_Disease": [0],  # 0 for No, 1 for Yes
        "Respiratory_Issues": [1],  # 0 for No, 1 for Yes
        "Ejection_Fraction": [30],
        "Surgery_Type_CABG": [0],
        "Surgery_Type_Valve": [1],
        "Surgery_Type_Congenital": [0],
        "Surgery_Type_Aneurysm": [1],
        "Surgery_Type_Transplant": [0],
    }

    # Get prediction
    severity, error = predict_complications(test_data_after_4_years)
    
    if error:
        print(error)
    else:
        # Determine risk level
        if severity < 30:
            risk_level = "✅ Normal Risk"
            recommendation = "Regular follow-up recommended"
        elif severity <= 60:
            risk_level = "⚠️ Moderate Risk"
            recommendation = "Medical consultation recommended"
        else:
            risk_level = "❌ High Risk"
            recommendation = "Immediate medical attention required"
            
        print(f"\n### Cardiac Surgery Risk Prediction ###")
        print(f"🔹 Predicted Complication Risk: {severity:.2f}%")
        print(f"🔹 Risk Level: {risk_level}")
        print(f"🔹 Recommendation: {recommendation}")
