import pandas as pd
import numpy as np
import os
from datetime import datetime

# Define dataset folder path
dataset_path = r"C:\Code\cardiac_surgery_app\datasets"

# Ensure dataset folder exists
os.makedirs(dataset_path, exist_ok=True)

# Generate unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"cardiac_surgery_data_{timestamp}.csv"
file_path = os.path.join(dataset_path, file_name)

# Function to generate synthetic data
def generate_data(num_samples):
    np.random.seed(42)

    # Demographics
    age = np.random.randint(30, 80, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples)

    # Medical history
    diabetes = np.round(np.random.uniform(70, 136, size=num_samples)).astype(int)
    hypertension = np.round(np.random.uniform(90, 180, size=num_samples)).astype(int)
    kidney_disease = np.round(np.random.uniform(10, 90, size=num_samples)).astype(int)
    respiratory_issues = np.round(np.random.uniform(1, 100, size=num_samples)).astype(int)

    # Cardiac function
    ejection_fraction = np.random.randint(30, 70, size=num_samples)

    # Type of surgery
    surgery_type = np.random.choice(
        ["CABG", "Valve", "Congenital", "Aneurysm", "Transplant"], size=num_samples
    )

    # Complications within 4 years post-surgery (1 for complications, 0 for no complications)
    complications_4_years = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "Kidney_Disease": kidney_disease,
            "Respiratory_Issues": respiratory_issues,
            "Ejection_Fraction": ejection_fraction,
            "Surgery_Type": surgery_type,
            "Complications_4_Years_Post_Surgery": complications_4_years,
        }
    )

    # Convert categorical variables into numerical using one-hot encoding
    data = pd.get_dummies(data, columns=["Gender", "Surgery_Type"], drop_first=True)

    return data

# Generate dataset
main_dataset = generate_data(5000)

# Save dataset with timestamp
main_dataset.to_csv(file_path, index=False)

print(f"Dataset saved successfully as: {file_name}")

def generate_data(num_samples):
    np.random.seed(42)

    # Demographics
    age = np.random.randint(30, 85, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples)

    # Medical history with real-world risk correlations
    diabetes = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])  # 30% have diabetes
    hypertension = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])  # 40% have hypertension
    kidney_disease = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])  # 15% have kidney disease
    respiratory_issues = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])  # 10% have respiratory issues

    # Cardiac function (Lower ejection fraction = higher risk)
    ejection_fraction = np.random.normal(50, 10, size=num_samples).astype(int)
    ejection_fraction = np.clip(ejection_fraction, 30, 70)

    # Surgery Type (Real-world probability distribution)
    surgery_type = np.random.choice(
        ["CABG", "Valve", "Congenital", "Aneurysm", "Transplant"],
        size=num_samples,
        p=[0.5, 0.3, 0.1, 0.05, 0.05]  # CABG is more common
    )

    # Risk of complications based on actual conditions
    base_risk = 0.1  # Baseline risk of 10%
    base_risk += 0.2 * diabetes + 0.15 * hypertension + 0.25 * (ejection_fraction < 40)

    # Assign complications using probability
    complications_4_years = np.random.choice([0, 1], size=num_samples, p=[1 - base_risk, base_risk])

    # Create DataFrame
    data = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Kidney_Disease": kidney_disease,
        "Respiratory_Issues": respiratory_issues,
        "Ejection_Fraction": ejection_fraction,
        "Surgery_Type": surgery_type,
        "Complications_4_Years_Post_Surgery": complications_4_years,
    })

    # Convert categorical variables into numerical using one-hot encoding
    data = pd.get_dummies(data, columns=["Gender", "Surgery_Type"], drop_first=True)

    return data
