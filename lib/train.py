import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define dataset and model paths
model_path = r"C:\Code\cardiac_surgery_app\models"
data_path = r"C:\Code\cardiac_surgery_app\datasets"
files = ["main_cardiac_surgery_data.csv", "scenario_2_cardiac_surgery_data.csv", "scenario_3_cardiac_surgery_data.csv"]

# Load datasets
datasets = []
for file in files:
    file_path = os.path.join(data_path, file)
    if os.path.exists(file_path):
        datasets.append(pd.read_csv(file_path))
    else:
        print(f"Warning: {file_path} not found. Skipping.")

# Ensure at least one dataset is loaded
if not datasets:
    raise FileNotFoundError("No valid datasets found. Exiting.")

# Combine datasets
combined_dataset = pd.concat(datasets, ignore_index=True)

# Prepare features (X) and target variable (y)
X = combined_dataset.drop(columns=["Complications_4_Years_Post_Surgery"])
y = combined_dataset["Complications_4_Years_Post_Surgery"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.head())

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Train model
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)

# Expand hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300, 500],  # More trees = better predictions
    "max_depth": [None, 10, 20, 30],  # Allow deeper trees
    "min_samples_split": [2, 5, 10, 15],  # Minimum samples to split nodes
    "min_samples_leaf": [1, 2, 5, 10],  # Minimum samples in leaf nodes
}

# Use GridSearchCV with more cross-validation folds
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)


grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Ensure model directory exists
os.makedirs(model_path, exist_ok=True)

# Save model and feature names inside model folder
joblib.dump(best_model, os.path.join(model_path, "best_cardiac_surgery_model.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(model_path, "model_feature_names.joblib"))

# Evaluate model
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))
