# main.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import os

from src.data_preprocessing import load_and_preprocess

# Load and preprocess dataset
data_path = "data/student-mat.csv"
X_train, X_test, y_train, y_test, full_feature_columns, scaler = load_and_preprocess(data_path)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model and tools
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(full_feature_columns, "models/feature_columns.pkl")
