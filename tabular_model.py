import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_DIR = "data"
MODELS_DIR = "models"
TABULAR_FILE = os.path.join(DATA_DIR, "welding_data.csv")
MODEL_STRENGTH_PATH = os.path.join(MODELS_DIR, "rf_strength_model.pkl")
MODEL_PASSFAIL_PATH = os.path.join(MODELS_DIR, "rf_passfail_model.pkl")
MODEL_DEFECT_PATH = os.path.join(MODELS_DIR, "rf_defect_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)

def train_tabular_models():
    print("Loading data...")
    if not os.path.exists(TABULAR_FILE):
        print(f"Error: {TABULAR_FILE} not found. Run data_gen.py first.")
        return

    df = pd.read_csv(TABULAR_FILE)
    
    # Features and Targets
    feature_cols = ["Current", "Voltage", "TravelSpeed", "WireFeedRate", "GasFlow", "Temperature", "PlateThickness", "MaterialType"]
    
    # Preprocessing
    le_material = LabelEncoder()
    df['MaterialType'] = le_material.fit_transform(df['MaterialType'])
    
    le_defect = LabelEncoder()
    df['DefectType_Encoded'] = le_defect.fit_transform(df['DefectType'])

    X = df[feature_cols]
    y_strength = df['TensileStrength']
    y_passfail = df['Passed']
    y_defect = df['DefectType_Encoded']

    # Split data
    X_train, X_test, y_str_train, y_str_test, y_pf_train, y_pf_test, y_def_train, y_def_test = train_test_split(
        X, y_strength, y_passfail, y_defect, test_size=0.2, random_state=42
    )

    # --- 1. Tensile Strength Model (Regression) ---
    print("Training Tensile Strength Model...")
    rf_strength = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_strength.fit(X_train, y_str_train)
    
    y_str_pred = rf_strength.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_str_test, y_str_pred))
    print(f"Tensile Strength RMSE: {rmse:.2f}")

    # --- 2. Pass/Fail Model (Classification) ---
    print("Training Pass/Fail Model...")
    rf_passfail = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_passfail.fit(X_train, y_pf_train)
    
    y_pf_pred = rf_passfail.predict(X_test)
    acc_pf = accuracy_score(y_pf_test, y_pf_pred)
    print(f"Pass/Fail Accuracy: {acc_pf:.2f}")

    # --- 3. Defect Type Model (Classification) ---
    print("Training Defect Type Model...")
    rf_defect = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_defect.fit(X_train, y_def_train)
    
    y_def_pred = rf_defect.predict(X_test)
    y_def_pred = rf_defect.predict(X_test)
    print("Defect Classification Report:")
    target_names = [str(c) for c in le_defect.classes_]
    print(classification_report(y_def_test, y_def_pred, target_names=target_names))

    # --- Save Models ---
    print("Saving models...")
    with open(MODEL_STRENGTH_PATH, 'wb') as f:
        pickle.dump(rf_strength, f)
        
    with open(MODEL_PASSFAIL_PATH, 'wb') as f:
        pickle.dump(rf_passfail, f)
        
    with open(MODEL_DEFECT_PATH, 'wb') as f:
        pickle.dump(rf_defect, f)

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump({'MaterialType': le_material, 'DefectType': le_defect}, f)

    print("All tabular models trained and saved.")

if __name__ == "__main__":
    train_tabular_models()
