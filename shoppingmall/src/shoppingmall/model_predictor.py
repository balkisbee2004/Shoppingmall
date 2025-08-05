import pandas as pd
import numpy as np
import joblib

def load_trained_model():
    """Load the trained esophageal cancer prediction model"""
    model_data = joblib.load('esophageal_cancer_model.pkl')
    return model_data

def make_prediction(patient_data, model_data):
    """
    Make a prediction for a patient's vital status
    
    Parameters:
    patient_data: dict or pandas DataFrame with patient information
    model_data: loaded model data from load_trained_model()
    
    Returns:
    prediction: 'Alive' or 'Dead'
    probability: confidence score (0-1)
    """
    
    # Extract model components
    clf = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    num_imputer = model_data['num_imputer']
    cat_imputer = model_data['cat_imputer']
    onehot_encoder = model_data['onehot_encoder']
    num_cols = model_data['num_cols']
    cat_cols = model_data['cat_cols']
    
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy()
    
    # Ensure all required columns are present
    for col in num_cols + cat_cols:
        if col not in df.columns:
            df[col] = np.nan  # Will be handled by imputers
    
    # Select only the columns used in training
    X_num = df[num_cols]
    X_cat = df[cat_cols]
    
    # Handle missing values
    X_num_imputed = num_imputer.transform(X_num)
    X_cat_imputed = cat_imputer.transform(X_cat)
    
    # One-hot encode categorical features
    if onehot_encoder is not None and len(cat_cols) > 0:
        X_cat_encoded = onehot_encoder.transform(X_cat_imputed)
        X_processed = np.hstack((X_num_imputed, X_cat_encoded))
    else:
        X_processed = X_num_imputed
    
    # Scale features
    X_scaled = scaler.transform(X_processed)
    
    # Make prediction
    prediction_encoded = clf.predict(X_scaled)[0]
    probabilities = clf.predict_proba(X_scaled)[0]
    
    # Convert back to original label
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = max(probabilities)
    
    return prediction, confidence

# Example usage
if __name__ == "__main__":
    print("Loading the esophageal cancer prediction model...")
    model = load_trained_model()
    print("Model loaded successfully!")
    
    print(f"\nModel Details:")
    print(f"- Number of features: {len(model['feature_names'])}")
    print(f"- Numeric features: {len(model['num_cols'])}")
    print(f"- Categorical features: {len(model['cat_cols'])}")
    print(f"- Model type: {type(model['model']).__name__}")
    
    print("\nThe model is ready to make predictions!")
    print("To use the model:")
    print("1. Load the model: model = load_trained_model()")
    print("2. Make prediction: prediction, confidence = make_prediction(patient_data, model)")
    print("\nExample patient_data format:")
    print("patient_data = {")
    for i, col in enumerate(model['num_cols'][:3]):  # Show first 3 numeric columns
        print(f"    '{col}': 65.5,")
    for i, col in enumerate(model['cat_cols'][:3]):  # Show first 3 categorical columns
        print(f"    '{col}': 'some_value',")
    print("    # ... include other relevant features")
    print("}")
