import pandas as pd
import numpy as np
import joblib

# Load the complete model pipeline
print("Loading the trained model...")
model_data = joblib.load('esophageal_cancer_model.pkl')

# Extract components
clf = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
num_imputer = model_data['num_imputer']
cat_imputer = model_data['cat_imputer']
onehot_encoder = model_data['onehot_encoder']
num_cols = model_data['num_cols']
cat_cols = model_data['cat_cols']
feature_names = model_data['feature_names']

print("Model loaded successfully!")
print(f"Model type: {type(clf).__name__}")
print(f"Number of features: {len(feature_names)}")
print(f"Numeric columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

# Function to make predictions on new data
def predict_vital_status(new_data_df):
    """
    Make predictions on new data using the trained model
    
    Parameters:
    new_data_df: pandas DataFrame with the same structure as training data
    
    Returns:
    predictions: array of predictions (0=Alive, 1=Dead)
    probabilities: array of prediction probabilities
    """
    
    # Separate numeric and categorical features
    X_num_new = new_data_df[num_cols]
    X_cat_new = new_data_df[cat_cols]
    
    # Handle missing values
    X_num_imputed = num_imputer.transform(X_num_new)
    X_cat_imputed = cat_imputer.transform(X_cat_new)
    
    # One-hot encode categorical features
    if onehot_encoder is not None and len(cat_cols) > 0:
        X_cat_encoded = onehot_encoder.transform(X_cat_imputed)
        X_processed = np.hstack((X_num_imputed, X_cat_encoded))
    else:
        X_processed = X_num_imputed
    
    # Scale features
    X_scaled = scaler.transform(X_processed)
    
    # Make predictions
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)
    
    # Convert back to original labels
    original_predictions = label_encoder.inverse_transform(predictions)
    
    return original_predictions, probabilities

print("\nModel is ready for predictions!")
print("Use the 'predict_vital_status(new_data_df)' function to make predictions on new data.")

# Display model performance metrics
print(f"\nModel Information:")
print(f"- Random Forest with {clf.n_estimators} trees")
print(f"- Trained on {len(feature_names)} features")
print(f"- Classes: {label_encoder.classes_}")
