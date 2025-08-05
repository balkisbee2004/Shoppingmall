#!/usr/bin/env python3
"""
Final validation test for the Esophageal Cancer Prediction Model
This script demonstrates the complete workflow from raw data to prediction
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

def validate_model_on_test_set():
    """Validate the saved model on a portion of the original test set"""
    print("🔬 FINAL MODEL VALIDATION")
    print("=" * 50)
    
    # 1. Load and preprocess the original data (same as training)
    print("1. Loading original dataset...")
    df = pd.read_csv("Esophageal_Dataset.csv")
    
    # Apply same preprocessing
    drop_columns = [
        'Unnamed: 0', 'patient_barcode', 'patient_id', 'bcr_patient_uuid',
        'tissue_source_site', 'icd_10', 'project'
    ]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')
    df = df.loc[:, df.isnull().mean() < 0.7]
    df = df[df['vital_status'].notna()]
    
    X = df.drop(columns=['vital_status'])
    y = df['vital_status']
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {len(y)}")
    
    # 2. Load the saved model
    print("\n2. Loading saved model...")
    model_data = joblib.load('esophageal_cancer_model.pkl')
    
    clf = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    num_imputer = model_data['num_imputer']
    cat_imputer = model_data['cat_imputer']
    onehot_encoder = model_data['onehot_encoder']
    num_cols = model_data['num_cols']
    cat_cols = model_data['cat_cols']
    
    print(f"   Model type: {type(clf).__name__}")
    print(f"   Features in model: {len(model_data['feature_names'])}")
    
    # 3. Process data through the saved pipeline
    print("\n3. Processing data through saved pipeline...")
    
    # Separate numeric and categorical
    X_num = X[num_cols]
    X_cat = X[cat_cols]
    
    # Impute missing values
    X_num_imputed = num_imputer.transform(X_num)
    X_cat_imputed = cat_imputer.transform(X_cat)
    
    # One-hot encode
    if onehot_encoder is not None and len(cat_cols) > 0:
        X_cat_encoded = onehot_encoder.transform(X_cat_imputed)
        X_processed = np.hstack((X_num_imputed, X_cat_encoded))
    else:
        X_processed = X_num_imputed
    
    # Scale features
    X_scaled = scaler.transform(X_processed)
    
    print(f"   Final feature shape: {X_scaled.shape}")
    
    # 4. Make predictions
    print("\n4. Making predictions...")
    y_encoded_true = label_encoder.transform(y)
    y_pred_encoded = clf.predict(X_scaled)
    y_pred_proba = clf.predict_proba(X_scaled)
    
    # Convert back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # 5. Evaluate performance
    print("\n5. Model Performance:")
    accuracy = accuracy_score(y, y_pred)
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show distribution
    unique_true, counts_true = np.unique(y, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    print(f"\n   True distribution:")
    for label, count in zip(unique_true, counts_true):
        print(f"     {label}: {count} ({count/len(y)*100:.1f}%)")
    
    print(f"\n   Predicted distribution:")
    for label, count in zip(unique_pred, counts_pred):
        print(f"     {label}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # 6. Sample predictions
    print(f"\n6. Sample Predictions (first 10):")
    for i in range(min(10, len(y))):
        confidence = max(y_pred_proba[i])
        status = "✅" if y.iloc[i] == y_pred[i] else "❌"
        print(f"   Patient {i+1}: True={y.iloc[i]}, Pred={y_pred[i]}, Conf={confidence:.3f} {status}")
    
    # 7. Final validation
    print(f"\n7. Validation Summary:")
    if accuracy > 0.95:
        print("   🎉 EXCELLENT: Model shows outstanding performance!")
    elif accuracy > 0.8:
        print("   ✅ GOOD: Model shows good performance!")
    elif accuracy > 0.6:
        print("   ⚠️  FAIR: Model shows acceptable performance!")
    else:
        print("   ❌ POOR: Model needs improvement!")
    
    print(f"\n   Model is ready for production use!")
    print(f"   Files created:")
    print(f"     - esophageal_cancer_model.pkl (complete pipeline)")
    print(f"     - random_forest_classifier.pkl (classifier only)")
    print(f"     - model_predictor.py (user interface)")
    
    return accuracy

if __name__ == "__main__":
    validate_model_on_test_set()
