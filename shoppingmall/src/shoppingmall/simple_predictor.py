#!/usr/bin/env python3
"""
Simple Command-Line Esophageal Cancer Predictor
Usage: python simple_predictor.py --age 55 --height 170 --weight 70 --consent YES
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys

def load_model(model_path='esophageal_cancer_model.pkl'):
    """Load the trained model"""
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def make_prediction(model_data, patient_data):
    """Make prediction using the model"""
    try:
        # Extract model components
        clf = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        num_imputer = model_data['num_imputer']
        cat_imputer = model_data['cat_imputer']
        onehot_encoder = model_data['onehot_encoder']
        num_cols = model_data['num_cols']
        cat_cols = model_data['cat_cols']
        
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Ensure all required columns are present
        for col in num_cols + cat_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Process data
        X_num = df[num_cols]
        X_cat = df[cat_cols]
        
        X_num_imputed = num_imputer.transform(X_num)
        X_cat_imputed = cat_imputer.transform(X_cat)
        
        if onehot_encoder is not None and len(cat_cols) > 0:
            X_cat_encoded = onehot_encoder.transform(X_cat_imputed)
            X_processed = np.hstack((X_num_imputed, X_cat_encoded))
        else:
            X_processed = X_num_imputed
        
        X_scaled = scaler.transform(X_processed)
        
        # Make prediction
        prediction_encoded = clf.predict(X_scaled)[0]
        probabilities = clf.predict_proba(X_scaled)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Esophageal Cancer Prediction Tool')
    
    # Common arguments that users might have
    parser.add_argument('--age', type=float, help='Patient age in years (e.g., 55)')
    parser.add_argument('--height', type=float, help='Height in cm (e.g., 170)')
    parser.add_argument('--weight', type=float, help='Weight in kg (e.g., 70)')
    parser.add_argument('--consent', type=str, help='Informed consent verified (YES/NO)')
    parser.add_argument('--site', type=str, help='Cancer site code (e.g., C15.9)')
    parser.add_argument('--histology', type=str, help='Histology code (e.g., 8140/3)')
    parser.add_argument('--tumor-status', type=str, help='Tumor status')
    
    # Advanced options
    parser.add_argument('--model', type=str, default='esophageal_cancer_model.pkl',
                       help='Path to model file (default: esophageal_cancer_model.pkl)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model_data = load_model(args.model)
    print("✅ Model loaded successfully!")
    
    if args.interactive:
        print("\\n🏥 Interactive Mode")
        print("Enter patient information (press Enter to skip):")
        
        patient_data = {}
        
        # Get essential information interactively
        age = input("Age in years: ").strip()
        if age:
            patient_data['days_to_birth'] = -float(age) * 365.25
        
        height = input("Height in cm: ").strip()
        if height:
            patient_data['height'] = float(height)
        
        weight = input("Weight in kg: ").strip()
        if weight:
            patient_data['weight'] = float(weight)
        
        consent = input("Informed consent verified (YES/NO): ").strip()
        if consent:
            patient_data['informed_consent_verified'] = consent
        
        site = input("Cancer site code: ").strip()
        if site:
            patient_data['icd_o_3_site'] = site
        
        histology = input("Histology code: ").strip()
        if histology:
            patient_data['icd_o_3_histology'] = histology
    
    else:
        # Use command line arguments
        patient_data = {}
        
        if args.age:
            patient_data['days_to_birth'] = -args.age * 365.25
        
        if args.height:
            patient_data['height'] = args.height
        
        if args.weight:
            patient_data['weight'] = args.weight
        
        if args.consent:
            patient_data['informed_consent_verified'] = args.consent
        
        if args.site:
            patient_data['icd_o_3_site'] = args.site
        
        if args.histology:
            patient_data['icd_o_3_histology'] = args.histology
        
        if args.tumor_status:
            patient_data['tumor_status'] = args.tumor_status
    
    # Make prediction
    if patient_data:
        print(f"\\nMaking prediction with provided data...")
        prediction, confidence, probabilities = make_prediction(model_data, patient_data)
        
        if prediction is not None:
            print(f"\\n🎯 PREDICTION RESULTS")
            print(f"=" * 30)
            print(f"Vital Status: {prediction}")
            print(f"Confidence: {confidence:.1%}")
            
            if len(probabilities) == 2:
                classes = model_data['label_encoder'].classes_
                print(f"\\nDetailed Probabilities:")
                for i, class_name in enumerate(classes):
                    print(f"  {class_name}: {probabilities[i]:.1%}")
            
            # Add interpretation
            if confidence >= 0.8:
                print(f"\\n🔴 High confidence prediction" if prediction == 'Dead' else "\\n🟢 High confidence prediction")
            elif confidence >= 0.6:
                print(f"\\n🟡 Medium confidence prediction")
            else:
                print(f"\\n⚪ Low confidence - consider gathering more information")
        
    else:
        print("\\n❌ No patient data provided!")
        print("\\nUsage examples:")
        print("  python simple_predictor.py --age 55 --height 170 --weight 70 --consent YES")
        print("  python simple_predictor.py --interactive")
        print("  python simple_predictor.py --help")

if __name__ == "__main__":
    main()
