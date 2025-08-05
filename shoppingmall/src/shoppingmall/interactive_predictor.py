#!/usr/bin/env python3
"""
Esophageal Cancer Prediction Program
Interactive interface for predicting patient vital status
"""

import pandas as pd
import numpy as np
import joblib
import sys

class EsophagealCancerPredictor:
    def __init__(self, model_path='esophageal_cancer_model.pkl'):
        """Initialize the predictor with the saved model"""
        print("🏥 Esophageal Cancer Prediction System")
        print("=" * 50)
        print("Loading model...")
        
        try:
            self.model_data = joblib.load(model_path)
            self.clf = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.label_encoder = self.model_data['label_encoder']
            self.num_imputer = self.model_data['num_imputer']
            self.cat_imputer = self.model_data['cat_imputer']
            self.onehot_encoder = self.model_data['onehot_encoder']
            self.num_cols = self.model_data['num_cols']
            self.cat_cols = self.model_data['cat_cols']
            self.feature_names = self.model_data['feature_names']
            
            print("✅ Model loaded successfully!")
            print(f"   Model type: {type(self.clf).__name__}")
            print(f"   Total features: {len(self.feature_names)}")
            print(f"   Numeric features: {len(self.num_cols)}")
            print(f"   Categorical features: {len(self.cat_cols)}")
            print()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def get_feature_info(self):
        """Display information about required features"""
        print("📋 REQUIRED PATIENT INFORMATION")
        print("=" * 50)
        print("Numeric Features (enter numbers):")
        for i, col in enumerate(self.num_cols, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nCategorical Features (enter text):")
        for i, col in enumerate(self.cat_cols, 1):
            print(f"  {i:2d}. {col}")
        print()
    
    def get_user_input_interactive(self):
        """Get patient data interactively from user"""
        print("📝 ENTER PATIENT DATA")
        print("=" * 50)
        print("Enter patient information (press Enter to skip/use default):")
        print()
        
        patient_data = {}
        
        # Get numeric features
        print("NUMERIC FEATURES:")
        print("-" * 20)
        for col in self.num_cols:
            while True:
                try:
                    user_input = input(f"{col}: ").strip()
                    if user_input == "":
                        patient_data[col] = np.nan  # Will be handled by imputer
                        break
                    else:
                        patient_data[col] = float(user_input)
                        break
                except ValueError:
                    print("  ⚠️  Please enter a valid number or press Enter to skip")
        
        print("\nCATEGORICAL FEATURES:")
        print("-" * 20)
        for col in self.cat_cols:
            user_input = input(f"{col}: ").strip()
            if user_input == "":
                patient_data[col] = np.nan  # Will be handled by imputer
            else:
                patient_data[col] = user_input
        
        return patient_data
    
    def get_user_input_simple(self):
        """Get essential patient data with simplified input"""
        print("📝 SIMPLIFIED PATIENT DATA ENTRY")
        print("=" * 50)
        print("Enter the most important patient information:")
        print("(Press Enter to skip any field)")
        print()
        
        patient_data = {}
        
        # Initialize all features with NaN (will be imputed)
        for col in self.num_cols + self.cat_cols:
            patient_data[col] = np.nan
        
        # Essential numeric features
        essential_numeric = {
            'days_to_birth': 'Age in days (negative number, e.g., -20000 for ~55 years old)',
            'height': 'Height in cm (e.g., 170)',
            'weight': 'Weight in kg (e.g., 70)',
        }
        
        print("ESSENTIAL NUMERIC INFORMATION:")
        for col, description in essential_numeric.items():
            if col in self.num_cols:
                while True:
                    try:
                        user_input = input(f"{description}: ").strip()
                        if user_input == "":
                            break
                        else:
                            patient_data[col] = float(user_input)
                            break
                    except ValueError:
                        print("  ⚠️  Please enter a valid number or press Enter to skip")
        
        # Essential categorical features
        essential_categorical = {
            'informed_consent_verified': 'Informed consent verified (YES/NO)',
            'icd_o_3_site': 'Cancer site code (e.g., C15.9)',
            'icd_o_3_histology': 'Histology code (e.g., 8140/3)',
            'tumor_status': 'Tumor status',
            'vital_status': 'Current vital status (if known for validation)'
        }
        
        print("\nESSENTIAL CATEGORICAL INFORMATION:")
        for col, description in essential_categorical.items():
            if col in self.cat_cols:
                user_input = input(f"{description}: ").strip()
                if user_input != "":
                    patient_data[col] = user_input
        
        return patient_data
    
    def predict(self, patient_data):
        """Make prediction for given patient data"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Ensure all required columns are present
            for col in self.num_cols + self.cat_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Select and order columns as in training
            X_num = df[self.num_cols]
            X_cat = df[self.cat_cols]
            
            # Handle missing values
            X_num_imputed = self.num_imputer.transform(X_num)
            X_cat_imputed = self.cat_imputer.transform(X_cat)
            
            # One-hot encode categorical features
            if self.onehot_encoder is not None and len(self.cat_cols) > 0:
                X_cat_encoded = self.onehot_encoder.transform(X_cat_imputed)
                X_processed = np.hstack((X_num_imputed, X_cat_encoded))
            else:
                X_processed = X_num_imputed
            
            # Scale features
            X_scaled = self.scaler.transform(X_processed)
            
            # Make prediction
            prediction_encoded = self.clf.predict(X_scaled)[0]
            probabilities = self.clf.predict_proba(X_scaled)[0]
            
            # Convert back to original label
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = max(probabilities)
            
            # Get probabilities for each class
            alive_prob = probabilities[0] if self.label_encoder.classes_[0] == 'Alive' else probabilities[1]
            dead_prob = probabilities[1] if self.label_encoder.classes_[1] == 'Dead' else probabilities[0]
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'alive_probability': alive_prob,
                'dead_probability': dead_prob
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def display_results(self, results):
        """Display prediction results in a user-friendly format"""
        if results is None:
            return
        
        print("\n🎯 PREDICTION RESULTS")
        print("=" * 50)
        print(f"Predicted Vital Status: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.1%}")
        print()
        print("Detailed Probabilities:")
        print(f"  • Alive: {results['alive_probability']:.1%}")
        print(f"  • Dead:  {results['dead_probability']:.1%}")
        print()
        
        # Interpretation
        if results['confidence'] >= 0.8:
            confidence_level = "HIGH"
            emoji = "🔴" if results['prediction'] == 'Dead' else "🟢"
        elif results['confidence'] >= 0.6:
            confidence_level = "MEDIUM"
            emoji = "🟡"
        else:
            confidence_level = "LOW"
            emoji = "⚪"
        
        print(f"{emoji} Confidence Level: {confidence_level}")
        
        if results['confidence'] < 0.6:
            print("⚠️  Low confidence - consider gathering more patient information")
        
        print()
    
    def run_interactive(self):
        """Run the interactive prediction program"""
        while True:
            print("\n🏥 ESOPHAGEAL CANCER PREDICTION SYSTEM")
            print("=" * 50)
            print("Choose an option:")
            print("1. Enter complete patient data (all features)")
            print("2. Enter simplified patient data (essential features only)")
            print("3. Show required features list")
            print("4. Exit")
            print()
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                patient_data = self.get_user_input_interactive()
                results = self.predict(patient_data)
                self.display_results(results)
                
            elif choice == '2':
                patient_data = self.get_user_input_simple()
                results = self.predict(patient_data)
                self.display_results(results)
                
            elif choice == '3':
                self.get_feature_info()
                
            elif choice == '4':
                print("👋 Thank you for using the Esophageal Cancer Prediction System!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            
            input("\nPress Enter to continue...")

def main():
    """Main program entry point"""
    try:
        predictor = EsophagealCancerPredictor()
        predictor.run_interactive()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
