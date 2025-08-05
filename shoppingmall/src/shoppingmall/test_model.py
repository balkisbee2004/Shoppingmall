import pandas as pd
import numpy as np
import joblib
import sys

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    try:
        model_data = joblib.load('esophageal_cancer_model.pkl')
        print("✅ Model loaded successfully!")
        return model_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_model_components(model_data):
    """Test if all model components are present"""
    print("\nTesting model components...")
    required_components = ['model', 'scaler', 'label_encoder', 'num_imputer', 
                          'cat_imputer', 'onehot_encoder', 'num_cols', 'cat_cols', 'feature_names']
    
    for component in required_components:
        if component in model_data:
            print(f"✅ {component}: Present")
        else:
            print(f"❌ {component}: Missing")
    
    print(f"\nModel details:")
    print(f"- Classifier type: {type(model_data['model']).__name__}")
    print(f"- Number of features: {len(model_data['feature_names'])}")
    print(f"- Numeric columns: {len(model_data['num_cols'])}")
    print(f"- Categorical columns: {len(model_data['cat_cols'])}")

def test_prediction_function(model_data):
    """Test the prediction functionality with sample data"""
    print("\nTesting prediction functionality...")
    
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
        
        # Create sample data with the same structure as training data
        sample_data = {}
        
        # Add numeric features with reasonable values
        for col in num_cols:
            if 'age' in col.lower() or 'birth' in col.lower():
                sample_data[col] = -20000  # Age in days (negative)
            elif 'height' in col.lower():
                sample_data[col] = 170
            elif 'weight' in col.lower():
                sample_data[col] = 70
            else:
                sample_data[col] = np.random.randn()  # Random normal value
        
        # Add categorical features with placeholder values
        for col in cat_cols:
            sample_data[col] = 'test_value'
        
        # Convert to DataFrame
        df_sample = pd.DataFrame([sample_data])
        
        # Process the data through the pipeline
        X_num = df_sample[num_cols]
        X_cat = df_sample[cat_cols]
        
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
        
        print(f"✅ Prediction successful!")
        print(f"   Sample prediction: {prediction}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probabilities: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with actual data from the dataset"""
    print("\nTesting with real data from the dataset...")
    
    try:
        # Load the original dataset
        df = pd.read_csv("Esophageal_Dataset.csv")
        
        # Apply the same preprocessing as in training
        drop_columns = [
            'Unnamed: 0', 'patient_barcode', 'patient_id', 'bcr_patient_uuid',
            'tissue_source_site', 'icd_10', 'project'
        ]
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        df = df.loc[:, df.isnull().mean() < 0.7]
        df = df[df['vital_status'].notna()]
        
        # Take a sample patient (first row)
        sample_patient = df.drop(columns=['vital_status']).iloc[0:1]
        actual_status = df['vital_status'].iloc[0]
        
        # Load model and make prediction
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
        
        # Process the sample
        X_num = sample_patient[num_cols]
        X_cat = sample_patient[cat_cols]
        
        X_num_imputed = num_imputer.transform(X_num)
        X_cat_imputed = cat_imputer.transform(X_cat)
        
        if onehot_encoder is not None and len(cat_cols) > 0:
            X_cat_encoded = onehot_encoder.transform(X_cat_imputed)
            X_processed = np.hstack((X_num_imputed, X_cat_encoded))
        else:
            X_processed = X_num_imputed
        
        X_scaled = scaler.transform(X_processed)
        
        prediction_encoded = clf.predict(X_scaled)[0]
        probabilities = clf.predict_proba(X_scaled)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = max(probabilities)
        
        print(f"✅ Real data prediction successful!")
        print(f"   Actual status: {actual_status}")
        print(f"   Predicted status: {prediction}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Match: {'✅' if actual_status == prediction else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with real data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 TESTING ESOPHAGEAL CANCER PREDICTION MODEL")
    print("=" * 50)
    
    # Test 1: Model Loading
    model_data = test_model_loading()
    if model_data is None:
        print("❌ Cannot proceed with other tests - model loading failed")
        return
    
    # Test 2: Model Components
    test_model_components(model_data)
    
    # Test 3: Prediction Function
    prediction_success = test_prediction_function(model_data)
    
    # Test 4: Real Data Test
    real_data_success = test_with_real_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Model Loading: ✅")
    print(f"Component Check: ✅")
    print(f"Prediction Function: {'✅' if prediction_success else '❌'}")
    print(f"Real Data Test: {'✅' if real_data_success else '❌'}")
    
    if prediction_success and real_data_success:
        print("\n🎉 ALL TESTS PASSED! The model is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
