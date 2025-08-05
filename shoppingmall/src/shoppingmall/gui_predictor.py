#!/usr/bin/env python3
"""
GUI Version of Esophageal Cancer Predictor
Simple graphical interface using tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
import sys
import threading

class EsophagealCancerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Esophageal Cancer Prediction System")
        self.root.geometry("800x600")
        
        # Load model
        self.model_data = None
        self.load_model()
        
        # Create GUI
        self.create_widgets()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model_data = joblib.load('esophageal_cancer_model.pkl')
            self.status_text = "✅ Model loaded successfully!"
        except Exception as e:
            self.status_text = f"❌ Error loading model: {e}"
            messagebox.showerror("Error", f"Could not load model: {e}")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🏥 Esophageal Cancer Prediction System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status
        status_label = ttk.Label(main_frame, text=self.status_text)
        status_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Patient Information", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Essential input fields
        self.entries = {}
        
        # Numeric fields
        numeric_fields = [
            ("Age (years)", "age"),
            ("Height (cm)", "height"),
            ("Weight (kg)", "weight")
        ]
        
        row = 0
        for label_text, key in numeric_fields:
            ttk.Label(input_frame, text=label_text + ":").grid(row=row, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(input_frame, width=20)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
            self.entries[key] = entry
            row += 1
        
        # Categorical fields
        categorical_fields = [
            ("Informed Consent", "consent"),
            ("Cancer Site Code", "site"),
            ("Histology Code", "histology"),
            ("Tumor Status", "tumor_status")
        ]
        
        for label_text, key in categorical_fields:
            ttk.Label(input_frame, text=label_text + ":").grid(row=row, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(input_frame, width=20)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
            self.entries[key] = entry
            row += 1
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Predict button
        predict_btn = ttk.Button(button_frame, text="🎯 Make Prediction", 
                                command=self.make_prediction, style="Accent.TButton")
        predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear Fields", 
                              command=self.clear_fields)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Example button
        example_btn = ttk.Button(button_frame, text="📝 Load Example", 
                                command=self.load_example)
        example_btn.pack(side=tk.LEFT)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=70)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        input_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def clear_fields(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.results_text.delete('1.0', tk.END)
    
    def load_example(self):
        """Load example patient data"""
        example_data = {
            "age": "55",
            "height": "170",
            "weight": "70",
            "consent": "YES",
            "site": "C15.9",
            "histology": "8140/3",
            "tumor_status": "WITH TUMOR"
        }
        
        for key, value in example_data.items():
            if key in self.entries:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, value)
    
    def get_patient_data(self):
        """Get patient data from input fields"""
        patient_data = {}
        
        # Get values from entries
        age = self.entries["age"].get().strip()
        if age:
            try:
                patient_data['days_to_birth'] = -float(age) * 365.25
            except ValueError:
                raise ValueError("Age must be a number")
        
        height = self.entries["height"].get().strip()
        if height:
            try:
                patient_data['height'] = float(height)
            except ValueError:
                raise ValueError("Height must be a number")
        
        weight = self.entries["weight"].get().strip()
        if weight:
            try:
                patient_data['weight'] = float(weight)
            except ValueError:
                raise ValueError("Weight must be a number")
        
        consent = self.entries["consent"].get().strip()
        if consent:
            patient_data['informed_consent_verified'] = consent
        
        site = self.entries["site"].get().strip()
        if site:
            patient_data['icd_o_3_site'] = site
        
        histology = self.entries["histology"].get().strip()
        if histology:
            patient_data['icd_o_3_histology'] = histology
        
        tumor_status = self.entries["tumor_status"].get().strip()
        if tumor_status:
            patient_data['tumor_status'] = tumor_status
        
        return patient_data
    
    def predict_patient(self, patient_data):
        """Make prediction for patient data"""
        if self.model_data is None:
            return None
        
        try:
            # Extract model components
            clf = self.model_data['model']
            scaler = self.model_data['scaler']
            label_encoder = self.model_data['label_encoder']
            num_imputer = self.model_data['num_imputer']
            cat_imputer = self.model_data['cat_imputer']
            onehot_encoder = self.model_data['onehot_encoder']
            num_cols = self.model_data['num_cols']
            cat_cols = self.model_data['cat_cols']
            
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
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'classes': label_encoder.classes_
            }
            
        except Exception as e:
            raise Exception(f"Prediction error: {e}")
    
    def make_prediction(self):
        """Make prediction and display results"""
        try:
            # Get patient data
            patient_data = self.get_patient_data()
            
            if not patient_data:
                messagebox.showwarning("Warning", "Please enter at least some patient information.")
                return
            
            # Make prediction
            results = self.predict_patient(patient_data)
            
            if results is None:
                messagebox.showerror("Error", "Model not loaded properly.")
                return
            
            # Display results
            self.display_results(results, patient_data)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def display_results(self, results, patient_data):
        """Display prediction results"""
        self.results_text.delete('1.0', tk.END)
        
        # Header
        self.results_text.insert(tk.END, "🎯 PREDICTION RESULTS\\n")
        self.results_text.insert(tk.END, "=" * 50 + "\\n\\n")
        
        # Prediction
        prediction = results['prediction']
        confidence = results['confidence']
        
        self.results_text.insert(tk.END, f"Predicted Vital Status: {prediction}\\n")
        self.results_text.insert(tk.END, f"Confidence: {confidence:.1%}\\n\\n")
        
        # Detailed probabilities
        self.results_text.insert(tk.END, "Detailed Probabilities:\\n")
        for i, class_name in enumerate(results['classes']):
            prob = results['probabilities'][i]
            self.results_text.insert(tk.END, f"  • {class_name}: {prob:.1%}\\n")
        
        # Confidence level
        if confidence >= 0.8:
            level = "HIGH"
            emoji = "🔴" if prediction == 'Dead' else "🟢"
        elif confidence >= 0.6:
            level = "MEDIUM"
            emoji = "🟡"
        else:
            level = "LOW"
            emoji = "⚪"
        
        self.results_text.insert(tk.END, f"\\n{emoji} Confidence Level: {level}\\n")
        
        if confidence < 0.6:
            self.results_text.insert(tk.END, "\\n⚠️  Low confidence - consider gathering more patient information\\n")
        
        # Input summary
        self.results_text.insert(tk.END, "\\n" + "-" * 30 + "\\n")
        self.results_text.insert(tk.END, "Input Data Summary:\\n")
        for key, value in patient_data.items():
            self.results_text.insert(tk.END, f"  {key}: {value}\\n")

def main():
    """Main function to run the GUI"""
    try:
        root = tk.Tk()
        app = EsophagealCancerGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("Please make sure the model file 'esophageal_cancer_model.pkl' exists in the current directory.")

if __name__ == "__main__":
    main()
