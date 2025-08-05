# Esophageal Cancer Prediction System - User Guide

## 🏥 Overview
This system provides multiple ways to predict the vital status (Alive/Dead) of esophageal cancer patients using a trained machine learning model.

## 📁 Available Programs

### 1. **Interactive Terminal Program** (`interactive_predictor.py`)
**Best for**: Step-by-step guided input

**How to run:**
```bash
python interactive_predictor.py
```

**Features:**
- Menu-driven interface
- Option for complete or simplified data entry
- Real-time validation
- Detailed result explanation

### 2. **Command-Line Program** (`simple_predictor.py`)
**Best for**: Quick predictions with known parameters

**How to run:**
```bash
# Basic usage
python simple_predictor.py --age 55 --height 170 --weight 70 --consent YES

# Interactive mode
python simple_predictor.py --interactive

# Full example
python simple_predictor.py --age 55 --height 170 --weight 70 --consent YES --site C15.9 --histology 8140/3
```

**Available options:**
- `--age`: Patient age in years
- `--height`: Height in cm
- `--weight`: Weight in kg
- `--consent`: Informed consent (YES/NO)
- `--site`: Cancer site code
- `--histology`: Histology code
- `--tumor-status`: Tumor status
- `--interactive`: Run in interactive mode

### 3. **Graphical Interface** (`gui_predictor.py`)
**Best for**: User-friendly visual interface

**How to run:**
```bash
python gui_predictor.py
```

**Features:**
- Easy-to-use forms
- Example data loading
- Visual result display
- Error handling with pop-ups

### 4. **Python API** (`model_predictor.py`)
**Best for**: Integration into other applications

**How to use:**
```python
from model_predictor import load_trained_model, make_prediction

# Load model once
model = load_trained_model()

# Make predictions
patient_data = {
    'days_to_birth': -20000,  # Age in days (negative)
    'height': 170,
    'weight': 70,
    'informed_consent_verified': 'YES'
    # ... other features
}

prediction, confidence = make_prediction(patient_data, model)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
```

## 📊 Input Data Guide

### Essential Information (Minimum Required)
These fields provide the best prediction accuracy:

1. **Age** (days_to_birth)
   - Format: Negative number (age in days)
   - Example: -20000 (≈ 55 years old)
   - Calculation: -(age_in_years × 365.25)

2. **Height** (height)
   - Format: Number in centimeters
   - Example: 170

3. **Weight** (weight)
   - Format: Number in kilograms
   - Example: 70

4. **Informed Consent** (informed_consent_verified)
   - Format: Text (typically YES/NO)
   - Example: "YES"

5. **Cancer Site Code** (icd_o_3_site)
   - Format: ICD-O-3 site code
   - Example: "C15.9"

6. **Histology Code** (icd_o_3_histology)
   - Format: ICD-O-3 histology code
   - Example: "8140/3"

### Additional Information (Optional but Helpful)
The model can use 66 total features. Missing values are automatically handled through imputation.

## 🎯 Understanding Results

### Prediction Output
- **Vital Status**: "Alive" or "Dead"
- **Confidence**: Percentage (0-100%)
- **Detailed Probabilities**: Breakdown for each class

### Confidence Levels
- **🟢 HIGH (≥80%)**: Very reliable prediction
- **🟡 MEDIUM (60-79%)**: Reliable prediction
- **⚪ LOW (<60%)**: Consider gathering more information

### Example Output
```
🎯 PREDICTION RESULTS
Predicted Vital Status: Alive
Confidence: 82.0%

Detailed Probabilities:
  • Alive: 82.0%
  • Dead:  18.0%

🟢 Confidence Level: HIGH
```

## 🚀 Quick Start Examples

### Example 1: Basic Command Line
```bash
python simple_predictor.py --age 55 --height 170 --weight 70 --consent YES
```

### Example 2: Interactive Mode
```bash
python interactive_predictor.py
# Choose option 2 for simplified input
# Enter patient information when prompted
```

### Example 3: GUI Mode
```bash
python gui_predictor.py
# Click "Load Example" to see sample data
# Modify fields as needed
# Click "Make Prediction"
```

### Example 4: Python Integration
```python
from model_predictor import load_trained_model, make_prediction

model = load_trained_model()
patient = {
    'days_to_birth': -20075,  # 55 years old
    'height': 175,
    'weight': 80,
    'informed_consent_verified': 'YES',
    'icd_o_3_site': 'C15.9',
    'icd_o_3_histology': '8140/3'
}

prediction, confidence = make_prediction(patient, model)
print(f"Result: {prediction} ({confidence:.1%} confidence)")
```

## ⚠️ Important Notes

1. **Missing Data**: The system handles missing values automatically using trained imputers
2. **Data Quality**: More complete information generally leads to better predictions
3. **Medical Disclaimer**: This is a research tool - always consult medical professionals for clinical decisions
4. **Model Limitations**: The model was trained on specific data and may not generalize to all populations

## 🔧 Troubleshooting

### Common Issues

**Error: "No module named 'pandas'"**
- Solution: Install dependencies: `pip install pandas numpy scikit-learn joblib`

**Error: "Model file not found"**
- Solution: Ensure `esophageal_cancer_model.pkl` is in the same directory

**GUI not working**
- Solution: Install tkinter: `sudo apt-get install python3-tk` (Linux) or ensure tkinter is available

**Low confidence predictions**
- Solution: Provide more patient information, especially the essential fields listed above

### Getting Help
- Check that all required files are present
- Verify input data formats match the examples
- Ensure the Python environment has all dependencies installed

## 📋 File Checklist
Make sure you have these files:
- ✅ `esophageal_cancer_model.pkl` (492KB) - Main model file
- ✅ `interactive_predictor.py` - Interactive terminal interface
- ✅ `simple_predictor.py` - Command-line interface
- ✅ `gui_predictor.py` - Graphical interface
- ✅ `model_predictor.py` - Python API
- ✅ `main.py` - Original training script
