import pandas as pd
import numpy as np
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Step 1: Load the data
print("Loading data...", flush=True)
df = pd.read_csv("Esophageal_Dataset.csv")
print(f"Data loaded. Shape: {df.shape}", flush=True)

# Step 2: Drop irrelevant or ID columns
print("Dropping irrelevant columns...", flush=True)
drop_columns = [
    'Unnamed: 0', 'patient_barcode', 'patient_id', 'bcr_patient_uuid',
    'tissue_source_site', 'icd_10', 'project'
]
df.drop(columns=drop_columns, inplace=True, errors='ignore')
print(f"After dropping columns. Shape: {df.shape}", flush=True)

# Step 3: Drop columns with too many missing values (>70%)
print("Dropping columns with >70% missing values...", flush=True)
df = df.loc[:, df.isnull().mean() < 0.7]
print(f"After dropping high-missing columns. Shape: {df.shape}", flush=True)

# Step 4: Drop rows where target is missing
print("Dropping rows with missing target...", flush=True)
df = df[df['vital_status'].notna()]
print(f"After dropping missing targets. Shape: {df.shape}", flush=True)

# Step 5: Separate features and target
print("Separating features and target...", flush=True)
X = df.drop(columns=['vital_status'])
y = df['vital_status']
print(f"Features shape: {X.shape}, Target shape: {y.shape}", flush=True)

# Step 6: Encode the target
print("Encoding target...", flush=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Alive=0, Dead=1
print(f"Target encoded. Unique values: {np.unique(y_encoded)}", flush=True)

# Step 7: Handle missing values
print("Handling missing values...", flush=True)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
print(f"Numeric columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}", flush=True)

print("Filling missing values for numeric columns...", flush=True)
num_imputer = SimpleImputer(strategy='mean')
X_num = num_imputer.fit_transform(X[num_cols])
print(f"Numeric data shape after imputation: {X_num.shape}", flush=True)

print("Filling missing values for categorical columns...", flush=True)
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat = cat_imputer.fit_transform(X[cat_cols])
print(f"Categorical data shape after imputation: {X_cat.shape}", flush=True)

# Step 8: One-hot encode categorical features
print("One-hot encoding categorical features...", flush=True)
if len(cat_cols) > 0:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Use sparse_output instead of sparse
    X_cat_encoded = onehot.fit_transform(X_cat)
    print(f"Categorical features encoded. Shape: {X_cat_encoded.shape}", flush=True)
else:
    X_cat_encoded = np.array([]).reshape(len(X), 0)
    print("No categorical features to encode.", flush=True)

# Step 9: Combine numeric and categorical features
print("Combining numeric and categorical features...", flush=True)
if X_cat_encoded.shape[1] > 0:
    X_processed = np.hstack((X_num, X_cat_encoded))
else:
    X_processed = X_num
print(f"Combined features shape: {X_processed.shape}", flush=True)

# Step 10: Scale features
print("Scaling features...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
print(f"Features scaled. Shape: {X_scaled.shape}", flush=True)

# Step 11: Split data
print("Splitting data...", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}", flush=True)

# Step 12: Train the model
print("Training Random Forest model...", flush=True)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model training completed!", flush=True)

# Step 13: Evaluate the model
print("Making predictions...", flush=True)
y_pred = clf.predict(X_test)
print("Predictions completed!", flush=True)
print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 14: Save the trained model and preprocessing components
print("\n=== Saving Model ===", flush=True)

# Get feature names for the final model
if len(cat_cols) > 0:
    try:
        cat_feature_names = onehot.get_feature_names_out(cat_cols).tolist()
    except:
        # Fallback for older scikit-learn versions
        cat_feature_names = [f"cat_{i}" for i in range(X_cat_encoded.shape[1])]
    all_feature_names = list(num_cols) + cat_feature_names
else:
    all_feature_names = list(num_cols)

model_data = {
    'model': clf,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'num_imputer': num_imputer,
    'cat_imputer': cat_imputer,
    'onehot_encoder': onehot if len(cat_cols) > 0 else None,
    'num_cols': num_cols.tolist(),
    'cat_cols': cat_cols.tolist(),
    'feature_names': all_feature_names
}

# Save the complete model pipeline
joblib.dump(model_data, 'esophageal_cancer_model.pkl')
print("Model saved as 'esophageal_cancer_model.pkl'", flush=True)

# Also save just the trained classifier separately
joblib.dump(clf, 'random_forest_classifier.pkl')
print("Classifier saved as 'random_forest_classifier.pkl'", flush=True)

print("\nModel training and saving completed successfully!", flush=True)
