import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Step 1: Load the data
print("Loading data...")
df = pd.read_csv("Esophageal_Dataset.csv")
print(f"Data loaded. Shape: {df.shape}")

# Step 2: Drop irrelevant or ID columns
print("Dropping irrelevant columns...")
drop_columns = [
    'Unnamed: 0', 'patient_barcode', 'patient_id', 'bcr_patient_uuid',
    'tissue_source_site', 'icd_10', 'project'
]
df.drop(columns=drop_columns, inplace=True, errors='ignore')
print(f"After dropping columns. Shape: {df.shape}")

# Step 3: Drop columns with too many missing values (>70%)
print("Dropping columns with >70% missing values...")
df = df.loc[:, df.isnull().mean() < 0.7]
print(f"After dropping high-missing columns. Shape: {df.shape}")

# Step 4: Drop rows where target is missing
print("Dropping rows with missing target...")
df = df[df['vital_status'].notna()]
print(f"After dropping missing targets. Shape: {df.shape}")

# Step 5: Separate features and target
print("Separating features and target...")
X = df.drop(columns=['vital_status'])
y = df['vital_status']
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Step 6: Encode the target
print("Encoding target...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Alive=0, Dead=1
print(f"Target encoded. Unique values: {np.unique(y_encoded)}")

# Step 7: Handle missing values
print("Handling missing values...")
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
print(f"Numeric columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")

print("Processing complete!")
