import sys
print("Testing joblib import...", flush=True)
import joblib
print("Joblib imported successfully!", flush=True)

# Test basic functionality
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Creating a simple model...", flush=True)
clf = RandomForestClassifier(n_estimators=10, random_state=42)
X = np.random.random((100, 5))
y = np.random.randint(0, 2, 100)
clf.fit(X, y)

print("Saving model...", flush=True)
joblib.dump(clf, 'test_model.pkl')
print("Model saved successfully!", flush=True)

print("Loading model...", flush=True)
loaded_clf = joblib.load('test_model.pkl')
print("Model loaded successfully!", flush=True)

print("Test completed!", flush=True)
