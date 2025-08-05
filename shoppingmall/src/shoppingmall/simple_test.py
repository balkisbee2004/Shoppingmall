import sys
print("Starting script...", flush=True)
sys.stdout.flush()

import pandas as pd
print("Pandas imported successfully", flush=True)
sys.stdout.flush()

print("Loading CSV file...", flush=True)
sys.stdout.flush()
df = pd.read_csv("Esophageal_Dataset.csv")
print(f"CSV loaded. Shape: {df.shape}", flush=True)
sys.stdout.flush()

print("Script completed successfully!", flush=True)
sys.stdout.flush()
