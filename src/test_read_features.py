import pandas as pd
import os

features_path = os.path.join("data", "Gambardella", "features.tsv.gz")
print("DEBUG: Reading features from:", features_path)
try:
    features = pd.read_csv(features_path, header=None, sep="\t", compression="gzip")
    print("DEBUG: Features loaded, shape:", features.shape)
    print(features.head())
except Exception as e:
    print("ERROR while reading features.tsv.gz:", e)
    import traceback
    traceback.print_exc() 