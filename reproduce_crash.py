import pandas as pd
import joblib
import time
import os

# Mock the data_loader functions to use local lite files
def load_canonical_data_lite():
    print("Loading canonical_products_lite.parquet...")
    start = time.time()
    df = pd.read_parquet("canonical_products_lite.parquet")
    print(f"Loaded in {time.time() - start:.2f}s")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def load_features_data_lite():
    print("Loading feature_data_lite.parquet...")
    start = time.time()
    df = pd.read_parquet("feature_data_lite.parquet")
    print(f"Loaded raw in {time.time() - start:.2f}s")
    
    print("Performing get_dummies...")
    start = time.time()
    df_model = pd.get_dummies(df, columns=['supermarket', 'category'], drop_first=True)
    df_model = df_model.dropna()
    print(f"Processed in {time.time() - start:.2f}s")
    print(f"Memory: {df_model.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df_model

def reproduce():
    print("Starting reproduction...")
    try:
        load_canonical_data_lite()
        load_features_data_lite()
        print("SUCCESS: No crash detected locally.")
    except Exception as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    reproduce()
