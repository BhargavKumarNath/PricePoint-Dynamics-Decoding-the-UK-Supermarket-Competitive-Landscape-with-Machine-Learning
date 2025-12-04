import pandas as pd
import joblib
import time
import os

def verify_sampling():
    print("Verifying sampling optimization...")
    
    if os.path.exists("feature_data_lite.parquet") and os.path.exists("price_predictor_lgbm.joblib"):
        # Load model to get expected features
        model = joblib.load("price_predictor_lgbm.joblib")
        model_features = model.feature_name_
        
        # Load lite data
        print("Loading lite data...")
        df = pd.read_parquet("feature_data_lite.parquet")
        print(f"Full rows: {len(df)}")
        
        # Sample BEFORE processing
        print("Sampling 1000 rows...")
        df_sample = df.sample(1000, random_state=42)
        
        # Process sample
        print("Processing sample...")
        start = time.time()
        df_encoded = pd.get_dummies(df_sample, columns=['supermarket', 'category'], drop_first=True)
        
        # Align columns
        # Add missing columns with 0
        missing_cols = set(model_features) - set(df_encoded.columns)
        for c in missing_cols:
            df_encoded[c] = 0
            
        df_encoded = df_encoded[model_features]
        
        print(f"Processed in {time.time() - start:.2f}s")
        print(f"Memory: {df_encoded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Verify prediction works
        print("Verifying prediction...")
        try:
            model.predict(df_encoded)
            print("Prediction successful!")
        except Exception as e:
            print(f"Prediction failed: {e}")
            
    else:
        print("Files not found.")

if __name__ == "__main__":
    verify_sampling()
