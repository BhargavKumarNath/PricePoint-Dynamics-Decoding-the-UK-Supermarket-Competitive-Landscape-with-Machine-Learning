import pandas as pd
import joblib
import os

def check_features():
    print("Checking feature cardinality...")
    
    # Load lite feature data
    if os.path.exists("feature_data_lite.parquet"):
        df = pd.read_parquet("feature_data_lite.parquet")
        print(f"Unique supermarkets: {df['supermarket'].nunique()}")
        print(f"Unique categories: {df['category'].nunique()}")
        
        # Load model
        if os.path.exists("price_predictor_lgbm.joblib"):
            model = joblib.load("price_predictor_lgbm.joblib")
            model_features = model.feature_name_
            print(f"\nModel expects {len(model_features)} features.")
            print(f"Sample features: {model_features[:5]}")
            
            # Simulate get_dummies
            df_dummies = pd.get_dummies(df, columns=['supermarket', 'category'], drop_first=True)
            print(f"\nGenerated columns: {len(df_dummies.columns)}")
            
            # Check overlap
            missing = set(model_features) - set(df_dummies.columns)
            extra = set(df_dummies.columns) - set(model_features)
            
            print(f"Missing features: {len(missing)}")
            print(f"Extra features (unused): {len(extra)}")
            
            if len(extra) > 0:
                print("Optimization Opportunity: Drop unused columns immediately.")
                
    else:
        print("feature_data_lite.parquet not found.")

if __name__ == "__main__":
    check_features()
