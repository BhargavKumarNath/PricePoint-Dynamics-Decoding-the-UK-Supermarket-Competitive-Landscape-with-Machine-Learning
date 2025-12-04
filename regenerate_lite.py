import pandas as pd
import joblib
import os

def create_lite_files():
    print("Starting file regeneration...")
    
    # 1. Create canonical_products_lite.parquet
    print("\nProcessing canonical_products_e5.parquet...")
    canonical_path = r"data\02_processed\canonical_products_e5.parquet"
    if os.path.exists(canonical_path):
        cols_to_keep = ['supermarket', 'prices', 'canonical_name', 'own_brand', 'date']
        df_canonical = pd.read_parquet(canonical_path, columns=cols_to_keep)
        
        df_canonical['supermarket'] = df_canonical['supermarket'].astype('category')
        df_canonical['own_brand'] = df_canonical['own_brand'].astype('category')
        df_canonical['canonical_name'] = df_canonical['canonical_name'].astype('category')
        df_canonical['prices'] = pd.to_numeric(df_canonical['prices'], downcast='float')
        df_canonical['date'] = pd.to_datetime(df_canonical['date'])
        
        df_canonical.to_parquet("canonical_products_lite.parquet", compression='snappy')
        print("canonical_products_lite.parquet created.")
    
    # 2. Create feature_data_lite.parquet
    print("\nProcessing feature_engineered_data.parquet...")
    feature_path = r"data\02_processed\feature_engineered_data.parquet"
    model_path = "price_predictor_lgbm.joblib"
    
    if os.path.exists(feature_path) and os.path.exists(model_path):
        model = joblib.load(model_path)
        model_features = model.feature_name_
        
        df_sample = pd.read_parquet(feature_path, engine='pyarrow')
        all_cols = df_sample.columns.tolist()
        
        cols_to_keep = set()
        if 'supermarket' in all_cols: cols_to_keep.add('supermarket')
        if 'category' in all_cols: cols_to_keep.add('category')
        for feat in model_features:
            if feat in all_cols:
                cols_to_keep.add(feat)
        
        df_features = df_sample[list(cols_to_keep)].copy()
        
        for col in df_features.select_dtypes(include=['object']).columns:
            df_features[col] = df_features[col].astype('category')
        for col in df_features.select_dtypes(include=['float64']).columns:
            df_features[col] = pd.to_numeric(df_features[col], downcast='float')
            
        df_features.to_parquet("feature_data_lite.parquet", compression='snappy')
        print("feature_data_lite.parquet created.")

if __name__ == "__main__":
    create_lite_files()
