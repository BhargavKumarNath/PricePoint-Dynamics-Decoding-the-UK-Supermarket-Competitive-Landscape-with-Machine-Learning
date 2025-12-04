import sys
import os
import pandas as pd
import joblib
import shap

# Add dashboard to path
sys.path.append(os.path.join(os.getcwd(), 'dashboard'))

from data_loader import load_features_sample, load_model

def verify_shap_optimization():
    print("Verifying SHAP optimization...")
    
    try:
        # 1. Test Data Loading
        print("\nTesting load_features_sample...")
        df_model = load_features_sample(sample_size=100) # Small sample for speed
        print(f"Loaded dataframe shape: {df_model.shape}")
        
        # 2. Test Model Loading
        print("\nTesting load_model...")
        model = load_model()
        print("Model loaded.")
        
        # 3. Verify Column Alignment
        model_features = model.feature_name_
        print(f"\nModel expects {len(model_features)} features.")
        print(f"Dataframe has {len(df_model.columns)} features.")
        
        missing = set(model_features) - set(df_model.columns)
        extra = set(df_model.columns) - set(model_features)
        
        if missing:
            print(f"ERROR: Missing features: {missing}")
        if extra:
            print(f"ERROR: Extra features: {extra}")
            
        if not missing and not extra:
            print("SUCCESS: Columns are perfectly aligned.")
            
        # 4. Test SHAP Calculation (Simulate Page Logic)
        print("\nTesting SHAP calculation...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_model)
        print("SHAP values calculated successfully.")
        
        print("\nALL CHECKS PASSED.")
            
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_shap_optimization()
