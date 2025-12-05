"""
Pre-compute SHAP values locally to avoid memory issues on Streamlit Cloud.
Run this script on your local machine with sufficient RAM.
"""
import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path

print("=" * 60)
print("SHAP Pre-computation Script")
print("=" * 60)

# Configuration
SAMPLE_SIZE = 8000
FEATURE_DATA_PATH = r"data\02_processed\feature_engineered_data.parquet"
MODEL_PATH = "price_predictor_lgbm.joblib"
OUTPUT_DIR = Path("shap_precomputed")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n1. Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"   ✓ Model loaded. Expects {len(model.feature_name_)} features.")

print(f"\n2. Loading feature data from {FEATURE_DATA_PATH}...")
df = pd.read_parquet(FEATURE_DATA_PATH, engine='pyarrow')
print(f"   ✓ Loaded {len(df):,} rows")

# Sample the data
print(f"\n3. Sampling {SAMPLE_SIZE:,} rows for SHAP analysis...")
if len(df) > SAMPLE_SIZE:
    # Stratified sampling to get diverse examples
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df.copy()
print(f"   ✓ Sample created: {len(df_sample):,} rows")

# Check if data needs encoding
print("\n4. Checking data format...")
if 'supermarket' in df_sample.columns and 'category' in df_sample.columns:
    print("   → Data needs one-hot encoding...")
    df_encoded = pd.get_dummies(df_sample, columns=['supermarket', 'category'], drop_first=True)
else:
    print("   → Data is already encoded")
    df_encoded = df_sample.copy()

# Ensure numeric data only
df_encoded = df_encoded.select_dtypes(include=['number'])

# Align with model features
print("\n5. Aligning columns with model features...")
model_features = model.feature_name_

# Add missing columns
missing_cols = set(model_features) - set(df_encoded.columns)
if missing_cols:
    print(f"   → Adding {len(missing_cols)} missing columns")
    for col in missing_cols:
        df_encoded[col] = 0

# Reorder to match model
df_encoded = df_encoded[model_features]

# Drop NaN rows
original_len = len(df_encoded)
df_encoded = df_encoded.dropna()
if len(df_encoded) < original_len:
    print(f"   → Dropped {original_len - len(df_encoded)} rows with NaN values")

print(f"   ✓ Final dataset: {df_encoded.shape}")

# Save the sample data
print("\n6. Saving sample data...")
sample_output = OUTPUT_DIR / "shap_sample_data.parquet"
df_encoded.to_parquet(sample_output, compression='snappy')
print(f"   ✓ Saved to {sample_output}")

# Compute SHAP values
print("\n7. Computing SHAP values (this may take several minutes)...")
print("   → Creating TreeExplainer...")
explainer = shap.TreeExplainer(model)

print("   → Calculating SHAP values...")
shap_values = explainer.shap_values(df_encoded)
print(f"   ✓ SHAP values computed: shape {shap_values.shape}")

# Save SHAP values
print("\n8. Saving SHAP values...")
shap_output = OUTPUT_DIR / "shap_values.npy"
np.save(shap_output, shap_values)
print(f"   ✓ Saved to {shap_output}")

# Save base value
base_value_output = OUTPUT_DIR / "shap_base_value.txt"
with open(base_value_output, 'w') as f:
    f.write(str(explainer.expected_value))
print(f"   ✓ Saved base value to {base_value_output}")

# Save feature names for reference
feature_names_output = OUTPUT_DIR / "feature_names.txt"
with open(feature_names_output, 'w') as f:
    f.write('\n'.join(model_features))
print(f"   ✓ Saved feature names to {feature_names_output}")

# Create summary statistics
print("\n9. Creating summary statistics...")
shap_importance = pd.DataFrame({
    'feature': model_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

summary_output = OUTPUT_DIR / "shap_feature_importance.csv"
shap_importance.to_csv(summary_output, index=False)
print(f"   ✓ Saved feature importance to {summary_output}")

print("\n" + "=" * 60)
print("SHAP Pre-computation Complete!")
print("=" * 60)
print(f"\nGenerated files in '{OUTPUT_DIR}/' directory:")
print(f"  1. shap_sample_data.parquet    ({sample_output.stat().st_size / 1024**2:.1f} MB)")
print(f"  2. shap_values.npy             ({shap_output.stat().st_size / 1024**2:.1f} MB)")
print(f"  3. shap_base_value.txt")
print(f"  4. feature_names.txt")
print(f"  5. shap_feature_importance.csv")
print("\nNext steps:")
print("  1. Upload these files to Google Drive")
print("  2. Get shareable links for each file")
print("  3. Update FILES_TO_DOWNLOAD in data_loader.py")
print("  4. Deploy to Streamlit Cloud")