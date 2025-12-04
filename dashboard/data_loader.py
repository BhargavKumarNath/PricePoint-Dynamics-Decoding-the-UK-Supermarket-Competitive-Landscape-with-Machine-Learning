import streamlit as st
import pandas as pd
import joblib
import gdown  
import os

FILES_TO_DOWNLOAD = {
    "canonical_products_e5.parquet": "1-xIpUJ3NbIQRqUGjBtJDp9wRMGXkKKh_",
    "feature_engineered_data.parquet": "13LNbB761rBccmqQsHpQPR8a9xa4-EDUc",
    "price_predictor_lgbm.joblib": "1_btm3rfSbA-RJQZZTBxDPKchP0gkxIWN"
}

def download_file_from_google_drive(id, destination):
    """Downloads a file from Google Drive, handling large files correctly."""
    if os.path.exists(destination):
        return 

    with st.spinner(f'Downloading required asset: {os.path.basename(destination)}... This may take a moment.'):
        url = f'https://drive.google.com/uc?id={id}'
        gdown.download(url, destination, quiet=False)
    st.success(f"Downloaded {os.path.basename(destination)} successfully.")


@st.cache_data
def load_canonical_data():
    """Downloads and loads the canonical products dataset with memory optimization."""
    file_path = "canonical_products_e5.parquet"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    
    # Load only necessary columns
    columns = ['supermarket', 'prices', 'canonical_name', 'own_brand', 'date']
    df = pd.read_parquet(file_path, columns=columns)
    
    # Optimize memory usage
    df['supermarket'] = df['supermarket'].astype('category')
    df['own_brand'] = df['own_brand'].astype('category')
    df['canonical_name'] = df['canonical_name'].astype('category')
    df['prices'] = pd.to_numeric(df['prices'], downcast='float')
    df['date'] = pd.to_datetime(df['date'])
    
    return df

@st.cache_data
def get_raw_features_df():
    """Downloads and reads the raw feature-engineered parquet file with memory optimization."""
    file_path = "feature_engineered_data.parquet"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    
    # Load all columns but optimize types
    df = pd.read_parquet(file_path)
    
    # Convert object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    return df

@st.cache_data
def load_features_sample(sample_size=1000):
    """
    Loads a sample of the feature data, processes it for the model, 
    and aligns columns to match the model's expected input.
    """
    df = get_raw_features_df()
    
    # Sample BEFORE processing to save memory
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Perform one-hot encoding on the small sample
    df_encoded = pd.get_dummies(df, columns=['supermarket', 'category'], drop_first=True)
    
    # Load model to get expected features
    file_path = "price_predictor_lgbm.joblib"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    model = joblib.load(file_path)
    model_features = model.feature_name_
    
    # Add missing columns with 0
    missing_cols = set(model_features) - set(df_encoded.columns)
    for c in missing_cols:
        df_encoded[c] = 0
        
    # Ensure order and drop extra columns
    df_encoded = df_encoded[model_features]
    
    return df_encoded

@st.cache_resource
def load_model():
    """Downloads and loads the trained LightGBM model."""
    file_path = "price_predictor_lgbm.joblib"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    return joblib.load(file_path)