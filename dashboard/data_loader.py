import streamlit as st
import pandas as pd
import joblib
import gdown  
import os
import numpy as np

FILES_TO_DOWNLOAD = {
    "canonical_products_e5.parquet": "1-xIpUJ3NbIQRqUGjBtJDp9wRMGXkKKh_",
    "feature_engineered_data.parquet": "13LNbB761rBccmqQsHpQPR8a9xa4-EDUc",
    "price_predictor_lgbm.joblib": "1_btm3rfSbA-RJQZZTBxDPKchP0gkxIWN",
    "shap_sample_data.parquet": "1V-gMTu5ygSO0LDNNgFiLaHEyaUdA7ABu",
    "shap_values.npy": "1HOfYnCzvaMgqjM57V2btxyOO71ElHVWi",
    "shap_base_value.txt": "1dFwJ_sV4rXJH3Bb_9emvnChSlpi1g6L4",
    "market_dispersion.parquet": "1dSKBRx2baiuXcMFwNvgc8VcMs388z3Et",
    "price_leadership.parquet": "1C0eBU4Qr7_gKeNwCMvkNYxVWd0881Eef"
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
    
    # Load specific columns with pyarrow engine
    columns = ['supermarket', 'prices', 'canonical_name', 'own_brand', 'date']
    df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    
    # Optimize types
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
    
    df = pd.read_parquet(
        file_path, 
        engine='pyarrow', 
        dtype_backend='pyarrow'
    )
    return df

@st.cache_data
def load_shap_sample_data():
    """
    Loads the PRE-COMPUTED sample data used for SHAP analysis.
    This is much faster and more memory-efficient than computing on-the-fly.
    """
    file_path = "shap_sample_data.parquet"
    file_id = FILES_TO_DOWNLOAD.get(file_path)
    
    if not file_id or file_id == "YOUR_GOOGLE_DRIVE_ID_HERE":
        st.error("⚠️ Pre-computed SHAP data not configured. Please run precompute_shap_values.py locally and upload the files.")
        return None
    
    download_file_from_google_drive(file_id, file_path)
    df = pd.read_parquet(file_path, engine='pyarrow')
    return df

@st.cache_data
def load_shap_values():
    """
    Loads the PRE-COMPUTED SHAP values.
    Returns: (shap_values, base_value) tuple
    """
    shap_file = "shap_values.npy"
    base_file = "shap_base_value.txt"
    
    shap_id = FILES_TO_DOWNLOAD.get(shap_file)
    base_id = FILES_TO_DOWNLOAD.get(base_file)
    
    if not shap_id or shap_id == "YOUR_GOOGLE_DRIVE_ID_HERE":
        st.error("⚠️ Pre-computed SHAP values not configured. Please run precompute_shap_values.py locally and upload the files.")
        return None, None
    
    # Download SHAP values
    download_file_from_google_drive(shap_id, shap_file)
    shap_values = np.load(shap_file)
    
    # Download base value
    download_file_from_google_drive(base_id, base_file)
    with open(base_file, 'r') as f:
        base_value = float(f.read().strip())
    
    return shap_values, base_value

@st.cache_resource
def load_model():
    """Downloads and loads the trained LightGBM model."""
    file_path = "price_predictor_lgbm.joblib"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    return joblib.load(file_path)

@st.cache_data
def load_market_dispersion():
    """
    Loads the PRE-COMPUTED market dispersion time series.
    Returns: pandas Series with date index
    """
    file_path = "market_dispersion.parquet"
    file_id = FILES_TO_DOWNLOAD.get(file_path)
    
    if not file_id or file_id == "YOUR_GOOGLE_DRIVE_ID_HERE":
        st.error("⚠️ Pre-computed market dispersion data not configured. Please run precompute_market_dynamics.py locally and upload the files.")
        return None
    
    download_file_from_google_drive(file_id, file_path)
    df = pd.read_parquet(file_path, engine='pyarrow')
    return df['dispersion']

@st.cache_data
def load_price_leadership():
    """
    Loads the PRE-COMPUTED price leadership analysis.
    Returns: pandas DataFrame with leader/follower relationships
    """
    file_path = "price_leadership.parquet"
    file_id = FILES_TO_DOWNLOAD.get(file_path)
    
    if not file_id or file_id == "YOUR_GOOGLE_DRIVE_ID_HERE":
        st.error("⚠️ Pre-computed price leadership data not configured. Please run precompute_market_dynamics.py locally and upload the files.")
        return None
    
    download_file_from_google_drive(file_id, file_path)
    df = pd.read_parquet(file_path, engine='pyarrow')
    return df