import streamlit as st
import pandas as pd
import joblib
import gdown  
import os

FILES_TO_DOWNLOAD = {
    "canonical_products_e5.parquet": "1YS2k15a6hhu15_KmRpMLo_bh76BxwFLt",
    "feature_engineered_data.parquet": "1RIs1hNG60uMgwJGWKAHNAWXeDHf7hscf",
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
    """Downloads and loads the canonical products dataset."""
    file_path = "canonical_products_e5.parquet"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    return pd.read_parquet(file_path)

@st.cache_data
def get_raw_features_df():
    """Downloads and reads the raw feature-engineered parquet file."""
    file_path = "feature_engineered_data.parquet"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    return pd.read_parquet(file_path)

@st.cache_data
def load_features_data():
    """Loads and preprocesses the feature-engineered dataset for the model."""
    df = get_raw_features_df() 
    df_model = pd.get_dummies(df, columns=['supermarket', 'category'], drop_first=True)
    df_model = df_model.dropna()
    return df_model

@st.cache_resource
def load_model():
    """Downloads and loads the trained LightGBM model."""
    file_path = "price_predictor_lgbm.joblib"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    return joblib.load(file_path)