import streamlit as st
import pandas as pd
import joblib
import requests
import os

# --- Configuration for all your files ---
FILES_TO_DOWNLOAD = {
    "canonical_products_e5.parquet": "1n6YLOF71Pg3nZ8IAuFI8LcoY_yY-J65_",
    "feature_engineered_data.parquet": "1q7e5bMiR6-e-2QjLuR0EOmLu6W-OFATK",
    "price_predictor_lgbm.joblib": "1tpfhYrNqNRNyP1jh9nDxb0aaLleXLqnK"
}

# --- Reusable Download Function ---
def download_file_from_google_drive(id, destination):
    """Downloads a file from Google Drive to a local path."""
    URL = "https://docs.google.com/uc?export=download&id="
    
    if os.path.exists(destination):
        return

    session = requests.Session()
    response = session.get(URL + id, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with st.spinner(f'Downloading required asset: {os.path.basename(destination)}...'):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    st.success(f"Downloaded {os.path.basename(destination)} successfully.")

# --- Cached Loading Functions ---
@st.cache_data
def load_canonical_data():
    """Downloads and loads the canonical products dataset."""
    file_path = "canonical_products_e5.parquet"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    return pd.read_parquet(file_path)

@st.cache_data
def load_features_data():
    """Downloads and loads the feature-engineered dataset."""
    file_path = "feature_engineered_data.parquet"
    file_id = FILES_TO_DOWNLOAD[file_path]
    download_file_from_google_drive(file_id, file_path)
    df = pd.read_parquet(file_path)
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