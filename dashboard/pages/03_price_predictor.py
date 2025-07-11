import streamlit as st
import pandas as pd
import joblib
import requests
import os

st.set_page_config(layout="wide")
st.title("Interactive Price Predictor")
st.markdown("Select a product's features using the sidebar to get a real time price prediction from our LightGBM model")

def download_file_from_google_drive(id, destination):
    """Downloads a file from Google Drive to a local path."""
    URL = "https://docs.google.com/uc?export=download&id="
    
    # Check if file already exists to avoid re-downloading
    if os.path.exists(destination):
        print(f"{destination} already exists. Skipping download.")
        return

    session = requests.Session()
    response = session.get(URL + id, stream=True)
    
    # Display a message while downloading
    with st.spinner(f'Downloading {os.path.basename(destination)}... This may take a moment.'):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {destination} successfully.")


# Load model
# @st.cache_resource
# def load_model_and_data():
#     model = joblib.load("models/price_predictor_lgbm.joblib")
#     df = pd.read_parquet("data/02_processed/feature_engineered_data.parquet")
#     return model, df

# model, df = load_model_and_data()
@st.cache_resource
def load_model():
    """Downloads and loads the trained model."""
    file_id = '1tpfhYrNqNRNyP1jh9nDxb0aaLleXLqnK'
    file_path = 'price_predictor_lgbm.joblib'
    download_file_from_google_drive(file_id, file_path)
    return joblib.load(file_path)

@st.cache_data
def load_feature_data():
    """Downloads and loads the feature-engineered dataset."""
    file_id = '1q7e5bMiR6-e-2QjLuR0EOmLu6W-OFATK'
    file_path = 'feature_engineered_data.parquet'
    download_file_from_google_drive(file_id, file_path)
    return pd.read_parquet(file_path)

# Call the functions to get the model and data
model = load_model()
df = load_feature_data()



st.sidebar.header("Product Features")

price_lag_1d = st.sidebar.number_input(
    "Price Yesterday (£)",
    min_value=0.0, max_value=float(df["price_lag_1d"].max()),
    value=1.50, step=0.10
)

price_rol_mean_7d = st.sidebar.slider(
    "7-Day Average Price (£)",
    min_value=0.0, max_value=float(df["price_rol_mean_7d"].max()),
    value=1.55, step=0.05
)

price_rol_max_7d = st.sidebar.slider(
    "7-Day Max Price (£)",
    min_value=0.0, max_value=float(df["price_rol_max_7d"].max()),
    value=2.00, step=0.05
)

supermarket = st.sidebar.selectbox(
    "Supermarket",
    options=['Aldi', 'ASDA', 'Morrisons', 'Sains', 'Tesco']
)

# CREATE FEATURE VECTOR from INPUTS
def prepare_input_data(user_input, original_columns):
    input_df = pd.DataFrame([user_input])
    # One hot encode the supermarket
    input_df["supermarket"] = supermarket
    input_df = pd.get_dummies(input_df, columns=["supermarket"])

    # Align columns with the model's training columns
    model_columns = pd.DataFrame(columns=original_columns)
    final_df = pd.concat([model_columns, input_df]).fillna(0)
    return final_df[original_columns]

# PREDICTION
if st.button("Product Price"):
    user_data = {
        'price_lag_1d': price_lag_1d,
        'price_rol_mean_7d': price_rol_mean_7d,
        'price_rol_max_7d': price_rol_max_7d,
        'price_rol_min_7d': price_rol_mean_7d * 0.9,
        'price_diff_1d': price_lag_1d - (price_rol_mean_7d * 0.95),
    }

    model_features = model.feature_name_
    input_vector = prepare_input_data(user_data, model_features)
    prediction = model.predict(input_vector)[0]
    st.success(f"Predicted Price: £{prediction:.2f}")
    st.balloons()
    
