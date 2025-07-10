import streamlit as st
import pandas as pd
import joblib
st.set_page_config(layout="wide")
st.title("Interactive Price Predictor")
st.markdown("Select a product's features using the sidebar to get a real time price prediction from our LightGBM model")

# Load model
@st.cache_resource
def load_model_and_data():
    model = joblib.load("models/price_predictor_lgbm.joblib")
    df = pd.read_parquet("data/02_processed/feature_engineered_data.parquet")
    return model, df

model, df = load_model_and_data()

# Use Inputs in sidebar
st.sidebar.header("Product Features")

# Create input widgets for the most important features (based on SHAP)
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
    
