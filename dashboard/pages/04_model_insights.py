import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(page_title="Model Insights", layout="wide")

# Custom Styling Function for Plots
def set_plot_style():
    """Sets a consistent, dark-themed style for all matplotlib plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#0E1117', 
        'figure.facecolor': '#0E1117',
        'axes.edgecolor': '#B0B0B0',
        'axes.labelcolor': '#B0B0B0',
        'xtick.color': '#B0B0B0',
        'ytick.color': '#B0B0B0',
        'text.color': '#FFFFFF', 
        'legend.facecolor': '#1E1E1E',
    })

st.markdown("<h1 style='text-align: center; color: white;'>🧠 Model Insights & Explainable AI (XAI)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This page delves into the 'brain' of our price prediction model. We use SHAP to understand not just *what* the model predicts, but *why*.</p>", unsafe_allow_html=True)
st.divider()

# Load Model and Data (Cached)
@st.cache_resource
def load_model():
    return joblib.load("models/price_predictor_lgbm.joblib")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/02_processed/feature_engineered_data.parquet")
    df_model = pd.get_dummies(df, columns=['supermarket', 'category'], drop_first=True)
    df_model = df_model.dropna()
    return df_model

model = load_model()
df_model = load_data()

# Model Performance
with st.container(border=True):
    st.subheader("Model Performance")
    st.markdown("The LightGBM model was trained on over 4.9 million data points and tested on a hold-out set of the final week's data (~460k records).")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", "£0.14", help="On average, the model's prediction is off by 14 pence.")
    col2.metric("Root Mean Squared Error (RMSE)", "£0.37", help="Penalizes larger errors more heavily.")
    col3.metric("Dataset Size", "9.5M+", help="Total records analyzed in the project.")

st.divider()

# Prepare data for SHAP
X_sample = df_model[model.feature_name_].sample(1000, random_state=42)

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def calculate_shap_values(_explainer, _X_sample):
    return _explainer.shap_values(_X_sample)

explainer = get_shap_explainer(model)
shap_values_sample = calculate_shap_values(explainer, X_sample)

# Global Feature Importance
with st.container(border=True):
    st.subheader("Global Feature Importance: What Drives Prices Overall?")
    st.markdown("The plots below show which features have the most predictive power across the entire dataset.")
    
    tab1, tab2 = st.tabs(["Bar Chart (Average Impact)", "Beeswarm Plot (Impact Distribution)"])

    with tab1:
        fig, ax = plt.subplots()
        set_plot_style()
        shap.summary_plot(shap_values_sample, X_sample, plot_type="bar", show=False)
        ax.set_xlabel("mean(|SHAP value|) (average impact on model output)", color="white") 
        plt.tick_params(axis='x', colors='white') 
        plt.tick_params(axis='y', colors='white') 
        st.pyplot(fig, use_container_width=True)

    with tab2:
        fig2, ax2 = plt.subplots()
        set_plot_style()
        shap.summary_plot(shap_values_sample, X_sample, show=False)
        ax2.set_xlabel("SHAP value (impact on model output)", color="white")
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        cbar = plt.gcf().axes[-1]
        cbar.tick_params(colors='white')
        cbar.set_ylabel(cbar.get_ylabel(), color='white')
        st.pyplot(fig2, use_container_width=True)

    with st.expander("How to Read These Plots"):
        st.markdown("""
        - **Bar Chart:** Ranks features by their average impact on the model's predictions.
        - **Beeswarm Plot:** Shows the distribution of impacts for each feature. Each dot is a prediction.
            - **Position:** Right of zero means it pushed the price prediction higher; left means lower.
            - **Color:** Red dots are high feature values; blue dots are low feature values.
        """)

st.divider()
with st.container(border=True):
    st.subheader("🔬 Local Prediction Explanations")
    st.markdown("Let's break down a **single prediction**. Select a product instance below to see how the model arrived at its forecast.")

    random_index = st.selectbox(
        "Select a random product instance to analyze:",
        options=X_sample.index,
        help="Each number represents a unique product on a specific day."
    )

    if random_index:
        instance = X_sample.loc[[random_index]]
        shap_value_instance = explainer.shap_values(instance)
        prediction = model.predict(instance)[0]
        base_value = explainer.expected_value

        st.markdown(f"##### Explaining Prediction for Instance #{random_index}")
        st.metric(label="Model's Price Prediction", value=f"£{prediction:.2f}")

        st.markdown("##### Force Plot Breakdown")

        force_plot = shap.force_plot(
            base_value,
            shap_value_instance,
            instance,
            show=False,
            matplotlib=False
        )

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        
        components.html(shap_html, height=200)

        with st.expander("How to Read This Force Plot"):
            st.markdown(f"""
            This plot shows the forces pushing the prediction for this single instance.
            - The **base value ({base_value:.2f})** is the average prediction over the dataset.
            - **<span style='color: #ff0d57;'>Red arrows</span>** show features that pushed the prediction **higher**.
            - **<span style='color: #008bfb;'>Blue arrows</span>** show features that pushed the prediction **lower**.
            - The final **bold** number is the model's output for this specific instance.
            """, unsafe_allow_html=True)