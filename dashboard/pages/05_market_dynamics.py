import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config
import requests
import os

st.set_page_config(page_title="Market Dynamics", layout="wide")

def set_plot_style():
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

st.markdown("<h1 style='text-align: center; color: white;'>🌐 Market Dynamics & Price Leadership</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This section explores the strategic interactions between retailers over time, answering the question: **Who leads, and who follows?**</p>", unsafe_allow_html=True)
st.divider()

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


@st.cache_data
def load_and_process_dynamics_data():
    file_id = '1n6YLOF71Pg3nZ8IAuFI8LcoY_yY-J65_'
    file_path = 'canonical_products_e5.parquet'
    download_file_from_google_drive(file_id, file_path)

    df = pd.read_parquet(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    daily_stats = df.groupby(['canonical_name', 'date'])['prices'].agg(['mean', 'std']).reset_index()
    daily_stats['dispersion'] = np.where(daily_stats['mean'] > 0, daily_stats['std'] / daily_stats['mean'], 0)
    market_dispersion = daily_stats.groupby('date')['dispersion'].mean()
    
    # Price Leadership Calculation
    price_pivot = df.pivot_table(index='date', columns=['supermarket', 'canonical_name'], values='prices').ffill()
    supermarkets = df['supermarket'].unique()
    leader_results = []
    
    sampled_products = np.random.choice(price_pivot.columns.get_level_values(1).unique(), 300, replace=False)
    
    for leader in supermarkets:
        for follower in supermarkets:
            if leader == follower: continue
            lags = []
            for product in sampled_products:
                try:
                    series1 = price_pivot[leader][product]
                    series2 = price_pivot[follower][product]
                    if series1.isnull().any() or series2.isnull().any() or series1.var() == 0 or series2.var() == 0:
                        continue
                    
                    max_lag = 7
                    corrs = [series1.corr(series2.shift(lag)) for lag in range(-max_lag, max_lag + 1)]
                    if np.nanmax(np.abs(corrs)) > 0.15:
                        lag_val = np.arange(-max_lag, max_lag + 1)[np.nanargmax(np.abs(corrs))]
                        lags.append(lag_val)
                except KeyError:
                    continue
            
            if lags:
                median_lag = np.median(lags)
                leader_results.append({
                    'leader': leader, 'follower': follower, 'median_lag_days': median_lag
                })
    
    leader_df = pd.DataFrame(leader_results)
    return market_dispersion, leader_df

market_dispersion, leader_df = load_and_process_dynamics_data()

tab1, tab2 = st.tabs(["Market Competitiveness", "Price Leadership Network"])

with tab1:
    with st.container(border=True):
        st.header("Market Competitiveness Over Time")
        st.markdown("This chart measures **price dispersion** (the average variation of prices for the same product across stores). A lower value indicates a more competitive/price-matched market.")
        
        set_plot_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        market_dispersion.plot(ax=ax, label='Mean Daily Dispersion', alpha=0.7, color='#1f77b4')
        market_dispersion.rolling(7).mean().plot(ax=ax, linestyle='--', color='#ff7f0e', label='7-Day Rolling Average', linewidth=2)
        ax.set_title("Market-Wide Price Dispersion", fontsize=16)
        ax.set_ylabel("Avg. Coefficient of Variation")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

with tab2:
    with st.container(border=True):
        st.header("Price Leadership Network")
        st.markdown("This network graph visualizes our cross-correlation analysis. An arrow from `A` to `B` means `A`'s price changes tend to precede `B`'s.")
        
        nodes = []
        edges = []
        
        node_colors = {'Aldi': '#FF4B4B', 'ASDA': '#3DDC97', 'Morrisons': '#FFAF4B', 'Sains': '#966DFF', 'Tesco': '#4B8BFF'}

        for retailer in leader_df['leader'].unique():
            nodes.append(Node(id=retailer, 
                             label=retailer, 
                             size=25,
                             color=node_colors.get(retailer, '#FFFFFF'),
                             font={'color': 'white', 'size': 18}))

        for _, row in leader_df.iterrows():
            if abs(row['median_lag_days']) > 0:
                lag_label = f"{abs(row['median_lag_days']):.0f} days"
                if row['median_lag_days'] > 0:
                    source, target = row['leader'], row['follower']
                else:
                    source, target = row['follower'], row['leader']
                
                edges.append(Edge(source=source, 
                                 target=target, 
                                 label=lag_label,
                                 color='#808080',
                                 font={'color': 'white', 'size': 14, 'strokeWidth': 0},
                                 arrows='to'))
        
        config = Config(width='100%', # Make the graph responsive
                        height=600, 
                        directed=True,
                        physics={
                            "forceAtlas2Based": {
                                "gravitationalConstant": -50,
                                "centralGravity": 0.01,
                                "springLength": 230,
                                "springConstant": 0.08,
                            },
                            "minVelocity": 0.75,
                            "solver": "forceAtlas2Based",
                        },
                        interaction={'navigationButtons': True, 'tooltipDelay': 200},
                        nodeHighlightBehavior=True)
        
        # --- THIS IS THE FIX FOR THE LAYOUT ---
        # Create 3 columns: an empty one, one for the graph, and another empty one.
        # This forces the graph into the center of the page.
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
             agraph(nodes=nodes, edges=edges, config=config)
        
        st.divider()
        st.subheader("Leadership Data")
        st.dataframe(leader_df.sort_values('median_lag_days', ascending=False), use_container_width=True)
        st.markdown("""
        **Insight:** The graph and table clearly show that **Aldi is a primary price-setter**. Arrows consistently point from Aldi to the "Big Four," with a lag of several days. Among the "Big Four," the relationships are much faster and more reciprocal, indicating a tight, reactive competitive cluster.
        """)