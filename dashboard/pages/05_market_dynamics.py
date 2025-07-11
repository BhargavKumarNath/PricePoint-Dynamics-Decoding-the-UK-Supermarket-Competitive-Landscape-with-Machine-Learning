import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config
from data_loader import load_canonical_data  # <-- IMPORT from your new loader

# --- Page Configuration ---
st.set_page_config(page_title="Market Dynamics", layout="wide")

# --- Custom Styling Function for Matplotlib Plots ---
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

# --- Header ---
st.markdown("<h1 style='text-align: center; color: white;'>🌐 Market Dynamics & Price Leadership</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This section explores the strategic interactions between retailers over time, answering the question: **Who leads, and who follows?**</p>", unsafe_allow_html=True)
st.divider()

#  Data Loading and Processing (Cached) 
@st.cache_data
def process_dynamics_data():
    """
    Loads data using the central loader and performs all heavy calculations
    for the market dynamics page.
    """
    df = load_canonical_data()  
    df['date'] = pd.to_datetime(df['date'])
    
    #  Price Dispersion Calculation 
    daily_stats = df.groupby(['canonical_name', 'date'])['prices'].agg(['mean', 'std']).reset_index()
    daily_stats['dispersion'] = np.where(daily_stats['mean'] > 0, daily_stats['std'] / daily_stats['mean'], 0)
    market_dispersion = daily_stats.groupby('date')['dispersion'].mean()
    
    #  Price Leadership Calculation 
    price_pivot = df.pivot_table(index='date', columns=['supermarket', 'canonical_name'], values='prices').ffill()
    supermarkets = df['supermarket'].unique()
    leader_results = []
    
    # Using a smaller sample for faster dashboard loading
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
                    if np.nanmax(np.abs(corrs)) > 0.15: # Use a threshold
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

# Call the function to get the processed data
market_dispersion, leader_df = process_dynamics_data()

# Use Tabs for Different Analyses 
tab1, tab2 = st.tabs(["📊 Market Competitiveness", "👑 Price Leadership Network"])

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
        st.markdown("""
        **Insight:** After a volatile period in mid-January (likely post-holiday sales), the market settled into a **stable equilibrium**. The level of price difference between retailers is not escalating into a price war, nor is it diminishing. Each retailer is holding its strategic ground.
        """)

with tab2:
    with st.container(border=True):
        st.header("Price Leadership Network")
        st.markdown("This network graph visualizes our cross-correlation analysis. An arrow from `A` to `B` means `A`'s price changes tend to precede `B`'s.")
        
        nodes = []
        edges = []
        
        node_colors = {'Aldi': '#FF4B4B', 'ASDA': '#3DDC97', 'Morrisons': '#FFAF4B', 'Sains': '#966DFF', 'Tesco': '#4B8BFF'}

        # Ensure all supermarkets are added as nodes, even if they have no connections in the sample
        all_supermarkets = df['supermarket'].unique()
        for retailer in all_supermarkets:
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
        
        config = Config(width='100%',
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
        
        # Center the graph using columns
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
             agraph(nodes=nodes, edges=edges, config=config)
        
        st.divider()
        st.subheader("Leadership Data")
        st.dataframe(leader_df.sort_values('median_lag_days', ascending=False), use_container_width=True)
        st.markdown("""
        **Insight:** The graph and table clearly show that **Aldi is a primary price-setter**. Arrows consistently point from Aldi to the "Big Four," with a lag of several days. Among the "Big Four," the relationships are much faster and more reciprocal, indicating a tight, reactive competitive cluster.
        """)