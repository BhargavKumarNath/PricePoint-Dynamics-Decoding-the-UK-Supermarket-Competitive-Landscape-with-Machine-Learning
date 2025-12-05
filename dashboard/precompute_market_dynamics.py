"""
Pre-compute market dynamics data locally to avoid memory issues on Streamlit Cloud.
Run this script on your local machine with sufficient RAM.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

print("=" * 60)
print("Market Dynamics Pre-computation Script")
print("=" * 60)

# Configuration
POSSIBLE_PATHS = [
    r"data\02_processed\canonical_products_e5.parquet",
]

CANONICAL_DATA_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        CANONICAL_DATA_PATH = path
        break

if CANONICAL_DATA_PATH is None:
    print("\n❌ ERROR: Could not find canonical_products_e5.parquet")
    print("\nSearched in:")
    for path in POSSIBLE_PATHS:
        print(f"  - {os.path.abspath(path)}")
    print("\nPlease ensure:")
    print("  1. You have the canonical_products_e5.parquet file")
    print("  2. You're running this script from the project root directory")
    print("  3. Or update CANONICAL_DATA_PATH manually in the script")
    sys.exit(1)

OUTPUT_DIR = Path("market_dynamics_precomputed")
SAMPLE_SIZE = 1000  

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n1. Loading canonical products data from {CANONICAL_DATA_PATH}...")
try:
    df = pd.read_parquet(CANONICAL_DATA_PATH, engine='pyarrow')
    df['date'] = pd.to_datetime(df['date'])
    print(f"   ✓ Loaded {len(df):,} rows")
    print(f"   ✓ Columns: {list(df.columns)}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
except Exception as e:
    print(f"\n❌ ERROR loading data: {e}")
    sys.exit(1)

# Calculate price dispersion
print("\n2. Calculating market-wide price dispersion...")
try:
    daily_stats = df.groupby(['canonical_name', 'date'], observed=True)['prices'].agg(['mean', 'std']).reset_index()
    daily_stats['dispersion'] = np.where(daily_stats['mean'] > 0, daily_stats['std'] / daily_stats['mean'], 0)
    market_dispersion = daily_stats.groupby('date', observed=True)['dispersion'].mean().sort_index()
    
    dispersion_output = OUTPUT_DIR / "market_dispersion.parquet"
    market_dispersion.to_frame('dispersion').to_parquet(dispersion_output, compression='snappy')
    print(f"   ✓ Saved to {dispersion_output}")
    print(f"   ✓ Date range: {market_dispersion.index.min()} to {market_dispersion.index.max()}")
    print(f"   ✓ Data points: {len(market_dispersion)}")
except Exception as e:
    print(f"\n❌ ERROR calculating dispersion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Price leadership analysis
print(f"\n3. Analyzing price leadership (sampling {SAMPLE_SIZE} products)...")

try:
    # Get products that appear in multiple supermarkets
    product_counts = df.groupby('canonical_name')['supermarket'].nunique()
    common_products = product_counts[product_counts >= 3].index
    print(f"   → Found {len(common_products):,} products in 3+ stores")
    
    if len(common_products) == 0:
        print("   ❌ No common products found across stores!")
        sys.exit(1)
    
    # Sample products
    sampled_products = np.random.choice(common_products, min(SAMPLE_SIZE, len(common_products)), replace=False)
    print(f"   → Sampling {len(sampled_products):,} products for analysis")
    
    # Create pivot table
    print("   → Creating pivot table...")
    df_sampled = df[df['canonical_name'].isin(sampled_products)].copy()
    price_pivot = df_sampled.pivot_table(
        index='date', 
        columns=['supermarket', 'canonical_name'], 
        values='prices'
    ).ffill()
    print(f"   ✓ Pivot table created: {price_pivot.shape}")
    
    # Cross-correlation analysis
    print("   → Running cross-correlation analysis...")
    supermarkets = df['supermarket'].unique()
    print(f"   → Supermarkets: {list(supermarkets)}")
    leader_results = []
    
    for i, leader in enumerate(supermarkets):
        print(f"      Analyzing {leader} as leader ({i+1}/{len(supermarkets)})...")
        for follower in supermarkets:
            if leader == follower: 
                continue
                
            lags = []
            for product in sampled_products:
                try:
                    if (leader, product) not in price_pivot.columns or (follower, product) not in price_pivot.columns:
                        continue
                        
                    series1 = price_pivot[leader][product]
                    series2 = price_pivot[follower][product]
                    
                    if series1.isnull().any() or series2.isnull().any() or series1.var() == 0 or series2.var() == 0:
                        continue
    
                    max_lag = 7
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        corrs = [series1.corr(series2.shift(lag)) for lag in range(-max_lag, max_lag + 1)]
                    
                    if np.nanmax(np.abs(corrs)) > 0.15:
                        lag_val = np.arange(-max_lag, max_lag + 1)[np.nanargmax(np.abs(corrs))]
                        lags.append(lag_val)
                except (KeyError, ValueError):
                    continue
    
            if lags:
                median_lag = np.median(lags)
                leader_results.append({
                    'leader': leader, 
                    'follower': follower, 
                    'median_lag_days': median_lag,
                    'n_products_analyzed': len(lags)
                })
                print(f"         {leader} → {follower}: {median_lag:.1f} days (from {len(lags)} products)")
    
    leader_df = pd.DataFrame(leader_results)
    leader_df = leader_df[leader_df['median_lag_days'] != 0].copy()
    
    leadership_output = OUTPUT_DIR / "price_leadership.parquet"
    leader_df.to_parquet(leadership_output, compression='snappy')
    print(f"   ✓ Saved to {leadership_output}")
    print(f"   ✓ Found {len(leader_df)} leadership relationships")
    
except Exception as e:
    print(f"\n❌ ERROR in leadership analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary statistics
print("\n4. Summary Statistics:")
print(f"   Market Dispersion: avg={market_dispersion.mean():.3f}, latest={market_dispersion.iloc[-1]:.3f}")
if not leader_df.empty:
    top_leader = leader_df['leader'].mode()[0]
    print(f"   Primary Price Leader: {top_leader}")
    print(f"   Total Product Comparisons: {leader_df['n_products_analyzed'].sum():,}")
else:
    print("   ⚠️  No clear leadership patterns found")

print("\n" + "=" * 60)
print("Market Dynamics Pre-computation Complete!")
print("=" * 60)
print(f"\nGenerated files in '{OUTPUT_DIR}/' directory:")
print(f"  1. market_dispersion.parquet    ({dispersion_output.stat().st_size / 1024:.1f} KB)")
print(f"  2. price_leadership.parquet     ({leadership_output.stat().st_size / 1024:.1f} KB)")
print("\nNext steps:")
print("  1. Upload these files to Google Drive")
print("  2. Get shareable links for each file")
print("  3. Update FILES_TO_DOWNLOAD in data_loader.py")
print("  4. Replace 05_market_dynamics.py with the pre-computed version")
print("  5. Deploy to Streamlit Cloud")