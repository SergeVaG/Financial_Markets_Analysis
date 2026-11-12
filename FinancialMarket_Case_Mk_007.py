import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import itertools

print("📥 Step 1: Downloading complete, live NASDAQ ticker directory...")
# Official NASDAQ live symbol directory
nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

# Request the file using standard headers to prevent basic security blocks
req = urllib.request.Request(nasdaq_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    # Read the pipe-delimited file
    nasdaq_raw = pd.read_csv(response, sep="|")

# The last line of the NASDAQ file contains a creation timestamp footer; we drop it
nasdaq_raw = nasdaq_raw.iloc[:-1]

# Extract all active tickers into a clean Python list
full_nasdaq_tickers = nasdaq_raw['Symbol'].dropna().unique().tolist()
print(f"✅ Success! Found {len(full_nasdaq_tickers)} active listings on the exchange.")

# ---------------------------------------------------------------------
# DEVELOPMENT SAFEGUARD (Recommended)
# Running all 3,500+ stocks at once through yfinance will throttle your IP.
# We will pull a representative sample of 150 assets for testing. 
# You can increase this slice once you integrate rate-limiting or proxies.
# ---------------------------------------------------------------------
sample_size = 150
sample_universe = [t.replace('.', '-') for t in full_nasdaq_tickers[:sample_size]]
download_list = sample_universe + ['^GSPC']

print(f"\n📥 Step 2: Extracting historical data matrix for {sample_size} sample stocks...")
raw_data = yf.download(download_list, start="2024-01-01", end="2026-11-01", threads=True)
close_prices = raw_data['Close']

stock_prices = close_prices[sample_universe]
market_prices = close_prices['^GSPC']

missing_threshold = 0.05 * len(close_prices)
stock_prices = stock_prices.dropna(thresh=len(close_prices) - missing_threshold, axis=1)
active_tickers = stock_prices.columns.tolist()

stock_returns = stock_prices.pct_change().dropna()
market_returns = market_prices.pct_change().dropna()

# =====================================================================
# 2. FEATURE ENGINEERING & SCALE
# =====================================================================
print("⚙️ Engineering 5 behavioral features...")
features = pd.DataFrame(index=active_tickers)
features['Annual_Return'] = stock_returns.mean() * 252
features['Volatility'] = stock_returns.std() * np.sqrt(252)

market_variance = market_returns.var()
features['Beta'] = [stock_returns[t].cov(market_returns) / market_variance for t in active_tickers]
features['Momentum'] = (stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1

max_dd_list = []
for t in active_tickers:
    wealth_index = (1 + stock_returns[t]).cumprod()
    max_dd_list.append(((wealth_index - wealth_index.cummax()) / wealth_index.cummax()).min())
features['Max_Drawdown'] = max_dd_list

features = features.replace([np.inf, -np.inf], np.nan).dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow optimization iteration
wcss = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Render Final Massive Universe Elbow Plot
plt.figure(figsize=(8, 4.5))
plt.plot(k_range, wcss, marker='s', linestyle='-', color='indigo', linewidth=2)
plt.title(f'Elbow Curve: Scaled NASDAQ Directory Subset ({len(features)} Assets)', fontweight='bold')
plt.xlabel('Cluster Counts (K)')
plt.ylabel('Inertia / WCSS')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================================================
# 3. RUN THE FINAL CLUSTERING MODEL (K = 4)
# =====================================================================
print("🎯 Fitting final clustering model (K = 4)...")
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
features['Cluster'] = kmeans.fit_predict(scaled_features)

# =====================================================================
# 4. ISOLATE AND FILTER THE HIGHEST-PERFORMING CLUSTER
# =====================================================================
# Rank clusters by their group-level Sharpe Ratio (Return / Volatility)
cluster_performance = features.groupby('Cluster').agg({
    'Annual_Return': 'mean',
    'Volatility': 'mean'
})
cluster_performance['Group_Sharpe'] = cluster_performance['Annual_Return'] / cluster_performance['Volatility']

# Find the best cluster index
top_cluster_id = cluster_performance['Group_Sharpe'].idxmax()
top_cluster_stocks = features[features['Cluster'] == top_cluster_id]

print("\n" + "="*50)
print(f"🌟 HIGHEST PERFORMING CLUSTER ID: Group {top_cluster_id}")
print(f"📊 Average Cluster Metrics: Return: {cluster_performance.loc[top_cluster_id, 'Annual_Return']:.2%}, Volatility: {cluster_performance.loc[top_cluster_id, 'Volatility']:.2%}")
print(f"📋 Total Tickers in this Cluster: {len(top_cluster_stocks)}")
print("="*50)

print("\n✨ Tickers inside the Top-Performing Cluster:")
print(top_cluster_stocks.index.tolist())

print("\n🔎 Detailed breakdown of the top tickers in this cluster:")
print(top_cluster_stocks.round(4))

# Generate all unique pairs of our 5 features
feature_cols = ['Annual_Return', 'Volatility', 'Beta', 'Momentum', 'Max_Drawdown']
feature_pairs = list(itertools.combinations(feature_cols, 2))

# Setup a grid plot layout (e.g., 5 rows x 2 columns to show 10 combinations)
fig, axes = plt.subplots(5, 2, figsize=(14, 22))
axes = axes.flatten()

for i, (col_x, col_y) in enumerate(feature_pairs):
    ax = axes[i]
    
    # Scatter points colored by the global 5-feature cluster label
    scatter = ax.scatter(features[col_x], features[col_y], 
                         c=features['Cluster'], cmap='plasma', s=150, alpha=0.8, edgecolors='black')
    
    # Label every stock ticker on this specific projection
    for ticker in features.index:
        ax.annotate(ticker, (features.loc[ticker, col_x], features.loc[ticker, col_y]),
                    textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9, fontweight='semibold')
        
    ax.set_title(f'{col_y} vs {col_x}', fontsize=11, fontweight='bold')
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.grid(True, linestyle=':', alpha=0.6)

# Turn off any unused subplot frames if necessary
for j in range(len(feature_pairs), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Stock Clusters Displayed Across Unique 2D Projections', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()
