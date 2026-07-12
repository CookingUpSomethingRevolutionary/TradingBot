import yfinance as yf
import pandas as pd
import numpy as np

# 1. Fetch data for the various decade assets
# Using proxy tickers that yfinance can track historically
tickers = {
    "SP500": "^GSPC",     # Market Trend Filter & 80s/90s Growth
    "NASDAQ": "^NDX",     # 2010s+ Growth
    "GOLD": "GC=F",       # 1970s / 2000s Defense
    "BONDS": "TLT"        # Modern Defensive Proxy (Note: TLT starts in 2002, we use Cash/SP500 proxies prior)
}

print("Fetching multi-asset historical data...")
data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start="1970-01-01", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    data[name] = df

# Align data frames to a unified timeline starting in 1975
df_main = pd.DataFrame(index=data["SP500"].index)
df_main["SP500_Close"] = data["SP500"]["Close"]

# Calculate our core market filter on the S&P 500
change = df_main['SP500_Close'].diff()
gain = change.mask(change < 0, 0)
loss = -change.mask(change > 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
df_main['RSI'] = 100 - (100 / (1 + rs))
import yfinance as yf
import pandas as pd
import numpy as np

# 1. Define the Global Macro Asset Universe
asset_tickers = {
    "US_Stocks": "SPY",
    "Tech_Stocks": "QQQ",
    "Gold": "GLD",
    "Bonds": "TLT",
    "Real_Estate": "VNQ"
}

print("Fetching historical data for asset universe...")
raw_data = {}
for name, ticker in asset_tickers.items():
    df = yf.download(ticker, start="2004-01-01", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    raw_data[name] = df['Close']

# Combine into a single master DataFrame
df_universe = pd.DataFrame(raw_data).dropna()

# Use S&P 500 (SPY) to calculate our core macro trend filter
change = df_universe['US_Stocks'].diff()
gain = change.mask(change < 0, 0)
loss = -change.mask(change > 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
df_universe['SPY_RSI'] = 100 - (100 / (1 + rs))
df_universe['SPY_EMA50'] = df_universe['US_Stocks'].ewm(span=50, adjust=False).mean()

# 2. Strategy Parameters
lookback_period = 252  # 12 months of trading days to calculate momentum
portfolio_value = 10000
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

# Track current holdings: {asset_name: shares_held}
current_holdings = {} 

print("Running Dynamic Momentum Ranking Simulation (2005 - 2026)...")

# We loop day-by-day starting after the first lookback year
for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    
    # Core Market Health Check
    spy_price = current_row['US_Stocks']
    spy_ema = current_row['SPY_EMA50']
    spy_rsi = current_row['SPY_RSI']
    
    # REBALANCING RULE: To minimize trading costs, we re-rank on the first trading day of every year
    if timestamp.strftime('%m-%d') == df_universe.index[df_universe.index.year == timestamp.year][0].strftime('%m-%d') or idx == lookback_period:
        
        # Calculate 12-month returns for all assets to rank them
        lookback_row = df_universe.iloc[idx - lookback_period]
        momentum_scores = {}
        
        for asset in asset_tickers.keys():
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
            
        # Sort assets by performance (highest to lowest)
        ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
        
        # Determine our 2 target assets based on macro market health
        if spy_price > spy_ema and spy_rsi < 70:
            # Market is healthy: Hold the top 2 absolute momentum leaders
            target_assets = ranked_assets[:2]
        else:
            # Defensive Regime: Force allocation into the top 2 safest momentum assets
            # Filter out pure equities (US & Tech) if the broader market is crashing
            safe_pool = [a for a in ranked_assets if a in ['Gold', 'Bonds', 'Real_Estate']]
            target_assets = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
            
        # Check if we need to change our portfolio setup
        if set(current_holdings.keys()) != set(target_assets):
            # Liquidate everything to cash first
            if current_holdings:
                total_cash = 0
                for asset, shares in current_holdings.items():
                    total_cash += shares * current_row[asset]
                portfolio_value = total_cash
                current_holdings = {}
                
            # Split cash 50/50 into the new top 2 ranked assets
            allocation = portfolio_value / 2
            for asset in target_assets:
                current_holdings[asset] = allocation / current_row[asset]

# Final day Mark-to-Market
final_row = df_universe.iloc[-1]
final_portfolio_value = 0
for asset, shares in current_holdings.items():
    final_portfolio_value += shares * final_row[asset]

bh_final_value = bh_shares * final_row['US_Stocks']

# 3. Print Results
print("\n" + "="*60)
print("       DYNAMIC MOMENTUM RANKING PORTFOLIO VERDICT")
print("="*60)
print(f"Timeline Covered:                 2005 to 2026")
print(f"Final Value (Top 2 Momentum):     ${final_portfolio_value:,.2f}")
print(f"Final Value (S&P 500 Buy & Hold): ${bh_final_value:,.2f}")
print("-" * 60)
print(f"Strategy Net Return:             {((final_portfolio_value - 10000)/10000)*100:+.2f}%")
print(f"Benchmark Net Return:            {((bh_final_value - 10000)/10000)*100:+.2f}%")
print("="*60)