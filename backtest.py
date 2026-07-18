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

# --- FRICTION CONTROL FACTOR ---
# Models 0.10% (10 basis points) of total execution drag per trade 
# to simulate bid-ask spreads, broker execution slippage, and SEC/TAF regulatory fees.
FRICTION_PCT = 0.0010  

print("Fetching historical data for asset universe...")
raw_data = {}
for name, ticker in asset_tickers.items():
    df = yf.download(ticker, start="2004-11-01", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    raw_data[name] = df['Close']

df_universe = pd.DataFrame(raw_data).dropna()

# Technical Indicators Pipeline
change = df_universe['US_Stocks'].diff()
gain = change.mask(change < 0, 0)
loss = -change.mask(change > 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean().replace(0, 0.00001)

rs = avg_gain / avg_loss
df_universe['SPY_RSI'] = 100 - (100 / (1 + rs))
df_universe['SPY_EMA50'] = df_universe['US_Stocks'].ewm(span=50, adjust=False).mean()

lookback_period = 252  
initial_capital = 10000
portfolio_value = initial_capital
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

current_holdings = {} 
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Friction-Adjusted Strategy Simulation (Weekly Rebalance)...")
last_rebalanced_week = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_week = (timestamp.isocalendar()[0], timestamp.isocalendar()[1])
    
    spy_price = current_row['US_Stocks']
    spy_ema = current_row['SPY_EMA50']
    spy_rsi = current_row['SPY_RSI']
    
    # Weekly Rebalance Gate
    if last_rebalanced_week is None or current_year_week != last_rebalanced_week:
        last_rebalanced_week = current_year_week
        lookback_row = df_universe.iloc[idx - lookback_period]
        momentum_scores = {}
        
        for asset in asset_tickers.keys():
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
            
        ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
        
        if spy_price > spy_ema and spy_rsi < 70:
            target_assets = ranked_assets[:2]
        else:
            safe_pool = [a for a in ranked_assets if a in ['Gold', 'Bonds', 'Real_Estate']]
            target_assets = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
            
        # Evaluate current book allocation states to calculate friction penalties
        total_liquid_cash = 0
        current_valuations = {}
        
        for asset in asset_tickers.keys():
            shares = current_holdings.get(asset, 0)
            asset_value = shares * current_row[asset]
            current_valuations[asset] = asset_value
            total_liquid_cash += asset_value
            
        if total_liquid_cash == 0:
            total_liquid_cash = portfolio_value
            
        ideal_allocation = total_liquid_cash / 2
        total_friction_drag = 0
        
        # Calculate exactly how much dollar value moves to compute realistic friction
        for asset in asset_tickers.keys():
            current_val = current_valuations.get(asset, 0)
            target_val = ideal_allocation if asset in target_assets else 0
            trade_volume = abs(target_val - current_val)
            total_friction_drag += trade_volume * FRICTION_PCT
            
        # Apply slippage/fee friction to net portfolio equity
        portfolio_value = total_liquid_cash - total_friction_drag
        
        # Re-distribute the net portfolio equity into target units
        allocation_allowance = portfolio_value / 2
        current_holdings = {}
        for asset in target_assets:
            current_holdings[asset] = allocation_allowance / current_row[asset]

    # Process Mark-to-Market Timeline Recording
    daily_portfolio_value = sum(shares * current_row[asset] for asset, shares in current_holdings.items()) if current_holdings else portfolio_value
    daily_bh_value = bh_shares * current_row['US_Stocks']
    
    date_timeline.append(timestamp)
    equity_timeline.append(daily_portfolio_value)
    benchmark_timeline.append(daily_bh_value)

results_df = pd.DataFrame({
    "Strategy_Equity": equity_timeline,
    "Benchmark_Equity": benchmark_timeline
}, index=date_timeline)

results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Friction-modeled backtest completed and written to 'backtest_results.csv'")