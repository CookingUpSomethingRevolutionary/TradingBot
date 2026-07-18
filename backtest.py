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

lookback_period = 252  
initial_capital = 10000
portfolio_value = initial_capital
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

current_holdings = {} 
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Dynamic Momentum Ranking Simulation (Weekly Rebalance)...")

# Track the calendar week number to detect when a new week begins
last_rebalanced_week = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    
    # .isocalendar() returns (year, week_number, weekday)
    # Using both year and week number prevents overlap between years
    current_year_week = (timestamp.isocalendar()[0], timestamp.isocalendar()[1])
    
    spy_price = current_row['US_Stocks']
    spy_ema = current_row['SPY_EMA50']
    spy_rsi = current_row['SPY_RSI']
    
    # REBALANCING TRIGGER: Triggers on the very first trading day available in any given week
    if last_rebalanced_week is None or current_year_week != last_rebalanced_week:
        last_rebalanced_week = current_year_week
        lookback_row = df_universe.iloc[idx - lookback_period]
        momentum_scores = {}
        
        # Calculate trailing 1-year performance momentum
        for asset in asset_tickers.keys():
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
            
        ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
        
        # Filter logic via Macro Conditions
        if spy_price > spy_ema and spy_rsi < 70:
            target_assets = ranked_assets[:2]
        else:
            safe_pool = [a for a in ranked_assets if a in ['Gold', 'Bonds', 'Real_Estate']]
            target_assets = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
            
        # Execute allocation adjustment if targets changed
        if set(current_holdings.keys()) != set(target_assets):
            if current_holdings:
                total_cash = 0
                for asset, shares in current_holdings.items():
                    total_cash += shares * current_row[asset]
                portfolio_value = total_cash
                current_holdings = {}
                
            allocation = portfolio_value / 2
            for asset in target_assets:
                current_holdings[asset] = allocation / current_row[asset]

    # --- RECORD REGULAR MARK-TO-MARKET VALUATIONS ---
    daily_portfolio_value = 0
    if current_holdings:
        for asset, shares in current_holdings.items():
            daily_portfolio_value += shares * current_row[asset]
    else:
        daily_portfolio_value = portfolio_value
        
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
print("✅ Weekly historical backtest completed! Metrics successfully written to 'backtest_results.csv'")