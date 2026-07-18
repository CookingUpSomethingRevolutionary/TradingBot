import yfinance as yf
import pandas as pd
import numpy as np

# 11 Official Select Sector SPDR ETFs
sector_tickers = {
    "XLK": "XLK", "XLV": "XLV", "XLF": "XLF", "XLY": "XLY", 
    "XLI": "XLI", "XLP": "XLP", "XLE": "XLE", "XLB": "XLB", 
    "XLU": "XLU", "XLRE": "XLRE", "XLC": "XLC"
}
BENCHMARK_TICKER = "SPY"

FRICTION_PCT = 0.0010  
DRIFT_BUFFER = 0.05    

print("Fetching historical data from maximum sector inception boundaries (Dec 1998)...")
raw_data = {}
for name, ticker in sector_tickers.items():
    df = yf.download(ticker, start="1998-12-16", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    raw_data[name] = df['Close']

df_spy = yf.download(BENCHMARK_TICKER, start="1998-12-16", interval="1d", progress=False)
if isinstance(df_spy.columns, pd.MultiIndex):
    df_spy.columns = df_spy.columns.droplevel(1)

# Compile master dataframe without dropping rows due to missing new assets
df_universe = pd.DataFrame(raw_data)
df_universe['US_Stocks'] = df_spy['Close']

# Only drop rows where the core benchmark index data is missing
df_universe = df_universe.dropna(subset=['US_Stocks'])

# Pre-calculate Benchmark Macro Trend Indicators (50 EMA & RSI)
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

print("Running Lookahead-Bias-Free Sector Rotation Simulation...")
last_rebalanced_week = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_week = (timestamp.isocalendar()[0], timestamp.isocalendar()[1])
    
    spy_price = current_row['US_Stocks']
    spy_ema = current_row['SPY_EMA50']
    spy_rsi = current_row['SPY_RSI']
    
    total_portfolio_value = sum(shares * current_row[asset] for asset, shares in current_holdings.items()) if current_holdings else portfolio_value
    
    # WEEKLY EXECUTION GATE
    if last_rebalanced_week is None or current_year_week != last_rebalanced_week:
        last_rebalanced_week = current_year_week
        lookback_row = df_universe.iloc[idx - lookback_period]
        
        # Dynamic Active Universe Discovery: Filter for assets trading both now and 1 year ago
        active_sectors = []
        for sector in sector_tickers.keys():
            if pd.notna(current_row[sector]) and pd.notna(lookback_row[sector]):
                active_sectors.append(sector)
                
        # Calculate momentum ranking scores for active assets only
        momentum_scores = {}
        for asset in active_sectors:
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
            
        ranked_sectors = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
        
        # Macro Filter Rules
        if spy_price > spy_ema and spy_rsi < 70:
            # Bull Market Regime: Pick the top 2 sectors with the strongest structural momentum
            target_assets = ranked_sectors[:2]
        else:
            # Bear/Overbought Regime: Pick defensive sectors only if they are trading and available
            defensive_pool = ['XLV', 'XLP', 'XLU']
            available_defensive = [s for s in ranked_sectors if s in defensive_pool]
            target_assets = available_defensive[:2] if len(available_defensive) >= 2 else ranked_sectors[:2]
            
        # Execute allocation changes if targets switch
        if set(current_holdings.keys()) != set(target_assets):
            total_friction_drag = 0
            
            # Formulate friction safely by avoiding 0 * NaN lookups on un-launched assets
            for asset in sector_tickers.keys():
                current_val = current_holdings[asset] * current_row[asset] if asset in current_holdings else 0.0
                ideal_val = total_portfolio_value * 0.50 if asset in target_assets else 0.0
                trade_volume = abs(ideal_val - current_val)
                total_friction_drag += trade_volume * FRICTION_PCT
                
            portfolio_value = total_portfolio_value - total_friction_drag
            current_holdings = {}
            
            allocation_slice = portfolio_value / 2
            for asset in target_assets:
                current_holdings[asset] = allocation_slice / current_row[asset]
                
            total_portfolio_value = sum(shares * current_row[asset] for asset, shares in current_holdings.items())

    date_timeline.append(timestamp)
    equity_timeline.append(total_portfolio_value)
    benchmark_timeline.append(bh_shares * current_row['US_Stocks'])

results_df = pd.DataFrame({"Strategy_Equity": equity_timeline, "Benchmark_Equity": benchmark_timeline}, index=date_timeline)
results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Multi-Inception Sector Backtest completed successfully with fixed math!")