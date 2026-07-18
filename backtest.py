import yfinance as yf
import pandas as pd
import numpy as np
import time

BENCHMARK_TICKER = "SPY"
FRICTION_PCT = 0.0010  # 10 bps transaction drag
DRIFT_BUFFER = 0.05    

# 1. Load the user's monthly S&P 500 historical constituent mapping file
print("Loading monthly survivorship-free constituent matrix...")
try:
    universe_map = pd.read_csv("sp500_monthly_2016_present.csv", parse_dates=["Date"], index_col="Date")
except FileNotFoundError:
    print("CRITICAL ERROR: sp500_monthly_2016_present.csv not found in the working directory!")
    exit(1)

# Compile a master unique set of all symbols that ever entered the index to batch download data
all_historical_tickers = set()
for tickers_str in universe_map["Tickers"].dropna():
    for t in tickers_str.split(","):
        all_historical_tickers.add(t.strip().replace('.', '-'))

# Set dates to ensure baseline lookback availability for the 2016 start line
start_date = "2015-01-01"
end_date = "2026-07-18"

print("Downloading benchmark index data...")
df_spy = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date, interval="1d", progress=False)
df_universe = pd.DataFrame(index=df_spy.index)
df_universe['US_Stocks'] = df_spy['Close'] if not isinstance(df_spy.columns, pd.MultiIndex) else df_spy['Close'][BENCHMARK_TICKER]
df_universe['SPY_SMA200'] = df_universe['US_Stocks'].rolling(window=200).mean()

print(f"Downloading historical price data for all {len(all_historical_tickers)} unique S&P 500 symbols...")
ticker_list = list(all_historical_tickers)
chunk_size = 40
close_prices = pd.DataFrame(index=df_universe.index)

for i in range(0, len(ticker_list), chunk_size):
    chunk = ticker_list[i:i+chunk_size]
    print(f"Downloading symbol chunk {i//chunk_size + 1} of {len(ticker_list)//chunk_size + 1}...")
    chunk_data = yf.download(chunk, start=start_date, end=end_date, interval="1d", progress=False)
    if isinstance(chunk_data.columns, pd.MultiIndex):
        chunk_close = chunk_data['Close']
    else:
        chunk_close = chunk_data
    for col in chunk_close.columns:
        close_prices[col] = chunk_close[col]
    time.sleep(1)

close_prices = close_prices.ffill()

lookback_period = 252
initial_capital = 10000
portfolio_value = initial_capital

# Align start index to ensure we are at or past 2016-01-01 and have baseline lookback rows
start_idx = 0
for i, timestamp in enumerate(df_universe.index):
    if timestamp >= pd.Timestamp("2016-01-01") and i >= lookback_period:
        start_idx = i
        break

bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[start_idx]

current_holdings = {}  
cash_pool = portfolio_value
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Monthly Bias-Free Rebalancing Loop...")
last_rebalanced_month = None

for idx in range(start_idx, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_month = (timestamp.year, timestamp.month)
    
    spy_price = current_row['US_Stocks']
    spy_sma = current_row['SPY_SMA200']
    
    total_portfolio_value = cash_pool + sum(
        shares * close_prices.loc[timestamp, asset] 
        for asset, shares in current_holdings.items() if asset in close_prices.columns and pd.notna(close_prices.loc[timestamp, asset])
    )
    
    if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
        last_rebalanced_month = current_year_month
        
        # Pull the closest preceding monthly pool row from the uploaded CSV file
        closest_date = universe_map.index[universe_map.index <= timestamp].max()
        if pd.isna(closest_date):
            continue
            
        tickers_string = universe_map.loc[closest_date, "Tickers"]
        active_constituents = [t.strip().replace('.', '-') for t in tickers_string.split(",")]
        
        lookback_timestamp = df_universe.index[idx - lookback_period]
        valid_pool = []
        for ticker in active_constituents:
            if ticker in close_prices.columns:
                if pd.notna(close_prices.loc[timestamp, ticker]) and pd.notna(close_prices.loc[lookback_timestamp, ticker]):
                    valid_pool.append(ticker)
        
        # Calculate momentum metrics: M = \frac{P_{end} - P_{start}}{P_{start}}
        momentum_scores = {}
        for asset in valid_pool:
            p_start = close_prices.loc[lookback_timestamp, asset]
            p_end = close_prices.loc[timestamp, asset]
            if p_start > 0:
                momentum_scores[asset] = (p_end - p_start) / p_start
            
        valid_candidates = [asset for asset, score in momentum_scores.items() if score > 0]
        ranked_stocks = sorted(valid_candidates, key=momentum_scores.get, reverse=True)
        
        target_weights = {ticker: 0.0 for ticker in all_historical_tickers}
        target_weights["Cash"] = 0.0
        
        if pd.notna(spy_sma) and spy_price > spy_sma and len(ranked_stocks) > 0:
            leaders = ranked_stocks[:5]
            weight_per_asset = 1.0 / len(leaders)
            for asset in leaders:
                target_weights[asset] = weight_per_asset
        else:
            target_weights["Cash"] = 1.0
            
        current_weights = {t: 0.0 for t in all_historical_tickers}
        for asset in list(current_holdings.keys()):
            if asset in close_prices.columns:
                p = close_prices.loc[timestamp, asset]
                current_weights[asset] = (current_holdings[asset] * p) / total_portfolio_value if total_portfolio_value > 0 else 0.0
        current_weights["Cash"] = cash_pool / total_portfolio_value if total_portfolio_value > 0 else 0.0
        
        trigger_trade = False
        for asset in target_weights.keys():
            if abs(target_weights[asset] - current_weights.get(asset, 0.0)) > DRIFT_BUFFER:
                trigger_trade = True
                break
                
        if trigger_trade:
            total_friction_drag = 0
            for asset in current_holdings.keys():
                if asset in close_prices.columns:
                    current_val = current_holdings[asset] * close_prices.loc[timestamp, asset]
                    ideal_val = total_portfolio_value * target_weights.get(asset, 0.0)
                    total_friction_drag += abs(ideal_val - current_val) * FRICTION_PCT
            
            net_portfolio_value = total_portfolio_value - total_friction_drag
            current_holdings = {}
            cash_pool = net_portfolio_value * target_weights["Cash"]
            
            for asset, weight in target_weights.items():
                if asset != "Cash" and weight > 0:
                    p = close_prices.loc[timestamp, asset]
                    if pd.notna(p) and p > 0:
                        current_holdings[asset] = (net_portfolio_value * weight) / p
                        
            total_portfolio_value = cash_pool + sum(shares * close_prices.loc[timestamp, asset] for asset, shares in current_holdings.items())

    date_timeline.append(timestamp)
    equity_timeline.append(total_portfolio_value)
    benchmark_timeline.append(bh_shares * spy_price)

results_df = pd.DataFrame({"Strategy_Equity": equity_timeline, "Benchmark_Equity": benchmark_timeline}, index=date_timeline)
results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Bias-Free Dynamic S&P 500 Backtest complete!")