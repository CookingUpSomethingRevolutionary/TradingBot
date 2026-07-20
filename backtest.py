import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os

BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 10000

print("Loading monthly survivorship-free constituent matrix...")
try:
    universe_map = pd.read_csv("sp500_monthly_2016_present.csv", parse_dates=["Date"], index_col="Date")
except FileNotFoundError:
    print("CRITICAL ERROR: sp500_monthly_2016_present.csv not found!")
    exit(1)

all_historical_tickers = set()
for tickers_str in universe_map["Tickers"].dropna():
    for t in tickers_str.split(","):
        all_historical_tickers.add(t.strip().replace('.', '-'))

ticker_list = list(all_historical_tickers)[:50] # LIMITING TO 50 FOR SPEED. Remove `[:50]` for full S&P 500
start_date, end_date = "2020-01-01", "2026-07-18"

print("Downloading OHLCV data for technical indicators...")
df_spy = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date, interval="1d", progress=False)

data_dict = {}
chunk_size = 20
for i in range(0, len(ticker_list), chunk_size):
    chunk = ticker_list[i:i+chunk_size]
    print(f"Downloading chunk {i//chunk_size + 1}...")
    chunk_data = yf.download(chunk, start=start_date, end=end_date, interval="1d", progress=False)
    
    for ticker in chunk:
        try:
            if isinstance(chunk_data.columns, pd.MultiIndex):
                single_stock = chunk_data.xs(ticker, level=1, axis=1).dropna()
            else:
                single_stock = chunk_data.dropna()
                
            if len(single_stock) > 100:
                # Calculate Indicators using pandas_ta
                single_stock.ta.ema(length=100, append=True) # 20-week approx
                single_stock.ta.rsi(length=14, append=True)
                single_stock.ta.cmf(length=20, append=True)
                single_stock.ta.atr(length=14, append=True)
                data_dict[ticker] = single_stock
        except Exception:
            pass

print("Running Daily Technical Rebalancing Loop...")
portfolio_value = INITIAL_CAPITAL
cash = INITIAL_CAPITAL
positions = {} # format: {ticker: {'shares': x, 'entry_price': y, 'atr_at_entry': z}}
equity_curve = []
dates = df_spy.index[df_spy.index >= pd.Timestamp("2021-01-01")]

for date in dates:
    # 1. Check open positions for Exits (Stop Loss / Take Profit)
    tickers_to_remove = []
    for ticker, pos_data in positions.items():
        if ticker in data_dict and date in data_dict[ticker].index:
            current_close = data_dict[ticker].loc[date, 'Close']
            entry = pos_data['entry_price']
            atr = pos_data['atr_at_entry']
            
            # Risk Rules: Stop Loss at 2x ATR, Take profit at 3x ATR
            if current_close < (entry - 2 * atr) or current_close > (entry + 3 * atr):
                cash += pos_data['shares'] * current_close
                tickers_to_remove.append(ticker)
                
    for t in tickers_to_remove:
        del positions[t]

    # 2. Scan for New Entries
    if len(positions) < 5: # Max 5 active trades
        for ticker, df in data_dict.items():
            if date in df.index and ticker not in positions:
                row = df.loc[date]
                # Technical Strategy Logic
                if pd.notna(row['EMA_100']) and pd.notna(row['RSI_14']) and pd.notna(row['CMF_20']):
                    is_uptrend = row['Close'] > row['EMA_100']
                    has_volume = row['CMF_20'] > 0.05
                    has_momentum = 50 < row['RSI_14'] < 70
                    
                    if is_uptrend and has_volume and has_momentum:
                        available_cash_per_trade = cash / (5 - len(positions))
                        shares = available_cash_per_trade // row['Close']
                        if shares > 0:
                            positions[ticker] = {
                                'shares': shares,
                                'entry_price': row['Close'],
                                'atr_at_entry': row['ATRr_14']
                            }
                            cash -= shares * row['Close']
            if len(positions) >= 5:
                break
                
    # Calculate daily equity
    daily_val = cash
    for ticker, pos in positions.items():
        if ticker in data_dict and date in data_dict[ticker].index:
            daily_val += pos['shares'] * data_dict[ticker].loc[date, 'Close']
    equity_curve.append(daily_val)

results_df = pd.DataFrame({"Strategy_Equity": equity_curve}, index=dates)
results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Technical Backtest complete!")