import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
import time

BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 10000.0

# Start 2024 to build indicator lookback, but we only evaluate from 2025 onwards
TEST_START = "2024-01-01" 
EVAL_START = pd.Timestamp("2025-01-01")
TEST_END = "2026-07-18"

if os.path.exists("best_params.json"):
    with open("best_params.json", "r") as f:
        PARAMS = json.load(f)
    print("Loaded Out-of-Sample parameters from best_params.json")
else:
    print("CRITICAL ERROR: Run optimize.py first to generate best_params.json")
    exit(1)

print("Loading monthly constituent matrix for testing period...")
universe_map = pd.read_csv("sp500_monthly_2016_present.csv", parse_dates=["Date"], index_col="Date")
universe_map = universe_map[(universe_map.index >= EVAL_START) & (universe_map.index <= pd.Timestamp(TEST_END))]

all_historical_tickers = set()
for tickers_str in universe_map["Tickers"].dropna():
    for t in tickers_str.split(","):
        all_historical_tickers.add(t.strip().replace('.', '-'))

ticker_list = list(all_historical_tickers)
print(f"Downloading OOS market data for {len(ticker_list)} symbols...")

df_spy = yf.download(BENCHMARK_TICKER, start=TEST_START, end=TEST_END, progress=False)
spy_df = df_spy.copy()
if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df = spy_df.xs(BENCHMARK_TICKER, level=1, axis=1)
spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)

data_dict = {}
chunk_size = 40
for i in range(0, len(ticker_list), chunk_size):
    chunk = ticker_list[i:i+chunk_size]
    chunk_data = yf.download(chunk, start=TEST_START, end=TEST_END, progress=False)
    
    for ticker in chunk:
        try:
            if isinstance(chunk_data.columns, pd.MultiIndex):
                df = chunk_data.xs(ticker, level=1, axis=1).dropna().copy()
            else:
                df = chunk_data.dropna().copy()
                
            if len(df) > PARAMS['ema_len']:
                df['EMA'] = ta.ema(df['Close'], length=PARAMS['ema_len'])
                df['RSI'] = ta.rsi(df['Close'], length=PARAMS['rsi_len'])
                df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=PARAMS['cmf_len'])
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                data_dict[ticker] = df
        except Exception:
            continue
    time.sleep(0.5)

cash = INITIAL_CAPITAL
positions = {}
dates = spy_df.index[spy_df.index >= EVAL_START]

equity_timeline = []
benchmark_timeline = []
spy_initial_price = spy_df.loc[dates[0], 'Close']

last_rebalanced_month = None
active_pool = []

for date in dates:
    # Update active S&P 500 pool for the current month
    current_year_month = (date.year, date.month)
    if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
        last_rebalanced_month = current_year_month
        closest_date = universe_map.index[universe_map.index <= date].max()
        if pd.notna(closest_date):
            active_pool = [t.strip().replace('.', '-') for t in universe_map.loc[closest_date, "Tickers"].split(",")]

    # Position Exits
    to_remove = []
    for t, pos in positions.items():
        if date in data_dict[t].index:
            curr_close = data_dict[t].loc[date, 'Close']
            sl_price = pos['entry'] - (PARAMS['sl_mult'] * pos['atr'])
            tp_price = pos['entry'] + (PARAMS['tp_mult'] * pos['atr'])

            if curr_close < sl_price or curr_close > tp_price:
                cash += pos['shares'] * curr_close
                to_remove.append(t)
    for t in to_remove:
        del positions[t]

    spy_bullish = spy_df.loc[date, 'Close'] > spy_df.loc[date, 'SMA200'] if date in spy_df.index else False

    # Dynamic Entries against the Out-of-Sample Active Pool
    if spy_bullish and len(positions) < 4:
        for t in active_pool:
            if t in data_dict and date in data_dict[t].index and t not in positions:
                row = data_dict[t].loc[date]
                if pd.isna(row['EMA']) or pd.isna(row['RSI']) or pd.isna(row['CMF']):
                    continue

                if row['Close'] > row['EMA'] and row['CMF'] > PARAMS['cmf_thresh'] and PARAMS['rsi_lower'] < row['RSI'] < PARAMS['rsi_upper']:
                    total_portfolio_val = cash + sum(p['shares'] * data_dict[tk].loc[date, 'Close'] for tk, p in positions.items() if date in data_dict[tk].index)
                    risk_capital = total_portfolio_val * 0.015 
                    risk_per_share = row['ATR'] * PARAMS['sl_mult']

                    if risk_per_share > 0:
                        shares = int(risk_capital // risk_per_share)
                        cost = shares * row['Close']
                        if 0 < cost <= cash:
                            positions[t] = {'shares': shares, 'entry': row['Close'], 'atr': row['ATR']}
                            cash -= cost
            if len(positions) >= 4:
                break

    daily_equity = cash + sum(pos['shares'] * data_dict[t].loc[date, 'Close'] for t, pos in positions.items() if date in data_dict[t].index)
    benchmark_val = (INITIAL_CAPITAL / spy_initial_price) * spy_df.loc[date, 'Close']

    equity_timeline.append(daily_equity)
    benchmark_timeline.append(benchmark_val)

results_df = pd.DataFrame({"Strategy_Equity": equity_timeline, "Benchmark_Equity": benchmark_timeline}, index=dates)
results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Out-of-Sample Backtest dataset exported to backtest_results.csv")