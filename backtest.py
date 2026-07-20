import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 10000.0

# Load optimized parameters or fallback to defaults
if os.path.exists("best_params.json"):
    with open("best_params.json", "r") as f:
        PARAMS = json.load(f)
    print("Loaded optimized parameters from best_params.json")
else:
    PARAMS = {
        "ema_len": 100, "rsi_len": 14, "rsi_lower": 45, "rsi_upper": 70,
        "cmf_len": 20, "cmf_thresh": 0.05, "sl_mult": 2.0, "tp_mult": 3.0
    }
    print("Using default strategy parameters...")

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AMD', 'GOOGL', 'COST', 'AVGO', 'NFLX']
START_DATE, END_DATE = "2021-01-01", "2026-07-18"

print("Downloading backtest market historical data...")
raw_data = yf.download(TICKERS + [BENCHMARK_TICKER], start=START_DATE, end=END_DATE, progress=False)

data_dict = {}
for t in TICKERS:
    df = raw_data.xs(t, level=1, axis=1).dropna().copy()
    df['EMA'] = ta.ema(df['Close'], length=PARAMS['ema_len'])
    df['RSI'] = ta.rsi(df['Close'], length=PARAMS['rsi_len'])
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=PARAMS['cmf_len'])
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    data_dict[t] = df

spy_df = raw_data.xs(BENCHMARK_TICKER, level=1, axis=1).copy()
spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)

cash = INITIAL_CAPITAL
positions = {}
dates = spy_df.index[spy_df.index >= pd.Timestamp("2022-01-01")]

equity_timeline = []
benchmark_timeline = []
spy_initial_price = spy_df.loc[dates[0], 'Close']

for date in dates:
    # 1. Handle Active Position Exits
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

    # 2. Check Macro Regime
    spy_bullish = spy_df.loc[date, 'Close'] > spy_df.loc[date, 'SMA200'] if date in spy_df.index else False

    # 3. Handle Position Entries
    if spy_bullish and len(positions) < 4:
        for t, df in data_dict.items():
            if date in df.index and t not in positions:
                row = df.loc[date]
                if pd.isna(row['EMA']) or pd.isna(row['RSI']) or pd.isna(row['CMF']):
                    continue

                if row['Close'] > row['EMA'] and row['CMF'] > PARAMS['cmf_thresh'] and PARAMS['rsi_lower'] < row['RSI'] < PARAMS['rsi_upper']:
                    total_portfolio_val = cash + sum(
                        p['shares'] * data_dict[tk].loc[date, 'Close']
                        for tk, p in positions.items() if date in data_dict[tk].index
                    )
                    risk_capital = total_portfolio_val * 0.015  # 1.5% Risk allocation per trade
                    risk_per_share = row['ATR'] * PARAMS['sl_mult']

                    if risk_per_share > 0:
                        shares = int(risk_capital // risk_per_share)
                        cost = shares * row['Close']
                        if 0 < cost <= cash:
                            positions[t] = {'shares': shares, 'entry': row['Close'], 'atr': row['ATR']}
                            cash -= cost
            if len(positions) >= 4:
                break

    # Calculate Daily Valuations
    daily_equity = cash + sum(
        pos['shares'] * data_dict[t].loc[date, 'Close']
        for t, pos in positions.items() if date in data_dict[t].index
    )
    benchmark_val = (INITIAL_CAPITAL / spy_initial_price) * spy_df.loc[date, 'Close']

    equity_timeline.append(daily_equity)
    benchmark_timeline.append(benchmark_val)

results_df = pd.DataFrame({
    "Strategy_Equity": equity_timeline,
    "Benchmark_Equity": benchmark_timeline
}, index=dates)

results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Technical Backtest Complete! Saved to backtest_results.csv")