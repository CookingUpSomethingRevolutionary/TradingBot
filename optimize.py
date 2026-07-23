import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import optuna
import json
import logging
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCHMARK_TICKER = "SPY"
TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"

print("Loading monthly survivorship-free constituent matrix...")
try:
    universe_map = pd.read_csv("sp500_monthly_2016_present.csv", parse_dates=["Date"], index_col="Date")
    # Filter matrix to training period
    universe_map = universe_map[(universe_map.index >= pd.Timestamp(TRAIN_START)) & (universe_map.index <= pd.Timestamp(TRAIN_END))]
except FileNotFoundError:
    print("CRITICAL ERROR: sp500_monthly_2016_present.csv not found!")
    exit(1)

all_historical_tickers = set()
for tickers_str in universe_map["Tickers"].dropna():
    for t in tickers_str.split(","):
        all_historical_tickers.add(t.strip().replace('.', '-'))

ticker_list = list(all_historical_tickers)
print(f"Fetching raw market data for {len(ticker_list)} historical constituents...")

# Batch download to handle 500+ stocks
raw_data = pd.DataFrame()
chunk_size = 40
df_spy = yf.download(BENCHMARK_TICKER, start=TRAIN_START, end=TRAIN_END, progress=False)

# To optimize RAM and speed inside the trial, we download and store a dictionary of DataFrames
stock_data_cache = {}
for i in range(0, len(ticker_list), chunk_size):
    chunk = ticker_list[i:i+chunk_size]
    print(f"Downloading chunk {i//chunk_size + 1}...")
    chunk_data = yf.download(chunk, start=TRAIN_START, end=TRAIN_END, progress=False)
    
    for ticker in chunk:
        try:
            if isinstance(chunk_data.columns, pd.MultiIndex):
                single_stock = chunk_data.xs(ticker, level=1, axis=1).dropna()
            else:
                single_stock = chunk_data.dropna()
            if len(single_stock) > 100:
                stock_data_cache[ticker] = single_stock
        except Exception:
            continue
    time.sleep(0.5)

def objective(trial):
    ema_len = trial.suggest_int('ema_len', 50, 150, step=10)
    rsi_len = trial.suggest_int('rsi_len', 7, 21)
    rsi_lower = trial.suggest_int('rsi_lower', 40, 55)
    rsi_upper = trial.suggest_int('rsi_upper', 65, 80)
    cmf_len = trial.suggest_int('cmf_len', 10, 30)
    cmf_thresh = trial.suggest_float('cmf_thresh', 0.0, 0.12, step=0.02)
    sl_mult = trial.suggest_float('sl_mult', 1.5, 3.0, step=0.5)
    tp_mult = trial.suggest_float('tp_mult', 2.5, 5.0, step=0.5)

    # Compute indicators for this trial
    trial_data = {}
    for t, df in stock_data_cache.items():
        temp_df = df.copy()
        temp_df['EMA'] = ta.ema(temp_df['Close'], length=ema_len)
        temp_df['RSI'] = ta.rsi(temp_df['Close'], length=rsi_len)
        temp_df['CMF'] = ta.cmf(temp_df['High'], temp_df['Low'], temp_df['Close'], temp_df['Volume'], length=cmf_len)
        temp_df['ATR'] = ta.atr(temp_df['High'], temp_df['Low'], temp_df['Close'], length=14)
        trial_data[t] = temp_df

    spy_df = df_spy.copy()
    spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)

    cash = 10000.0
    positions = {}
    dates = spy_df.index[spy_df.index >= pd.Timestamp("2022-01-01")]
    
    last_rebalanced_month = None
    active_pool = []

    for date in dates:
        # Dynamic Pool Update (Survivorship Bias Elimination)
        current_year_month = (date.year, date.month)
        if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
            last_rebalanced_month = current_year_month
            closest_date = universe_map.index[universe_map.index <= date].max()
            if pd.notna(closest_date):
                active_pool = [t.strip().replace('.', '-') for t in universe_map.loc[closest_date, "Tickers"].split(",")]

        # Check exits
        to_remove = []
        for t, pos in positions.items():
            if date in trial_data[t].index:
                curr_close = trial_data[t].loc[date, 'Close']
                if curr_close < (pos['entry'] - sl_mult * pos['atr']) or curr_close > (pos['entry'] + tp_mult * pos['atr']):
                    cash += pos['shares'] * curr_close
                    to_remove.append(t)
        for t in to_remove:
            del positions[t]

        spy_bullish = spy_df.loc[date, 'Close'] > spy_df.loc[date, 'SMA200'] if date in spy_df.index else False

        # Scan entries (ONLY from the active_pool for this specific month)
        if spy_bullish and len(positions) < 4:
            for t in active_pool:
                if t in trial_data and date in trial_data[t].index and t not in positions:
                    row = trial_data[t].loc[date]
                    if pd.isna(row['EMA']) or pd.isna(row['RSI']) or pd.isna(row['CMF']):
                        continue

                    if row['Close'] > row['EMA'] and row['CMF'] > cmf_thresh and rsi_lower < row['RSI'] < rsi_upper:
                        portfolio_total = cash + sum(p['shares'] * trial_data[tk].loc[date, 'Close'] for tk, p in positions.items() if date in trial_data[tk].index)
                        risk_amount = portfolio_total * 0.015
                        risk_per_share = row['ATR'] * sl_mult

                        if risk_per_share > 0:
                            shares = int(risk_amount // risk_per_share)
                            cost = shares * row['Close']
                            if 0 < cost <= cash:
                                positions[t] = {'shares': shares, 'entry': row['Close'], 'atr': row['ATR']}
                                cash -= cost
                if len(positions) >= 4:
                    break

    final_val = cash
    for t, pos in positions.items():
        if dates[-1] in trial_data[t].index:
            final_val += pos['shares'] * trial_data[t].loc[dates[-1], 'Close']

    return final_val

if __name__ == "__main__":
    print("Running Bias-Free Optuna Search... (This will take time)")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30) # Reduced to 30 trials to balance compute time

    print("\n🚀 IN-SAMPLE OPTIMIZATION COMPLETE!")
    print(f"Optimal Terminal Equity: ${study.best_value:,.2f}")
    
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ Exported configuration to best_params.json")