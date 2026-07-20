import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import optuna
import json
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

BENCHMARK_TICKER = "SPY"
START_DATE = "2021-01-01"
END_DATE = "2026-07-18"

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AMD', 'GOOGL', 'COST', 'AVGO', 'NFLX']

print(f"Fetching market data for {len(TICKERS)} tickers...")
raw_data = yf.download(TICKERS + [BENCHMARK_TICKER], start=START_DATE, end=END_DATE, progress=False)

def objective(trial):
    ema_len = trial.suggest_int('ema_len', 50, 150, step=10)
    rsi_len = trial.suggest_int('rsi_len', 7, 21)
    rsi_lower = trial.suggest_int('rsi_lower', 40, 55)
    rsi_upper = trial.suggest_int('rsi_upper', 65, 80)
    cmf_len = trial.suggest_int('cmf_len', 10, 30)
    cmf_thresh = trial.suggest_float('cmf_thresh', 0.0, 0.15, step=0.02)
    sl_mult = trial.suggest_float('sl_mult', 1.5, 3.0, step=0.5)
    tp_mult = trial.suggest_float('tp_mult', 2.5, 5.0, step=0.5)

    data_dict = {}
    for t in TICKERS:
        df = raw_data.xs(t, level=1, axis=1).dropna().copy()
        if len(df) > ema_len:
            df['EMA'] = ta.ema(df['Close'], length=ema_len)
            df['RSI'] = ta.rsi(df['Close'], length=rsi_len)
            df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=cmf_len)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            data_dict[t] = df

    spy_df = raw_data.xs(BENCHMARK_TICKER, level=1, axis=1).copy()
    spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)

    cash = 10000
    positions = {}
    dates = spy_df.index[spy_df.index >= pd.Timestamp("2022-01-01")]

    for date in dates:
        # Check exits
        to_remove = []
        for t, pos in positions.items():
            if date in data_dict[t].index:
                curr_close = data_dict[t].loc[date, 'Close']
                if curr_close < (pos['entry'] - sl_mult * pos['atr']) or curr_close > (pos['entry'] + tp_mult * pos['atr']):
                    cash += pos['shares'] * curr_close
                    to_remove.append(t)
        for t in to_remove:
            del positions[t]

        # Check SPY macro regime
        spy_bullish = spy_df.loc[date, 'Close'] > spy_df.loc[date, 'SMA200'] if date in spy_df.index else False

        # Scan entries
        if spy_bullish and len(positions) < 4:
            for t, df in data_dict.items():
                if date in df.index and t not in positions:
                    row = df.loc[date]
                    if pd.isna(row['EMA']) or pd.isna(row['RSI']) or pd.isna(row['CMF']):
                        continue

                    if row['Close'] > row['EMA'] and row['CMF'] > cmf_thresh and rsi_lower < row['RSI'] < rsi_upper:
                        portfolio_total = cash + sum(
                            p['shares'] * data_dict[tk].loc[date, 'Close']
                            for tk, p in positions.items() if date in data_dict[tk].index
                        )
                        risk_amount = portfolio_total * 0.015  # 1.5% Risk Sizing
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
        if dates[-1] in data_dict[t].index:
            final_val += pos['shares'] * data_dict[t].loc[dates[-1], 'Close']

    return final_val

if __name__ == "__main__":
    print("Running Optuna Hyperparameter Optimization (50 Trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\n🚀 OPTIMIZATION COMPLETE!")
    print(f"Optimal Strategy Portfolio Output: ${study.best_value:,.2f}")
    print("Top Parameters Found:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")

    # Save parameters locally
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ Saved parameters to best_params.json")