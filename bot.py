import os
import sys
import json
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("CRITICAL ERROR: Alpaca API keys missing from environment!")
    sys.exit(1)

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Load parameters
if os.path.exists("best_params.json"):
    with open("best_params.json", "r") as f:
        PARAMS = json.load(f)
else:
    PARAMS = {
        "ema_len": 100, "rsi_len": 14, "rsi_lower": 45, "rsi_upper": 70,
        "cmf_len": 20, "cmf_thresh": 0.05, "sl_mult": 2.0, "tp_mult": 3.0
    }

WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AMD', 'GOOGL', 'COST', 'AVGO', 'NFLX']
BENCHMARK = "SPY"

def get_indicators(symbol):
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1)
    df = df.dropna()
    if len(df) > PARAMS['ema_len']:
        df['EMA'] = ta.ema(df['Close'], length=PARAMS['ema_len'])
        df['RSI'] = ta.rsi(df['Close'], length=PARAMS['rsi_len'])
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=PARAMS['cmf_len'])
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        return df.iloc[-1]
    return None

def check_market_regime():
    spy_df = yf.download(BENCHMARK, period="1y", interval="1d", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df = spy_df.xs(BENCHMARK, level=1, axis=1)
    spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)
    latest = spy_df.iloc[-1]
    return latest['Close'] > latest['SMA200']

def manage_positions():
    print("Scanning active positions for ATR Stop/Take-Profit exits...")
    positions = trading_client.get_all_positions()

    for pos in positions:
        tech = get_indicators(pos.symbol)
        if tech is not None:
            curr_p = float(pos.current_price)
            avg_entry = float(pos.avg_entry_price)
            atr = tech['ATR']

            stop_loss = avg_entry - (PARAMS['sl_mult'] * atr)
            take_profit = avg_entry + (PARAMS['tp_mult'] * atr)

            if curr_p < stop_loss or curr_p > take_profit:
                print(f"Risk rule triggered for {pos.symbol}. Liquidating position...")
                trading_client.submit_order(
                    order_data=MarketOrderRequest(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                )

def scan_entries():
    if not check_market_regime():
        print("Market Regime: SPY below 200 SMA (Bearish). Entry scans paused.")
        return

    positions = trading_client.get_all_positions()
    open_symbols = [p.symbol for p in positions]

    if len(open_symbols) >= 4:
        print("Portfolio fully allocated (Max 4 trades).")
        return

    account = trading_client.get_account()
    portfolio_equity = float(account.portfolio_value)
    cash_available = float(account.cash)

    risk_budget = portfolio_equity * 0.015  # 1.5% Risk Sizing

    for ticker in WATCHLIST:
        if ticker in open_symbols:
            continue

        tech = get_indicators(ticker)
        if tech is not None:
            is_uptrend = tech['Close'] > tech['EMA']
            has_volume = tech['CMF'] > PARAMS['cmf_thresh']
            has_momentum = PARAMS['rsi_lower'] < tech['RSI'] < PARAMS['rsi_upper']

            if is_uptrend and has_volume and has_momentum:
                risk_per_share = tech['ATR'] * PARAMS['sl_mult']
                shares_to_buy = int(risk_budget // risk_per_share)
                cost = shares_to_buy * tech['Close']

                if shares_to_buy > 0 and cost <= cash_available:
                    print(f"BUY SIGNAL IDENTIFIED -> Ticker: {ticker} | Shares: {shares_to_buy}")
                    trading_client.submit_order(
                        order_data=MarketOrderRequest(
                            symbol=ticker,
                            qty=shares_to_buy,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                    )
                    open_symbols.append(ticker)
                    if len(open_symbols) >= 4:
                        break

if __name__ == "__main__":
    if trading_client.get_clock().is_open:
        manage_positions()
        scan_entries()
    else:
        print("Market is closed. Bot standing by.")