import os
import sys
import json
import time
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ---------------------------------------------------------
# 1. Initialization & Configuration
# ---------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("CRITICAL ERROR: Alpaca credentials missing from environment!")
    sys.exit(1)

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

if os.path.exists("best_params.json"):
    with open("best_params.json", "r") as f:
        PARAMS = json.load(f)
    print("Loaded optimized parameters from best_params.json")
else:
    print("CRITICAL ERROR: best_params.json missing. Run optimize.py first.")
    sys.exit(1)

BENCHMARK = "SPY"
MAX_POSITIONS = 4
RISK_BUDGET = 0.015  # Risk 1.5% of total portfolio equity per position

# ---------------------------------------------------------
# 2. Data Fetching Functions
# ---------------------------------------------------------
def get_current_universe():
    """Reads the local CSV to get the most up-to-date S&P 500 tickers."""
    try:
        universe_map = pd.read_csv("sp500_monthly_2016_present.csv", parse_dates=["Date"], index_col="Date")
        latest_row = universe_map.iloc[-1]
        active_symbols = [t.strip().replace('.', '-') for t in latest_row["Tickers"].split(",")]
        return active_symbols
    except Exception as e:
        print(f"Error loading universe matrix: {e}")
        return []

def check_market_regime():
    """Checks if SPY is above its 200-day SMA."""
    spy_df = yf.download(BENCHMARK, period="1y", interval="1d", progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df = spy_df.xs(BENCHMARK, level=1, axis=1)
    spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)
    latest = spy_df.iloc[-1]
    return latest['Close'] > latest['SMA200']

def batch_download_and_calculate(tickers):
    """Downloads OHLCV data in chunks and calculates technical indicators."""
    print(f"Batch downloading data for {len(tickers)} tickers...")
    chunk_size = 50
    valid_data = {}
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        chunk_data = yf.download(chunk, period="1y", interval="1d", progress=False)
        
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
                    valid_data[ticker] = df.iloc[-1]
            except Exception:
                continue
        time.sleep(0.5) # Prevent rate limiting
        
    return valid_data

# ---------------------------------------------------------
# 3. Core Rebalancing Logic
# ---------------------------------------------------------
def rebalance_portfolio():
    print("Initiating Weekly Portfolio Rebalance...")
    
    account = trading_client.get_account()
    total_equity = float(account.portfolio_value)
    current_positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    
    # Check Macro Regime
    is_bull_market = check_market_regime()
    if not is_bull_market:
        print("Market Regime: SPY below 200 SMA (Bearish). Liquidating all positions to cash.")
        target_portfolio = {} # Empty portfolio means sell everything
    else:
        # Build Target Portfolio
        universe = get_current_universe()
        universe.extend(list(current_positions.keys())) # Ensure current holdings are evaluated
        universe = list(set(universe)) # Remove duplicates
        
        market_data = batch_download_and_calculate(universe)
        
        passed_candidates = []
        for ticker, data in market_data.items():
            if pd.isna(data['EMA']) or pd.isna(data['RSI']) or pd.isna(data['CMF']):
                continue
            
            # The Technical Filters
            if data['Close'] > data['EMA'] and data['CMF'] > PARAMS['cmf_thresh']:
                if PARAMS['rsi_lower'] < data['RSI'] < PARAMS['rsi_upper']:
                    passed_candidates.append({
                        'symbol': ticker,
                        'rsi': data['RSI'],
                        'atr': data['ATR'],
                        'close': data['Close']
                    })
        
        # Rank by RSI (Momentum) and select top candidates
        ranked_candidates = sorted(passed_candidates, key=lambda x: x['rsi'], reverse=True)
        top_targets = ranked_candidates[:MAX_POSITIONS]
        
        # Calculate target shares for each top stock
        target_portfolio = {}
        for target in top_targets:
            risk_per_share = target['atr'] * PARAMS['sl_mult']
            if risk_per_share > 0:
                target_shares = int((total_equity * RISK_BUDGET) // risk_per_share)
                if target_shares > 0:
                    target_portfolio[target['symbol']] = target_shares

    # ---------------------------------------------------------
    # 4. Execution (Sells First, then Buys)
    # ---------------------------------------------------------
    print("\n--- Generating Execution Plan ---")
    
    sells = {}
    buys = {}
    
    # Evaluate Sells (Liquidations and Trims)
    for ticker, current_qty in current_positions.items():
        if ticker not in target_portfolio:
            sells[ticker] = current_qty # Full liquidation
            print(f"SELL PLAN: Liquidating {current_qty} shares of {ticker} (Fell out of Top 4)")
        else:
            target_qty = target_portfolio[ticker]
            if current_qty > target_qty:
                sells[ticker] = current_qty - target_qty
                print(f"SELL PLAN: Trimming {sells[ticker]} shares of {ticker} to reach target weight")
                
    # Evaluate Buys (New Positions and Additions)
    for ticker, target_qty in target_portfolio.items():
        current_qty = current_positions.get(ticker, 0)
        if target_qty > current_qty:
            buys[ticker] = target_qty - current_qty
            print(f"BUY PLAN: Accumulating {buys[ticker]} shares of {ticker} to reach target weight")

    # Execute Orders
    if not sells and not buys:
        print("Portfolio is perfectly balanced. No trades required.")
        return

    print("\nExecuting SELL orders to free up capital...")
    for ticker, qty in sells.items():
        trading_client.submit_order(
            order_data=MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        )
        
    print("Waiting 3 seconds for SELLS to clear...")
    time.sleep(3) # Give Alpaca a moment to settle cash
    
    print("\nExecuting BUY orders...")
    for ticker, qty in buys.items():
        try:
            trading_client.submit_order(
                order_data=MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            )
        except Exception as e:
            print(f"Failed to buy {ticker}: {e} (Likely insufficient settled cash)")

    print("✅ Weekly Rebalance Complete.")

if __name__ == "__main__":
    if trading_client.get_clock().is_open:
        rebalance_portfolio()
    else:
        print("Market closed. Execution paused.")