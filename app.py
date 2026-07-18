import pandas as pd
import numpy as np
import time
import sys
import os
import json

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
STATE_FILE = "bot_state.json"
MAX_DRAWDOWN_LIMIT = 0.15
DRIFT_BUFFER = 0.05    

if not API_KEY or not SECRET_KEY:
    print("CRITICAL ERROR: Alpaca API keys are missing!")
    sys.exit(1)

try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    account = trading_client.get_account()
except Exception as e:
    print(f"Error authenticating with Alpaca API: {e}")
    sys.exit(1)

# Combined Active Asset Universe Dictionary
core_universe = {"US_Stocks": "SPY", "Tech_Stocks": "QQQ", "Gold": "GLD", "Bonds": "TLT", "Real_Estate": "VNQ"}
CASH_PROXY_TICKER = "BIL"

def load_or_init_state(current_equity):
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception: pass
    return {"peak_equity": current_equity, "is_tripped": False, "last_rebalanced_month": 0}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e: print(f"⚠️ State save error: {e}")

def calculate_dynamic_targets():
    print("Fetching historical data arrays for Ensemble Matrix calculations...")
    ensemble_scores = {}
    asset_sma200s = {}
    asset_prices = {}
    
    start_date = pd.Timestamp.now() - pd.DateOffset(months=16)
    all_tickers = list(core_universe.values()) + [CASH_PROXY_TICKER]
    
    request_params = StockBarsRequest(symbol_or_symbols=all_tickers, timeframe=TimeFrame.Day, start=start_date)
    bars_df = data_client.get_stock_bars(request_params).df
    
    # Store immediate spot pricing for target allocations
    for t in all_tickers:
        asset_prices[t] = float(bars_df.loc[t]['close'].iloc[-1])
        
    # Calculate scores across the core equity/commodity assets
    for name, ticker in core_universe.items():
        df_ticker = bars_df.loc[ticker].copy()
        df_ticker['SMA200'] = df_ticker['close'].rolling(window=200).mean()
        
        p_now = float(df_ticker['close'].iloc[-1])
        p_3m = float(df_ticker['close'].iloc[-63])
        p_6m = float(df_ticker['close'].iloc[-126])
        p_12m = float(df_ticker['close'].iloc[-252])
        
        # Upgrade 2: Apply multi-window tracking weights
        ret_3m = (p_now - p_3m) / p_3m
        ret_6m = (p_now - p_6m) / p_6m
        ret_12m = (p_now - p_12m) / p_12m
        
        ensemble_scores[ticker] = (0.40 * ret_3m) + (0.30 * ret_6m) + (0.30 * ret_12m)
        asset_sma200s[ticker] = float(df_ticker['SMA200'].iloc[-1])

    ranked_assets = sorted(ensemble_scores, key=ensemble_scores.get, reverse=True)
    
    spy_p = asset_prices['SPY']
    spy_sma = asset_sma200s['SPY']
    
    print(f"\n--- Multi-Window Health Matrix ---")
    print(f"SPY Spot: ${spy_p:.2f} | SPY Macro 200 SMA Filter: ${spy_sma:.2f}")
    
    if spy_p > spy_sma:
        print("Regime Status: Structural Bull Trend. Allocating to ensemble leaders.")
        candidates = ranked_assets[:2]
    else:
        print("Regime Status: Defensive Caution Flag. Restricting to safe haven filters.")
        safe_tickers = ['GLD', 'TLT', 'VNQ']
        safe_pool = [a for a in ranked_assets if a in safe_tickers]
        candidates = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
        
    # Absolute Momentum verification with automated yield parking transitions
    target_weights = {}
    for ticker in candidates:
        if asset_prices[ticker] > asset_sma200s[ticker]:
            print(f"  -> {ticker} Validated (Above 200 SMA). Target weight allocation: 50%")
            target_weights[ticker] = target_weights.get(ticker, 0.0) + 0.50
        else:
            print(f"  -> {ticker} Fails Absolute Trend. Routing 50% allocation to Yield Parking ({CASH_PROXY_TICKER}).")
            target_weights[CASH_PROXY_TICKER] = target_weights.get(CASH_PROXY_TICKER, 0.0) + 0.50
            
    return target_weights, asset_prices

def run_live_rebalance():
    print("Initiating Systematic Rebalance Loop...")
    account_data = trading_client.get_account()
    total_portfolio_equity = float(account_data.portfolio_value)
    
    state = load_or_init_state(total_portfolio_equity)
    current_month = pd.Timestamp.now().month
    
    if state.get("is_tripped", False):
        print("❌ EXECUTION BLOCKED: Circuit breaker state is locked.")
        sys.exit(1)
        
    if total_portfolio_equity > state["peak_equity"]:
        state["peak_equity"] = total_portfolio_equity
        
    if ((state["peak_equity"] - total_portfolio_equity) / state["peak_equity"]) >= MAX_DRAWDOWN_LIMIT:
        print("🚨🚨 CIRCUIT BREAKER VIOLATION: Liquidating all open positions...")
        state["is_tripped"] = True
        save_state(state)
        for p in trading_client.get_all_positions():
            trading_client.submit_order(MarketOrderRequest(symbol=p.symbol, qty=int(p.qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
        sys.exit(1)

    if state.get("last_rebalanced_month") == current_month:
        print("Execution Gate Checked: Rebalance cycle already performed this month.")
        sys.exit(0)

    target_weights, current_prices = calculate_dynamic_targets()
    open_positions = {p.symbol: int(p.qty) for p in trading_client.get_all_positions()}
    
    # 5% Drift Tolerance Gate Check
    trigger_trade = False
    if set(target_weights.keys()) != set(open_positions.keys()):
        trigger_trade = True
    else:
        for ticker, target_w in target_weights.items():
            actual_w = (open_positions[ticker] * current_prices[ticker]) / total_portfolio_equity
            if abs(actual_w - target_w) > DRIFT_BUFFER:
                trigger_trade = True
                break

    if not trigger_trade:
        print("✅ Portfolio weights sit cleanly within the 5% buffer zone. Transaction churn bypassed.")
        state["last_rebalanced_month"] = current_month
        save_state(state)
        sys.exit(0)

    print("Drift threshold exceeded. Commencing rebalance execution pipeline...")
    
    # Sell phase for out-of-bounds metrics
    for symbol in list(open_positions.keys()):
        if symbol not in target_weights:
            print(f"Liquidating asset assignment: {symbol}")
            trading_client.submit_order(MarketOrderRequest(symbol=symbol, qty=open_positions[symbol], side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            del open_positions[symbol]
            time.sleep(2)

    # Buy/Adjust weights phase
    updated_account = trading_client.get_account()
    total_portfolio_equity = float(updated_account.portfolio_value)
    
    for ticker, target_w in target_weights.items():
        asset_price = current_prices[ticker]
        target_shares_qty = int((total_portfolio_equity * target_w) // asset_price)
        current_qty = open_positions.get(ticker, 0)
        
        if current_qty == 0 and target_shares_qty > 0:
            print(f"Deploying Allocation: {target_shares_qty} shares of {ticker}")
            trading_client.submit_order(MarketOrderRequest(symbol=ticker, qty=target_shares_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        elif current_qty != target_shares_qty:
            qty_delta = target_shares_qty - current_qty
            side = OrderSide.BUY if qty_delta > 0 else OrderSide.SELL
            print(f"Modifying Position Balance -> {ticker}: {abs(qty_delta)} shares ({side.name})")
            trading_client.submit_order(MarketOrderRequest(symbol=ticker, qty=abs(qty_delta), side=side, time_in_force=TimeInForce.DAY))

    state["last_rebalanced_month"] = current_month
    save_state(state)
    print("\nProduction Rebalance Loop Successfully Concluded.")

if __name__ == "__main__":
    if trading_client.get_clock().is_open:
        run_live_rebalance()
    else:
        print("Market closed. Standing by.")