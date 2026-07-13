import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import sys
import os

# ==========================================
# 1. AUTHENTICATION & CONFIGURATION (SECURE)
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# --- TEMP DIAGNOSTIC LOGS ---
if API_KEY:
    print(f"DIAGNOSTIC: API_KEY starts with '{API_KEY[:4]}' and is {len(API_KEY)} characters long.")
if SECRET_KEY:
    print(f"DIAGNOSTIC: SECRET_KEY starts with '{SECRET_KEY[:4]}' and is {len(SECRET_KEY)} characters long.")
# -----------------------------
BASE_URL = "https://paper-api.alpaca.markets/v2"  # Paper trading environment gateway URL

# Enforce a strict validation check on credentials before starting execution
if not API_KEY or not SECRET_KEY:
    print("CRITICAL ERROR: Alpaca API keys are missing! Check your GitHub Secrets configuration.")
    sys.exit(1)

try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
    account = api.get_account()
except Exception as e:
    print(f"Error authenticating with Alpaca API: {e}")
    sys.exit(1)

# Asset Universe Mapping (Matching our unified backtest framework)
asset_universe = {
    "US_Stocks": "SPY",
    "Tech_Stocks": "QQQ",
    "Gold": "GLD",
    "Bonds": "TLT",
    "Real_Estate": "VNQ"
}

# ==========================================
# 2. CORE SYSTEM LOGIC ENGINE
# ==========================================
def calculate_dynamic_targets():
    """
    Downloads historical data from Alpaca using free-tier IEX feeds to
    calculate trailing 12-month momentum and apply macro trend filters on SPY.
    """
    print("Fetching active historical trend matrices...")
    momentum_scores = {}
    spy_data = None
    
    # Establish an explicit 15-month trailing data range to guarantee 252 trading bars
    start_date = (pd.Timestamp.now() - pd.DateOffset(months=15)).strftime('%Y-%m-%d')
    
    for name, ticker in asset_universe.items():
        # Force feed='iex' to bypass subscription authorization rejections
        bars = api.get_bars(ticker, tradeapi.rest.TimeFrame.Day, start=start_date, feed='iex').df
        
        if bars.empty:
            print(f"CRITICAL: No data returned for asset ticker: {ticker}")
            sys.exit(1)
            
        # Strip outer column labels if the API returns a MultiIndex format
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.droplevel(1)
            
        if len(bars) < 252:
            print(f"CRITICAL: Insufficient bars returned for {ticker} (Got {len(bars)}, need 252)")
            sys.exit(1)
            
        if ticker == "SPY":
            spy_data = bars.copy()
            
        # Calculate trailing 12-month (252 trading sessions) absolute return momentum
        start_price = float(bars['close'].iloc[-252])
        end_price = float(bars['close'].iloc[-1])
        momentum_scores[ticker] = (end_price - start_price) / start_price
    
    # Sort entire asset universe from high velocity performance to low performance
    ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
    
    # Extract structural trend components from SPY
    spy_close = spy_data['close']
    spy_ema50 = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
    current_spy_price = float(spy_close.iloc[-1])
    
    # Calculate Alpha Relative Strength Index (RSI-14)
    change = spy_close.diff()
    gain = change.mask(change < 0, 0)
    loss = -change.mask(change > 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    # Prevent divide-by-zero crashes if loss calculation resolves to flat zero
    avg_loss = avg_loss.replace(0, 0.00001)
    rs = avg_gain / avg_loss
    current_spy_rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    
    print(f"\n--- Market Health Report ---")
    print(f"SPY Price: ${current_spy_price:.2f} | 50 EMA: ${spy_ema50:.2f} | RSI: {current_spy_rsi:.2f}")
    print(f"Momentum Scores: { {k: f'{v:+.2%}' for k, v in momentum_scores.items()} }")
    
    # Determine execution strategy map corresponding to macro thresholds
    if current_spy_price > spy_ema50 and current_spy_rsi < 70:
        print("Regime Status: Healthy Market Bull Run. Selecting absolute momentum leaders.")
        # Resolve user class names to concrete asset symbols for execution
        return [asset_universe[name] for name in ranked_assets[:2]]
    else:
        print("Regime Status: Volatile/Overbought Market Danger. Executing safety allocation rules.")
        # Restrict acceptable target allocations to historical defensive proxies
        safe_pool = [a for a in ranked_assets if a in ['Gold', 'Bonds', 'Real_Estate']]
        selected = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
        return [asset_universe[name] for name in selected]

# ==========================================
# 3. PRODUCTION BALANCER EXECUTION
# ==========================================
def run_live_rebalance():
    print("Initiating Systematic Rebalance Loop...")
    target_tickers = calculate_dynamic_targets()
    print(f"Identified Action Target Assets: {target_tickers}")
    
    # Fetch structural open positions
    open_positions = {p.symbol: int(p.qty) for p in api.list_positions()}
    
    # Phase A: Liquidate assets no longer meeting current ranking parameters
    for symbol in list(open_positions.keys()):
        if symbol not in target_tickers:
            print(f"Liquidating out-of-bounds asset: {symbol}")
            api.submit_order(
                symbol=symbol,
                qty=open_positions[symbol],
                side='sell',
                type='market',
                time_in_force='day'
            )
            del open_positions[symbol]
            
    # Pause execution briefly to let exchange matching systems clear matching settlements
    if not open_positions:
        print("Waiting for settlement clearing cycles...")
        time.sleep(10)
        
    # Phase B: Recalculate portfolio limits and equally distribute target positions
    updated_account = api.get_account()
    total_portfolio_equity = float(updated_account.portfolio_value)
    target_capital_allocation = total_portfolio_equity / 2
    
    print(f"\nTotal Liquid Capital: ${total_portfolio_equity:,.2f} | Target Allocation per Asset: ${target_capital_allocation:,.2f}")
    
    for ticker in target_tickers:
        # Check if the asset is already established to prevent duplicate executions
        if ticker in open_positions:
            print(f"Asset {ticker} is already established in the active portfolio. Skipping.")
            continue
            
        # Pull active current quotes using market transactions
        current_market_trade = api.get_latest_trade(ticker)
        asset_price = float(current_market_trade.price)
        
        # Calculate whole asset shares using integer floor division
        target_shares_qty = int(target_capital_allocation // asset_price)
        
        if target_shares_qty > 0:
            print(f"Routing Production Market BUY Order: {target_shares_qty} shares of {ticker}")
            api.submit_order(
                symbol=ticker,
                qty=target_shares_qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
    print("\nSystem Rebalance Successfully Completed.")

# ==========================================
# 4. SCHEDULER ENTRY HOOK
# ==========================================
if __name__ == "__main__":
    market_clock = api.get_clock()
    if market_clock.is_open:
        run_live_rebalance()
    else:
        print(f"Execution Halted: The stock market is currently closed. Next Open: {market_clock.next_open}")