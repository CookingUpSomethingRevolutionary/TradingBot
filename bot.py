import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import sys
import os

# ==========================================
# 1. AUTHENTICATION & CONFIGURATION (SECURE)
# ==========================================
# Reads secrets injected into the environment by GitHub Actions
API_KEY = os.getenv(ALPACA_API_KEY)
SECRET_KEY = os.getenv(ALPACA_SECRET_KEY)
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading environment URL

# Enforce a strict key validation guardrail
if not API_KEY or not SECRET_KEY:
    print("CRITICAL ERROR: Alpaca API keys are missing! Check your GitHub Secrets configuration.")
    sys.exit(1)

try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
    account = api.get_account()
except Exception as e:
    print(f"Error authenticating with Alpaca API: {e}")
    sys.exit(1)

# Asset Universe Mapping (Exactly matching our historical framework)
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
    Downloads historical data from Alpaca to calculate trailing 12-month 
    momentum and applies macro trend filters (EMA & RSI) on SPY.
    """
    print("Fetching active historical trend matrices...")
    momentum_scores = {}
    spy_data = None
    
    for name, ticker in asset_universe.items():
        # Fetch 265 trading bars to comfortably compute a rolling 50 EMA and RSI
        bars = api.get_bars(ticker, tradeapi.rest.TimeFrame.Day, limit=265).df
        if bars.empty or len(bars) < 252:
            print(f"CRITICAL: Insufficient historical data returned for asset: {ticker}")
            sys.exit(1)
            
        if ticker == "SPY":
            spy_data = bars.copy()
            
        # Calculate standard 12-month (252 trading days) performance momentum
        start_price = float(bars['close'].iloc[-252])
        end_price = float(bars['close'].iloc[-1])
        momentum_scores[ticker] = (end_price - start_price) / start_price
    
    # Sort assets by performance (Highest to Lowest velocity)
    ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
    
    # Calculate Macro Trend Indicators on SPY
    spy_close = spy_data['close']
    spy_ema50 = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
    current_spy_price = float(spy_close.iloc[-1])
    
    # Calculate RSI
    change = spy_close.diff()
    gain = change.mask(change < 0, 0)
    loss = -change.mask(change > 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    current_spy_rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    
    print(f"\n--- Market Health Report ---")
    print(f"SPY Price: ${current_spy_price:.2f} | 50 EMA: ${spy_ema50:.2f} | RSI: {current_spy_rsi:.2f}")
    print(f"Momentum Scores: { {k: f'{v:+.2%}' for k, v in momentum_scores.items()} }")
    
    # Determine targets matching risk thresholds
    if current_spy_price > spy_ema50 and current_spy_rsi < 70:
        print("Regime Status: Healthy Market Bull Run. Selecting absolute momentum leaders.")
        return ranked_assets[:2]
    else:
        print("Regime Status: Volatile/Overbought Market Danger. Executing safety allocation rules.")
        # Filter down into safe asset classes exclusively
        safe_pool = [a for a in ranked_assets if a in ['GLD', 'TLT', 'VNQ']]
        if len(safe_pool) >= 2:
            return safe_pool[:2]
        else:
            return ranked_assets[:2]

# ==========================================
# 3. PRODUCTION BALANCER EXECUTION
# ==========================================
def run_live_rebalance():
    print("Initiating Systematic Rebalance Loop...")
    target_tickers = calculate_dynamic_targets()
    print(f"Identified Action Target Assets: {target_tickers}")
    
    # Fetch current open positions
    open_positions = {p.symbol: int(p.qty) for p in api.list_positions()}
    
    # Phase A: Liquidate assets no longer meeting ranking criteria
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
            
    # Pause execution briefly to let exchange settlement cycles process
    if not open_positions:
        print("Waiting for settlement clearing cycles...")
        time.sleep(10)
        
    # Phase B: Re-verify buying liquidity and distribute 50/50 targets
    updated_account = api.get_account()
    total_portfolio_equity = float(updated_account.portfolio_value)
    target_capital_allocation = total_portfolio_equity / 2
    
    print(f"\nTotal Liquid Capital: ${total_portfolio_equity:,.2f} | Target Allocation per Asset: ${target_capital_allocation:,.2f}")
    
    for ticker in target_tickers:
        # Check if the asset is already established to prevent double buying
        if ticker in open_positions:
            print(f"Asset {ticker} is already established in the active portfolio. Skipping.")
            continue
            
        # Get active quote price
        current_market_trade = api.get_latest_trade(ticker)
        asset_price = float(current_market_trade.price)
        
        # Calculate whole integer stock share quantities
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