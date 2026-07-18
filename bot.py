import pandas as pd
import numpy as np
import time
import sys
import os

# Modern alpaca-py structural imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 1. AUTHENTICATION & INITIALIZATION
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

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
    print("Fetching active historical trend matrices via alpaca-py...")
    momentum_scores = {}
    spy_data = None
    
    start_date = pd.Timestamp.now() - pd.DateOffset(months=15)
    
    for name, ticker in asset_universe.items():
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_date
        )
        
        bars_df = data_client.get_stock_bars(request_params).df
        
        if bars_df.empty:
            print(f"CRITICAL: No data returned for asset ticker: {ticker}")
            sys.exit(1)
            
        df_ticker = bars_df.loc[ticker] if ticker in bars_df.index.levels[0] else bars_df
        
        if len(df_ticker) < 252:
            print(f"CRITICAL: Insufficient bars returned for {ticker} (Got {len(df_ticker)}, need 252)")
            sys.exit(1)
            
        if ticker == "SPY":
            spy_data = df_ticker.copy()
            
        start_price = float(df_ticker['close'].iloc[-252])
        end_price = float(df_ticker['close'].iloc[-1])
        
        # FIX: Track scores directly by their string names ("Gold", "US_Stocks") to match structural logic
        momentum_scores[name] = (end_price - start_price) / start_price
    
    ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
    
    spy_close = spy_data['close']
    spy_ema50 = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
    current_spy_price = float(spy_close.iloc[-1])
    
    change = spy_close.diff()
    gain = change.mask(change < 0, 0)
    loss = -change.mask(change > 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean().replace(0, 0.00001)
    
    current_spy_rsi = float((100 - (100 / (1 + (avg_gain / avg_loss)))).iloc[-1])
    
    print(f"\n--- Market Health Report ---")
    print(f"SPY Price: ${current_spy_price:.2f} | 50 EMA: ${spy_ema50:.2f} | RSI: {current_spy_rsi:.2f}")
    
    if current_spy_price > spy_ema50 and current_spy_rsi < 70:
        print("Regime Status: Healthy Market Bull Run. Selecting momentum leaders.")
        return [asset_universe[name] for name in ranked_assets[:2]]
    else:
        print("Regime Status: Volatile Risk-Off Warning. Executing safety allocation rules.")
        # FIX: "Gold", "Bonds", and "Real_Estate" match the ranked_assets tracking strings perfectly now
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
    
    positions = trading_client.get_all_positions()
    open_positions = {p.symbol: int(p.qty) for p in positions}
    
    for symbol in list(open_positions.keys()):
        if symbol not in target_tickers:
            print(f"Liquidating out-of-bounds asset: {symbol}")
            
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=open_positions[symbol],
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order_data=market_order_data)
            del open_positions[symbol]
            
    if not open_positions:
        print("Waiting for settlement clearing cycles...")
        time.sleep(10)
        
    updated_account = trading_client.get_account()
    total_portfolio_equity = float(updated_account.portfolio_value)
    target_capital_allocation = total_portfolio_equity / 2
    
    print(f"\nTotal Liquid Capital: ${total_portfolio_equity:,.2f} | Target Allocation per Asset: ${target_capital_allocation:,.2f}")
    
    for ticker in target_tickers:
        if ticker in open_positions:
            print(f"Asset {ticker} is already established. Skipping.")
            continue
            
        request_params = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Day, limit=1)
        latest_bars = data_client.get_stock_bars(request_params).df
        asset_price = float(latest_bars['close'].iloc[-1])
        
        target_shares_qty = int(target_capital_allocation // asset_price)
        
        if target_shares_qty > 0:
            print(f"Routing Production Market BUY Order: {target_shares_qty} shares of {ticker}")
            
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=target_shares_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order_data=market_order_data)
            
    print("\nSystem Rebalance Successfully Completed.")

if __name__ == "__main__":
    clock = trading_client.get_clock()
    if clock.is_open:
        run_live_rebalance()
    else:
        print(f"Execution Halted: The stock market is currently closed. Next Open: {clock.next_open}")