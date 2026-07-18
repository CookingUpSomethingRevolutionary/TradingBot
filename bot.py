import pandas as pd
import numpy as np
import time
import sys
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("CRITICAL ERROR: Alpaca API keys are missing!")
    sys.exit(1)

try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
except Exception as e:
    print(f"Auth Error: {e}")
    sys.exit(1)

# Core 30 liquid Mega-Cap Tech/Growth stocks
stock_universe = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", 
    "COST", "PEP", "CSCO", "ADBE", "TXN", "NFLX", "AMD", "QCOM", 
    "INTC", "AMAT", "ISRG", "HON", "AMGN", "SBUX", "GILD", "BKNG", 
    "MDLZ", "ADI", "LRCX", "VRTX", "PANW", "SNPS"
]
BENCHMARK_TICKER = "SPY"

def calculate_production_targets():
    print("Scanning Individual Stock Momentum Matrices...")
    momentum_scores = {}
    
    # We need at least ~300 days of history to calculate a 252-day lookback + 200 SMA
    start_date = pd.Timestamp.now() - pd.DateOffset(days=400) 
    
    all_tickers = stock_universe + [BENCHMARK_TICKER]
    
    # Fetch Data
    request_params = StockBarsRequest(symbol_or_symbols=all_tickers, timeframe=TimeFrame.Day, start=start_date)
    bars_df = data_client.get_stock_bars(request_params).df
    
    spy_data = bars_df.loc[BENCHMARK_TICKER].copy()
    
    # Score the stocks
    for ticker in stock_universe:
        try:
            df_ticker = bars_df.loc[ticker]
            # Need at least 252 bars for 1-year momentum
            if len(df_ticker) >= 252:
                start_p = float(df_ticker['close'].iloc[-252])
                end_p = float(df_ticker['close'].iloc[-1])
                momentum_scores[ticker] = (end_p - start_p) / start_p
        except KeyError:
            continue  # Skip if stock data didn't fetch properly
        
    valid_candidates = [t for t, score in momentum_scores.items() if score > 0]
    ranked_stocks = sorted(valid_candidates, key=momentum_scores.get, reverse=True)
    
    # Calculate SPY 200 SMA
    spy_close = spy_data['close']
    spy_sma200 = float(spy_close.rolling(window=200).mean().iloc[-1])
    current_spy_price = float(spy_close.iloc[-1])
    
    print(f"\n--- Institutional Regime Matrix ---")
    print(f"SPY Price: ${current_spy_price:.2f} | SPY SMA200: ${spy_sma200:.2f}")
    
    # If SPY > 200 SMA, Buy Top 5 Stocks
    if current_spy_price > spy_sma200 and len(ranked_stocks) > 0:
        leaders = ranked_stocks[:5]
        print(f"Regime Status: Bullish Extension. Target allocations: {leaders}")
        return leaders
    else:
        print("Regime Status: Risk Mitigation Alert. Safely offloading capital to Cash.")
        return []  # Empty array signals a 100% Cash pivot

def run_live_rebalance():
    print("Initiating Systematic Rebalance Loop...")
    target_tickers = calculate_production_targets()
    
    positions = trading_client.get_all_positions()
    open_positions = {p.symbol: int(p.qty) for p in positions}
    
    liquidated_any = False
    for symbol in list(open_positions.keys()):
        if symbol not in target_tickers:
            print(f"Liquidating asset allocation to preserve cash: {symbol}")
            trading_client.submit_order(
                order_data=MarketOrderRequest(symbol=symbol, qty=open_positions[symbol], side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            )
            del open_positions[symbol]
            liquidated_any = True
            
    if liquidated_any:
        time.sleep(15)  # Wait for fills to settle and cash to return to buying power
        
    if not target_tickers:
        print("Capital protection routing executed successfully. Portfolio holds 100% Cash.")
        return
        
    updated_account = trading_client.get_account()
    total_portfolio_equity = float(updated_account.portfolio_value)
    target_capital_allocation = total_portfolio_equity / len(target_tickers)
    
    for ticker in target_tickers:
        if ticker in open_positions:
            continue  # Already holding, skip (or you could calculate drift here later)
            
        request_params = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Day, limit=1)
        latest_bars = data_client.get_stock_bars(request_params).df
        asset_price = float(latest_bars['close'].iloc[-1])
        target_shares_qty = int(target_capital_allocation // asset_price)
        
        if target_shares_qty > 0:
            print(f"Deploying Capital Matrix -> Ticker: {ticker} | Shares: {target_shares_qty}")
            try:
                trading_client.submit_order(
                    order_data=MarketOrderRequest(symbol=ticker, qty=target_shares_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                )
            except Exception:
                # Fallback if precision fails slightly due to real-time price slippage
                adjusted_qty = int((target_capital_allocation * 0.95) // asset_price)
                if adjusted_qty > 0:
                    trading_client.submit_order(
                        order_data=MarketOrderRequest(symbol=ticker, qty=adjusted_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    )

if __name__ == "__main__":
    if trading_client.get_clock().is_open:
        run_live_rebalance()
    else:
        print("Execution Halted: The stock market is currently closed.")