import pandas as pd
import numpy as np
import time
import sys
import os
import yfinance as yf

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

BENCHMARK_TICKER = "SPY"

def calculate_production_targets():
    print("Scraping live active S&P 500 constituent lists...")
    wiki_tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    active_symbols = wiki_tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    all_tickers = active_symbols + [BENCHMARK_TICKER]
    
    print("Downloading historical reference parameters...")
    df = yf.download(all_tickers, period="2y", interval="1d", progress=False)
    df_close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    df_close = df_close.ffill()
    
    momentum_scores = {}
    for ticker in active_symbols:
        if ticker in df_close.columns:
            series = df_close[ticker].dropna()
            if len(series) >= 252:
                start_p = float(series.iloc[-252])
                end_p = float(series.iloc[-1])
                momentum_scores[ticker] = (end_p - start_p) / start_p

    valid_candidates = [t for t, score in momentum_scores.items() if score > 0]
    ranked_stocks = sorted(valid_candidates, key=momentum_scores.get, reverse=True)
    
    spy_series = df_close[BENCHMARK_TICKER]
    spy_sma200 = float(spy_series.rolling(window=200).mean().iloc[-1])
    current_spy_price = float(spy_series.iloc[-1])
    
    print(f"\n--- S&P 500 Market Regime Framework ---")
    print(f"SPY Spot: ${current_spy_price:.2f} | SPY 200-SMA: ${spy_sma200:.2f}")
    
    if current_spy_price > spy_sma200 and len(ranked_stocks) > 0:
        leaders = ranked_stocks[:5]
        print(f"Regime Metric: Bullish. Target Picks: {leaders}")
        return leaders
    else:
        print("Regime Metric: Bearish Risk Avoidance. Moving allocations to Cash.")
        return []

def run_live_rebalance():
    print("Initiating Rebalance Execution Loop...")
    target_tickers = calculate_production_targets()
    
    positions = trading_client.get_all_positions()
    open_positions = {p.symbol: int(p.qty) for p in positions}
    
    liquidated_any = False
    for symbol in list(open_positions.keys()):
        if symbol not in target_tickers:
            print(f"Selling asset: {symbol}")
            trading_client.submit_order(
                order_data=MarketOrderRequest(symbol=symbol, qty=open_positions[symbol], side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            )
            del open_positions[symbol]
            liquidated_any = True
            
    if liquidated_any:
        time.sleep(15) 
        
    if not target_tickers:
        print("Capital protection completed. Portfolio matching 100% cash.")
        return
        
    updated_account = trading_client.get_account()
    total_portfolio_equity = float(updated_account.portfolio_value)
    target_capital_allocation = total_portfolio_equity / len(target_tickers)
    
    for ticker in target_tickers:
        if ticker in open_positions:
            continue  
            
        request_params = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Day, limit=1)
        latest_bars = data_client.get_stock_bars(request_params).df
        asset_price = float(latest_bars['close'].iloc[-1])
        target_shares_qty = int(target_capital_allocation // asset_price)
        
        if target_shares_qty > 0:
            print(f"Purchasing Asset -> Ticker: {ticker} | Shares: {target_shares_qty}")
            try:
                trading_client.submit_order(
                    order_data=MarketOrderRequest(symbol=ticker, qty=target_shares_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                )
            except Exception:
                adjusted_qty = int((target_capital_allocation * 0.95) // asset_price)
                if adjusted_qty > 0:
                    trading_client.submit_order(
                        order_data=MarketOrderRequest(symbol=ticker, qty=adjusted_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    )

if __name__ == "__main__":
    if trading_client.get_clock().is_open:
        run_live_rebalance()
    else:
        print("Execution Paused: Market is closed.")