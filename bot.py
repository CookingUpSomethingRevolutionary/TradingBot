import os
import sys
import pandas as pd
import pandas_ta as ta
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

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def get_technical_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
    df = df.dropna()
    if len(df) > 100:
        df.ta.ema(length=100, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.cmf(length=20, append=True)
        df.ta.atr(length=14, append=True)
        return df.iloc[-1]
    return None

def manage_open_positions():
    print("Checking active positions for ATR Stop-Loss / Take-Profit...")
    positions = trading_client.get_all_positions()
    
    for pos in positions:
        tech = get_technical_data(pos.symbol)
        if tech is not None:
            current_price = float(pos.current_price)
            avg_entry = float(pos.avg_entry_price)
            current_atr = tech['ATRr_14']
            
            # Stop Loss: 2x ATR, Take Profit: 3x ATR
            if current_price < (avg_entry - 2 * current_atr) or current_price > (avg_entry + 3 * current_atr):
                print(f"Risk threshold hit for {pos.symbol}. Liquidating position.")
                trading_client.submit_order(
                    order_data=MarketOrderRequest(symbol=pos.symbol, qty=pos.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                )

def scan_for_entries():
    print("Scanning universe for new technical setups...")
    # Mock watchlist for speed in live bot - replace with your CSV reading logic if desired
    watchlist = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'META', 'GOOGL'] 
    
    positions = trading_client.get_all_positions()
    open_symbols = [p.symbol for p in positions]
    
    if len(open_symbols) >= 5:
        print("Portfolio full. No new entries allowed.")
        return

    account = trading_client.get_account()
    buying_power = float(account.buying_power)
    allocation_per_trade = buying_power / (5 - len(open_symbols))
    
    for ticker in watchlist:
        if ticker in open_symbols:
            continue
            
        tech = get_technical_data(ticker)
        if tech is not None:
            is_uptrend = tech['Close'] > tech['EMA_100']
            has_volume = tech['CMF_20'] > 0.05
            has_momentum = 50 < tech['RSI_14'] < 70
            
            if is_uptrend and has_volume and has_momentum:
                shares_to_buy = int(allocation_per_trade // tech['Close'])
                if shares_to_buy > 0:
                    print(f"Technical Setup Found! Buying {shares_to_buy} shares of {ticker}")
                    trading_client.submit_order(
                        order_data=MarketOrderRequest(symbol=ticker, qty=shares_to_buy, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    )
                    open_symbols.append(ticker)
                    if len(open_symbols) >= 5:
                        break

if __name__ == "__main__":
    if trading_client.get_clock().is_open:
        manage_open_positions()
        scan_for_entries()
    else:
        print("Execution Paused: Market is closed.")