import yfinance as yf
import pandas as pd
import numpy as np

# Core 30 liquid Mega-Cap Tech/Growth stocks
stock_universe = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", 
    "COST", "PEP", "CSCO", "ADBE", "TXN", "NFLX", "AMD", "QCOM", 
    "INTC", "AMAT", "ISRG", "HON", "AMGN", "SBUX", "GILD", "BKNG", 
    "MDLZ", "ADI", "LRCX", "VRTX", "PANW", "SNPS"
]
BENCHMARK_TICKER = "SPY"

FRICTION_PCT = 0.0010  # 10 bps transaction drag
DRIFT_BUFFER = 0.05    

print("Fetching historical data for momentum universe (since 2015)...")
all_tickers = stock_universe + [BENCHMARK_TICKER]
df = yf.download(all_tickers, start="2015-01-01", interval="1d", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df_close = df['Close']
else:
    df_close = df

# Compile master dataframe
df_universe = df_close.copy()
df_universe['US_Stocks'] = df_universe[BENCHMARK_TICKER]

# Only drop rows where the market benchmark itself is missing
df_universe = df_universe.dropna(subset=['US_Stocks'])

# Forward-fill minor intra-day data drops for active tickers
df_universe = df_universe.ffill()

# Pre-calculate Benchmark Macro Trend Indicator (200 SMA)
df_universe['SPY_SMA200'] = df_universe['US_Stocks'].rolling(window=200).mean()

lookback_period = 252  # 12-month momentum
initial_capital = 10000
portfolio_value = initial_capital
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

current_holdings = {}  
cash_pool = portfolio_value
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Lookahead-Bias-Free Monthly Top-5 Stock Rotation...")
last_rebalanced_month = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_month = (timestamp.year, timestamp.month)
    
    spy_price = current_row['US_Stocks']
    spy_sma = current_row['SPY_SMA200']
    
    # Safely calculate daily valuation
    total_portfolio_value = cash_pool + sum(
        shares * current_row[asset] 
        for asset, shares in current_holdings.items() if pd.notna(current_row[asset])
    )
    
    # MONTHLY REBALANCE GATE
    if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
        last_rebalanced_month = current_year_month
        lookback_row = df_universe.iloc[idx - lookback_period]
        
        # Dynamic Active Universe Discovery (Ensure stock was publicly trading 1 year ago)
        active_stocks = []
        for stock in stock_universe:
            if pd.notna(current_row[stock]) and pd.notna(lookback_row[stock]):
                active_stocks.append(stock)
                
        # Calculate 12-Month Momentum
        momentum_scores = {}
        for asset in active_stocks:
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
            
        valid_candidates = [asset for asset, score in momentum_scores.items() if score > 0]
        ranked_stocks = sorted(valid_candidates, key=momentum_scores.get, reverse=True)
        
        target_weights = {ticker: 0.0 for ticker in stock_universe}
        target_weights["Cash"] = 0.0
        
        # Macro Filter Rules: SPY > 200 SMA
        if pd.notna(spy_sma) and spy_price > spy_sma and len(ranked_stocks) > 0:
            leaders = ranked_stocks[:5]  # Buy Top 5 Stocks
            weight_per_asset = 1.0 / len(leaders)
            for asset in leaders:
                target_weights[asset] = weight_per_asset
        else:
            target_weights["Cash"] = 1.0  # Bear Market -> 100% Cash
            
        # Rebalancing calculations
        current_weights = {}
        for asset in stock_universe:
            if asset in current_holdings:
                current_weights[asset] = (current_holdings[asset] * current_row[asset]) / total_portfolio_value if total_portfolio_value > 0 else 0.0
            else:
                current_weights[asset] = 0.0
        current_weights["Cash"] = cash_pool / total_portfolio_value if total_portfolio_value > 0 else 0.0
        
        trigger_trade = False
        for asset in target_weights.keys():
            if abs(target_weights[asset] - current_weights.get(asset, 0.0)) > DRIFT_BUFFER:
                trigger_trade = True
                break
                
        if trigger_trade:
            total_friction_drag = 0
            for asset in stock_universe:
                current_val = current_holdings[asset] * current_row[asset] if asset in current_holdings and pd.notna(current_row[asset]) else 0.0
                ideal_val = total_portfolio_value * target_weights.get(asset, 0.0)
                total_friction_drag += abs(ideal_val - current_val) * FRICTION_PCT
                
            net_portfolio_value = total_portfolio_value - total_friction_drag
            
            current_holdings = {}
            cash_pool = net_portfolio_value * target_weights["Cash"]
            
            for asset in stock_universe:
                asset_weight = target_weights.get(asset, 0.0)
                if asset_weight > 0 and pd.notna(current_row[asset]):
                    current_holdings[asset] = (net_portfolio_value * asset_weight) / current_row[asset]
                    
            total_portfolio_value = cash_pool + sum(shares * current_row[asset] for asset, shares in current_holdings.items() if pd.notna(current_row[asset]))

    date_timeline.append(timestamp)
    equity_timeline.append(total_portfolio_value)
    benchmark_timeline.append(bh_shares * current_row['US_Stocks'])

results_df = pd.DataFrame({"Strategy_Equity": equity_timeline, "Benchmark_Equity": benchmark_timeline}, index=date_timeline)
results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Momentum Backtest completed successfully!")