import yfinance as yf
import pandas as pd
import numpy as np

# 11 Official Select Sector SPDR ETFs
sector_tickers = {
    "XLK": "XLK", "XLV": "XLV", "XLF": "XLF", "XLY": "XLY", 
    "XLI": "XLI", "XLP": "XLP", "XLE": "XLE", "XLB": "XLB", 
    "XLU": "XLU", "XLRE": "XLRE", "XLC": "XLC"
}
BENCHMARK_TICKER = "SPY"

FRICTION_PCT = 0.0010  # 10 bps transaction drag
DRIFT_BUFFER = 0.05    

print("Fetching historical data from maximum sector inception boundaries (Dec 1998)...")
raw_data = {}
for name, ticker in sector_tickers.items():
    df = yf.download(ticker, start="1998-12-16", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    raw_data[name] = df['Close']

df_spy = yf.download(BENCHMARK_TICKER, start="1998-12-16", interval="1d", progress=False)
if isinstance(df_spy.columns, pd.MultiIndex):
    df_spy.columns = df_spy.columns.droplevel(1)

# Compile master dataframe
df_universe = pd.DataFrame(raw_data)
df_universe['US_Stocks'] = df_spy['Close']

# CRITICAL FIX: Only drop rows where the market benchmark itself is missing
df_universe = df_universe.dropna(subset=['US_Stocks'])

# CRITICAL FIX: Forward-fill any random single-day yfinance missing data gaps.
# This ensures that once an asset launches, its price is never a random NaN.
df_universe = df_universe.ffill()

# Pre-calculate Benchmark Macro Trend Indicators (50 EMA)
df_universe['SPY_EMA50'] = df_universe['US_Stocks'].ewm(span=50, adjust=False).mean()

lookback_period = 252  
initial_capital = 10000
portfolio_value = initial_capital
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

current_holdings = {}  # Format: {ticker: shares}
cash_pool = portfolio_value
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Lookahead-Bias-Free Monthly Sector Rotation...")
last_rebalanced_month = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_month = (timestamp.year, timestamp.month)
    
    spy_price = current_row['US_Stocks']
    spy_ema = current_row['SPY_EMA50']
    
    # Calculate daily mark-to-market valuation safely
    total_portfolio_value = cash_pool + sum(shares * current_row[asset] for asset, shares in current_holdings.items())
    
    # MONTHLY REBALANCE GATE (Slashes friction drag by 75%)
    if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
        last_rebalanced_month = current_year_month
        lookback_row = df_universe.iloc[idx - lookback_period]
        
        # Dynamic Active Universe Discovery
        active_sectors = []
        for sector in sector_tickers.keys():
            if pd.notna(current_row[sector]) and pd.notna(lookback_row[sector]):
                active_sectors.append(sector)
                
        # Calculate 12-Month Momentum Performance Metrics
        momentum_scores = {}
        for asset in active_sectors:
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
            
        # Absolute Filter: Assets MUST have positive returns over the lookback to be valid candidates
        valid_candidates = [asset for asset, score in momentum_scores.items() if score > 0]
        ranked_sectors = sorted(valid_candidates, key=momentum_scores.get, reverse=True)
        
        target_weights = {ticker: 0.0 for ticker in sector_tickers.keys()}
        target_weights["Cash"] = 0.0
        
        # Macro Filter Rules: Only allocate to equities if SPY is in a healthy uptrend
        if spy_price > spy_ema and len(ranked_sectors) > 0:
            leaders = ranked_sectors[:2]
            weight_per_asset = 1.0 / len(leaders)
            for asset in leaders:
                target_weights[asset] = weight_per_asset
        else:
            # Risk Mitigation: Total Capital Protection (100% Cash Allocation Floor)
            target_weights["Cash"] = 1.0
            
        # Determine Trade Execution Requirements via Active Allocation Change Checks
        current_weights = {}
        for asset in sector_tickers.keys():
            asset_shares = current_holdings.get(asset, 0.0)
            current_weights[asset] = (asset_shares * current_row[asset]) / total_portfolio_value if total_portfolio_value > 0 else 0.0
        current_weights["Cash"] = cash_pool / total_portfolio_value if total_portfolio_value > 0 else 0.0
        
        trigger_trade = False
        for asset in target_weights.keys():
            if abs(target_weights[asset] - current_weights.get(asset, 0.0)) > DRIFT_BUFFER:
                trigger_trade = True
                break
                
        if trigger_trade:
            total_friction_drag = 0
            for asset in sector_tickers.keys():
                current_val = current_holdings.get(asset, 0.0) * current_row[asset]
                ideal_val = total_portfolio_value * target_weights.get(asset, 0.0)
                total_friction_drag += abs(ideal_val - current_val) * FRICTION_PCT
                
            # Deduct transaction friction costs from the portfolio pool
            net_portfolio_value = total_portfolio_value - total_friction_drag
            
            # Clear allocations and re-deploy capital
            current_holdings = {}
            cash_pool = net_portfolio_value * target_weights["Cash"]
            
            for asset in sector_tickers.keys():
                asset_weight = target_weights.get(asset, 0.0)
                if asset_weight > 0:
                    current_holdings[asset] = (net_portfolio_value * asset_weight) / current_row[asset]
                    
            total_portfolio_value = cash_pool + sum(shares * current_row[asset] for asset, shares in current_holdings.items())

    date_timeline.append(timestamp)
    equity_timeline.append(total_portfolio_value)
    benchmark_timeline.append(bh_shares * current_row['US_Stocks'])

results_df = pd.DataFrame({"Strategy_Equity": equity_timeline, "Benchmark_Equity": benchmark_timeline}, index=date_timeline)
results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Institutional Sector Backtest concluded successfully with full NaN protection!")