import yfinance as yf
import pandas as pd
import numpy as np

# 1. Expand the Asset Universe to include our yield-bearing Cash Proxy (BIL)
asset_tickers = {
    "US_Stocks": "SPY",
    "Tech_Stocks": "QQQ",
    "Gold": "GLD",
    "Bonds": "TLT",
    "Real_Estate": "VNQ",
    "T_Bills": "BIL"  # Upgrade 1: Yield Parking Vehicle
}

FRICTION_PCT = 0.0010  # 0.10% transaction friction penalty
DRIFT_BUFFER = 0.05    # 5% allocation drift tolerance band

print("Fetching historical data for asset universe...")
raw_data = {}
for name, ticker in asset_tickers.items():
    df = yf.download(ticker, start="2004-11-01", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    raw_data[name] = df['Close']

# Create universe dataframe
df_universe = pd.DataFrame(raw_data)

# Handle BIL pre-2007 inception gap dynamically by back-filling with a flat baseline 
# so the data engine doesn't discard 2004-2007 equity metrics.
df_universe['T_Bills'] = df_universe['T_Bills'].ffill().bfill()
df_universe = df_universe.dropna(subset=["US_Stocks", "Tech_Stocks", "Gold", "Bonds", "Real_Estate"])

# Pre-calculate 200-day SMAs for Absolute Momentum Filters
core_assets = ["US_Stocks", "Tech_Stocks", "Gold", "Bonds", "Real_Estate"]
for name in core_assets:
    df_universe[f'{name}_SMA200'] = df_universe[name].rolling(window=200).mean()

lookback_period = 252  
initial_capital = 10000
portfolio_value = initial_capital
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

current_holdings = {}  # asset_name: shares held
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Ensemble Dual-Momentum Engine with T-Bill Parking...")
last_rebalanced_month = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_month = (timestamp.year, timestamp.month)
    
    spy_price = current_row['US_Stocks']
    spy_sma200 = current_row['US_Stocks_SMA200']
    
    # Calculate Mark-to-Market Portfolio Net Valuation
    total_portfolio_value = sum(shares * current_row[asset] for asset, shares in current_holdings.items()) if current_holdings else portfolio_value
    
    # MONTHLY EXECUTION GATE
    if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
        last_rebalanced_month = current_year_month
        
        # Pull historical points for our multi-window calculations
        row_now = current_row
        row_3m = df_universe.iloc[idx - 63]
        row_6m = df_universe.iloc[idx - 126]
        row_12m = df_universe.iloc[idx - 252]
        
        # Upgrade 2: Multi-Window Ensemble Lookback Matrix Ranking (0.40/0.30/0.30 weighting)
        ensemble_scores = {}
        for asset in core_assets:
            ret_3m = (row_now[asset] - row_3m[asset]) / row_3m[asset]
            ret_6m = (row_now[asset] - row_6m[asset]) / row_6m[asset]
            ret_12m = (row_now[asset] - row_12m[asset]) / row_12m[asset]
            
            ensemble_scores[asset] = (0.40 * ret_3m) + (0.30 * ret_6m) + (0.30 * ret_12m)
            
        ranked_assets = sorted(ensemble_scores, key=ensemble_scores.get, reverse=True)
        
        # Macro Regime Filter (200 SMA on SPY)
        if spy_price > spy_sma200:
            candidate_assets = ranked_assets[:2]
        else:
            safe_pool = [a for a in ranked_assets if a in ['Gold', 'Bonds', 'Real_Estate']]
            candidate_assets = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
            
        # Absolute Momentum filter configuration with Dynamic T-Bill Yield Parking
        target_weights = {ticker: 0.0 for ticker in asset_tickers.keys()}
        
        for asset in candidate_assets:
            if current_row[asset] > current_row[f'{asset}_SMA200']:
                target_weights[asset] += 0.50
            else:
                # Upgrade 1: If asset trends downward, re-route capital into yield parking vehicle
                target_weights["T_Bills"] += 0.50
                
        # Allocation Drift Buffer Evaluation
        trigger_trade = False
        active_targets = {k: v for k, v in target_weights.items() if v > 0}
        
        if set(active_targets.keys()) != set(current_holdings.keys()):
            trigger_trade = True
        else:
            for asset, target_w in active_targets.items():
                current_w = (current_holdings[asset] * current_row[asset]) / total_portfolio_value
                if abs(current_w - target_w) > DRIFT_BUFFER:
                    trigger_trade = True
                    break
                    
        if trigger_trade:
            # Rebalance Engine execution phase
            current_valuations = {asset: current_holdings.get(asset, 0) * current_row[asset] for asset in asset_tickers.keys()}
            total_friction_drag = 0
            
            # Calculate total trade volume to compute realistic transaction friction
            for asset in asset_tickers.keys():
                ideal_val = total_portfolio_value * target_weights[asset]
                trade_volume = abs(ideal_val - current_valuations.get(asset, 0))
                total_friction_drag += trade_volume * FRICTION_PCT
                
            net_portfolio_value = total_portfolio_value - total_friction_drag
            
            # Deploy assets into target metrics
            current_holdings = {}
            for asset, target_w in target_weights.items():
                if target_w > 0:
                    current_holdings[asset] = (net_portfolio_value * target_w) / current_row[asset]
                    
            total_portfolio_value = sum(shares * current_row[asset] for asset, shares in current_holdings.items())

    daily_bh_value = bh_shares * current_row['US_Stocks']
    
    date_timeline.append(timestamp)
    equity_timeline.append(total_portfolio_value)
    benchmark_timeline.append(daily_bh_value)

results_df = pd.DataFrame({
    "Strategy_Equity": equity_timeline,
    "Benchmark_Equity": benchmark_timeline
}, index=date_timeline)

results_df.index.name = "Date"
results_df.to_csv("backtest_results.csv")
print("✅ Alpha-optimized backtest completed and saved to 'backtest_results.csv'!")