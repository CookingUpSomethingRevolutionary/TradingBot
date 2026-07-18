import yfinance as yf
import pandas as pd
import numpy as np

# 1. Define the Global Macro Asset Universe
asset_tickers = {
    "US_Stocks": "SPY",
    "Tech_Stocks": "QQQ",
    "Gold": "GLD",
    "Bonds": "TLT",
    "Real_Estate": "VNQ"
}

FRICTION_PCT = 0.0010  # Models 0.10% friction per trade volume
DRIFT_BUFFER = 0.05    # 5% allocation tolerance band

print("Fetching historical data for asset universe...")
raw_data = {}
for name, ticker in asset_tickers.items():
    df = yf.download(ticker, start="2004-11-01", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    raw_data[name] = df['Close']

df_universe = pd.DataFrame(raw_data).dropna()

# Calculate 200-day SMAs for ALL assets (Absolute Momentum Trend Filters)
for name in asset_tickers.keys():
    df_universe[f'{name}_SMA200'] = df_universe[name].rolling(window=200).mean()

lookback_period = 252  
initial_capital = 10000
portfolio_value = initial_capital
cash_balance = portfolio_value  # Track cash explicitly for absolute momentum sit-outs
bh_shares = portfolio_value / df_universe['US_Stocks'].iloc[lookback_period]

current_holdings = {} # asset_name: shares
equity_timeline = []
benchmark_timeline = []
date_timeline = []

print("Running Robust Dual-Momentum Simulation (Monthly Rebalance + Drift Buffer)...")
last_rebalanced_month = None

for idx in range(lookback_period, len(df_universe)):
    timestamp = df_universe.index[idx]
    current_row = df_universe.iloc[idx]
    current_year_month = (timestamp.year, timestamp.month)
    
    spy_price = current_row['US_Stocks']
    spy_sma200 = current_row['US_Stocks_SMA200']
    
    # Calculate Total Portfolio Value Mark-to-Market
    total_portfolio_value = cash_balance + sum(shares * current_row[asset] for asset, shares in current_holdings.items())
    
    # MONTHLY REBALANCE GATE
    if last_rebalanced_month is None or current_year_month != last_rebalanced_month:
        last_rebalanced_month = current_year_month
        lookback_row = df_universe.iloc[idx - lookback_period]
        
        # 1. Relative Momentum Ranking (1-Year Trailing Return)
        momentum_scores = {}
        for asset in asset_tickers.keys():
            momentum_scores[asset] = (current_row[asset] - lookback_row[asset]) / lookback_row[asset]
        ranked_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
        
        # 2. Macro Regime Filter (Upgrade D: 200 SMA on SPY)
        if spy_price > spy_sma200:
            candidate_assets = ranked_assets[:2]
        else:
            safe_pool = [a for a in ranked_assets if a in ['Gold', 'Bonds', 'Real_Estate']]
            candidate_assets = safe_pool[:2] if len(safe_pool) >= 2 else ranked_assets[:2]
            
        # 3. Absolute Momentum Filter (Upgrade C: Asset must be above its own 200 SMA)
        target_weights = {}
        for asset in candidate_assets:
            if current_row[asset] > current_row[f'{asset}_SMA200']:
                target_weights[asset] = 0.50
            else:
                target_weights[asset] = 0.0  # Fails trend filter, remains in cash
                
        # 4. Allocation Drift Buffer Check (Upgrade B)
        trigger_trade = False
        if set(target_weights.keys()) != set(current_holdings.keys()):
            trigger_trade = True
        else:
            for asset, target_w in target_weights.items():
                if target_w > 0:
                    current_w = (current_holdings[asset] * current_row[asset]) / total_portfolio_value
                    if abs(current_w - target_w) > DRIFT_BUFFER:
                        trigger_trade = True
                        break
                        
        if trigger_trade:
            # Liquidate all positions to cash to recalculate clean weights with friction
            total_cash = total_portfolio_value
            current_valuations = {asset: current_holdings.get(asset, 0) * current_row[asset] for asset in asset_tickers.keys()}
            
            total_friction_drag = 0
            # Calculate execution volume to apply realistic slippage/fees
            for asset in asset_tickers.keys():
                ideal_val = total_cash * target_weights.get(asset, 0)
                trade_volume = abs(ideal_val - current_valuations.get(asset, 0))
                total_friction_drag += trade_volume * FRICTION_PCT
                
            net_portfolio_value = total_cash - total_friction_drag
            
            # Re-allocate cash into validated targets
            current_holdings = {}
            cash_balance = net_portfolio_value
            
            for asset, target_w in target_weights.items():
                if target_w > 0:
                    allocation_dollars = net_portfolio_value * target_w
                    current_holdings[asset] = allocation_dollars / current_row[asset]
                    cash_balance -= allocation_dollars
                    
            total_portfolio_value = cash_balance + sum(shares * current_row[asset] for asset, shares in current_holdings.items())

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
print("✅ Upgraded institutional backtest completed successfully!")