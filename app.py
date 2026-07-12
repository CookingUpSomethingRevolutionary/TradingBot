import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# ==========================================
# PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="Dynamic Momentum Strategy Suite",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Global Macro Dynamic Momentum Suite")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Live Production Environment", "⏳ Historical Backtest Engine"])

asset_universe = {
    "US_Stocks": "SPY",
    "Tech_Stocks": "QQQ",
    "Gold": "GLD",
    "Bonds": "TLT",
    "Real_Estate": "VNQ"
}

# ==========================================
# TAB 1: LIVE PRODUCTION ENVIRONMENT
# ==========================================
with tab1:
    API_KEY = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
    SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    BASE_URL = "https://paper-api.alpaca.markets"

    if not API_KEY or not SECRET_KEY:
        st.warning("🔒 **Running in Preview Mode:** Connect your Alpaca secrets to unlock live account telemetry.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Equity", "$100,000.00", "Preview")
        col2.metric("System Mode", "Standby")
        col3.metric("API Gateway", "Disengaged")
    else:
        try:
            api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
            account = api.get_account()
            
            # Account summary metrics
            col1, col2, col3, col4 = st.columns(4)
            total_equity = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            
            col1.metric("Total Portfolio Equity", f"${total_equity:,.2f}")
            col2.metric("Available Cash Balance", f"${cash:,.2f}")
            col3.metric("Buying Multiplier Power", f"${buying_power:,.2f}")
            col4.metric("Engine Health", "Online", delta="Operational")
            
            st.markdown("---")
            st.subheader("📊 Current Strategy State Indicators")
            
            @st.cache_data(ttl=300)
            def run_live_indicators():
                momentum_scores = {}
                spy_bars = api.get_bars("SPY", tradeapi.rest.TimeFrame.Day, limit=265).df
                if spy_bars.empty: return None
                
                for name, ticker in asset_universe.items():
                    b = api.get_bars(ticker, tradeapi.rest.TimeFrame.Day, limit=265).df
                    if not b.empty and len(b) >= 252:
                        start_val = float(b['close'].iloc[-252])
                        end_val = float(b['close'].iloc[-1])
                        momentum_scores[ticker] = (end_val - start_val) / start_val
                
                spy_close = spy_bars['close']
                spy_ema50 = spy_close.ewm(span=50, adjust=False).mean()
                
                change = spy_close.diff()
                gain = change.mask(change < 0, 0).ewm(com=13, adjust=False).mean()
                loss = -change.mask(change > 0, 0).ewm(com=13, adjust=False).mean()
                spy_rsi = 100 - (100 / (1 + (gain / loss)))
                
                ranked = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
                curr_p, curr_e, curr_r = float(spy_close.iloc[-1]), float(spy_ema50.iloc[-1]), float(spy_rsi.iloc[-1])
                
                healthy = curr_p > curr_e and curr_r < 70
                regime = "Bull Market Run (Equities Active)" if healthy else "Defensive Mode Triggered (Safe Assets)"
                targets = ranked[:2] if healthy else [a for a in ranked if a in ['GLD', 'TLT', 'VNQ']][:2]
                
                return momentum_scores, curr_p, curr_e, curr_r, regime, targets, spy_close, spy_ema50

            engine_out = run_live_indicators()
            if engine_out:
                scores, spy_p, spy_e, spy_r, regime_str, target_list, spy_hist, ema_hist = engine_out
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("SPY Spot", f"${spy_p:.2f}")
                m2.metric("SPY 50 EMA Filter", f"${spy_e:.2f}")
                m3.metric("SPY RSI Parameter", f"{spy_r:.1f}")
                
                if "Bull" in regime_str:
                    m4.success(f"🟢 **Regime:**\n{regime_str}")
                else:
                    m4.warning(f"⚠️ **Regime:**\n{regime_str}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spy_hist.index, y=spy_hist.values, name='SPY Price', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=ema_hist.index, y=ema_hist.values, name='50 EMA', line=dict(color='#EF553B', dash='dash')))
                fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"🎯 **System Target Allocations:** {', '.join(target_list)}")
                
            st.markdown("---")
            st.subheader("📦 Currently Deployed Positions")
            live_pos = api.list_positions()
            if live_pos:
                p_records = []
                for p in live_pos:
                    p_records.append({
                        "Asset Ticker": p.symbol,
                        "Allocated Shares": int(p.qty),
                        "Avg Entry Price": f"${float(p.avg_entry_price):,.2f}",
                        "Current Valuation": f"${float(p.current_price):,.2f}",
                        "Market Net Value": float(p.market_value),
                        "Unrealized Growth": f"${float(p.unrealized_pl):+,.2f} ({float(p.unrealized_plpc)*100:+.2f}%)"
                    })
                st.dataframe(pd.DataFrame(p_records), hide_index=True, use_container_width=True)
            else:
                st.info("No active open positions found. Waiting for next schedule rebalance loop execution.")
                
        except Exception as e:
            st.error(f"Engine connection processing error: {e}")

# ==========================================
# TAB 2: HISTORICAL BACKTEST ENGINE (LIVE PARSING)
# ==========================================
with tab2:
    st.subheader("⏳ Mathematical Backtest Analytics (Parsed Verification)")
    
    csv_file = "backtest_results.csv"
    if not os.path.exists(csv_file):
        st.info("ℹ️ **Backtest Data File Missing:** Please run `python backtest.py` locally in your workspace terminal first to compile the underlying validation CSV matrix.")
    else:
        # Read the exact mathematical numbers generated by your script
        backtest_df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        
        strat_final = backtest_df["Strategy_Equity"].iloc[-1]
        bench_final = backtest_df["Benchmark_Equity"].iloc[-1]
        
        strat_return = ((strat_final - 10000) / 10000) * 100
        bench_return = ((bench_final - 10000) / 10000) * 100
        alpha_excess = strat_return - bench_return
        
        # Performance Indicators Rows
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("Strategy Portfolio Value", f"${strat_final:,.2f}", f"{strat_return:+.2f}% Total Return")
        b_col2.metric("S&P 500 Benchmark Value", f"${bench_final:,.2f}", f"{bench_return:+.2f}% Total Return")
        b_col3.metric("System Alpha Multiplier", f"{alpha_excess:+.2f}%", "Excess Market Capture", delta_color="normal")
        
        # Interactive Performance Chart
        st.markdown("### 📈 Strategic Growth Allocation Profiles")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Equity'], name="Dynamic Momentum Strategy Portfolio", line=dict(color="#FFB900", width=3)))
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Benchmark_Equity'], name="S&P 500 Buy & Hold Benchmark (SPY)", line=dict(color="#888888", width=1.5, dash='dash')))
        
        fig_hist.update_layout(
            template="plotly_dark", 
            height=450, 
            xaxis_title="Timeline Execution Date", 
            yaxis_title="Asset Allocation Growth Value ($)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Added bonus: Generate explicit statistical performance rows
        st.markdown("### 📋 Mathematical Integrity Performance Table")
        
        # Calculate trailing metrics
        strat_pcts = backtest_df["Strategy_Equity"].pct_change().dropna()
        bench_pcts = backtest_df["Benchmark_Equity"].pct_change().dropna()
        
        # Annualized Sharpe Matrix calculation (Assuming Risk-Free Rate = 0 for pure baseline mapping)
        strat_sharpe = (strat_pcts.mean() / strat_pcts.std()) * np.sqrt(252) if strat_pcts.std() != 0 else 0
        bench_sharpe = (bench_pcts.mean() / bench_pcts.std()) * np.sqrt(252) if bench_pcts.std() != 0 else 0
        
        stats_data = {
            "Performance Tracking Indicator": ["Starting Principle Capital", "Terminal Matrix Portfolio Valuation", "Compounded Cumulative Return %", "Estimated Annualized Sharpe Index Strategy Target"],
            "Dynamic Momentum Strategy Portfolio": [f"$10,000.00", f"${strat_final:,.2f}", f"{strat_return:+.2f}%", f"{strat_sharpe:.2f}"],
            "S&P 500 Buy & Hold Benchmark (SPY)": [f"$10,000.00", f"${bench_final:,.2f}", f"{bench_return:+.2f}%", f"{bench_sharpe:.2f}"]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)