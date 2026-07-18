import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

from alpaca.trading.client import TradingClient

st.set_page_config(page_title="Henry's Trading Bot", page_icon="⚡", layout="wide")
st.title("⚡ Henry's Dynamic S&P 500 Momentum Rotation System")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Live Production Environment", "⏳ Historical Backtest Engine"])

with tab1:
    API_KEY = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
    SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not API_KEY or not SECRET_KEY:
        st.warning("🔒 **Running in Preview Mode:** Connect your Alpaca secrets to unlock live account telemetry.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Equity", "$100,000.00", "Preview")
        col2.metric("System Mode", "Standby")
        col3.metric("API Gateway", "Disengaged")
    else:
        try:
            trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
            account = trading_client.get_account()
            
            col1, col2, col3, col4 = st.columns(4)
            total_equity = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            
            col1.metric("Total Portfolio Equity", f"${total_equity:,.2f}")
            col2.metric("Available Cash Balance", f"${cash:,.2f}")
            col3.metric("Buying Multiplier Power", f"${buying_power:,.2f}")
            col4.metric("Engine Health", "Online", delta="Operational")
        except Exception as e:
            st.error(f"Engine connection processing error: {e}")

with tab2:
    st.subheader("⏳ Mathematical Backtest Analytics (Bias-Free)")
    csv_file = "backtest_results.csv"
    
    if not os.path.exists(csv_file):
        st.info("ℹ️ **Backtest Data File Missing:** Please run `python backtest.py` locally first.")
    else:
        backtest_df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
        
        strat_final = backtest_df["Strategy_Equity"].iloc[-1]
        bench_final = backtest_df["Benchmark_Equity"].iloc[-1]
        
        strat_return = ((strat_final - 10000) / 10000) * 100
        bench_return = ((bench_final - 10000) / 10000) * 100
        alpha_excess = strat_return - bench_return
        
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("Strategy Portfolio Value", f"${strat_final:,.2f}", f"{strat_return:+.2f}% Total Return")
        b_col2.metric("S&P 500 Benchmark Value", f"${bench_final:,.2f}", f"{bench_return:+.2f}% Total Return")
        b_col3.metric("System Alpha Multiplier", f"{alpha_excess:+.2f}%", "Excess Market Capture")
        
        st.markdown("### 📈 Strategic Growth Allocation Profiles")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Equity'], name="Dynamic S&P 500 Top-5 Strategy", line=dict(color="#FFB900", width=3)))
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Benchmark_Equity'], name="S&P 500 Buy & Hold Benchmark (SPY)", line=dict(color="#888888", width=1.5, dash='dash')))
        fig_hist.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("### 📋 Mathematical Integrity Performance Table")
        strat_pcts = backtest_df["Strategy_Equity"].pct_change().dropna()
        bench_pcts = backtest_df["Benchmark_Equity"].pct_change().dropna()
        
        strat_sharpe = (strat_pcts.mean() / strat_pcts.std()) * np.sqrt(252) if strat_pcts.std() != 0 else 0
        bench_sharpe = (bench_pcts.mean() / bench_pcts.std()) * np.sqrt(252) if bench_pcts.std() != 0 else 0
        
        stats_data = {
            "Performance Tracking Indicator": ["Starting Principal Capital", "Terminal Portfolio Valuation", "Compounded Cumulative Return %", "Estimated Annualized Sharpe Index"],
            "Dynamic S&P 500 Strategy": [f"$10,000.00", f"${strat_final:,.2f}", f"{strat_return:+.2f}%", f"{strat_sharpe:.2f}"],
            "S&P 500 Buy & Hold (SPY)": [f"$10,000.00", f"${bench_final:,.2f}", f"{bench_return:+.2f}%", f"{bench_sharpe:.2f}"]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)