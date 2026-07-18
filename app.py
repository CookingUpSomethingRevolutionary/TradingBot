import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Henry's Trading Bot", page_icon="⚡", layout="wide")
st.title("⚡ Henry's 11-Sector SPDR Rotation System")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Live Production Environment", "⏳ Historical Backtest Engine"])

# ==========================================
# TAB 1: LIVE CHANNELS
# ==========================================
with tab1:
    st.info("System deployed on Lookahead-Bias-Free Sector Universe (XLK, XLV, XLF, XLY, XLI, XLP, XLE, XLB, XLU, XLRE, XLC).")
    st.markdown("Dynamic active asset filters running. Bind execution API environment strings to view live metrics.")

# ==========================================
# TAB 2: SIMULATION CHARTS
# ==========================================
with tab2:
    csv_file = "backtest_results.csv"
    if not os.path.exists(csv_file):
        st.info("Run `python backtest.py` locally to populate institutional simulation graphs.")
    else:
        backtest_df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        
        strat_final = backtest_df["Strategy_Equity"].iloc[-1]
        bench_final = backtest_df["Benchmark_Equity"].iloc[-1]
        
        strat_return = ((strat_final - 10000) / 10000) * 100
        bench_return = ((bench_final - 10000) / 10000) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("11-Sector Rotation Strategy Value", f"${strat_final:,.2f}", f"{strat_return:+.2f}% Total Return")
        c2.metric("S&P 500 Buy & Hold Value", f"${bench_final:,.2f}", f"{bench_return:+.2f}% Total Return")
        c3.metric("Bias-Free Net Alpha Spread", f"{strat_return - bench_return:+.2f}%")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Equity'], name="11-Sector SPDR Momentum Strategy", line=dict(color="#FFB900", width=2.5)))
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Benchmark_Equity'], name="S&P 500 Benchmark (SPY)", line=dict(color="#888888", width=1.5, dash='dash')))
        fig_hist.update_layout(template="plotly_dark", height=450, xaxis_title="Timeline Execution", yaxis_title="Equity Valuation ($)")
        st.plotly_chart(fig_hist, use_container_width=True)