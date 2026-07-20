import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import os

from alpaca.trading.client import TradingClient

st.set_page_config(page_title="Technical Trading Bot", page_icon="⚡", layout="wide")
st.title("⚡ Dynamic 4-Indicator Technical System")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Live Telemetry Dashboard", "⏳ Backtest Analytics & Rules"])

with tab1:
    API_KEY = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
    SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not API_KEY or not SECRET_KEY:
        st.warning("🔒 **Running in Preview Mode:** Connect your Alpaca secrets to view real-time account data.")
    else:
        try:
            trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
            account = trading_client.get_account()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio Equity", f"${float(account.portfolio_value):,.2f}")
            col2.metric("Available Cash", f"${float(account.cash):,.2f}")
            col3.metric("Active Positions", len(trading_client.get_all_positions()))
            col4.metric("Status", "Active", delta="Operational")
        except Exception as e:
            st.error(f"Alpaca API connection failed: {e}")

with tab2:
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            params = json.load(f)
        st.markdown("### ⚙️ Optuna Machine-Learned Parameters")
        p_cols = st.columns(4)
        p_cols[0].metric("EMA Length", params.get("ema_len"))
        p_cols[1].metric("RSI Range", f"{params.get('rsi_lower')} - {params.get('rsi_upper')}")
        p_cols[2].metric("CMF Threshold", f"{params.get('cmf_thresh'):.2f}")
        p_cols[3].metric("ATR Risk Multiplier", f"{params.get('sl_mult')}x SL / {params.get('tp_mult')}x TP")

    st.markdown("---")
    st.subheader("📈 Backtest Performance Analytics")
    csv_file = "backtest_results.csv"

    if not os.path.exists(csv_file):
        st.info("ℹ️ No backtest results found. Run `python backtest.py` to generate analytics.")
    else:
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
        strat_final = df["Strategy_Equity"].iloc[-1]
        bench_final = df["Benchmark_Equity"].iloc[-1]

        strat_ret = ((strat_final - 10000) / 10000) * 100
        bench_ret = ((bench_final - 10000) / 10000) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Strategy Portfolio Value", f"${strat_final:,.2f}", f"{strat_ret:+.2f}%")
        m2.metric("S&P 500 Benchmark Value", f"${bench_final:,.2f}", f"{bench_ret:+.2f}%")
        m3.metric("Alpha Generated", f"{(strat_ret - bench_ret):+.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Strategy_Equity'], name="Strategy", line=dict(color="#FFB900", width=3)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Benchmark_Equity'], name="S&P 500 (SPY)", line=dict(color="#888888", width=1.5, dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)