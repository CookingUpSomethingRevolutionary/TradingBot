import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

from alpaca.trading.client import TradingClient

st.set_page_config(page_title="Henry's Algo", page_icon="⚡", layout="wide")
st.title("⚡ Henry's 4-Indicator Technical Algo (EMA/RSI/CMF/ATR)")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Live Production Environment", "⏳ Historical Backtest Engine"])

with tab1:
    API_KEY = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
    SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not API_KEY or not SECRET_KEY:
        st.warning("🔒 **Running in Preview Mode:** Connect your Alpaca secrets to unlock live account telemetry.")
    else:
        try:
            trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
            account = trading_client.get_account()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Portfolio Equity", f"${float(account.portfolio_value):,.2f}")
            col2.metric("Available Cash Balance", f"${float(account.cash):,.2f}")
            col3.metric("Active Positions", len(trading_client.get_all_positions()))
            col4.metric("Engine Health", "Online", delta="Operational")
        except Exception as e:
            st.error(f"Engine connection processing error: {e}")

with tab2:
    st.subheader("⏳ Mathematical Backtest Analytics")
    csv_file = "backtest_results.csv"
    
    if not os.path.exists(csv_file):
        st.info("ℹ️ **Backtest Data File Missing:** Please run `python backtest.py` locally first.")
    else:
        backtest_df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
        strat_final = backtest_df["Strategy_Equity"].iloc[-1]
        strat_return = ((strat_final - 10000) / 10000) * 100
        
        st.metric("Strategy Portfolio Value", f"${strat_final:,.2f}", f"{strat_return:+.2f}% Total Return")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Equity'], name="Technical Algo Strategy", line=dict(color="#FFB900", width=3)))
        fig_hist.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)