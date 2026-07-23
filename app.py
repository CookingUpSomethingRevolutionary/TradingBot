import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import yfinance as yf
import pandas_ta as ta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

st.set_page_config(page_title="Henry's Trading Bot", page_icon="⚡", layout="wide")
st.title("⚡ Henry's Dynamic S&P 500 Rebalancing Engine")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Live Production Environment", "⏳ Historical Backtest Engine"])

# =========================================================
# TAB 1: LIVE PRODUCTION TELEMETRY DASHBOARD
# =========================================================
with tab1:
    API_KEY = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
    SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not API_KEY or not SECRET_KEY:
        st.warning("🔒 **Running in Preview Mode:** Connect your Alpaca secrets to unlock real-time account telemetry.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Equity", "$100,000.00", "Preview")
        col2.metric("Available Cash Balance", "$25,000.00")
        col3.metric("Buying Power", "$50,000.00")
        col4.metric("Engine Health", "Standby", delta_color="off")
    else:
        try:
            trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
            account = trading_client.get_account()
            positions = trading_client.get_all_positions()
            
            # Account Core Metrics
            total_equity = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            last_equity = float(account.last_equity)
            daily_change = total_equity - last_equity
            daily_change_pct = (daily_change / last_equity * 100) if last_equity > 0 else 0.0

            # Live SPY Market Regime Check
            try:
                spy_df = yf.download("SPY", period="1y", interval="1d", progress=False)
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df = spy_df.xs("SPY", level=1, axis=1)
                spy_df['SMA200'] = ta.sma(spy_df['Close'], length=200)
                latest_spy = spy_df.iloc[-1]
                spy_close = float(latest_spy['Close'])
                spy_sma = float(latest_spy['SMA200'])
                regime_bullish = spy_close > spy_sma
                regime_status = "Bullish" if regime_bullish else "Bearish"
            except Exception:
                regime_status = "Unknown"

            # -----------------------------------------------------
            # 1. Top Executive Metrics
            # -----------------------------------------------------
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Portfolio Equity", f"${total_equity:,.2f}", f"{daily_change_pct:+.2f}% Today")
            m2.metric("Available Cash", f"${cash:,.2f}")
            m3.metric("Buying Power", f"${buying_power:,.2f}")
            m4.metric("Active Holdings", f"{len(positions)} Positions")
            m5.metric("Market Regime", regime_status)

            st.markdown("---")

            # -----------------------------------------------------
            # 2. Portfolio Asset Allocation & Health Telemetry
            # -----------------------------------------------------
            col_chart, col_stats = st.columns([1, 1])

            with col_chart:
                st.subheader("📊 Portfolio Asset Allocation")
                
                allocation_data = []
                for p in positions:
                    mkt_val = float(p.market_value)
                    allocation_data.append({"Asset": p.symbol, "Value": mkt_val})
                
                # Add Cash balance to allocation donut
                allocation_data.append({"Asset": "CASH", "Value": cash})
                alloc_df = pd.DataFrame(allocation_data)
                
                fig_pie = px.pie(
                    alloc_df, 
                    values="Value", 
                    names="Asset", 
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_stats:
                st.subheader("⚡ Risk & Exposure Metrics")
                
                long_val = float(account.long_market_value)
                exposure_pct = (long_val / total_equity * 100) if total_equity > 0 else 0.0
                
                st.write(f"**Account Status:** `{account.status.value.upper()}`")
                st.write(f"**Long Market Value:** `${long_val:,.2f}`")
                st.write(f"**Cash Allocation:** `${cash:,.2f}` ({(cash / total_equity * 100):.1f}%)")
                
                st.progress(min(int(exposure_pct), 100), text=f"Equity Exposure: {exposure_pct:.1f}%")

            st.markdown("---")

            # -----------------------------------------------------
            # 3. Active Holdings Table
            # -----------------------------------------------------
            st.subheader("💼 Active Positions Detail")
            
            if positions:
                pos_list = []
                for p in positions:
                    mkt_val = float(p.market_value)
                    unrealized_pl = float(p.unrealized_pl)
                    unrealized_plpc = float(p.unrealized_plpc) * 100
                    weight = (mkt_val / total_equity * 100) if total_equity > 0 else 0.0
                    
                    pos_list.append({
                        "Ticker": p.symbol,
                        "Shares": float(p.qty),
                        "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
                        "Current Price": f"${float(p.current_price):,.2f}",
                        "Market Value": f"${mkt_val:,.2f}",
                        "Unrealized P&L ($)": f"${unrealized_pl:+,.2f}",
                        "Unrealized P&L (%)": f"{unrealized_plpc:+.2f}%",
                        "Portfolio Weight": f"{weight:.2f}%"
                    })
                
                pos_df = pd.DataFrame(pos_list)
                st.dataframe(pos_df, hide_index=True, use_container_width=True)
            else:
                st.info("No active positions held in portfolio. Capital is currently 100% in cash.")

            st.markdown("---")

            # -----------------------------------------------------
            # 4. Recent Order Execution Audit Trail
            # -----------------------------------------------------
            st.subheader("📜 Recent Executed & Pending Orders")
            
            try:
                orders_req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10)
                orders = trading_client.get_orders(orders_req)
                
                if orders:
                    order_list = []
                    for o in orders:
                        order_list.append({
                            "Submitted At": o.submitted_at.strftime("%Y-%m-%d %H:%M:%S") if o.submitted_at else "N/A",
                            "Symbol": o.symbol,
                            "Side": o.side.value.upper(),
                            "Qty": float(o.qty) if o.qty else 0,
                            "Order Type": o.type.value.upper(),
                            "Status": o.status.value.upper(),
                            "Filled Price": f"${float(o.filled_avg_price):,.2f}" if o.filled_avg_price else "N/A"
                        })
                    
                    order_df = pd.DataFrame(order_list)
                    st.dataframe(order_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No recent orders recorded.")
            except Exception as order_err:
                st.caption(f"Could not retrieve order log: {order_err}")

        except Exception as e:
            st.error(f"Engine connection processing error: {e}")

# =========================================================
# TAB 2: HISTORICAL BACKTEST ENGINE
# =========================================================
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