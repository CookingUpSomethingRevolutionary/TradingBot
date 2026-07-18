import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Modern alpaca-py structural imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="Henry's Trading Bot Portal",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Henry's Trading Bot Portal")
st.markdown("---")

# ==========================================
# SIDEBAR: INTERACTIVE SCENARIO CONTROLS
# ==========================================
st.sidebar.header("⚙️ Strategy Parameter Optimization")
st.sidebar.markdown("Adjust simulation variables below to evaluate different risk profiles dynamically.")

lookback_days = st.sidebar.slider(
    "Momentum Lookback Window (Days)",
    min_value=63,
    max_value=378,
    value=252,
    step=21,
    help="Default is 252 days (~1 year). Shorter periods respond faster to recent trends."
)

rsi_threshold = st.sidebar.slider(
    "SPY Overbought RSI Ceiling",
    min_value=60,
    max_value=85,
    value=70,
    step=5,
    help="Default is 70. Lower numbers make the risk-off filter switch to safe assets sooner."
)

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

    if not API_KEY or not SECRET_KEY:
        st.warning("🔒 **Running in Preview Mode:** Connect your Alpaca secrets to unlock live account telemetry.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Equity", "$100,000.00", "Preview")
        col2.metric("System Mode", "Standby")
        col3.metric("API Gateway", "Disengaged")
    else:
        try:
            trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
            data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
            account = trading_client.get_account()
            
            # 1. Main Telemetry Block
            col1, col2, col3 = st.columns(3)
            total_equity = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            
            col1.metric("Total Portfolio Equity", f"${total_equity:,.2f}")
            col2.metric("Available Cash Balance", f"${cash:,.2f}")
            col3.metric("Buying Multiplier Power", f"${buying_power:,.2f}")
            
            st.markdown("---")
            
            # Helper logic to fetch live indicator status
            def calculate_live_signals():
                start_date = pd.Timestamp.now() - pd.DateOffset(months=18)
                tickers = list(asset_universe.values())
                request_params = StockBarsRequest(symbol_or_symbols=tickers, timeframe=TimeFrame.Day, start=start_date)
                
                bars_df = data_client.get_stock_bars(request_params).df
                if bars_df.empty: return None
                
                spy_bars = bars_df.xs("SPY", level=0) if "SPY" in bars_df.index.levels[0] else bars_df
                
                momentum_scores = {}
                for name, ticker in asset_universe.items():
                    b = bars_df.xs(ticker, level=0) if ticker in bars_df.index.levels[0] else bars_df
                    if len(b) >= lookback_days:
                        start_val = float(b['close'].iloc[-lookback_days])
                        end_val = float(b['close'].iloc[-1])
                        momentum_scores[ticker] = (end_val - start_val) / start_val
                        
                spy_close = spy_bars['close']
                spy_ema50 = spy_close.ewm(span=50, adjust=False).mean()
                
                change = spy_close.diff()
                gain = change.mask(change < 0, 0).ewm(com=13, adjust=False).mean()
                loss = change.mask(change > 0, 0).abs().ewm(com=13, adjust=False).mean().replace(0, 0.00001)
                spy_rsi = 100 - (100 / (1 + (gain / loss)))
                
                return spy_close, spy_ema50, float(spy_rsi.iloc[-1]), momentum_scores

            signals = calculate_live_signals()
            
            if signals:
                spy_c, spy_e, curr_rsi, scores = signals
                curr_price = float(spy_c.iloc[-1])
                curr_ema = float(spy_e.iloc[-1])
                
                # Check health status using sidebar variables
                is_bull = curr_price > curr_ema and curr_rsi < rsi_threshold
                ranked = sorted(scores, key=scores.get, reverse=True)
                targets = ranked[:2] if is_bull else [a for a in ranked if a in ['GLD', 'TLT', 'VNQ']][:2]
                if not targets: targets = ranked[:2]
                
                # UPGRADE 1: High-impact Visual Market Status Callouts
                if is_bull:
                    st.success(f"### 🟢 Market Status: BULL REGIME\nEquities are structurally healthy. The bot is dynamically targeting core momentum acceleration nodes. **Current target assets:** {', '.join(targets)}")
                else:
                    st.warning(f"### 🛑 Market Status: DEFENSIVE REGIME\nRisk-off thresholds triggered (Price below EMA or RSI overbought). Capital protects itself in safety pools. **Current target assets:** {', '.join(targets)}")
                
                # Visual Chart Metrics Block
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("SPY Spot Price", f"${curr_price:.2f}", f"EMA: {curr_ema:.2f}")
                m_col2.metric("System RSI Metric", f"{curr_rsi:.1f}", f"Ceiling Trigger: {rsi_threshold}")
                m_col3.metric("System Health", "Operational", delta="Pipeline Live")
                
                # Macro trend tracking chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spy_c.index, y=spy_c.values, name='SPY Price', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=spy_e.index, y=spy_e.values, name='50 EMA', line=dict(color='#EF553B', dash='dash')))
                fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # Split bottom into balanced distribution sections
            bot_col1, bot_col2 = st.columns(2)
            
            with bot_col1:
                st.subheader("📦 Open Positions Balance")
                live_pos = trading_client.get_all_positions()
                if live_pos:
                    p_records = [{
                        "Ticker": p.symbol,
                        "Shares": int(p.qty),
                        "Entry Price": f"${float(p.avg_entry_price):,.2f}",
                        "Market Value": float(p.market_value),
                        "Growth": f"${float(p.unrealized_pl):+,.2f} ({float(p.unrealized_plpc)*100:+.2f}%)"
                    } for p in live_pos]
                    st.dataframe(pd.DataFrame(p_records), hide_index=True, use_container_width=True)
                else:
                    st.info("No active open positions. Awaiting Monday market open execution execution.")
                    
            with bot_col2:
                # UPGRADE 3: Historical Audit Logs
                st.subheader("📜 Recent Order Audits")
                try:
                    orders_req = GetOrdersRequest(status='filled', limit=5)
                    recent_orders = trading_client.get_orders(filter=orders_req)
                    if recent_orders:
                        order_log = [{
                            "Timestamp": o.filled_at.strftime('%m-%d %H:%M') if o.filled_at else "Pending",
                            "Asset": o.symbol,
                            "Action": str(o.side.value).upper(),
                            "Qty": o.qty,
                            "Execution Price": f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "Market"
                        } for o in recent_orders]
                        st.dataframe(pd.DataFrame(order_log), hide_index=True, use_container_width=True)
                    else:
                        st.text("No historical execution records found within standard lookback bounds.")
                except Exception as e:
                    st.text("Audit Log currently synchronization offline.")
                    
        except Exception as e:
            st.error(f"Engine Gateway connection synchronization failure: {e}")

# ==========================================
# TAB 2: HISTORICAL BACKTEST ENGINE
# ==========================================
with tab2:
    st.subheader("⏳ Mathematical Backtest Performance Matrix")
    
    csv_file = "backtest_results.csv"
    if not os.path.exists(csv_file):
        st.info("ℹ️ **Backtest Analytics Manifest Missing:** Run `python backtest.py` via your CLI to generate your equity calculation vectors.")
    else:
        backtest_df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        
        strat_final = backtest_df["Strategy_Equity"].iloc[-1]
        bench_final = backtest_df["Benchmark_Equity"].iloc[-1]
        strat_return = ((strat_final - 10000) / 10000) * 100
        bench_return = ((bench_final - 10000) / 10000) * 100
        
        # UPGRADE 4: Drawdown Risk Vector Calculation
        rolling_max = backtest_df['Strategy_Equity'].cummax()
        drawdown_series = (backtest_df['Strategy_Equity'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown_series.min()
        
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("Strategy Terminal Valuation", f"${strat_final:,.2f}", f"{strat_return:+.2f}% Total Return")
        b_col2.metric("S&P 500 Benchmark Value", f"${bench_final:,.2f}", f"{bench_return:+.2f}% Total Return")
        b_col3.metric("Maximum Peak-to-Trough Drawdown", f"{max_drawdown:.2f}%", "Historical Risk Vector Threshold", delta_color="inverse")
        
        # Plot Main Performance Growth Curve
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Equity'], name="Dynamic Momentum Strategy", line=dict(color="#FFB900", width=2.5)))
        fig_hist.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Benchmark_Equity'], name="S&P 500 Buy & Hold (SPY)", line=dict(color="#888888", width=1.5, dash='dash')))
        fig_hist.update_layout(
            template="plotly_dark", height=340, 
            xaxis_title="Timeline Date", yaxis_title="Equity Value ($)",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Plot Risk Drawdown Underneath
        st.markdown("### 📉 Historical System Peak Drawdown Map")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series.values, fill='tozeroy', name="Strategy Drawdown %", line=dict(color="#EF553B", width=1)))
        fig_dd.update_layout(
            template="plotly_dark", height=180,
            xaxis_title="Timeline Date", yaxis_title="Drawdown (%)",
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Analytics breakdown grid table
        strat_pcts = backtest_df["Strategy_Equity"].pct_change().dropna()
        bench_pcts = backtest_df["Benchmark_Equity"].pct_change().dropna()
        strat_sharpe = (strat_pcts.mean() / strat_pcts.std()) * np.sqrt(252) if strat_pcts.std() != 0 else 0
        bench_sharpe = (bench_pcts.mean() / bench_pcts.std()) * np.sqrt(252) if bench_pcts.std() != 0 else 0
        
        stats_data = {
            "Performance Metric Tracker": ["Initial Deposited Capital", "Terminal Matrix Portfolio Valuation", "Compounded Cumulative Return %", "Estimated Annualized Sharpe Ratio Coefficient"],
            "Dynamic Momentum Strategy Portfolio": ["$10,000.00", f"${strat_final:,.2f}", f"{strat_return:+.2f}%", f"{strat_sharpe:.2f}"],
            "S&P 500 Buy & Hold Benchmark (SPY)": ["$10,000.00", f"${bench_final:,.2f}", f"{bench_return:+.2f}%", f"{bench_sharpe:.2f}"]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)