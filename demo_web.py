import yfinance as yf
import pandas as pd
import numpy as np

# 1. Configuration & Data Fetching
tickers = {
    "SPY": "S&P 500 (Stocks)",
    "QQQ": "NASDAQ (Tech)",
    "GLD": "Gold (Defense)",
    "TLT": "Bonds (Safety)",
    "VNQ": "Real Estate"
}

print("Gathering real-time trend analytics...")
raw_data = yf.download(list(tickers.keys()), period="1y", interval="1d", progress=False)

if isinstance(raw_data.columns, pd.MultiIndex):
    close_prices = raw_data['Close']
else:
    close_prices = raw_data[['Close']]

close_prices = close_prices.ffill().bfill()

# 2. Analytics Engine Computations
# A. Trailing Momentum Standings (Relative Strength over past 252 days)
past_prices = close_prices.iloc[-252] if len(close_prices) >= 252 else close_prices.iloc[0]
current_prices = close_prices.iloc[-1]
momentum_scores = (((current_prices - past_prices) / past_prices) * 100).round(2)
momentum_sorted = momentum_scores.sort_values(ascending=False)

# B. S&P 500 Trend Indicators (EMA + RSI)
spy_series = close_prices['SPY']
spy_ema50 = spy_series.ewm(span=50, adjust=False).mean()

change = spy_series.diff()
gain = change.mask(change < 0, 0)
loss = -change.mask(change > 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
spy_rsi = 100 - (100 / (1 + rs))

# Extract vital point-in-time statistics
latest_spy_price = round(float(spy_series.iloc[-1]), 2)
latest_spy_ema = round(float(spy_ema50.iloc[-1]), 2)
latest_spy_rsi = round(float(spy_rsi.iloc[-1]), 2)

# Determine Regime Condition
if latest_spy_price > latest_spy_ema and latest_spy_rsi < 70:
    regime_status = "🟢 BULLISH REGIME (Healthy Market Expansion)"
    target_selection = f"{momentum_sorted.index[0]} & {momentum_sorted.index[1]}"
else:
    regime_status = "🔴 DEFENSIVE REGIME (High Volatility / Correction Risk)"
    safe_assets = [a for a in momentum_sorted.index if a in ['GLD', 'TLT', 'VNQ']]
    target_selection = f"{safe_assets[0]} & {safe_assets[1]}" if len(safe_assets) >= 2 else f"{momentum_sorted.index[0]} & {momentum_sorted.index[1]}"

# 3. Dynamic HTML Layout Compilation
print("Compiling interactive visual webpage template...")
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Macro Momentum Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 font-sans min-h-screen p-6">
    <div class="max-w-6xl mx-auto space-y-6">
        
        <header class="bg-gray-800 border border-gray-700 rounded-xl p-6 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <h1 class="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">🤖 Macro Momentum Dashboard</h1>
                <p class="text-gray-400 mt-1">Autonomous Era-Adaptive Global Asset Allocation Engine</p>
            </div>
            <div class="bg-gray-900 border border-gray-700 rounded-lg p-3 text-right">
                <span class="text-xs font-semibold uppercase tracking-wider text-gray-400 block">System Regime Status</span>
                <span class="text-lg font-bold">{regime_status}</span>
            </div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="bg-gray-800 border border-gray-700 rounded-xl p-5 shadow-md">
                <h3 class="text-gray-400 text-xs font-bold uppercase tracking-wider">SPY Price vs 50 EMA</h3>
                <p class="text-2xl font-black mt-2 text-blue-400">${latest_spy_price} <span class="text-xs font-normal text-gray-400">/ EMA: ${latest_spy_ema}</span></p>
            </div>
            <div class="bg-gray-800 border border-gray-700 rounded-xl p-5 shadow-md">
                <h3 class="text-gray-400 text-xs font-bold uppercase tracking-wider">SPY 14-Day RSI Indicator</h3>
                <p class="text-2xl font-black mt-2 text-purple-400">{latest_spy_rsi} <span class="text-xs font-normal text-gray-400">(Overbought Level: 70)</span></p>
            </div>
            <div class="bg-gray-800 border border-emerald-800 bg-emerald-950/20 rounded-xl p-5 shadow-md">
                <h3 class="text-emerald-400 text-xs font-bold uppercase tracking-wider">Active Tactical Portfolio Targets</h3>
                <p class="text-2xl font-black mt-2 text-emerald-400 tracking-wide">{target_selection}</p>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-gray-800 border border-gray-700 rounded-xl p-6 shadow-lg">
                <h2 class="text-lg font-bold mb-4 flex items-center gap-2">🏆 12-Month Momentum Rankings</h2>
                <div class="relative h-64"><canvas id="momentumChart"></canvas></div>
            </div>
            <div class="bg-gray-800 border border-gray-700 rounded-xl p-6 shadow-lg flex flex-col justify-between">
                <div>
                    <h2 class="text-lg font-bold mb-4">🌍 Available Asset Class Pool</h2>
                    <div class="divide-y divide-gray-700">
                        {"".join([f'<div class="py-3 flex justify-between items-center"><span class="font-medium text-gray-300">{t} <span class="text-xs text-gray-500">({tickers[t]})</span></span><span class="font-bold {"text-emerald-400" if momentum_scores[t] >= 0 else "text-red-400"}">{momentum_scores[t]:+.2f}%</span></div>' for t in momentum_sorted.index])}
                    </div>
                </div>
                <footer class="text-center text-xs text-gray-500 mt-4 pt-4 border-t border-gray-700">
                    Live operational analytics compiled automatically using production data matrices.
                </footer>
            </div>
        </div>

    </div>

    <script>
        // Inject data structure variables securely into ChartJS frontend instance
        const labels = {list(momentum_sorted.index)};
        const dataValues = {list(momentum_sorted.values)};
        const backgroundColors = dataValues.map(v => v >= 0 ? 'rgba(52, 211, 153, 0.65)' : 'rgba(248, 113, 113, 0.65)');
        const borderColors = dataValues.map(v => v >= 0 ? '#34d399' : '#f87171');

        const ctx = document.getElementById('momentumChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Return Velocity (%)',
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: '#374151' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#9ca3af', font: {{ weight: 'bold' }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

# Output static index.html file to root path structure
with open("index.html", "w") as f:
    f.write(html_content)

print("🎉 Complete! Web module written out to 'index.html'.")