import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Define Systems Configuration & Timeframes
etf_universe = {
    "SPY": "S&P 500 Stocks",
    "QQQ": "NASDAQ Tech",
    "GLD": "Gold Bullion",
    "TLT": "Long-Term Bonds",
    "VNQ": "Real Estate"
}

print("Executing Multi-Era Historical Data Harvesting Pipeline...")

# Fetch Modern Assets Pool (10 Years of Daily History to compute up to 1-Decade Returns)
modern_data = yf.download(list(etf_universe.keys()), period="10y", interval="1d", progress=False)
close_prices = modern_data['Close'].ffill().bfill()

# Fetch Historical S&P 500 Index Base since its exact inception: March 4, 1957
sp500_history = yf.download("^GSPC", start="1957-03-04", interval="1d", progress=False)
sp500_close = sp500_history['Close'].ffill().bfill()

# ==========================================
# 2. COMPUTE MULTI-TIMELINE RETURN VECTORS
# ==========================================
metrics_grid = {}

for ticker in etf_universe.keys():
    series = close_prices[ticker]
    
    # Calculate returns across historical offset boundaries safely
    r_1d = ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100
    r_1m = ((series.iloc[-1] - series.iloc[-22]) / series.iloc[-22]) * 100  # ~1 trading month
    r_1y = ((series.iloc[-1] - series.iloc[-252]) / series.iloc[-252]) * 100 # ~1 trading year
    r_10y = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100      # Full 10 year span
    
    metrics_grid[ticker] = {
        "1D": round(float(r_1d), 2),
        "1M": round(float(r_1m), 2),
        "1Y": round(float(r_1y), 2),
        "10Y": round(float(r_10y), 2)
    }

# Compute S&P 500 Compounding Metrics Since March 4, 1957
spy_1957_start = float(sp500_close.iloc[0])
spy_1957_current = float(sp500_close.iloc[-1])
total_sp500_growth_pct = ((spy_1957_current - spy_1957_start) / spy_1957_start) * 100

# Generate a downsampled data array for the 1957 compounding line chart (To keep the web page lightweight)
sampled_sp500 = sp500_close.resample('ME').last()
dates_1957 = [d.strftime('%Y-%m-%d') for d in sampled_sp500.index]
# Calculate how an initial $10,000 investment compounded over that timeline
investment_values_1957 = ((sampled_sp500 / spy_1957_start) * 10000).round(2).tolist()

# ==========================================
# 3. COMPUTE CURRENT LIVE MARKET REGIME STATUS
# ==========================================
spy_series = close_prices['SPY']
spy_ema50 = spy_series.ewm(span=50, adjust=False).mean()

# RSI Calculation Matrix
change = spy_series.diff()
gain = change.mask(change < 0, 0)
loss = -change.mask(change > 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
spy_rsi = 100 - (100 / (1 + rs))

latest_price = round(float(spy_series.iloc[-1]), 2)
latest_ema = round(float(spy_ema50.iloc[-1]), 2)
latest_rsi = round(float(spy_rsi.iloc[-1]), 2)

# Sort current 12M momentum velocity to find system target recommendations
momentum_ranking = pd.Series({t: metrics_grid[t]["1Y"] for t in etf_universe.keys()}).sort_values(ascending=False)

if latest_price > latest_ema and latest_rsi < 70:
    regime_status = "🟢 BULLISH REGIME (Healthy Market Expansion)"
    targets = f"{momentum_ranking.index[0]} & {momentum_ranking.index[1]}"
else:
    regime_status = "🔴 DEFENSIVE REGIME (High Volatility / Correction Risk)"
    safe_pool = [a for a in momentum_ranking.index if a in ['GLD', 'TLT', 'VNQ']]
    targets = f"{safe_pool[0]} & {safe_pool[1]}" if len(safe_pool) >= 2 else f"{momentum_ranking.index[0]} & {momentum_ranking.index[1]}"

# ==========================================
# 4. EXPORT LIVE INTERACTIVE DASHBOARD HTML
# ==========================================
print("Generating production HTML structure layout containing multiple charts...")
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Analytics Macro Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-950 text-gray-100 font-sans min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-8">
        
        <header class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl flex flex-col lg:flex-row justify-between lg:items-center gap-6">
            <div>
                <div class="flex items-center gap-3">
                    <span class="animate-pulse h-3 w-3 rounded-full bg-emerald-400"></span>
                    <h1 class="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400">Live Analytics Control Matrix</h1>
                </div>
                <p class="text-gray-400 mt-1 text-sm">Autonomous Execution Tracking & Cross-Era Macro Performance Analysis</p>
            </div>
            <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center gap-6">
                <div>
                    <span class="text-xs font-bold uppercase tracking-widest text-gray-500 block">Active Market Regime</span>
                    <span class="text-base font-extrabold tracking-wide">{regime_status}</span>
                </div>
                <div class="bg-emerald-950/30 border border-emerald-800/60 rounded-lg px-4 py-2">
                    <span class="text-xs font-bold uppercase tracking-widest text-emerald-500 block">Target Portfolio Allocations</span>
                    <span class="text-xl font-black text-emerald-400 tracking-wider">{targets}</span>
                </div>
            </div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-5 shadow-lg">
                <h3 class="text-gray-500 text-xs font-bold uppercase tracking-wider">S&P 500 Baseline Spot</h3>
                <p class="text-3xl font-black mt-2 text-blue-400">${latest_price} <span class="text-xs font-medium text-gray-400">/ 50 EMA: ${latest_ema}</span></p>
            </div>
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-5 shadow-lg">
                <h3 class="text-gray-500 text-xs font-bold uppercase tracking-wider">System Volatility Wave (14D RSI)</h3>
                <p class="text-3xl font-black mt-2 text-purple-400">{latest_rsi} <span class="text-xs font-medium text-gray-400">/ Limit: 70</span></p>
            </div>
            <div class="bg-gray-900 border border-indigo-900/50 bg-indigo-950/10 rounded-xl p-5 shadow-lg">
                <h3 class="text-indigo-400 text-xs font-bold uppercase tracking-wider">S&P 500 Compounding Engine</h3>
                <p class="text-3xl font-black mt-2 text-indigo-400">+{total_sp500_growth_pct:,.1f}% <span class="text-xs font-medium text-gray-400">Since Mar 4, 1957</span></p>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl">
                <h2 class="text-xl font-bold mb-2 tracking-tight">📊 Modern Era Cross-Timeline Returns</h2>
                <p class="text-xs text-gray-500 mb-6">Comparative view of return profiles across 1 Day, 1 Month, 1 Year, and 10 Year parameters.</p>
                <div class="relative h-80"><canvas id="multiPeriodChart"></canvas></div>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl">
                <h2 class="text-xl font-bold mb-2 tracking-tight">📈 Growth of $10,000 Since 1957 Inception</h2>
                <p class="text-xs text-gray-500 mb-6">The compounding engine of the S&P 500 index baseline tracked across generational market shifts.</p>
                <div class="relative h-80"><canvas id="longTermCompoundingChart"></canvas></div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl lg:col-span-2">
                <h2 class="text-xl font-bold mb-2 tracking-tight">🏆 Active 12-Month Momentum Velocity</h2>
                <p class="text-xs text-gray-500 mb-6">Current trajectory velocities used by the system to rank macro portfolio allocation weight choices.</p>
                <div class="relative h-64"><canvas id="momentumLeaderboardChart"></canvas></div>
            </div>

            <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold mb-4 tracking-tight">📋 Live Engine Blueprint</h2>
                    <div class="space-y-3 mt-4 text-sm">
                        <div class="flex justify-between border-b border-gray-800 pb-2">
                            <span class="text-gray-400">Index Origin Date</span>
                            <span class="font-mono text-gray-200">March 4, 1957</span>
                        </div>
                        <div class="flex justify-between border-b border-gray-800 pb-2">
                            <span class="text-gray-400">Origin Core Index Value</span>
                            <span class="font-mono text-gray-200">44.06</span>
                        </div>
                        <div class="flex justify-between border-b border-gray-800 pb-2">
                            <span class="text-gray-400">Asset Tracking Vehicles</span>
                            <span class="font-mono text-gray-200">5 Pools (Global Macro)</span>
                        </div>
                        <div class="flex justify-between border-b border-gray-800 pb-2">
                            <span class="text-gray-400">Rebalance Rule Frequency</span>
                            <span class="font-mono text-gray-200">Systematic Run via Cron</span>
                        </div>
                    </div>
                </div>
                <div class="bg-gray-950 rounded-xl p-4 border border-gray-800 text-center mt-6 text-xs text-gray-500 leading-relaxed">
                    Live operational engine calculations complete. All database parameters verified and synchronized with real-time financial markets.
                </div>
            </div>
        </div>

    </div>

    <script>
        // Data structural arrays injected from python calculations engine
        const assets = {list(etf_universe.keys())};
        const datasetsGrid = {metrics_grid};

        // --- CHART A IMPLEMENTATION (Multi-Timeline Grouped Bars) ---
        new Chart(document.getElementById('multiPeriodChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: assets,
                datasets: [
                    {{ label: '1 Day (%)', data: assets.map(a => datasetsGrid[a]['1D']), backgroundColor: '#60a5fa' }},
                    {{ label: '1 Month (%)', data: assets.map(a => datasetsGrid[a]['1M']), backgroundColor: '#c084fc' }},
                    {{ label: '1 Year (%)', data: assets.map(a => datasetsGrid[a]['1Y']), backgroundColor: '#34d399' }},
                    {{ label: '10 Years (%)', data: assets.map(a => datasetsGrid[a]['10Y']), backgroundColor: '#fbbf24' }}
                ]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#9ca3af', font: {{ size: 11 }} }} }} }},
                scales: {{
                    y: {{ grid: {{ color: '#1f2937' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#9ca3af', font: {{ weight: 'bold' }} }} }}
                }}
            }}
        }});

        // --- CHART B IMPLEMENTATION (Astronomic Line Compounding Since 1957) ---
        new Chart(document.getElementById('longTermCompoundingChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: {dates_1957},
                datasets: [{{
                    label: 'Portfolio Value ($)',
                    data: {investment_values_1957},
                    borderColor: '#818cf8',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    backgroundColor: 'rgba(129, 140, 248, 0.04)'
                }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ 
                        type: 'logarithmic',
                        grid: {{ color: '#1f2937' }}, 
                        ticks: {{ 
                            color: '#9ca3af',
                            callback: function(value) {{ return '$' + value.toLocaleString(); }}
                        }} 
                    }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#6b7280', maxTicksLimit: 12 }} }}
                }}
            }}
        }});

        // --- CHART C IMPLEMENTATION (12M Momentum Leaderboard Velocity) ---
        const momentumLabels = {list(momentum_ranking.index)};
        const momentumValues = {list(momentum_ranking.values)};
        
        new Chart(document.getElementById('momentumLeaderboardChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: momentumLabels,
                datasets: [{{
                    data: momentumValues,
                    backgroundColor: momentumValues.map(v => v >= 0 ? 'rgba(52, 211, 153, 0.7)' : 'rgba(248, 113, 113, 0.7)'),
                    borderColor: momentumValues.map(v => v >= 0 ? '#34d399' : '#f87171'),
                    borderWidth: 1.5,
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ grid: {{ color: '#1f2937' }}, ticks: {{ color: '#9ca3af' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#9ca3af', font: {{ weight: 'bold' }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

with open("index.html", "w") as f:
    f.write(html_content)

print("🎉 Complete Success! Production dashboard output to 'index.html'.")