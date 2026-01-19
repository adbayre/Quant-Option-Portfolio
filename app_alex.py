"""
================================================================================
CRR OPTION PRICING PLATFORM - FINAL
ESILV - Projet d'Innovation Industrielle (Pi2)
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CRR Pricing Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. THEME SYSTEM
# =============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def get_theme_colors():
    if st.session_state.theme == 'dark':
        return {
            'bg_primary': '#0a0e17',
            'bg_secondary': '#111827',
            'bg_card': '#1a2332',
            'bg_hover': '#243044',
            'accent_blue': '#3b82f6',
            'accent_cyan': '#06b6d4',
            'accent_green': '#10b981',
            'accent_red': '#ef4444',
            'accent_orange': '#f59e0b',
            'accent_purple': '#8b5cf6',
            'text_primary': '#f8fafc',
            'text_secondary': '#94a3b8',
            'text_muted': '#64748b',
            'border_color': '#1e293b',
            'chart_bg': 'rgba(17, 24, 39, 0.8)',
            'grid_color': '#1e293b'
        }
    else:
        return {
            'bg_primary': '#f8fafc',
            'bg_secondary': '#ffffff',
            'bg_card': '#ffffff',
            'bg_hover': '#f1f5f9',
            'accent_blue': '#2563eb',
            'accent_cyan': '#0891b2',
            'accent_green': '#059669',
            'accent_red': '#dc2626',
            'accent_orange': '#d97706',
            'accent_purple': '#7c3aed',
            'text_primary': '#0f172a',
            'text_secondary': '#475569',
            'text_muted': '#94a3b8',
            'border_color': '#e2e8f0',
            'chart_bg': 'rgba(255, 255, 255, 0.9)',
            'grid_color': '#e2e8f0'
        }

colors = get_theme_colors()

# =============================================================================
# 3. CSS STYLING
# =============================================================================

def generate_css(colors):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --bg-primary: {colors['bg_primary']};
    --bg-secondary: {colors['bg_secondary']};
    --bg-card: {colors['bg_card']};
    --accent-blue: {colors['accent_blue']};
    --accent-cyan: {colors['accent_cyan']};
    --accent-green: {colors['accent_green']};
    --accent-red: {colors['accent_red']};
    --accent-orange: {colors['accent_orange']};
    --accent-purple: {colors['accent_purple']};
    --text-primary: {colors['text_primary']};
    --text-secondary: {colors['text_secondary']};
    --text-muted: {colors['text_muted']};
    --border-color: {colors['border_color']};
}}

.stApp {{
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-primary) 100%);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.main .block-container {{
    padding: 2rem 3rem;
    max-width: 100%;
}}

h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: var(--text-primary) !important;
}}

.premium-card {{
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}}

.premium-card:hover {{
    border-color: var(--accent-blue);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
    transform: translateY(-2px);
}}

.accent-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-blue);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}}

.metric-card {{
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    border-color: var(--accent-cyan);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
}}

.metric-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}}

.metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}}

.metric-value.positive {{ color: var(--accent-green); }}
.metric-value.negative {{ color: var(--accent-red); }}
.metric-value.blue {{ color: var(--accent-blue); }}
.metric-value.cyan {{ color: var(--accent-cyan); }}

.section-title {{
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-blue) 0%, transparent 100%);
}}

.header-banner {{
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}

.header-banner::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-purple));
}}

.header-title {{
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 8px 0;
}}

.header-subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: var(--text-secondary);
}}

.greeks-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 16px 0;
}}

.greek-item {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}}

.greek-item:hover {{
    border-color: var(--accent-purple);
    background: var(--bg-card);
}}

.greek-symbol {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: var(--accent-purple);
}}

.greek-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    color: var(--text-primary);
}}

.greek-name {{
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.formula-box {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-left: 3px solid var(--accent-purple);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
    font-family: 'JetBrains Mono', monospace;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: var(--bg-secondary);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    padding: 12px 24px;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
    color: white !important;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-right: 1px solid var(--border-color);
}}

.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stSelectbox > div > div {{
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

.stSlider > div > div > div > div {{
    background: var(--accent-blue) !important;
}}

.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}}

.badge-success {{
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
    border: 1px solid var(--accent-green);
}}

.badge-info {{
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
}}

.badge-warning {{
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-orange);
    border: 1px solid var(--accent-orange);
}}

.comparison-row {{
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-color);
}}

.comparison-label {{
    color: var(--text-secondary);
}}

.comparison-value {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--text-primary);
}}

.stress-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}}

.stress-card.negative {{
    border-left: 3px solid var(--accent-red);
}}

.stress-card.positive {{
    border-left: 3px solid var(--accent-green);
}}

.live-indicator {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--accent-green);
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.live-dot {{
    width: 8px;
    height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.data-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid var(--accent-blue);
    border-radius: 6px;
    font-size: 11px;
    color: var(--accent-blue);
}}
</style>
"""

st.markdown(generate_css(colors), unsafe_allow_html=True)

# =============================================================================
# 4. DATA FUNCTIONS
# =============================================================================

INDEX_MAP = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM",
    "Euro Stoxx 50 (FEZ)": "FEZ",
    "CAC 40 (EWQ)": "EWQ",
    "DAX (EWG)": "EWG",
    "FTSE 100 (EWU)": "EWU",
    "Nikkei 225 (EWJ)": "EWJ",
    "Manual Input": "MANUAL"
}

PERIOD_MAP = {
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y"
}

@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        return df
    except Exception as e:
        return None

def get_vol_surface_data(ticker_symbol, spot_price, option_type):
    """Fetch option chain for volatility surface."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        exps = ticker.options[:6]
        if not exps:
            return None
        
        strikes, maturities, ivs = [], [], []
        
        for exp in exps:
            chain = ticker.option_chain(exp)
            data = chain.calls if option_type == "Call" else chain.puts
            
            T = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
            if T < 0.01:
                continue
            
            # Filter for liquidity and relevance
            mask = (data['impliedVolatility'] > 0.001) & \
                   (data['strike'] > spot_price * 0.5) & \
                   (data['strike'] < spot_price * 1.5)
            
            filtered = data[mask]
            strikes.extend(filtered['strike'].tolist())
            maturities.extend([T] * len(filtered))
            ivs.extend(filtered['impliedVolatility'].tolist())
        
        if len(strikes) > 10:
            return strikes, maturities, ivs
        return None
    except:
        return None

# =============================================================================
# 5. MATHEMATICAL FUNCTIONS
# =============================================================================

def black_scholes(S, K, T, r, sigma, option_type="Call"):
    """Calculate Black-Scholes price and Greeks."""
    if T <= 0 or sigma <= 0:
        return {"Price": 0, "Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
        theta = (-(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = -si.norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
        theta = (-(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2))
    
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1)
    
    return {
        "Price": price, "Delta": delta, "Gamma": gamma,
        "Vega": vega / 100, "Theta": theta / 365, "Rho": rho / 100
    }


def crr_binomial_tree(S, K, T, r, sigma, N, option_type="Call", return_tree=False):
    """Cox-Ross-Rubinstein binomial tree pricing."""
    if T <= 0 or sigma <= 0:
        return (0, None, None, None, 1, 1, 0.5) if return_tree else 0
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    option_tree = np.zeros((N + 1, N + 1))
    if option_type == "Call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    else:
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)
    
    discount = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
    
    delta_tree = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            if stock_tree[j, i + 1] != stock_tree[j + 1, i + 1]:
                delta_tree[j, i] = (option_tree[j, i + 1] - option_tree[j + 1, i + 1]) / \
                                   (stock_tree[j, i + 1] - stock_tree[j + 1, i + 1])
    
    if return_tree:
        return option_tree[0, 0], stock_tree, option_tree, delta_tree, u, d, p
    return option_tree[0, 0]


def crr_delta(S, K, T, r, sigma, N, option_type="Call"):
    """Calculate CRR delta."""
    if T <= 0 or sigma <= 0 or N < 1:
        return 0
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    Su, Sd = S * u, S * d
    Cu = crr_binomial_tree(Su, K, T - dt, r, sigma, N - 1, option_type)
    Cd = crr_binomial_tree(Sd, K, T - dt, r, sigma, N - 1, option_type)
    return (Cu - Cd) / (Su - Sd)


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths):
    """Simulate GBM price paths."""
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    dW = np.random.standard_normal((n_paths, n_steps))
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW[:, t-1])
    return paths


def simulate_hedging_strategy(S0, K, T, r, sigma, n_steps, n_paths, use_bs_delta=True):
    """Simulate delta hedging strategy."""
    dt = T / n_steps
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    
    bs_initial = black_scholes(S0, K, T, r, sigma, "Call")
    initial_option_price = bs_initial["Price"]
    
    portfolio_values = np.zeros((n_paths, n_steps + 1))
    deltas = np.zeros((n_paths, n_steps + 1))
    cash_accounts = np.zeros((n_paths, n_steps + 1))
    
    for i in range(n_paths):
        S = paths[i, 0]
        delta = black_scholes(S, K, T, r, sigma, "Call")["Delta"] if use_bs_delta else crr_delta(S, K, T, r, sigma, max(10, n_steps), "Call")
        deltas[i, 0] = delta
        cash_accounts[i, 0] = initial_option_price - delta * S
        portfolio_values[i, 0] = delta * S + cash_accounts[i, 0]
    
    for t in range(1, n_steps + 1):
        ttm = T - t * dt
        for i in range(n_paths):
            S = paths[i, t]
            if ttm > 0.001:
                new_delta = black_scholes(S, K, ttm, r, sigma, "Call")["Delta"] if use_bs_delta else crr_delta(S, K, ttm, r, sigma, max(10, n_steps - t), "Call")
            else:
                new_delta = 1.0 if S > K else 0.0
            
            old_delta = deltas[i, t - 1]
            cash_accounts[i, t] = cash_accounts[i, t - 1] * np.exp(r * dt) - (new_delta - old_delta) * S
            deltas[i, t] = new_delta
            portfolio_values[i, t] = new_delta * S + cash_accounts[i, t]
    
    final_payoffs = np.maximum(paths[:, -1] - K, 0)
    hedging_errors = portfolio_values[:, -1] - final_payoffs
    
    return {
        "paths": paths,
        "portfolio_values": portfolio_values,
        "deltas": deltas,
        "final_payoffs": final_payoffs,
        "hedging_errors": hedging_errors,
        "initial_price": initial_option_price
    }


def calculate_stress_scenarios(S, K, T, r, sigma, option_type):
    """Calculate stress test scenarios."""
    scenarios = [
        {"name": "Crash -20%", "spot_chg": -0.20, "vol_chg": 0.50},
        {"name": "Bear -10%", "spot_chg": -0.10, "vol_chg": 0.25},
        {"name": "Base Case", "spot_chg": 0.00, "vol_chg": 0.00},
        {"name": "Bull +10%", "spot_chg": 0.10, "vol_chg": -0.10},
        {"name": "Rally +20%", "spot_chg": 0.20, "vol_chg": -0.20},
    ]
    
    base_price = black_scholes(S, K, T, r, sigma, option_type)["Price"]
    results = []
    
    for scenario in scenarios:
        new_S = S * (1 + scenario["spot_chg"])
        new_sigma = max(0.05, sigma * (1 + scenario["vol_chg"]))
        new_price = black_scholes(new_S, K, T, r, new_sigma, option_type)["Price"]
        pnl = new_price - base_price
        pnl_pct = (pnl / base_price * 100) if base_price > 0 else 0
        
        results.append({
            "name": scenario["name"],
            "spot": new_S,
            "vol": new_sigma,
            "price": new_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
    
    return results

# =============================================================================
# 6. VISUALIZATION FUNCTIONS
# =============================================================================

def create_candlestick_chart(df, ticker):
    """Create candlestick chart."""
    if df is None or df.empty:
        return go.Figure()
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color=colors['accent_green'],
        decreasing_line_color=colors['accent_red']
    )])
    
    fig.update_layout(
        title=dict(text=f"{ticker} Price History", font=dict(color=colors['text_primary'], size=14)),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor=colors['chart_bg'],
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor=colors['grid_color'], rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor=colors['grid_color']),
        font=dict(color=colors['text_primary'])
    )
    
    return fig


def create_binomial_tree_figure(stock_tree, option_tree, delta_tree, N, u, d, p):
    """Create binomial tree visualization."""
    fig = go.Figure()
    max_display = min(N, 8)
    
    for i in range(max_display + 1):
        for j in range(i + 1):
            x, y = i, i - 2 * j
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=45, color=colors['accent_blue'], 
                            line=dict(width=2, color=colors['accent_cyan'])),
                text=f"S:{stock_tree[j,i]:.1f}<br>V:{option_tree[j,i]:.2f}",
                textposition="middle center",
                textfont=dict(size=8, color=colors['text_primary'], family="JetBrains Mono"),
                hovertemplate=f"Step {i}, Node {j}<br>Stock: ${stock_tree[j,i]:.2f}<br>Option: ${option_tree[j,i]:.2f}<extra></extra>",
                showlegend=False
            ))
            
            if i < max_display:
                for y_offset in [1, -1]:
                    fig.add_trace(go.Scatter(
                        x=[x, x + 1], y=[y, y + y_offset],
                        mode='lines',
                        line=dict(color=colors['border_color'], width=1.5),
                        hoverinfo='skip',
                        showlegend=False
                    ))
    
    fig.update_layout(
        title=dict(text=f"Binomial Tree - {max_display + 1} of {N + 1} Steps", font=dict(size=14, color=colors['text_primary'])),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_convergence_figure(S, K, T, r, sigma, option_type, max_steps=200):
    """Create convergence chart."""
    bs_price = black_scholes(S, K, T, r, sigma, option_type)["Price"]
    bs_delta = black_scholes(S, K, T, r, sigma, option_type)["Delta"]
    
    steps_range = list(range(5, max_steps + 1, 5))
    crr_prices = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in steps_range]
    crr_deltas = [crr_delta(S, K, T, r, sigma, n, option_type) for n in steps_range]
    price_errors = [abs(p - bs_price) for p in crr_prices]
    delta_errors = [abs(d - bs_delta) for d in crr_deltas]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Option Price Convergence", "Price Error",
                        "Delta Convergence", "Delta Error"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(x=steps_range, y=crr_prices, name="CRR Price",
                  line=dict(color=colors['accent_blue'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[steps_range[0], steps_range[-1]], y=[bs_price, bs_price],
                  name="BS Price", line=dict(color=colors['text_primary'], dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=steps_range, y=price_errors, name="Price Error",
                  line=dict(color=colors['accent_red'], width=2), fill='tozeroy',
                  fillcolor='rgba(239, 68, 68, 0.2)'), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=steps_range, y=crr_deltas, name="CRR Delta",
                  line=dict(color=colors['accent_green'], width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[steps_range[0], steps_range[-1]], y=[bs_delta, bs_delta],
                  name="BS Delta", line=dict(color=colors['text_primary'], dash='dash')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=steps_range, y=delta_errors, name="Delta Error",
                  line=dict(color=colors['accent_orange'], width=2), fill='tozeroy',
                  fillcolor='rgba(245, 158, 11, 0.2)'), row=2, col=2)
    
    fig.update_layout(
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor=colors['chart_bg'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary'])
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=colors['grid_color'])
    fig.update_yaxes(showgrid=True, gridcolor=colors['grid_color'])
    
    return fig


def create_vol_surface_real(strikes, maturities, ivs, ticker):
    """Create 3D volatility surface from real data."""
    if not strikes or len(strikes) < 10:
        return None
    
    grid_x, grid_y = np.mgrid[min(strikes):max(strikes):40j, min(maturities):max(maturities):20j]
    grid_z = griddata((strikes, maturities), ivs, (grid_x, grid_y), method='linear')
    
    fig = go.Figure(data=[go.Surface(
        z=grid_z * 100,
        x=grid_x,
        y=grid_y,
        colorscale='Viridis',
        hovertemplate='Strike: %{x:.0f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=f"{ticker} Implied Volatility Surface (Real Data)", font=dict(color=colors['text_primary'])),
        scene=dict(
            xaxis=dict(title='Strike', backgroundcolor=colors['bg_secondary'], gridcolor=colors['grid_color']),
            yaxis=dict(title='Maturity (Years)', backgroundcolor=colors['bg_secondary'], gridcolor=colors['grid_color']),
            zaxis=dict(title='IV (%)', backgroundcolor=colors['bg_secondary'], gridcolor=colors['grid_color']),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_vol_surface_synthetic(S, K, T, r, sigma):
    """Create synthetic volatility surface."""
    strikes = np.linspace(S * 0.80, S * 1.20, 20)
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0])
    
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    moneyness = np.log(K_grid / S)
    
    smile = 0.1 * moneyness**2
    skew = -0.15 * moneyness
    term = 0.02 * np.sqrt(T_grid)
    
    IV_surface = sigma + smile + skew + term
    IV_surface += np.random.normal(0, 0.005, IV_surface.shape)
    IV_surface = np.clip(IV_surface, 0.05, 1.0)
    
    fig = go.Figure(data=[go.Surface(
        z=IV_surface * 100,
        x=strikes,
        y=maturities,
        colorscale='Viridis',
        hovertemplate='Strike: %{x:.0f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Implied Volatility Surface (Synthetic)", font=dict(color=colors['text_primary'])),
        scene=dict(
            xaxis=dict(title='Strike', backgroundcolor=colors['bg_secondary'], gridcolor=colors['grid_color']),
            yaxis=dict(title='Maturity (Years)', backgroundcolor=colors['bg_secondary'], gridcolor=colors['grid_color']),
            zaxis=dict(title='IV (%)', backgroundcolor=colors['bg_secondary'], gridcolor=colors['grid_color']),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_hedging_figure(results, n_display=100):
    """Create hedging simulation visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Price Paths", "Hedging Error Distribution",
                        "Portfolio vs Payoff", "Delta Evolution"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    n_paths = results["paths"].shape[0]
    n_steps = results["paths"].shape[1]
    time_axis = np.linspace(0, 1, n_steps)
    
    for i in range(min(n_display, n_paths)):
        fig.add_trace(go.Scatter(
            x=time_axis, y=results["paths"][i],
            mode='lines',
            line=dict(width=0.5, color=f'rgba(59, 130, 246, 0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
    
    mean_path = results["paths"].mean(axis=0)
    fig.add_trace(go.Scatter(
        x=time_axis, y=mean_path,
        mode='lines',
        line=dict(width=2, color=colors['text_primary']),
        name='Mean Path'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=results["hedging_errors"],
        nbinsx=50,
        marker_color=colors['accent_blue'],
        opacity=0.7,
        name='Hedging Error'
    ), row=1, col=2)
    
    var_95 = np.percentile(results["hedging_errors"], 5)
    var_99 = np.percentile(results["hedging_errors"], 1)
    fig.add_vline(x=var_95, line_dash="dash", line_color=colors['accent_orange'], row=1, col=2)
    fig.add_vline(x=var_99, line_dash="dash", line_color=colors['accent_red'], row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=results["final_payoffs"],
        y=results["portfolio_values"][:, -1],
        mode='markers',
        marker=dict(size=4, color=colors['accent_blue'], opacity=0.5),
        name='Portfolio'
    ), row=2, col=1)
    
    max_payoff = results["final_payoffs"].max()
    fig.add_trace(go.Scatter(
        x=[0, max_payoff], y=[0, max_payoff],
        mode='lines',
        line=dict(color=colors['text_primary'], dash='dash'),
        name='Perfect Hedge'
    ), row=2, col=1)
    
    mean_delta = results["deltas"].mean(axis=0)
    fig.add_trace(go.Scatter(
        x=time_axis, y=mean_delta,
        mode='lines',
        line=dict(width=2, color=colors['accent_green']),
        name='Mean Delta'
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor=colors['chart_bg'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary'])
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=colors['grid_color'])
    fig.update_yaxes(showgrid=True, gridcolor=colors['grid_color'])
    
    return fig


def create_greeks_heatmap(S, K, T, r, sigma, greek, option_type):
    """Create Greeks heatmap."""
    spot_range = np.linspace(S * 0.7, S * 1.3, 25)
    vol_range = np.linspace(max(0.05, sigma * 0.3), sigma * 2, 25)
    
    Z = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            result = black_scholes(spot, K, T, r, vol, option_type)
            Z[i, j] = result[greek]
    
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=spot_range,
        y=vol_range * 100,
        colorscale='RdBu_r' if greek in ['Delta', 'Gamma', 'Vega'] else 'Viridis',
        colorbar=dict(title=greek),
        hovertemplate=f'Spot: %{{x:.1f}}<br>Vol: %{{y:.1f}}%<br>{greek}: %{{z:.4f}}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[S], y=[sigma * 100],
        mode='markers',
        marker=dict(size=12, color='white', symbol='x', line=dict(width=2, color='black')),
        name='Current',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text=f"{greek} Heatmap", font=dict(color=colors['text_primary'])),
        xaxis_title="Spot Price",
        yaxis_title="Volatility (%)",
        height=350,
        plot_bgcolor=colors['chart_bg'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary'])
    )
    
    return fig


def create_pnl_payoff_chart(S, K, T, r, sigma, option_type, premium):
    """Create P&L payoff diagram."""
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)
    
    if option_type == "Call":
        payoff = np.maximum(spot_range - K, 0)
        breakeven = K + premium
    else:
        payoff = np.maximum(K - spot_range, 0)
        breakeven = K - premium
    
    pnl = payoff - premium
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_range, y=payoff,
        mode='lines',
        name='Payoff at Expiry',
        line=dict(color=colors['accent_blue'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl,
        mode='lines',
        name='P&L',
        line=dict(color=colors['accent_green'], width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color=colors['text_muted'])
    fig.add_vline(x=K, line_dash="dot", line_color=colors['accent_orange'])
    fig.add_vline(x=S, line_dash="dash", line_color=colors['accent_purple'])
    
    fig.add_trace(go.Scatter(
        x=[breakeven], y=[0],
        mode='markers',
        marker=dict(size=10, color=colors['accent_red'], symbol='diamond'),
        name=f'Break-even: ${breakeven:.2f}'
    ))
    
    fig.update_layout(
        title="P&L Diagram",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="P&L ($)",
        height=400,
        plot_bgcolor=colors['chart_bg'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=True, gridcolor=colors['grid_color']),
        yaxis=dict(showgrid=True, gridcolor=colors['grid_color'])
    )
    
    return fig, breakeven


# =============================================================================
# 7. SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: {colors['accent_blue']}; margin: 0; font-size: 28px; font-weight: 700;">CRR</h2>
        <p style="color: {colors['text_muted']}; margin: 4px 0 0 0; font-size: 10px; letter-spacing: 3px;">PRICING PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Theme Toggle
    st.markdown(f'<p class="section-title">Theme</p>', unsafe_allow_html=True)
    new_theme = st.selectbox("Select Theme", ['dark', 'light'], 
                             index=0 if st.session_state.theme == 'dark' else 1,
                             key="theme_selector")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    
    st.markdown("---")
    
    # Data Source
    st.markdown(f'<p class="section-title">Data Source</p>', unsafe_allow_html=True)
    
    selected_index = st.selectbox("Index", list(INDEX_MAP.keys()), index=0)
    ticker_symbol = INDEX_MAP[selected_index]
    
    use_real_data = ticker_symbol != "MANUAL"
    
    if use_real_data:
        selected_period = st.selectbox("History Period", list(PERIOD_MAP.keys()), index=1)
        
        with st.spinner("Fetching Market Data..."):
            df_hist = get_stock_data(ticker_symbol, PERIOD_MAP[selected_period])
        
        if df_hist is not None and not df_hist.empty:
            current_price = float(df_hist['Close'].iloc[-1])
            returns = np.log(df_hist['Close'] / df_hist['Close'].shift(1)).dropna()
            hist_vol = float(returns.std() * np.sqrt(252))
            
            st.markdown(f"""
            <div class="data-badge">
                {ticker_symbol}: ${current_price:.2f} | Vol: {hist_vol:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            current_price = 100.0
            hist_vol = 0.20
            st.warning("Could not fetch data. Using defaults.")
    else:
        df_hist = None
        current_price = 100.0
        hist_vol = 0.20
    
    st.markdown("---")
    
    # Model Parameters
    st.markdown(f'<p class="section-title">Model Parameters</p>', unsafe_allow_html=True)
    
    if use_real_data:
        S = st.number_input("Spot Price (S)", value=float(current_price), min_value=0.1, step=1.0, format="%.2f")
        K = st.number_input("Strike Price (K)", value=float(current_price), min_value=0.1, step=1.0, format="%.2f")
        sigma = st.slider("Volatility", min_value=0.05, max_value=1.0, value=float(hist_vol), step=0.01, format="%.2f")
    else:
        S = st.number_input("Spot Price (S)", value=100.0, min_value=0.1, step=1.0, format="%.2f")
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=1.0, format="%.2f")
        sigma = st.slider("Volatility", min_value=0.05, max_value=1.0, value=0.20, step=0.01, format="%.2f")
    
    T = st.number_input("Maturity (Years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")
    r = st.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=0.5, step=0.001, format="%.4f")
    
    st.markdown("---")
    
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    N = st.slider("CRR Tree Steps", min_value=5, max_value=500, value=50, step=5)
    
    st.markdown("---")
    
    # Moneyness indicator
    moneyness = S / K
    if moneyness > 1.05:
        m_txt = "ITM" if option_type == "Call" else "OTM"
        m_col = colors['accent_green'] if option_type == "Call" else colors['accent_red']
    elif moneyness < 0.95:
        m_txt = "OTM" if option_type == "Call" else "ITM"
        m_col = colors['accent_red'] if option_type == "Call" else colors['accent_green']
    else:
        m_txt, m_col = "ATM", colors['accent_orange']
    
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 16px;">
        <div class="metric-label">Moneyness (S/K)</div>
        <div class="metric-value" style="color: {m_col};">{moneyness:.2%}</div>
        <span class="badge badge-info" style="margin-top: 8px;">{m_txt}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. MAIN CONTENT
# =============================================================================

# Header
st.markdown(f"""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">Cox-Ross-Rubinstein Option Pricing</h1>
            <p class="header-subtitle">Interactive platform for binomial option pricing, convergence analysis, and delta hedging simulation</p>
        </div>
        <div class="live-indicator">
            <span class="live-dot"></span>
            <span>{'LIVE' if use_real_data else 'MANUAL'}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Calculate prices
bs_result = black_scholes(S, K, T, r, sigma, option_type)
crr_price, stock_tree, option_tree, delta_tree, u, d, p = crr_binomial_tree(
    S, K, T, r, sigma, N, option_type, return_tree=True
)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard", "CRR Model", "Convergence", "Vol Surface", "Hedging", "P&L Analysis", "Theory"
])

# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================
with tab1:
    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Black-Scholes Price</div>
            <div class="metric-value">${bs_result['Price']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        price_diff = crr_price - bs_result['Price']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CRR Price ({N} steps)</div>
            <div class="metric-value blue">${crr_price:.4f}</div>
            <div style="font-size: 12px; color: {colors['accent_green'] if price_diff >= 0 else colors['accent_red']}; margin-top: 4px;">
                {price_diff:+.4f} vs BS
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Up Factor (u)</div>
            <div class="metric-value cyan">{u:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Down Factor (d)</div>
            <div class="metric-value cyan">{d:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk-Neutral Prob (p)</div>
            <div class="metric-value cyan">{p:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Greeks
    st.markdown(f'<p class="section-title">Greeks</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="greeks-grid">
        <div class="greek-item">
            <div class="greek-symbol">Delta</div>
            <div class="greek-value">{bs_result['Delta']:.4f}</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">Gamma</div>
            <div class="greek-value">{bs_result['Gamma']:.4f}</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">Vega</div>
            <div class="greek-value">{bs_result['Vega']:.4f}</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">Theta</div>
            <div class="greek-value">{bs_result['Theta']:.4f}</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">Rho</div>
            <div class="greek-value">{bs_result['Rho']:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        if df_hist is not None:
            st.plotly_chart(create_candlestick_chart(df_hist, ticker_symbol), use_container_width=True)
        else:
            # Quick convergence chart if no data
            steps_quick = list(range(10, 150, 5))
            crr_quick = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in steps_quick]
            
            fig_quick = go.Figure()
            fig_quick.add_trace(go.Scatter(x=steps_quick, y=crr_quick, mode='lines+markers',
                               name='CRR Price', line=dict(color=colors['accent_blue'], width=2), marker=dict(size=4)))
            fig_quick.add_hline(y=bs_result['Price'], line_dash="dash", line_color=colors['text_primary'],
                               annotation_text=f"BS = ${bs_result['Price']:.4f}")
            
            fig_quick.update_layout(
                title="Price Convergence Preview",
                height=350, margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor=colors['chart_bg'], paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Number of Steps", showgrid=True, gridcolor=colors['grid_color']),
                yaxis=dict(title="Option Price ($)", showgrid=True, gridcolor=colors['grid_color']),
                font=dict(color=colors['text_primary'])
            )
            st.plotly_chart(fig_quick, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Model Comparison")
        
        error_pct = abs(crr_price - bs_result['Price']) / bs_result['Price'] * 100 if bs_result['Price'] > 0 else 0
        
        st.markdown(f"""
        <div class="comparison-row">
            <span class="comparison-label">Black-Scholes</span>
            <span class="comparison-value">${bs_result['Price']:.4f}</span>
        </div>
        <div class="comparison-row">
            <span class="comparison-label">CRR ({N} steps)</span>
            <span class="comparison-value" style="color: {colors['accent_blue']};">${crr_price:.4f}</span>
        </div>
        <div class="comparison-row">
            <span class="comparison-label">Absolute Error</span>
            <span class="comparison-value" style="color: {colors['accent_red']};">${abs(crr_price - bs_result['Price']):.6f}</span>
        </div>
        <div class="comparison-row">
            <span class="comparison-label">Relative Error</span>
            <span class="comparison-value" style="color: {colors['accent_orange']};">{error_pct:.4f}%</span>
        </div>
        <div class="comparison-row" style="border: none;">
            <span class="comparison-label">CRR Delta</span>
            <span class="comparison-value" style="color: {colors['accent_green']};">{delta_tree[0, 0]:.4f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Stress Testing
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-title">Stress Testing</p>', unsafe_allow_html=True)
    
    stress_results = calculate_stress_scenarios(S, K, T, r, sigma, option_type)
    
    cols = st.columns(5)
    for i, res in enumerate(stress_results):
        with cols[i]:
            is_negative = res['pnl'] < 0
            card_class = "negative" if is_negative else "positive" if res['pnl'] > 0 else ""
            pnl_color = colors['accent_red'] if is_negative else colors['accent_green'] if res['pnl'] > 0 else colors['text_primary']
            
            st.markdown(f"""
            <div class="stress-card {card_class}">
                <div style="font-size: 12px; font-weight: 600; color: {colors['text_secondary']}; margin-bottom: 8px;">
                    {res['name']}
                </div>
                <div style="font-size: 20px; font-weight: 600; color: {colors['text_primary']};">
                    ${res['price']:.2f}
                </div>
                <div style="font-size: 14px; color: {pnl_color}; margin-top: 4px;">
                    {res['pnl']:+.2f} ({res['pnl_pct']:+.1f}%)
                </div>
                <div style="font-size: 10px; color: {colors['text_muted']}; margin-top: 4px;">
                    S={res['spot']:.0f} | Vol={res['vol']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: CRR MODEL
# =============================================================================
with tab2:
    st.markdown(f'<p class="section-title">Binomial Tree Visualization</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        tree_fig = create_binomial_tree_figure(stock_tree, option_tree, delta_tree, N, u, d, p)
        st.plotly_chart(tree_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### CRR Parameters")
        
        dt = T / N
        
        st.markdown(f"""
        <div class="formula-box">
            <div style="margin-bottom: 12px;">
                <span style="color: {colors['text_secondary']};">Time step:</span>
                <span style="color: {colors['accent_blue']}; float: right;">dt = T/N = {dt:.6f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: {colors['text_secondary']};">Up factor:</span>
                <span style="color: {colors['accent_green']}; float: right;">u = exp(sigma*sqrt(dt)) = {u:.6f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: {colors['text_secondary']};">Down factor:</span>
                <span style="color: {colors['accent_red']}; float: right;">d = 1/u = {d:.6f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: {colors['text_secondary']};">Risk-neutral prob:</span>
                <span style="color: {colors['accent_orange']}; float: right;">p = {p:.6f}</span>
            </div>
            <div>
                <span style="color: {colors['text_secondary']};">Discount factor:</span>
                <span style="color: {colors['accent_purple']}; float: right;">exp(-r*dt) = {np.exp(-r * dt):.6f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### No-Arbitrage Condition")
        no_arb = d < np.exp(r * dt) < u
        st.markdown(f"""
        <div class="formula-box">
            <div style="text-align: center;">d < exp(r*dt) < u</div>
            <div style="text-align: center; margin-top: 10px;">
                {d:.4f} < {np.exp(r * dt):.4f} < {u:.4f}
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <span class="badge {'badge-success' if no_arb else 'badge-warning'}">
                    {'SATISFIED' if no_arb else 'VIOLATED'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Current Hedge Ratio")
        current_delta = delta_tree[0, 0] if N > 0 else 0
        hedge_cost = current_delta * S
        bond_position = crr_price - hedge_cost
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Delta</div>
                <div class="metric-value blue">{current_delta:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Stock Position</div>
                <div class="metric-value positive">${hedge_cost:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3: CONVERGENCE
# =============================================================================
with tab3:
    st.markdown(f'<p class="section-title">CRR Convergence to Black-Scholes</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_steps_conv = st.slider("Maximum Steps", 50, 500, 200, 10, key="conv_steps")
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-label">BS Reference Price</div>
            <div class="metric-value" style="font-size: 20px;">${bs_result['Price']:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-label">BS Reference Delta</div>
            <div class="metric-value" style="font-size: 20px;">{bs_result['Delta']:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    conv_fig = create_convergence_figure(S, K, T, r, sigma, option_type, max_steps_conv)
    st.plotly_chart(conv_fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Theoretical Convergence")
        st.markdown("The CRR model converges to Black-Scholes as N approaches infinity:")
        st.latex(r"|C_{CRR}^N - C_{BS}| = O\left(\frac{1}{N}\right)")
        st.markdown("Doubling the steps roughly halves the error.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Observed Convergence")
        
        test_steps = [50, 100, 200]
        test_prices = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in test_steps]
        test_errors = [abs(p - bs_result['Price']) for p in test_prices]
        
        st.markdown(f"""
        | Steps | CRR Price | Error | Error x N |
        |-------|-----------|-------|-----------|
        | 50 | ${test_prices[0]:.6f} | {test_errors[0]:.6f} | {test_errors[0] * 50:.4f} |
        | 100 | ${test_prices[1]:.6f} | {test_errors[1]:.6f} | {test_errors[1] * 100:.4f} |
        | 200 | ${test_prices[2]:.6f} | {test_errors[2]:.6f} | {test_errors[2] * 200:.4f} |
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4: VOL SURFACE
# =============================================================================
with tab4:
    st.markdown(f'<p class="section-title">Implied Volatility Surface</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        data_source = st.radio(
            "Data Source",
            ["Real Market Data", "Synthetic Model"],
            index=0 if use_real_data else 1
        )
        
        if data_source == "Real Market Data" and use_real_data:
            load_vol = st.button("Load Volatility Surface", type="primary")
        else:
            load_vol = False
    
    with col2:
        if data_source == "Real Market Data":
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid {colors['accent_blue']};">
                <strong>Real Market Data</strong><br>
                <span style="font-size: 12px; color: {colors['text_secondary']};">
                    Fetches actual option chain data from {ticker_symbol} via Yahoo Finance.
                    Click the button to load the volatility surface.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(139, 92, 246, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid {colors['accent_purple']};">
                <strong>Synthetic Model</strong><br>
                <span style="font-size: 12px; color: {colors['text_secondary']};">
                    Generates a realistic volatility surface using smile and term structure effects.
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if data_source == "Real Market Data" and load_vol and use_real_data:
        with st.spinner("Fetching option chain data..."):
            vol_data = get_vol_surface_data(ticker_symbol, S, option_type)
        
        if vol_data:
            strikes, maturities, ivs = vol_data
            fig = create_vol_surface_real(strikes, maturities, ivs, ticker_symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data points to create surface. Try SPY or QQQ.")
        else:
            st.warning("Could not retrieve option chain data. Try a more liquid index.")
    
    elif data_source == "Synthetic Model":
        fig = create_vol_surface_synthetic(S, K, T, r, sigma)
        st.plotly_chart(fig, use_container_width=True)
    
    elif data_source == "Real Market Data" and not load_vol:
        st.info("Click 'Load Volatility Surface' to fetch real-time option chain data.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Volatility Smile")
        st.markdown("""
        The volatility smile is the empirical observation that options with strikes 
        far from spot tend to have higher implied volatilities than ATM options.
        
        **Causes:** Fat tails in return distributions, jump risk, supply/demand dynamics.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Term Structure")
        st.markdown("""
        The term structure describes how IV varies across different maturities.
        
        **Patterns:** Contango (IV increases with maturity), Backwardation (IV decreases), Flat.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 5: HEDGING
# =============================================================================
with tab5:
    st.markdown(f'<p class="section-title">Monte Carlo Delta Hedging Simulation</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_sim_paths = st.selectbox("Simulation Paths", [100, 500, 1000, 2000], index=1)
    with col2:
        n_sim_steps = st.selectbox("Rebalancing Steps", [12, 52, 252], index=1,
                                   format_func=lambda x: f"{x} ({'Monthly' if x==12 else 'Weekly' if x==52 else 'Daily'})")
    with col3:
        use_bs = st.selectbox("Delta Method", ["Black-Scholes", "CRR"], index=0)
    with col4:
        run_sim = st.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_sim or 'hedge_results' in st.session_state:
        if run_sim:
            with st.spinner("Running Monte Carlo simulation..."):
                results = simulate_hedging_strategy(S, K, T, r, sigma, n_sim_steps, n_sim_paths,
                                                    use_bs_delta=(use_bs == "Black-Scholes"))
                st.session_state['hedge_results'] = results
        
        results = st.session_state['hedge_results']
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        mean_error = results['hedging_errors'].mean()
        std_error = results['hedging_errors'].std()
        var_95 = np.percentile(results['hedging_errors'], 5)
        var_99 = np.percentile(results['hedging_errors'], 1)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("Mean Hedging Error", f"${mean_error:.4f}", colors['accent_green'] if mean_error >= 0 else colors['accent_red']),
            ("Std Dev Error", f"${std_error:.4f}", colors['text_primary']),
            ("VaR 95%", f"${var_95:.4f}", colors['accent_orange']),
            ("VaR 99%", f"${var_99:.4f}", colors['accent_red']),
            ("Initial Premium", f"${results['initial_price']:.4f}", colors['accent_blue'])
        ]
        
        for col, (label, value, color) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color: {color};">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        hedge_fig = create_hedging_figure(results, min(100, n_sim_paths))
        st.plotly_chart(hedge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 6: P&L ANALYSIS
# =============================================================================
with tab6:
    st.markdown(f'<p class="section-title">P&L and Payoff Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        pnl_fig, breakeven = create_pnl_payoff_chart(S, K, T, r, sigma, option_type, bs_result['Price'])
        st.plotly_chart(pnl_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Position Summary")
        
        max_loss = -bs_result['Price']
        max_profit = "Unlimited" if option_type == "Call" else K - bs_result['Price']
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-label">Premium Paid</div>
            <div class="metric-value">${bs_result['Price']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-label">Break-Even</div>
            <div class="metric-value cyan">${breakeven:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-label">Max Loss</div>
            <div class="metric-value negative">${max_loss:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Profit</div>
            <div class="metric-value positive">{max_profit if isinstance(max_profit, str) else f'${max_profit:.2f}'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Greeks Heatmaps
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-title">Greeks Heatmaps</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.plotly_chart(create_greeks_heatmap(S, K, T, r, sigma, "Delta", option_type), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.plotly_chart(create_greeks_heatmap(S, K, T, r, sigma, "Gamma", option_type), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.plotly_chart(create_greeks_heatmap(S, K, T, r, sigma, "Vega", option_type), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.plotly_chart(create_greeks_heatmap(S, K, T, r, sigma, "Theta", option_type), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 7: THEORY
# =============================================================================
with tab7:
    st.markdown(f'<p class="section-title">Mathematical Foundation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Cox-Ross-Rubinstein Model")
        st.markdown("The CRR model discretizes the continuous-time Black-Scholes framework:")
        st.latex(r"S_{t+\Delta t} = \begin{cases} S_t \cdot u & \text{prob } p \\ S_t \cdot d & \text{prob } 1-p \end{cases}")
        st.markdown("Parameters matching GBM moments:")
        st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}")
        st.latex(r"p = \frac{e^{r\Delta t} - d}{u - d}")
        st.markdown("Option price by backward induction:")
        st.latex(r"C_t = e^{-r\Delta t}[p \cdot C_{t+1}^u + (1-p) \cdot C_{t+1}^d]")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Black-Scholes Model")
        st.markdown("European call option formula:")
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
        st.markdown("Where:")
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
        st.markdown("Greeks:")
        st.latex(r"\Delta = N(d_1), \quad \Gamma = \frac{N'(d_1)}{S\sigma\sqrt{T}}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model extensions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-title">Model Limitations and Extensions</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### CRR Limitations")
        st.markdown("""
        - Constant volatility assumption
        - Discrete time steps
        - No jumps in asset prices
        - European options only (basic form)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Heston Model")
        st.latex(r"dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^v")
        st.markdown("""
        - Stochastic volatility
        - Captures volatility smile
        - Mean-reverting variance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### SABR Model")
        st.latex(r"dF_t = \sigma_t F_t^\beta dW_t^1")
        st.markdown("""
        - Popular in rates/FX markets
        - Analytical approximations
        - Calibrates to smile
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; padding: 20px; border-top: 1px solid {colors['border_color']};">
    <p style="color: {colors['text_muted']}; font-size: 12px; margin: 0;">
        CRR Pricing Platform - ESILV Pi2 Project - 2025
    </p>
    <p style="color: {colors['text_muted']}; font-size: 11px; margin: 8px 0 0 0;">
        Built with Streamlit | Mathematical models for educational purposes only
    </p>
</div>
""", unsafe_allow_html=True)