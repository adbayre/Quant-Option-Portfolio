import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CRR Pricing Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. CUSTOM CSS - Premium Financial Terminal Design
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-hover: #243044;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-orange: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-color: #1e293b;
    --glow-blue: rgba(59, 130, 246, 0.15);
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #0d1321 50%, var(--bg-primary) 100%);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 100%;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

/* Custom card component */
.premium-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, rgba(26,35,50,0.8) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}

.premium-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15), inset 0 1px 0 rgba(255,255,255,0.08);
    transform: translateY(-2px);
}

/* Card with accent border */
.accent-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-blue);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 20px var(--glow-blue);
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-value.positive { color: var(--accent-green); }
.metric-value.negative { color: var(--accent-red); }
.metric-value.blue { color: var(--accent-blue); }
.metric-value.cyan { color: var(--accent-cyan); }

.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin-top: 6px;
}

/* Section titles */
.section-title {
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
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-blue) 0%, transparent 100%);
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.header-banner::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-purple));
}

.header-title {
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
}

/* Greeks grid */
.greeks-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 16px 0;
}

.greek-item {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.greek-item:hover {
    border-color: var(--accent-purple);
    background: var(--bg-hover);
}

.greek-symbol {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: var(--accent-purple);
    margin-bottom: 4px;
}

.greek-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    color: var(--text-primary);
}

.greek-name {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Formula box */
.formula-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #151922 100%);
    border: 1px solid var(--border-color);
    border-left: 3px solid var(--accent-purple);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
    font-family: 'JetBrains Mono', monospace;
}

/* Tree node */
.tree-node {
    background: var(--bg-card);
    border: 2px solid var(--accent-blue);
    border-radius: 8px;
    padding: 8px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-primary);
    display: inline-block;
    margin: 4px;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--bg-secondary);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    padding: 12px 24px;
    border: none;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
    color: white !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
}

/* Input styling */
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px var(--glow-blue) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: var(--accent-blue) !important;
}

/* Checkbox */
.stCheckbox label span {
    color: var(--text-secondary) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
    border: 1px solid var(--accent-green);
}

.badge-info {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
}

/* Comparison table */
.comparison-row {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-color);
}

.comparison-label {
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
}

.comparison-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--text-primary);
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. MATHEMATICAL & DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol: str, period: str):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        return df  # <--- Only return the DataFrame (it is serializable)
    except Exception as e:
        return None

def get_vol_surface_data(ticker_obj, spot_price, option_type):
    """Fetch option chain and structure data for vol surface."""
    try:
        exps = ticker_obj.options[:6] # Limit to first 6 expirations for speed
        if not exps: return None
        strikes, tm, iv = [], [], []
        
        for exp in exps:
            chain = ticker_obj.option_chain(exp)
            data = chain.calls if option_type == "Call" else chain.puts
            
            T = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
            if T < 0.01: continue
            
            # Filter for liquidity and relevance
            mask = (data['impliedVolatility'] > 0.001) & \
                   (data['strike'] > spot_price * 0.5) & (data['strike'] < spot_price * 1.5)
            
            f = data[mask]
            strikes.extend(f['strike'].tolist())
            tm.extend([T]*len(f))
            iv.extend(f['impliedVolatility'].tolist())
            
        return strikes, tm, iv
    except: return None

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "Call") -> dict:
    if T <= 0 or sigma <= 0:
        return {k: 0.0 for k in ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
        theta = (-(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    else:  # Put
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

def crr_binomial_tree(S: float, K: float, T: float, r: float, sigma: float, 
                       N: int, option_type: str = "Call", 
                       return_tree: bool = False) -> tuple:
    if T <= 0 or sigma <= 0:
        return (0.0, None, None, None, 1, 1, 0.5) if return_tree else 0.0
    
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

def crr_delta(S: float, K: float, T: float, r: float, sigma: float, N: int, option_type: str = "Call") -> float:
    if T <= 0 or sigma <= 0 or N < 1: return 0.0
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    Su = S * u; Sd = S * d
    Cu = crr_binomial_tree(Su, K, T - dt, r, sigma, N - 1, option_type)
    Cd = crr_binomial_tree(Sd, K, T - dt, r, sigma, N - 1, option_type)
    return (Cu - Cd) / (Su - Sd)

def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int) -> np.ndarray:
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    dW = np.random.standard_normal((n_paths, n_steps))
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW[:, t-1])
    return paths

def simulate_hedging_strategy(S0: float, K: float, T: float, r: float, sigma: float,
                               n_steps: int, n_paths: int, use_bs_delta: bool = True) -> dict:
    dt = T / n_steps
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    bs_initial = black_scholes(S0, K, T, r, sigma, "Call")
    initial_option_price = bs_initial["Price"]
    portfolio_values = np.zeros((n_paths, n_steps + 1))
    cash_accounts = np.zeros((n_paths, n_steps + 1))
    stock_positions = np.zeros((n_paths, n_steps + 1))
    deltas = np.zeros((n_paths, n_steps + 1))
    
    for i in range(n_paths):
        S = paths[i, 0]
        if use_bs_delta: delta = black_scholes(S, K, T, r, sigma, "Call")["Delta"]
        else: delta = crr_delta(S, K, T, r, sigma, max(10, n_steps), "Call")
        deltas[i, 0] = delta
        stock_positions[i, 0] = delta * S
        cash_accounts[i, 0] = initial_option_price - delta * S
        portfolio_values[i, 0] = stock_positions[i, 0] + cash_accounts[i, 0]
    
    for t in range(1, n_steps + 1):
        ttm = T - t * dt
        for i in range(n_paths):
            S = paths[i, t]
            if ttm > 0.001:
                if use_bs_delta: new_delta = black_scholes(S, K, ttm, r, sigma, "Call")["Delta"]
                else: new_delta = crr_delta(S, K, ttm, r, sigma, max(10, n_steps - t), "Call")
            else: new_delta = 1.0 if S > K else 0.0
            old_delta = deltas[i, t - 1]
            cash_accounts[i, t] = cash_accounts[i, t - 1] * np.exp(r * dt)
            shares_traded = new_delta - old_delta
            cash_accounts[i, t] -= shares_traded * S
            deltas[i, t] = new_delta
            stock_positions[i, t] = new_delta * S
            portfolio_values[i, t] = stock_positions[i, t] + cash_accounts[i, t]
            
    final_payoffs = np.maximum(paths[:, -1] - K, 0)
    final_portfolio = portfolio_values[:, -1]
    hedging_errors = final_portfolio - final_payoffs
    return {
        "paths": paths, "portfolio_values": portfolio_values, "cash_accounts": cash_accounts,
        "stock_positions": stock_positions, "deltas": deltas, "final_payoffs": final_payoffs,
        "hedging_errors": hedging_errors, "initial_price": initial_option_price
    }

# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================

def create_candlestick_chart(df, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444'
    )])
    fig.update_layout(
        title=dict(text=f"{ticker} Price History", font=dict(color='#f8fafc', size=14)),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#1e293b', rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='#1e293b'),
        font=dict(color='#f8fafc')
    )
    return fig

def create_vol_surface_figure(strikes, tm, iv, ticker):
    if not strikes: return go.Figure()
    grid_x, grid_y = np.mgrid[min(strikes):max(strikes):40j, min(tm):max(tm):20j]
    grid_z = griddata((strikes, tm), iv, (grid_x, grid_y), method='linear')
    
    fig = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
    fig.update_layout(
        title=dict(text=f"{ticker} Implied Volatility Surface", font=dict(color='#f8fafc', size=14)),
        scene=dict(
            xaxis=dict(title='Strike', backgroundcolor='#111827', gridcolor='#1e293b', title_font=dict(color='#94a3b8')),
            yaxis=dict(title='Maturity (Years)', backgroundcolor='#111827', gridcolor='#1e293b', title_font=dict(color='#94a3b8')),
            zaxis=dict(title='IV', backgroundcolor='#111827', gridcolor='#1e293b', title_font=dict(color='#94a3b8'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def create_binomial_tree_figure(stock_tree, option_tree, delta_tree, N, u, d, p):
    fig = go.Figure()
    node_color = "#3b82f6"; line_color = "#1e293b"; text_color = "#f8fafc"
    max_display = min(N, 8)
    
    for i in range(max_display + 1):
        for j in range(i + 1):
            x = i; y = i - 2 * j
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=40, color=node_color, line=dict(width=2, color='#60a5fa')),
                text=f"S={stock_tree[j, i]:.1f}<br>V={option_tree[j, i]:.2f}",
                textposition="middle center",
                textfont=dict(size=8, color=text_color, family="JetBrains Mono"),
                hoverinfo='skip', showlegend=False
            ))
            if i < max_display:
                fig.add_trace(go.Scatter(x=[x, x + 1], y=[y, y + 1], mode='lines', line=dict(color=line_color, width=1.5), hoverinfo='skip', showlegend=False))
                fig.add_trace(go.Scatter(x=[x, x + 1], y=[y, y - 1], mode='lines', line=dict(color=line_color, width=1.5), hoverinfo='skip', showlegend=False))
    
    fig.add_annotation(x=0.5, y=0.5, text=f"p = {p:.4f}", showarrow=True, arrowhead=2, ax=30, ay=-30, font=dict(size=10, color="#10b981"), arrowcolor="#10b981")
    fig.add_annotation(x=0.5, y=-0.5, text=f"1-p = {1-p:.4f}", showarrow=True, arrowhead=2, ax=30, ay=30, font=dict(size=10, color="#ef4444"), arrowcolor="#ef4444")
    
    fig.update_layout(title=dict(text=f"Binomial Tree ({max_display + 1} steps displayed)", font=dict(size=14, color=text_color)),
                      showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
                      height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_convergence_figure(S, K, T, r, sigma, option_type, max_steps):
    bs_price = black_scholes(S, K, T, r, sigma, option_type)["Price"]
    bs_delta = black_scholes(S, K, T, r, sigma, option_type)["Delta"]
    steps_range = list(range(5, max_steps + 1, 5))
    crr_prices = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in steps_range]
    price_errors = [abs(p - bs_price) for p in crr_prices]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Price Convergence", "Price Error (|CRR - BS|)"), horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=steps_range, y=crr_prices, name="CRR Price", line=dict(color="#3b82f6", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[steps_range[0], steps_range[-1]], y=[bs_price, bs_price], name="BS Price", line=dict(color="#f8fafc", width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps_range, y=price_errors, name="Error", line=dict(color="#ef4444", width=2), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                      plot_bgcolor='rgba(17, 24, 39, 0.8)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'))
    fig.update_xaxes(showgrid=True, gridcolor='#1e293b'); fig.update_yaxes(showgrid=True, gridcolor='#1e293b')
    return fig

def create_hedging_pnl_figure(results, n_display=100):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Price Paths", "Hedging Error", "Portfolio vs Payoff", "Delta"), vertical_spacing=0.12, horizontal_spacing=0.1)
    n_paths = results["paths"].shape[0]; n_steps = results["paths"].shape[1]; time_axis = np.linspace(0, 1, n_steps)
    
    for i in range(min(n_display, n_paths)):
        fig.add_trace(go.Scatter(x=time_axis, y=results["paths"][i], mode='lines', line=dict(width=0.5, color=f'rgba(59, 130, 246, 0.3)'), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_axis, y=results["paths"].mean(axis=0), mode='lines', line=dict(width=2, color='#f8fafc'), name='Mean Path'), row=1, col=1)
    
    fig.add_trace(go.Histogram(x=results["hedging_errors"], nbinsx=50, marker_color='#3b82f6', opacity=0.7, name='Error'), row=1, col=2)
    fig.add_trace(go.Scatter(x=results["final_payoffs"], y=results["portfolio_values"][:, -1], mode='markers', marker=dict(size=4, color='#3b82f6', opacity=0.5), name='Portfolio'), row=2, col=1)
    max_payoff = results["final_payoffs"].max()
    fig.add_trace(go.Scatter(x=[0, max_payoff], y=[0, max_payoff], mode='lines', line=dict(color='#f8fafc', dash='dash'), name='Perfect Hedge'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=time_axis, y=results["deltas"].mean(axis=0), mode='lines', line=dict(width=2, color='#10b981'), name='Mean Delta'), row=2, col=2)
    fig.update_layout(height=600, showlegend=True, legend=dict(orientation="h", y=1.02), plot_bgcolor='rgba(17, 24, 39, 0.8)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'))
    fig.update_xaxes(showgrid=True, gridcolor='#1e293b'); fig.update_yaxes(showgrid=True, gridcolor='#1e293b')
    return fig

# =============================================================================
# 5. SIDEBAR - PARAMETERS
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #3b82f6; margin: 0; font-size: 24px; font-weight: 700;">CRR</h2>
        <p style="color: #64748b; margin: 5px 0 0 0; font-size: 11px; letter-spacing: 2px;">PRICING PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 1. ASSET SELECTION
    st.markdown('<p class="section-title">Asset Selection</p>', unsafe_allow_html=True)
    
    index_map = {
        "S&P 500 (US)": "SPY", "Nasdaq 100 (US)": "QQQ", "Dow Jones (US)": "DIA",
        "Russell 2000": "IWM", "VIX": "VXX", "Euro Stoxx 50": "FEZ",
        "CAC 40": "EWQ", "DAX": "EWG", "FTSE 100": "EWU", "Nikkei 225": "EWJ"
    }
    selected_name = st.selectbox("Index", list(index_map.keys()))
    ticker_symbol = index_map[selected_name]
    
    horizon_map = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    selected_horizon = st.selectbox("History Period", list(horizon_map.keys()), index=1)
    
    # Fetch Data
    with st.spinner("Fetching Market Data..."):
        # 1. Get the cached dataframe
        df_hist = get_stock_data(ticker_symbol, horizon_map[selected_horizon])
        
        # 2. Re-create the Ticker object instantly (it's lightweight)
        ticker_obj = yf.Ticker(ticker_symbol)
        
    if df_hist is not None and not df_hist.empty:
        current_price = df_hist['Close'].iloc[-1]
        # Calculate Hist Volatility
        returns = np.log(df_hist['Close']/df_hist['Close'].shift(1))
        hist_vol = returns.std() * np.sqrt(252)
        st.caption(f"Fetched {ticker_symbol}: S={current_price:.2f}, Vol={hist_vol:.2%}")
    else:
        st.error("Error fetching data")
        current_price = 100.0
        hist_vol = 0.20

    st.markdown("---")
    
    # 2. OPTION PARAMETERS
    st.markdown('<p class="section-title">Model Parameters</p>', unsafe_allow_html=True)
    
    S = st.number_input("Spot Price", value=float(current_price), min_value=0.1, step=1.0, format="%.2f")
    K = st.number_input("Strike Price", value=float(current_price), min_value=0.1, step=1.0, format="%.2f")
    T = st.number_input("Maturity (Years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")
    r = st.number_input("Risk-Free Rate", value=0.045, min_value=0.0, max_value=0.5, step=0.001, format="%.4f")
    sigma = st.slider("Volatility", min_value=0.01, max_value=2.0, value=float(hist_vol), step=0.01, format="%.2f")
    
    st.markdown("---")
    
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    N = st.slider("CRR Tree Steps", min_value=5, max_value=500, value=50, step=5)
    
    st.markdown("---")
    
    moneyness = S / K
    if moneyness > 1.05:
        m_txt, m_col = ("ITM", "#10b981") if option_type == "Call" else ("OTM", "#ef4444")
    elif moneyness < 0.95:
        m_txt, m_col = ("OTM", "#ef4444") if option_type == "Call" else ("ITM", "#10b981")
    else:
        m_txt, m_col = ("ATM", "#f59e0b")
    
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 16px;">
        <div class="metric-label">Moneyness</div>
        <div class="metric-value" style="color: {m_col};">{moneyness:.2%}</div>
        <span class="badge badge-info" style="margin-top: 8px;">{m_txt}</span>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 6. MAIN CONTENT
# =============================================================================

st.markdown("""
<div class="header-banner">
    <h1 class="header-title">Cox-Ross-Rubinstein Option Pricing</h1>
    <p class="header-subtitle">Real-time data integration, advanced visualization, and delta hedging simulation</p>
</div>
""", unsafe_allow_html=True)

# Calculate prices
bs_result = black_scholes(S, K, T, r, sigma, option_type)
crr_price, stock_tree, option_tree, delta_tree, u, d, p = crr_binomial_tree(S, K, T, r, sigma, N, option_type, return_tree=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard", "CRR Model", "Convergence", "Vol Surface", "Hedging", "Theory"
])

# --- TAB 1: DASHBOARD ---
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">BS Price</div><div class="metric-value">${bs_result['Price']:.4f}</div></div>""", unsafe_allow_html=True)
    with col2:
        diff = crr_price - bs_result['Price']
        st.markdown(f"""<div class="metric-card"><div class="metric-label">CRR Price ({N})</div><div class="metric-value blue">${crr_price:.4f}</div><div class="metric-delta" style="color: {'#10b981' if diff >= 0 else '#ef4444'};">{diff:+.4f} vs BS</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Delta</div><div class="metric-value cyan">{bs_result['Delta']:.4f}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Gamma</div><div class="metric-value cyan">{bs_result['Gamma']:.4f}</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Vega</div><div class="metric-value cyan">{bs_result['Vega']:.4f}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        if df_hist is not None:
            st.plotly_chart(create_candlestick_chart(df_hist, ticker_symbol), use_container_width=True)
        else:
            st.info("No market data available")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_info:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Greeks")
        greeks_html = f"""
        <div class="greeks-grid">
            <div class="greek-item"><div class="greek-value">{bs_result['Delta']:.4f}</div><div class="greek-name">Delta</div></div>
            <div class="greek-item"><div class="greek-value">{bs_result['Gamma']:.4f}</div><div class="greek-name">Gamma</div></div>
            <div class="greek-item"><div class="greek-value">{bs_result['Vega']:.4f}</div><div class="greek-name">Vega</div></div>
            <div class="greek-item"><div class="greek-value">{bs_result['Theta']:.4f}</div><div class="greek-name">Theta</div></div>
            <div class="greek-item"><div class="greek-value">{bs_result['Rho']:.4f}</div><div class="greek-name">Rho</div></div>
        </div>"""
        st.markdown(greeks_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: CRR MODEL ---
with tab2:
    st.markdown('<p class="section-title">Binomial Tree Visualization</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.plotly_chart(create_binomial_tree_figure(stock_tree, option_tree, delta_tree, N, u, d, p), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Parameters")
        st.markdown(f"""
        <div class="formula-box">
            <div><span style="color:#94a3b8;">dt:</span> <span style="color:#3b82f6; float:right;">{T/N:.6f}</span></div>
            <div><span style="color:#94a3b8;">u:</span> <span style="color:#10b981; float:right;">{u:.6f}</span></div>
            <div><span style="color:#94a3b8;">d:</span> <span style="color:#ef4444; float:right;">{d:.6f}</span></div>
            <div><span style="color:#94a3b8;">p:</span> <span style="color:#f59e0b; float:right;">{p:.6f}</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: CONVERGENCE ---
with tab3:
    st.markdown('<p class="section-title">CRR Convergence</p>', unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    max_steps_conv = st.slider("Max Steps", 50, 500, 200, 10)
    st.plotly_chart(create_convergence_figure(S, K, T, r, sigma, option_type, max_steps_conv), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: VOL SURFACE ---
with tab4:
    st.markdown('<p class="section-title">Real-Time Volatility Surface</p>', unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    if st.button("Load Volatility Surface"):
        with st.spinner("Fetching Option Chain Data..."):
            res = get_vol_surface_data(ticker_obj, S, option_type)
            if res:
                strikes, tm, iv = res
                st.plotly_chart(create_vol_surface_figure(strikes, tm, iv, ticker_symbol), use_container_width=True)
            else:
                st.warning("Could not retrieve sufficient option data.")
    else:
        st.info("Click to fetch real-time option chain data.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: HEDGING ---
with tab5:
    st.markdown('<p class="section-title">Monte Carlo Delta Hedging</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: n_sim_paths = st.selectbox("Paths", [100, 500, 1000], index=0)
    with col2: n_sim_steps = st.selectbox("Rebalancing", [12, 52, 252], index=1)
    with col3: use_bs = st.selectbox("Delta", ["Black-Scholes", "CRR"])
    with col4: run_sim = st.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_sim:
        with st.spinner("Simulating..."):
            res = simulate_hedging_strategy(S, K, T, r, sigma, n_sim_steps, n_sim_paths, (use_bs=="Black-Scholes"))
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.plotly_chart(create_hedging_pnl_figure(res, 100), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 6: THEORY ---
with tab6:
    st.markdown('<p class="section-title">Mathematical Foundation</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Cox-Ross-Rubinstein")
        st.latex(r"C_t = e^{-r\Delta t}[p \cdot C_{t+1}^u + (1-p) \cdot C_{t+1}^d]")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Black-Scholes")
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
        st.markdown('</div>', unsafe_allow_html=True)