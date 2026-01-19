"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CRR OPTION PRICING PLATFORM                               ║
║                    ESILV - Projet d'Innovation Industrielle                  ║
║                                                                              ║
║   A comprehensive platform for Cox-Ross-Rubinstein option pricing,          ║
║   demonstrating convergence to Black-Scholes and delta-hedging strategies   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

/* Animations */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px var(--glow-blue); }
    50% { box-shadow: 0 0 40px var(--glow-blue); }
}

.glow-pulse {
    animation: pulse-glow 2s infinite;
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Plotly chart containers */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. MATHEMATICAL FUNCTIONS
# =============================================================================

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "Call") -> dict:
    """
    Calculate Black-Scholes option price and Greeks.
    
    Parameters:
    -----------
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to maturity (in years)
    r : float - Risk-free interest rate
    sigma : float - Volatility
    option_type : str - "Call" or "Put"
    
    Returns:
    --------
    dict with Price, Delta, Gamma, Vega, Theta, Rho
    """
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
        "Price": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega / 100,  # Per 1% change in volatility
        "Theta": theta / 365,  # Daily theta
        "Rho": rho / 100  # Per 1% change in rate
    }


def crr_binomial_tree(S: float, K: float, T: float, r: float, sigma: float, 
                       N: int, option_type: str = "Call", 
                       return_tree: bool = False) -> tuple:
    """
    Cox-Ross-Rubinstein binomial tree option pricing.
    
    Parameters:
    -----------
    S : float - Current stock price
    K : float - Strike price  
    T : float - Time to maturity (in years)
    r : float - Risk-free interest rate
    sigma : float - Volatility
    N : int - Number of time steps
    option_type : str - "Call" or "Put"
    return_tree : bool - If True, return full price and option trees
    
    Returns:
    --------
    If return_tree=False: option price (float)
    If return_tree=True: (price, stock_tree, option_tree, delta_tree, u, d, p)
    """
    if T <= 0 or sigma <= 0:
        return (0.0, None, None, None, 1, 1, 0.5) if return_tree else 0.0
    
    # CRR parameters
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor (ensures recombining tree)
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    
    # Initialize stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Initialize option value tree at maturity
    option_tree = np.zeros((N + 1, N + 1))
    if option_type == "Call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    else:
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)
    
    # Backward induction
    discount = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
    
    # Calculate delta at each node
    delta_tree = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            if stock_tree[j, i + 1] != stock_tree[j + 1, i + 1]:
                delta_tree[j, i] = (option_tree[j, i + 1] - option_tree[j + 1, i + 1]) / \
                                   (stock_tree[j, i + 1] - stock_tree[j + 1, i + 1])
    
    if return_tree:
        return option_tree[0, 0], stock_tree, option_tree, delta_tree, u, d, p
    return option_tree[0, 0]


def crr_delta(S: float, K: float, T: float, r: float, sigma: float, 
              N: int, option_type: str = "Call") -> float:
    """Calculate CRR delta using the first step of the tree."""
    if T <= 0 or sigma <= 0 or N < 1:
        return 0.0
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    Su = S * u
    Sd = S * d
    
    Cu = crr_binomial_tree(Su, K, T - dt, r, sigma, N - 1, option_type)
    Cd = crr_binomial_tree(Sd, K, T - dt, r, sigma, N - 1, option_type)
    
    return (Cu - Cd) / (Su - Sd)


def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float, 
                        n_steps: int, n_paths: int) -> np.ndarray:
    """
    Simulate geometric Brownian motion paths.
    
    Returns array of shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate random increments
    dW = np.random.standard_normal((n_paths, n_steps))
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW[:, t-1])
    
    return paths


def simulate_hedging_strategy(S0: float, K: float, T: float, r: float, sigma: float,
                               n_steps: int, n_paths: int, use_bs_delta: bool = True) -> dict:
    """
    Simulate delta hedging strategy and compute P&L distribution.
    
    Returns dict with paths, portfolio values, hedging errors, etc.
    """
    dt = T / n_steps
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    
    # Initial option price
    bs_initial = black_scholes(S0, K, T, r, sigma, "Call")
    initial_option_price = bs_initial["Price"]
    
    # Arrays to store results
    portfolio_values = np.zeros((n_paths, n_steps + 1))
    cash_accounts = np.zeros((n_paths, n_steps + 1))
    stock_positions = np.zeros((n_paths, n_steps + 1))
    deltas = np.zeros((n_paths, n_steps + 1))
    
    # Initialize: sell option, set up hedge
    for i in range(n_paths):
        S = paths[i, 0]
        ttm = T
        
        if use_bs_delta:
            delta = black_scholes(S, K, ttm, r, sigma, "Call")["Delta"]
        else:
            delta = crr_delta(S, K, ttm, r, sigma, max(10, n_steps), "Call")
        
        deltas[i, 0] = delta
        stock_positions[i, 0] = delta * S
        cash_accounts[i, 0] = initial_option_price - delta * S
        portfolio_values[i, 0] = stock_positions[i, 0] + cash_accounts[i, 0]
    
    # Rebalance at each step
    for t in range(1, n_steps + 1):
        ttm = T - t * dt
        
        for i in range(n_paths):
            S = paths[i, t]
            
            # Calculate new delta
            if ttm > 0.001:
                if use_bs_delta:
                    new_delta = black_scholes(S, K, ttm, r, sigma, "Call")["Delta"]
                else:
                    new_delta = crr_delta(S, K, ttm, r, sigma, max(10, n_steps - t), "Call")
            else:
                new_delta = 1.0 if S > K else 0.0
            
            old_delta = deltas[i, t - 1]
            
            # Update positions
            # Cash grows at risk-free rate
            cash_accounts[i, t] = cash_accounts[i, t - 1] * np.exp(r * dt)
            
            # Rebalance: buy/sell shares
            shares_traded = new_delta - old_delta
            cash_accounts[i, t] -= shares_traded * S
            
            # Update values
            deltas[i, t] = new_delta
            stock_positions[i, t] = new_delta * S
            portfolio_values[i, t] = stock_positions[i, t] + cash_accounts[i, t]
    
    # Final P&L
    final_payoffs = np.maximum(paths[:, -1] - K, 0)
    final_portfolio = portfolio_values[:, -1]
    hedging_errors = final_portfolio - final_payoffs
    
    return {
        "paths": paths,
        "portfolio_values": portfolio_values,
        "cash_accounts": cash_accounts,
        "stock_positions": stock_positions,
        "deltas": deltas,
        "final_payoffs": final_payoffs,
        "hedging_errors": hedging_errors,
        "initial_price": initial_option_price
    }


# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================

def create_binomial_tree_figure(stock_tree: np.ndarray, option_tree: np.ndarray, 
                                 delta_tree: np.ndarray, N: int, 
                                 u: float, d: float, p: float) -> go.Figure:
    """Create interactive binomial tree visualization."""
    fig = go.Figure()
    
    # Color scheme
    node_color = "#3b82f6"
    line_color = "#1e293b"
    text_color = "#f8fafc"
    
    # Plot nodes and edges
    max_display = min(N, 8)  # Limit display for readability
    
    for i in range(max_display + 1):
        for j in range(i + 1):
            x = i
            y = i - 2 * j  # Center the tree
            
            # Node marker
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=40, color=node_color, line=dict(width=2, color='#60a5fa')),
                text=f"S={stock_tree[j, i]:.1f}<br>V={option_tree[j, i]:.2f}",
                textposition="middle center",
                textfont=dict(size=8, color=text_color, family="JetBrains Mono"),
                hovertemplate=f"<b>Step {i}, Node {j}</b><br>" +
                              f"Stock: ${stock_tree[j, i]:.2f}<br>" +
                              f"Option: ${option_tree[j, i]:.2f}<br>" +
                              f"Delta: {delta_tree[j, i] if i < N else 'N/A':.4f}<extra></extra>",
                showlegend=False
            ))
            
            # Draw edges to next nodes
            if i < max_display:
                # Up edge
                fig.add_trace(go.Scatter(
                    x=[x, x + 1], y=[y, y + 1],
                    mode='lines',
                    line=dict(color=line_color, width=1.5),
                    hoverinfo='skip',
                    showlegend=False
                ))
                # Down edge
                fig.add_trace(go.Scatter(
                    x=[x, x + 1], y=[y, y - 1],
                    mode='lines',
                    line=dict(color=line_color, width=1.5),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    # Add probability annotations
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"p = {p:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=30, ay=-30,
        font=dict(size=10, color="#10b981"),
        arrowcolor="#10b981"
    )
    fig.add_annotation(
        x=0.5, y=-0.5,
        text=f"1-p = {1-p:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=30, ay=30,
        font=dict(size=10, color="#ef4444"),
        arrowcolor="#ef4444"
    )
    
    fig.update_layout(
        title=dict(
            text=f"Binomial Tree (showing {max_display + 1} of {N + 1} steps)",
            font=dict(size=14, color=text_color)
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_convergence_figure(S: float, K: float, T: float, r: float, sigma: float, 
                               option_type: str, max_steps: int = 200) -> go.Figure:
    """Create CRR to Black-Scholes convergence visualization."""
    
    bs_price = black_scholes(S, K, T, r, sigma, option_type)["Price"]
    bs_delta = black_scholes(S, K, T, r, sigma, option_type)["Delta"]
    
    steps_range = list(range(5, max_steps + 1, 5))
    crr_prices = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in steps_range]
    crr_deltas = [crr_delta(S, K, T, r, sigma, n, option_type) for n in steps_range]
    
    # Price convergence errors
    price_errors = [abs(p - bs_price) for p in crr_prices]
    delta_errors = [abs(d - bs_delta) for d in crr_deltas]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Option Price Convergence", "Price Error (|CRR - BS|)",
                       "Delta Convergence", "Delta Error (|CRR - BS|)"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Price convergence
    fig.add_trace(
        go.Scatter(x=steps_range, y=crr_prices, name="CRR Price",
                   line=dict(color="#3b82f6", width=2), mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[steps_range[0], steps_range[-1]], y=[bs_price, bs_price],
                   name="BS Price", line=dict(color="#f8fafc", width=2, dash='dash')),
        row=1, col=1
    )
    
    # Price error
    fig.add_trace(
        go.Scatter(x=steps_range, y=price_errors, name="Price Error",
                   line=dict(color="#ef4444", width=2), mode='lines', fill='tozeroy',
                   fillcolor='rgba(239, 68, 68, 0.2)'),
        row=1, col=2
    )
    
    # Delta convergence
    fig.add_trace(
        go.Scatter(x=steps_range, y=crr_deltas, name="CRR Delta",
                   line=dict(color="#10b981", width=2), mode='lines'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[steps_range[0], steps_range[-1]], y=[bs_delta, bs_delta],
                   name="BS Delta", line=dict(color="#f8fafc", width=2, dash='dash')),
        row=2, col=1
    )
    
    # Delta error
    fig.add_trace(
        go.Scatter(x=steps_range, y=delta_errors, name="Delta Error",
                   line=dict(color="#f59e0b", width=2), mode='lines', fill='tozeroy',
                   fillcolor='rgba(245, 158, 11, 0.2)'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc')
    )
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                showgrid=True, gridcolor='#1e293b', zeroline=False,
                title_text="Number of Steps" if i == 2 else "",
                row=i, col=j
            )
            fig.update_yaxes(
                showgrid=True, gridcolor='#1e293b', zeroline=False,
                row=i, col=j
            )
    
    return fig


def create_hedging_pnl_figure(results: dict, n_display: int = 100) -> go.Figure:
    """Create hedging P&L visualization."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Sample Price Paths", "Hedging Error Distribution",
                       "Portfolio Value vs Option Payoff", "Delta Over Time"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    n_paths = results["paths"].shape[0]
    n_steps = results["paths"].shape[1]
    time_axis = np.linspace(0, 1, n_steps)
    
    # Sample paths
    for i in range(min(n_display, n_paths)):
        fig.add_trace(
            go.Scatter(x=time_axis, y=results["paths"][i], mode='lines',
                      line=dict(width=0.5, color=f'rgba(59, 130, 246, 0.3)'),
                      showlegend=False, hoverinfo='skip'),
            row=1, col=1
        )
    
    # Mean path
    mean_path = results["paths"].mean(axis=0)
    fig.add_trace(
        go.Scatter(x=time_axis, y=mean_path, mode='lines',
                  line=dict(width=2, color='#f8fafc'),
                  name='Mean Path'),
        row=1, col=1
    )
    
    # Hedging error histogram
    fig.add_trace(
        go.Histogram(x=results["hedging_errors"], nbinsx=50,
                    marker_color='#3b82f6', opacity=0.7,
                    name='Hedging Error'),
        row=1, col=2
    )
    
    # Add VaR lines
    var_95 = np.percentile(results["hedging_errors"], 5)
    var_99 = np.percentile(results["hedging_errors"], 1)
    fig.add_vline(x=var_95, line_dash="dash", line_color="#f59e0b", row=1, col=2)
    fig.add_vline(x=var_99, line_dash="dash", line_color="#ef4444", row=1, col=2)
    
    # Portfolio vs Payoff scatter
    fig.add_trace(
        go.Scatter(x=results["final_payoffs"], y=results["portfolio_values"][:, -1],
                  mode='markers', marker=dict(size=4, color='#3b82f6', opacity=0.5),
                  name='Portfolio vs Payoff'),
        row=2, col=1
    )
    # Perfect hedge line
    max_payoff = results["final_payoffs"].max()
    fig.add_trace(
        go.Scatter(x=[0, max_payoff], y=[0, max_payoff],
                  mode='lines', line=dict(color='#f8fafc', dash='dash'),
                  name='Perfect Hedge'),
        row=2, col=1
    )
    
    # Delta evolution (sample paths)
    for i in range(min(20, n_paths)):
        fig.add_trace(
            go.Scatter(x=time_axis, y=results["deltas"][i], mode='lines',
                      line=dict(width=0.5, color=f'rgba(16, 185, 129, 0.4)'),
                      showlegend=False, hoverinfo='skip'),
            row=2, col=2
        )
    
    # Mean delta
    mean_delta = results["deltas"].mean(axis=0)
    fig.add_trace(
        go.Scatter(x=time_axis, y=mean_delta, mode='lines',
                  line=dict(width=2, color='#10b981'),
                  name='Mean Delta'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc')
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='#1e293b', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#1e293b', zeroline=False)
    
    return fig


def create_greek_surface(S: float, K: float, T: float, r: float, sigma: float,
                          greek: str, option_type: str) -> go.Figure:
    """Create 3D surface for a specific Greek."""
    
    # Create ranges
    spot_range = np.linspace(S * 0.7, S * 1.3, 30)
    vol_range = np.linspace(sigma * 0.5, sigma * 1.5, 30)
    
    Z = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            result = black_scholes(spot, K, T, r, vol, option_type)
            Z[i, j] = result[greek]
    
    fig = go.Figure(data=[go.Surface(
        z=Z, x=spot_range, y=vol_range,
        colorscale='Viridis',
        contours_z=dict(show=True, usecolormap=True, project_z=True)
    )])
    
    fig.update_layout(
        title=dict(text=f"{greek} Surface", font=dict(color='#f8fafc', size=14)),
        scene=dict(
            xaxis=dict(title='Spot Price', backgroundcolor='#111827', 
                      gridcolor='#1e293b', title_font=dict(color='#94a3b8')),
            yaxis=dict(title='Volatility', backgroundcolor='#111827',
                      gridcolor='#1e293b', title_font=dict(color='#94a3b8')),
            zaxis=dict(title=greek, backgroundcolor='#111827',
                      gridcolor='#1e293b', title_font=dict(color='#94a3b8'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
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
    
    # Model Parameters
    st.markdown('<p class="section-title">📊 Model Parameters</p>', unsafe_allow_html=True)
    
    S = st.number_input("Spot Price (S₀)", value=100.0, min_value=1.0, step=1.0, format="%.2f")
    K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0, step=1.0, format="%.2f")
    T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")
    r = st.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=0.5, step=0.01, format="%.4f")
    sigma = st.slider("Volatility (σ)", min_value=0.05, max_value=1.0, value=0.20, step=0.01, format="%.2f")
    
    st.markdown("---")
    
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    N = st.slider("CRR Tree Steps", min_value=5, max_value=500, value=50, step=5)
    
    st.markdown("---")
    
    # Moneyness indicator
    moneyness = S / K
    if moneyness > 1.05:
        moneyness_text = "ITM" if option_type == "Call" else "OTM"
        moneyness_color = "#10b981" if option_type == "Call" else "#ef4444"
    elif moneyness < 0.95:
        moneyness_text = "OTM" if option_type == "Call" else "ITM"
        moneyness_color = "#ef4444" if option_type == "Call" else "#10b981"
    else:
        moneyness_text = "ATM"
        moneyness_color = "#f59e0b"
    
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 16px;">
        <div class="metric-label">Moneyness (S/K)</div>
        <div class="metric-value" style="color: {moneyness_color};">{moneyness:.2%}</div>
        <span class="badge badge-info" style="margin-top: 8px;">{moneyness_text}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 6. MAIN CONTENT - TABS
# =============================================================================

# Header
st.markdown("""
<div class="header-banner">
    <h1 class="header-title">Cox-Ross-Rubinstein Option Pricing</h1>
    <p class="header-subtitle">Interactive platform for binomial option pricing, convergence analysis, and delta hedging simulation</p>
</div>
""", unsafe_allow_html=True)

# Calculate prices
bs_result = black_scholes(S, K, T, r, sigma, option_type)
crr_price, stock_tree, option_tree, delta_tree, u, d, p = crr_binomial_tree(
    S, K, T, r, sigma, N, option_type, return_tree=True
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🌳 CRR Model", 
    "📈 Convergence", 
    "🛡️ Hedging Simulation",
    "📚 Theory"
])

# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================
with tab1:
    # Metrics row
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
        diff_color = "positive" if price_diff >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CRR Price ({N} steps)</div>
            <div class="metric-value blue">${crr_price:.4f}</div>
            <div class="metric-delta" style="color: {'#10b981' if price_diff >= 0 else '#ef4444'};">
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
    
    # Greeks section
    st.markdown('<p class="section-title">Greeks</p>', unsafe_allow_html=True)
    
    greeks_html = f"""
    <div class="greeks-grid">
        <div class="greek-item">
            <div class="greek-symbol">Δ</div>
            <div class="greek-value">{bs_result['Delta']:.4f}</div>
            <div class="greek-name">Delta</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">Γ</div>
            <div class="greek-value">{bs_result['Gamma']:.4f}</div>
            <div class="greek-name">Gamma</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">ν</div>
            <div class="greek-value">{bs_result['Vega']:.4f}</div>
            <div class="greek-name">Vega</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">Θ</div>
            <div class="greek-value">{bs_result['Theta']:.4f}</div>
            <div class="greek-name">Theta</div>
        </div>
        <div class="greek-item">
            <div class="greek-symbol">ρ</div>
            <div class="greek-value">{bs_result['Rho']:.4f}</div>
            <div class="greek-name">Rho</div>
        </div>
    </div>
    """
    st.markdown(greeks_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick convergence chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Price Convergence Preview")
        
        steps_quick = list(range(10, 150, 5))
        crr_quick = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in steps_quick]
        
        fig_quick = go.Figure()
        fig_quick.add_trace(go.Scatter(
            x=steps_quick, y=crr_quick,
            mode='lines+markers',
            name='CRR Price',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4)
        ))
        fig_quick.add_hline(y=bs_result['Price'], line_dash="dash", line_color="#f8fafc",
                           annotation_text=f"BS = ${bs_result['Price']:.4f}")
        
        fig_quick.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor='rgba(17, 24, 39, 0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Number of Steps", showgrid=True, gridcolor='#1e293b'),
            yaxis=dict(title="Option Price ($)", showgrid=True, gridcolor='#1e293b'),
            font=dict(color='#f8fafc')
        )
        st.plotly_chart(fig_quick, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### Model Comparison")
        
        error_pct = abs(crr_price - bs_result['Price']) / bs_result['Price'] * 100
        
        st.markdown(f"""
        <div class="comparison-row">
            <span class="comparison-label">Black-Scholes</span>
            <span class="comparison-value">${bs_result['Price']:.4f}</span>
        </div>
        <div class="comparison-row">
            <span class="comparison-label">CRR ({N} steps)</span>
            <span class="comparison-value" style="color: #3b82f6;">${crr_price:.4f}</span>
        </div>
        <div class="comparison-row">
            <span class="comparison-label">Absolute Error</span>
            <span class="comparison-value" style="color: #ef4444;">${abs(crr_price - bs_result['Price']):.6f}</span>
        </div>
        <div class="comparison-row">
            <span class="comparison-label">Relative Error</span>
            <span class="comparison-value" style="color: #f59e0b;">{error_pct:.4f}%</span>
        </div>
        <div class="comparison-row" style="border: none;">
            <span class="comparison-label">CRR Delta</span>
            <span class="comparison-value" style="color: #10b981;">{delta_tree[0, 0]:.4f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: CRR MODEL
# =============================================================================
with tab2:
    st.markdown('<p class="section-title">Binomial Tree Visualization</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        # Tree visualization
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
                <span style="color: #94a3b8;">Time step:</span>
                <span style="color: #3b82f6; float: right;">Δt = T/N = {dt:.6f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #94a3b8;">Up factor:</span>
                <span style="color: #10b981; float: right;">u = e^(σ√Δt) = {u:.6f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #94a3b8;">Down factor:</span>
                <span style="color: #ef4444; float: right;">d = 1/u = {d:.6f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #94a3b8;">Risk-neutral prob:</span>
                <span style="color: #f59e0b; float: right;">p = (e^(rΔt) - d)/(u - d) = {p:.6f}</span>
            </div>
            <div>
                <span style="color: #94a3b8;">Discount factor:</span>
                <span style="color: #8b5cf6; float: right;">e^(-rΔt) = {np.exp(-r * dt):.6f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### No-Arbitrage Condition")
        st.markdown("""
        For the model to be arbitrage-free, we need:
        """)
        
        no_arb = d < np.exp(r * dt) < u
        st.markdown(f"""
        <div class="formula-box">
            <div style="text-align: center;">
                d < e<sup>rΔt</sup> < u
            </div>
            <div style="text-align: center; margin-top: 10px;">
                {d:.4f} < {np.exp(r * dt):.4f} < {u:.4f}
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <span class="badge {'badge-success' if no_arb else 'badge-danger'}">
                    {'✓ SATISFIED' if no_arb else '✗ VIOLATED'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Delta hedging explanation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Delta Hedging in CRR</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Replicating Portfolio")
        st.markdown("""
        At each node, we can replicate the option payoff with a portfolio of:
        - **φ** shares of the underlying
        - **ψ** units of the risk-free bond
        
        The number of shares (delta) at node (i,j) is:
        """)
        st.latex(r"\phi_{i,j} = \frac{C^u_{i+1,j} - C^d_{i+1,j+1}}{S_{i+1,j}^u - S_{i+1,j+1}^d} = \frac{C^u - C^d}{S(u-d)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Current Hedge Ratio")
        
        current_delta = delta_tree[0, 0] if N > 0 else 0
        shares_to_hold = current_delta
        hedge_cost = shares_to_hold * S
        bond_position = crr_price - hedge_cost
        
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div class="metric-card">
                <div class="metric-label">Delta (φ)</div>
                <div class="metric-value blue">{current_delta:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Shares to Hold</div>
                <div class="metric-value">{shares_to_hold:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Stock Position</div>
                <div class="metric-value positive">${hedge_cost:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bond Position</div>
                <div class="metric-value {'positive' if bond_position >= 0 else 'negative'}">${bond_position:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3: CONVERGENCE
# =============================================================================
with tab3:
    st.markdown('<p class="section-title">CRR Convergence to Black-Scholes</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    
    # Convergence parameters
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
    
    # Convergence analysis
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Theoretical Convergence")
        st.markdown("""
        The CRR model converges to Black-Scholes as N → ∞. The rate of convergence is:
        """)
        st.latex(r"|C_{CRR}^N - C_{BS}| = O\left(\frac{1}{N}\right)")
        st.markdown("""
        This means:
        - Doubling the steps roughly halves the error
        - The error decreases linearly with the number of steps
        - More steps = more accurate but more computation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Observed Convergence")
        
        # Calculate actual convergence rate
        test_steps = [50, 100, 200]
        test_prices = [crr_binomial_tree(S, K, T, r, sigma, n, option_type) for n in test_steps]
        test_errors = [abs(p - bs_result['Price']) for p in test_prices]
        
        st.markdown(f"""
        | Steps | CRR Price | Error | Error × N |
        |-------|-----------|-------|-----------|
        | 50 | ${test_prices[0]:.6f} | {test_errors[0]:.6f} | {test_errors[0] * 50:.4f} |
        | 100 | ${test_prices[1]:.6f} | {test_errors[1]:.6f} | {test_errors[1] * 100:.4f} |
        | 200 | ${test_prices[2]:.6f} | {test_errors[2]:.6f} | {test_errors[2] * 200:.4f} |
        """)
        
        st.markdown("""
        *If Error × N is roughly constant, convergence is O(1/N)*
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4: HEDGING SIMULATION
# =============================================================================
with tab4:
    st.markdown('<p class="section-title">Monte Carlo Delta Hedging Simulation</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_sim_paths = st.selectbox("Simulation Paths", [100, 500, 1000, 5000], index=1)
    with col2:
        n_sim_steps = st.selectbox("Rebalancing Steps", [12, 52, 252], index=1,
                                   format_func=lambda x: f"{x} ({'Monthly' if x==12 else 'Weekly' if x==52 else 'Daily'})")
    with col3:
        use_bs = st.selectbox("Delta Method", ["Black-Scholes", "CRR"], index=0)
    with col4:
        run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    if run_sim or 'hedge_results' in st.session_state:
        if run_sim:
            with st.spinner("Running Monte Carlo simulation..."):
                results = simulate_hedging_strategy(
                    S, K, T, r, sigma, n_sim_steps, n_sim_paths,
                    use_bs_delta=(use_bs == "Black-Scholes")
                )
                st.session_state['hedge_results'] = results
        
        results = st.session_state['hedge_results']
        
        # Metrics
        st.markdown("<br>", unsafe_allow_html=True)
        
        mean_error = results['hedging_errors'].mean()
        std_error = results['hedging_errors'].std()
        var_95 = np.percentile(results['hedging_errors'], 5)
        var_99 = np.percentile(results['hedging_errors'], 1)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Hedging Error</div>
                <div class="metric-value {'positive' if mean_error >= 0 else 'negative'}">${mean_error:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Std Dev Error</div>
                <div class="metric-value">${std_error:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">VaR 95%</div>
                <div class="metric-value negative">${var_95:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">VaR 99%</div>
                <div class="metric-value negative">${var_99:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Initial Premium</div>
                <div class="metric-value blue">${results['initial_price']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        hedge_fig = create_hedging_pnl_figure(results, min(100, n_sim_paths))
        st.plotly_chart(hedge_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="accent-card">', unsafe_allow_html=True)
            st.markdown("#### Hedging Performance")
            st.markdown(f"""
            - **Mean Error**: The average hedging error of **${mean_error:.4f}** indicates 
              {'slight over-hedging' if mean_error > 0 else 'slight under-hedging'} on average.
            - **Std Dev**: The error standard deviation of **${std_error:.4f}** represents 
              the typical hedging uncertainty.
            - **Error/Premium Ratio**: {abs(std_error/results['initial_price'])*100:.2f}% of the initial premium.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="accent-card">', unsafe_allow_html=True)
            st.markdown("#### Sources of Hedging Error")
            st.markdown("""
            1. **Discrete Rebalancing**: Real hedging can't be continuous
            2. **Gamma Risk**: Delta changes between rebalancing dates
            3. **Model Risk**: True volatility may differ from assumed σ
            4. **Transaction Costs**: Not included in this simulation
            """)
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 5: THEORY
# =============================================================================
with tab5:
    st.markdown('<p class="section-title">Mathematical Foundation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Cox-Ross-Rubinstein Model")
        
        st.markdown("""
        The CRR model (1979) discretizes the continuous-time Black-Scholes framework 
        into a binomial tree. At each time step Δt = T/N:
        """)
        
        st.latex(r"S_{t+\Delta t} = \begin{cases} S_t \cdot u & \text{with probability } p \\ S_t \cdot d & \text{with probability } 1-p \end{cases}")
        
        st.markdown("Where the parameters are chosen to match the first two moments of GBM:")
        
        st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = e^{-\sigma\sqrt{\Delta t}} = \frac{1}{u}")
        
        st.latex(r"p = \frac{e^{r\Delta t} - d}{u - d}")
        
        st.markdown("""
        The option price is computed by **backward induction** from the terminal payoff:
        """)
        
        st.latex(r"C_t = e^{-r\Delta t}[p \cdot C_{t+1}^u + (1-p) \cdot C_{t+1}^d]")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### Black-Scholes Model")
        
        st.markdown("""
        The continuous-time limit of CRR gives the Black-Scholes formula. 
        For a European call option:
        """)
        
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
        
        st.markdown("Where:")
        
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}")
        
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
        
        st.markdown("And N(·) is the standard normal CDF.")
        
        st.markdown("#### The Greeks")
        
        st.latex(r"\Delta = \frac{\partial C}{\partial S} = N(d_1)")
        st.latex(r"\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S\sigma\sqrt{T}}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Delta Hedging Theory
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### Delta Hedging Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Discrete CRR Hedging
        
        In the binomial model, perfect replication is possible. At each node, hold:
        """)
        
        st.latex(r"\phi_t = \frac{C_{t+1}^u - C_{t+1}^d}{S_t(u-d)}")
        
        st.markdown("""
        This creates a **self-financing** portfolio that exactly replicates 
        the option payoff at maturity.
        """)
    
    with col2:
        st.markdown("""
        #### Continuous BS Hedging
        
        In the continuous limit, the hedge ratio becomes:
        """)
        
        st.latex(r"\Delta_t = \frac{\partial C}{\partial S}(t, S_t)")
        
        st.markdown("""
        **Key insight**: As Δt → 0, the CRR delta converges to the BS delta:
        """)
        
        st.latex(r"\lim_{N \to \infty} \phi_t^{CRR} = \Delta_t^{BS}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Limitations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Model Limitations & Extensions</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### CRR Limitations")
        st.markdown("""
        - Constant volatility assumption
        - Discrete time steps
        - Computational cost for many steps
        - No jumps in asset prices
        - European options only (basic form)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### Heston Model")
        st.markdown("""
        Adds stochastic volatility:
        """)
        st.latex(r"dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^v")
        st.markdown("""
        - Captures volatility smile
        - More realistic dynamics
        - Requires numerical methods
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="accent-card">', unsafe_allow_html=True)
        st.markdown("#### SABR Model")
        st.markdown("""
        Stochastic Alpha Beta Rho model:
        """)
        st.latex(r"dF_t = \sigma_t F_t^\beta dW_t^1")
        st.latex(r"d\sigma_t = \alpha \sigma_t dW_t^2")
        st.markdown("""
        - Popular in rates/FX markets
        - Analytical approximations exist
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; border-top: 1px solid #1e293b;">
    <p style="color: #64748b; font-size: 12px; margin: 0;">
        CRR Pricing Platform • ESILV - Projet d'Innovation Industrielle • 2025
    </p>
    <p style="color: #475569; font-size: 11px; margin: 8px 0 0 0;">
        Built with Streamlit • Mathematical models for educational purposes only
    </p>
</div>
""", unsafe_allow_html=True)
