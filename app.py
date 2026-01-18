import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Options Analytics Workstation",
    page_icon="🔵",
    layout="wide",
)

# Custom CSS to force the FactSet Blue/Slate theme onto standard Streamlit components
st.markdown("""
<style>
    /* IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Source+Code+Pro:wght@400;600&display=swap');

    /* VARIABLES */
    :root {
        --primary: #1C97F3;
        --bg-panel: #1E2329;
        --text-main: #FFFFFF;
        --text-muted: #90A4AE;
        --border: #2C333D;
    }

    /* GLOBAL */
    .stApp {
        background-color: #0F1216;
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Roboto', sans-serif !important;
        font-weight: 300 !important;
        margin-bottom: 0px;
    }
    
    /* OVERRIDE NATIVE CONTAINERS (to match FactSet look) */
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 0px; /* Sharp corners */
    }

    /* METRICS */
    div[data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-size: 24px !important;
        font-weight: 300 !important;
    }
    div[data-testid="stMetricLabel"] label {
        color: var(--text-muted);
        font-weight: 600;
        font-size: 11px;
    }
    
    /* INPUTS */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #15191F !important;
        border: 1px solid var(--border) !important;
        color: white !important;
        border-radius: 0px !important;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        color: var(--text-muted);
        border-radius: 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
        border: 1px solid var(--primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL FUNCTIONS
# -----------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type="Call"):
    if T <= 0 or sigma <= 0: return {k: 0.0 for k in ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = -si.norm.cdf(-d1)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1)
    theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)
    return {"Price": price, "Delta": delta, "Gamma": gamma, "Vega": vega/100, "Theta": theta/365}

def crr_binomial_tree(S, K, T, r, sigma, N, option_type="Call"):
    if T <= 0 or sigma <= 0: return 0.0
    dt = T / N; u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; p = (np.exp(r * dt) - d) / (u - d)
    asset_prices = np.zeros(N + 1); asset_prices[0] = S * d**N
    for i in range(1, N + 1): asset_prices[i] = asset_prices[i - 1] * (u / d)
    option_values = np.zeros(N + 1)
    if option_type == "Call": option_values = np.maximum(0, asset_prices - K)
    else: option_values = np.maximum(0, K - asset_prices)
    discount = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = discount * (p * option_values[j + 1] + (1 - p) * option_values[j])
    return option_values[0]

def get_vol_surface_data(ticker_obj, spot):
    try:
        exps = ticker_obj.options[:6]
        if not exps: return None
        strikes, tm, iv = [], [], []
        for exp in exps:
            chain = ticker_obj.option_chain(exp).calls
            T = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
            if T < 0.01: continue
            mask = (chain['impliedVolatility'] > 0.01) & (chain['impliedVolatility'] < 2.0) & \
                   (chain['strike'] > spot*0.7) & (chain['strike'] < spot*1.3)
            f = chain[mask]
            strikes.extend(f['strike'].tolist())
            tm.extend([T]*len(f))
            iv.extend(f['impliedVolatility'].tolist())
        return strikes, tm, iv
    except: return None

# -----------------------------------------------------------------------------
# 3. LAYOUT & LOGIC
# -----------------------------------------------------------------------------

# Header
st.markdown("""
<div style="display:flex; align-items:center; margin-bottom:15px;">
    <h2 style="color:#1C97F3; font-weight:700; margin-right:10px;">DEEPBLUE</h2>
    <span style="color:#90A4AE; letter-spacing:1px; font-size:14px;">OPTIONS ANALYTICS</span>
</div>
""", unsafe_allow_html=True)

# Main Columns [1, 3] Layout
col_left, col_right = st.columns([1, 3])

# --- LEFT COLUMN: INPUTS & METRICS ---
with col_left:
    # 1. Ticker Selection Panel
    input_container = st.container(border=True)
    with input_container:
        st.markdown("**ASSET SELECTION**")
        ticker = st.text_input("TICKER", value="SPY", label_visibility="collapsed").upper()
        
        # Load Data
        try:
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(period="6mo")
            if df.empty: st.error("Invalid Ticker"); st.stop()
            current_price = df['Close'].iloc[-1]
            hist_vol = np.log(df['Close']/df['Close'].shift(1)).std() * np.sqrt(252)
        except:
            st.error("Data Error"); st.stop()
    
    # 2. Parameters Panel
    param_container = st.container(border=True)
    with param_container:
        st.markdown("**OPTION PARAMETERS**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            strike = st.number_input("STRIKE", value=float(round(current_price, 0)), step=1.0)
            ttm = st.number_input("YEARS TO EXP", value=0.5, step=0.1)
            sigma = st.number_input("VOLATILITY", value=hist_vol, step=0.01, format="%.3f")
        with col_p2:
            r = st.number_input("RISK FREE", value=0.045, step=0.001, format="%.3f")
            opt_type = st.selectbox("TYPE", ["Call", "Put"])
            steps = st.number_input("STEPS (N)", value=50, min_value=10)

    # 3. Output Metrics Panel
    bs_res = black_scholes(current_price, strike, ttm, r, sigma, opt_type)
    crr_res = crr_binomial_tree(current_price, strike, ttm, r, sigma, steps, opt_type)

    metric_container = st.container(border=True)
    with metric_container:
        st.markdown("**VALUATION**")
        st.metric("BLACK-SCHOLES", f"${bs_res['Price']:.4f}")
        st.metric("BINOMIAL (CRR)", f"${crr_res:.4f}", delta=f"{(crr_res-bs_res['Price']):.4f}")
        st.markdown("---")
        
        # Greeks Grid
        g1, g2 = st.columns(2)
        g1.metric("DELTA", f"{bs_res['Delta']:.3f}")
        g1.metric("GAMMA", f"{bs_res['Gamma']:.3f}")
        g2.metric("THETA", f"{bs_res['Theta']:.3f}")
        g2.metric("VEGA", f"{bs_res['Vega']:.3f}")

# --- RIGHT COLUMN: VISUALIZATIONS ---
with col_right:
    
    # 1. Price History Chart
    hist_container = st.container(border=True)
    with hist_container:
        st.markdown(f"**{ticker} PRICE HISTORY (6M)**")
        fig_price = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                         line=dict(color='#1C97F3', width=2), fill='tozeroy', 
                                         fillcolor='rgba(28, 151, 243, 0.1)'))
        fig_price.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), 
                                paper_bgcolor="#1E2329", plot_bgcolor="#1E2329",
                                xaxis=dict(showgrid=True, gridcolor="#2C333D"),
                                yaxis=dict(showgrid=True, gridcolor="#2C333D"))
        st.plotly_chart(fig_price, use_container_width=True)

    # 2. Tabs for Advanced Analytics
    tab_conv, tab_vol, tab_chain = st.tabs(["CONVERGENCE ANALYSIS", "VOLATILITY SURFACE", "OPTION DESK"])
    
    # Tab A: Convergence
    with tab_conv:
        conv_container = st.container(border=True)
        with conv_container:
            step_rng = range(10, 150, 5)
            crr_vals = [crr_binomial_tree(current_price, strike, ttm, r, sigma, n, opt_type) for n in step_rng]
            
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=list(step_rng), y=crr_vals, mode='lines+markers', name='CRR', line=dict(color='#00C853')))
            fig_conv.add_hline(y=bs_res['Price'], line_dash="dash", line_color="white", annotation_text="BS Limit")
            
            fig_conv.update_layout(height=350, margin=dict(t=30, b=10, l=10, r=10),
                                   paper_bgcolor="#1E2329", plot_bgcolor="#1E2329",
                                   title="BINOMIAL CONVERGENCE TO BLACK-SCHOLES",
                                   xaxis_title="STEPS (N)", yaxis_title="PRICE",
                                   xaxis=dict(showgrid=True, gridcolor="#2C333D"),
                                   yaxis=dict(showgrid=True, gridcolor="#2C333D"))
            st.plotly_chart(fig_conv, use_container_width=True)

    # Tab B: Vol Surface
    with tab_vol:
        vol_container = st.container(border=True)
        with vol_container:
            if st.button("LOAD SURFACE DATA (COMPUTE HEAVY)"):
                with st.spinner("Calculating..."):
                    res = get_vol_surface_data(t_obj, current_price)
                    if res:
                        strikes, tm, iv = res
                        # Interpolate
                        grid_x, grid_y = np.mgrid[min(strikes):max(strikes):40j, min(tm):max(tm):20j]
                        grid_z = griddata((strikes, tm), iv, (grid_x, grid_y), method='linear')
                        
                        fig_vol = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
                        fig_vol.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10),
                                              paper_bgcolor="#1E2329",
                                              scene=dict(xaxis_title='Strike', yaxis_title='Time', zaxis_title='IV',
                                                         xaxis=dict(backgroundcolor="#1E2329", gridcolor="#2C333D"),
                                                         yaxis=dict(backgroundcolor="#1E2329", gridcolor="#2C333D"),
                                                         zaxis=dict(backgroundcolor="#1E2329", gridcolor="#2C333D")))
                        st.plotly_chart(fig_vol, use_container_width=True)
                    else:
                        st.warning("Not enough data for surface.")
            else:
                st.info("Click to load Volatility Surface")

    # Tab C: Option Chain
    with tab_chain:
        chain_container = st.container(border=True)
        with chain_container:
            try:
                exps = t_obj.options
                if exps:
                    selected_exp = st.selectbox("EXPIRATION", exps)
                    chain = t_obj.option_chain(selected_exp)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("CALLS")
                        st.dataframe(chain.calls[['strike','lastPrice','impliedVolatility','volume']], hide_index=True, height=300)
                    with c2:
                        st.markdown("PUTS")
                        st.dataframe(chain.puts[['strike','lastPrice','impliedVolatility','volume']], hide_index=True, height=300)
            except:
                st.info("No option data available")