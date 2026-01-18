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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Source+Code+Pro:wght@400;600&display=swap');
    :root { --primary: #1C97F3; --bg-panel: #1E2329; --text-muted: #90A4AE; --border: #2C333D; }
    .stApp { background-color: #0F1216; font-family: 'Roboto', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Roboto', sans-serif !important; font-weight: 300 !important; margin-bottom: 0px; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div { background-color: var(--bg-panel); border: 1px solid var(--border); border-radius: 0px; }
    div[data-testid="stMetricValue"] { color: var(--primary) !important; font-size: 24px !important; font-weight: 300 !important; }
    div[data-testid="stMetricLabel"] label { color: var(--text-muted); font-weight: 600; font-size: 11px; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div { background-color: #15191F !important; border: 1px solid var(--border) !important; color: white !important; border-radius: 0px !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { height: 40px; background-color: var(--bg-panel); border: 1px solid var(--border); color: var(--text-muted); border-radius: 0px; }
    .stTabs [aria-selected="true"] { background-color: var(--primary) !important; color: white !important; border: 1px solid var(--primary) !important; }
    .stPills button { background-color: #15191F !important; border: 1px solid var(--border) !important; color: #fff !important; border-radius: 0px !important; }
    .stPills button[aria-selected="true"] { background-color: var(--primary) !important; border-color: var(--primary) !important; }
    
    /* CUSTOM CLASSES FOR HEDGING UI */
    .hedge-box { border: 1px solid var(--border); padding: 15px; margin-bottom: 10px; background: #15191F; }
    .hedge-rec { font-family: 'Source Code Pro', monospace; color: #00C853; font-size: 18px; font-weight: 600; }
    .hedge-warn { font-family: 'Source Code Pro', monospace; color: #FF3D00; font-size: 18px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL FUNCTIONS
# -----------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type="Call"):
    if T <= 0 or sigma <= 0: return {k: 0.0 for k in ["Price", "Delta", "Gamma", "Vega", "Theta"]}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1) / 100 
    
    if option_type == "Call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = -si.norm.cdf(-d1)
        theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)

    return {"Price": price, "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta/365}

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

def get_vol_surface_data(ticker_obj, spot, option_type="Call"):
    try:
        exps = ticker_obj.options[:6]
        if not exps: return None
        strikes, tm, iv = [], [], []
        for exp in exps:
            chain = ticker_obj.option_chain(exp)
            data = chain.calls if option_type == "Call" else chain.puts
            T = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
            if T < 0.01: continue
            mask = (data['impliedVolatility'] > 0.001) & \
                   (data['strike'] > spot*0.5) & (data['strike'] < spot*1.5)
            f = data[mask]
            strikes.extend(f['strike'].tolist())
            tm.extend([T]*len(f))
            iv.extend(f['impliedVolatility'].tolist())
        return strikes, tm, iv
    except: return None

# -----------------------------------------------------------------------------
# 3. LAYOUT & LOGIC
# -----------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; align-items:center; margin-bottom:15px;">
    <h2 style="color:#1C97F3; font-weight:700; margin-right:10px;">FACTSET</h2>
    <span style="color:#90A4AE; letter-spacing:1px; font-size:14px;">OPTIONS ANALYTICS</span>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 3])

# --- LEFT COLUMN (INPUTS) ---
with col_left:
    # 1. Asset Selection
    input_container = st.container(border=True)
    with input_container:
        st.markdown("**ASSET SELECTION**")
        index_map = {
            "S&P 500 (US)": "SPY", "Nasdaq 100 (US)": "QQQ", "Dow Jones (US)": "DIA",
            "Russell 2000": "IWM", "VIX": "VXX", "Euro Stoxx 50": "FEZ",
            "CAC 40": "EWQ", "DAX": "EWG", "FTSE 100": "EWU", "Nikkei 225": "EWJ"
        }
        selected_name = st.selectbox("INDEX", list(index_map.keys()), label_visibility="collapsed")
        ticker = index_map[selected_name]
        
        st.markdown('<div style="margin-top: 10px; margin-bottom: 5px; font-size: 11px; font-weight: 600; color: #90A4AE;">TIME HORIZON</div>', unsafe_allow_html=True)
        horizon_map = {"3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
        selected_horizon = st.pills("Horizon", list(horizon_map.keys()), default="6M", label_visibility="collapsed")
        st.caption(f"Symbol: {ticker}")

        try:
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(period=horizon_map[selected_horizon])
            if df.empty: st.error("No Data"); st.stop()
            current_price = df['Close'].iloc[-1]
            hist_vol = np.log(df['Close']/df['Close'].shift(1)).std() * np.sqrt(252)
        except Exception as e: st.error(f"Error: {e}"); st.stop()

    # 2. Parameters
    param_container = st.container(border=True)
    with param_container:
        st.markdown("**OPTION PARAMETERS (HEDGE INSTRUMENT)**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            strike = st.number_input("STRIKE", value=float(round(current_price, 0)), step=1.0)
            ttm = st.number_input("YEARS TO EXP", value=0.5, step=0.1)
            sigma = st.number_input("VOLATILITY", value=hist_vol, step=0.01, format="%.3f")
        with col_p2:
            r = st.number_input("RISK FREE", value=0.045, step=0.001, format="%.3f")
            opt_type = st.selectbox("TYPE", ["Put", "Call"], index=0) # Default to Put for hedging
            steps = st.number_input("STEPS (N)", value=50, min_value=10)

    bs_res = black_scholes(current_price, strike, ttm, r, sigma, opt_type)
    
    # Simple Metrics
    metric_container = st.container(border=True)
    with metric_container:
        st.markdown(f"**VALUATION ({opt_type.upper()})**")
        st.metric("PRICE", f"${bs_res['Price']:.4f}")
        st.markdown("---")
        g1, g2 = st.columns(2)
        g1.metric("DELTA", f"{bs_res['Delta']:.3f}")
        g1.metric("GAMMA", f"{bs_res['Gamma']:.3f}")

# --- RIGHT COLUMN (ANALYTICS) ---
with col_right:
    # 1. Price Chart (Candlestick)
    hist_container = st.container(border=True)
    with hist_container:
        st.markdown(f"**{ticker} PRICE HISTORY ({selected_horizon})**")
        
        # Create Candlestick Chart
        fig_price = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#00C853', # Green for Up
            decreasing_line_color='#FF3D00', # Red for Down
            name='OHLC'
        )])
        
        # Update layout for a clean "TradingView" look
        fig_price.update_layout(
            height=350,  # Slightly taller to see details
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#1E2329",
            plot_bgcolor="#1E2329",
            xaxis=dict(
                showgrid=True, 
                gridcolor="#2C333D",
                rangeslider=dict(visible=False) # Hide the bottom slider to save space
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor="#2C333D"
            ),
            showlegend=False
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # NEW TABS (Added Hedging Lab)
    tab_hedge, tab_conv, tab_vol, tab_chain = st.tabs(["HEDGING LAB", "CONVERGENCE", "VOLATILITY SURFACE", "OPTION DESK"])
    
    # --- 1. HEDGING LAB (UPDATED LOGIC) ---
    with tab_hedge:
        hedge_container = st.container(border=True)
        with hedge_container:
            c_h1, c_h2 = st.columns([1, 2])
            
            with c_h1:
                st.markdown("**PORTFOLIO CONFIG**")
                # User inputs their STOCK position
                shares = st.number_input("SHARES HELD", value=100, step=10, help="How many shares of the index ETF do you own?")
                
                st.markdown("<br>**STRESS TEST MODE**", unsafe_allow_html=True)
                stress_on = st.toggle("SIMULATE MARKET CRASH", value=False)
                
                # Apply Stress
                if stress_on:
                    sim_price = current_price * 0.85 # -15% Spot
                    sim_vol = sigma + 0.20           # +20% Vol
                    # Recalculate Option stats under stress
                    sim_bs = black_scholes(sim_price, strike, ttm, r, sim_vol, opt_type)
                    st.caption(f"📉 Spot: ${sim_price:.2f} | 📈 Vol: {sim_vol:.2%}")
                else:
                    sim_bs = bs_res
                    sim_price = current_price

            with c_h2:
                # 1. Calculate Portfolio Delta (Stock Delta is 1.0 per share)
                port_delta = shares * 1.0
                
                # 2. Calculate Option Delta (Per Contract = Delta * 100)
                # Note: Put Delta is negative. Call Delta is positive.
                opt_delta_per_contract = sim_bs['Delta'] * 100
                
                # 3. Calculate Hedge Required
                # Goal: Net Delta = 0
                # Formula: Shares + (Contracts * ContractDelta) = 0
                # Contracts = -Shares / ContractDelta
                
                if abs(opt_delta_per_contract) > 0.001:
                    hedge_contracts_req = -port_delta / opt_delta_per_contract
                else:
                    hedge_contracts_req = 0

                # Formatting for UI
                rec_action = "BUY" if hedge_contracts_req > 0 else "SELL"
                abs_contracts = abs(hedge_contracts_req)
                color_class = "hedge-warn" if stress_on else "hedge-rec"
                
                st.markdown('<div class="hedge-box">', unsafe_allow_html=True)
                st.markdown(f"**PORTFOLIO EXPOSURE (SHARES)**: <span style='color:white'>${(shares * sim_price):,.0f}</span>", unsafe_allow_html=True)
                st.markdown(f"**OPTION DELTA (x100)**: <span style='color:white'>{opt_delta_per_contract:.2f}</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("##### 🛡️ HEDGE RECOMMENDATION")
                st.markdown(f'<div class="{color_class}">{rec_action} {abs_contracts:.1f} {opt_type.upper()} CONTRACTS</div>', unsafe_allow_html=True)
                st.caption(f"Traded against Strike ${strike} {opt_type}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if stress_on and opt_type == "Put":
                    st.info("""
                    **💡 INSIGHT:** Notice that in a crash, Put Delta becomes more negative (moves toward -100). 
                    Your options become "stronger" hedges. If you hedge fully now, you might be **over-hedged** (net short) if the market crashes.
                    """)

    # --- 2. CONVERGENCE ---
    with tab_conv:
        conv_container = st.container(border=True)
        with conv_container:
            step_rng = range(10, 100, 5)
            crr_vals = [crr_binomial_tree(current_price, strike, ttm, r, sigma, n, opt_type) for n in step_rng]
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=list(step_rng), y=crr_vals, mode='lines+markers', name='CRR', line=dict(color='#00C853')))
            fig_conv.add_hline(y=bs_res['Price'], line_dash="dash", line_color="white", annotation_text="BS Limit")
            fig_conv.update_layout(height=350, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor="#1E2329", plot_bgcolor="#1E2329",
                                   title=f"CONVERGENCE ({opt_type.upper()})", xaxis_title="STEPS (N)", yaxis_title="PRICE",
                                   xaxis=dict(showgrid=True, gridcolor="#2C333D"), yaxis=dict(showgrid=True, gridcolor="#2C333D"))
            st.plotly_chart(fig_conv, use_container_width=True)

    # --- 3. VOL SURFACE ---
    with tab_vol:
        vol_container = st.container(border=True)
        with vol_container:
            if st.button(f"LOAD {opt_type.upper()} SURFACE"):
                with st.spinner("Calculating..."):
                    res = get_vol_surface_data(t_obj, current_price, opt_type)
                    if res:
                        strikes, tm, iv = res
                        grid_x, grid_y = np.mgrid[min(strikes):max(strikes):40j, min(tm):max(tm):20j]
                        grid_z = griddata((strikes, tm), iv, (grid_x, grid_y), method='linear')
                        fig_vol = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
                        fig_vol.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="#1E2329",
                                              scene=dict(xaxis_title='Strike', yaxis_title='Time', zaxis_title='IV',
                                                         xaxis=dict(backgroundcolor="#1E2329", gridcolor="#2C333D"),
                                                         yaxis=dict(backgroundcolor="#1E2329", gridcolor="#2C333D"),
                                                         zaxis=dict(backgroundcolor="#1E2329", gridcolor="#2C333D")))
                        st.plotly_chart(fig_vol, use_container_width=True)
                    else: st.warning("Not enough data.")
            else: st.info(f"Click to load {opt_type} Volatility Surface")

    # --- 4. OPTION DESK ---
    with tab_chain:
        chain_container = st.container(border=True)
        with chain_container:
            try:
                exps = t_obj.options
                if exps:
                    selected_exp = st.selectbox("EXPIRATION", exps)
                    chain = t_obj.option_chain(selected_exp)
                    if opt_type == "Call": st.dataframe(chain.calls[['strike','lastPrice','impliedVolatility','volume']], hide_index=True, use_container_width=True)
                    else: st.dataframe(chain.puts[['strike','lastPrice','impliedVolatility','volume']], hide_index=True, use_container_width=True)
            except: st.info("No option data available")