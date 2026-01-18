import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    /* CUSTOM CLASSES */
    .hedge-box { border: 1px solid var(--border); padding: 15px; margin-bottom: 10px; background: #15191F; }
    .hedge-rec { font-family: 'Source Code Pro', monospace; color: #00C853; font-size: 18px; font-weight: 600; }
    .hedge-warn { font-family: 'Source Code Pro', monospace; color: #FF3D00; font-size: 18px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL FUNCTIONS
# -----------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type="Call"):
    # Avoid division by zero for very small T
    if T <= 1e-5: 
        # Intrinsic value at expiration
        if option_type == "Call": return {"Price": max(0, S-K), "Delta": 1.0 if S>K else 0.0}
        else: return {"Price": max(0, K-S), "Delta": -1.0 if K>S else 0.0}
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = -si.norm.cdf(-d1)
        
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1) / 100 
    theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) # Simplified Theta
    
    return {"Price": price, "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta/365}

def crr_binomial_tree(S, K, T, r, sigma, N, option_type="Call"):
    if T <= 0: return max(0, S-K) if option_type=="Call" else max(0, K-S)
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
            mask = (data['impliedVolatility'] > 0.001) & (data['strike'] > spot*0.5) & (data['strike'] < spot*1.5)
            f = data[mask]
            strikes.extend(f['strike'].tolist())
            tm.extend([T]*len(f))
            iv.extend(f['impliedVolatility'].tolist())
        return strikes, tm, iv
    except: return None

# -----------------------------------------------------------------------------
# 3. BACKTEST ENGINE
# -----------------------------------------------------------------------------
def run_hedging_backtest(df, strike, risk_free, vol, opt_type):
    """
    Simulates Delta Hedging a Short Option position over the historical dataframe.
    """
    history = []
    
    # We assume we SOLD the option at t=0
    # Initial Setup
    start_date = df.index[0]
    end_date = df.index[-1]
    total_days = (end_date - start_date).days
    
    # Portfolio State
    cash = 0
    shares_held = 0
    
    for date, row in df.iterrows():
        S = row['Close']
        days_remaining = (end_date - date).days
        T = days_remaining / 365.0
        
        # Calculate BS Theoretical Price & Delta
        if days_remaining > 0:
            res = black_scholes(S, strike, T, risk_free, vol, opt_type)
            theo_price = res['Price']
            delta = res['Delta']
        else:
            # Expiration
            theo_price = max(0, S - strike) if opt_type == "Call" else max(0, strike - S)
            delta = 0 # No hedge needed after exp
            
        # DELTA HEDGING LOGIC
        # We are SHORT the option, so we have negative Delta exposure.
        # To hedge, we need +Delta (Buy Shares).
        # Target Shares = -1 * (-1 * Delta * 100) = Delta * 100
        # Wait, if we Short Call (Neg Delta for us), we need Pos Delta (Buy Shares).
        # Short Call Delta is negative? No, Long Call Delta is positive.
        # Position Delta = -1 (Short Contract) * CallDelta (~0.5) = -0.5
        # Hedge needed = +0.5 (Buy 50 shares)
        
        target_shares = delta * 100 # Standard 100x multiplier
        
        trade_shares = target_shares - shares_held
        cost = trade_shares * S
        
        # Update Portfolio
        shares_held = target_shares
        cash -= cost # Spend cash to buy shares
        
        # Calculate Portfolio Value (PnL)
        # Portfolio = Cash + Stock Value - Option Liability
        stock_val = shares_held * S
        option_liability = theo_price * 100
        total_pnl = cash + stock_val - option_liability
        
        history.append({
            "Date": date,
            "Spot": S,
            "Delta": delta,
            "Hedge Shares": shares_held,
            "PnL": total_pnl,
            "Option Price": theo_price
        })
        
    return pd.DataFrame(history).set_index("Date")

# -----------------------------------------------------------------------------
# 4. LAYOUT
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

    bs_res = black_scholes(current_price, strike, ttm, r, sigma, opt_type)
    crr_res = crr_binomial_tree(current_price, strike, ttm, r, sigma, steps, opt_type)
    
    metric_container = st.container(border=True)
    with metric_container:
        st.markdown(f"**VALUATION ({opt_type.upper()})**")
        st.metric("BLACK-SCHOLES", f"${bs_res['Price']:.4f}")
        st.metric("BINOMIAL (CRR)", f"${crr_res:.4f}", delta=f"{(crr_res-bs_res['Price']):.4f}")
        st.markdown("---")
        g1, g2 = st.columns(2)
        g1.metric("DELTA", f"{bs_res['Delta']:.3f}")
        g1.metric("GAMMA", f"{bs_res['Gamma']:.3f}")

# --- RIGHT COLUMN (ANALYTICS) ---
with col_right:
    # Price Chart
    hist_container = st.container(border=True)
    with hist_container:
        st.markdown(f"**{ticker} PRICE HISTORY ({selected_horizon})**")
        fig_price = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#00C853', decreasing_line_color='#FF3D00', name='OHLC'
        )])
        fig_price.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="#1E2329", plot_bgcolor="#1E2329",
                                xaxis=dict(showgrid=True, gridcolor="#2C333D", rangeslider=dict(visible=False)),
                                yaxis=dict(showgrid=True, gridcolor="#2C333D"), showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

    # --- TABS ---
    tab_hedge, tab_robust, tab_conv, tab_vol, tab_chain = st.tabs(["HEDGING LAB", "ROBUSTNESS TEST", "CONVERGENCE", "VOL SURFACE", "OPTION DESK"])
    
    # 1. HEDGING LAB (Interactive)
    with tab_hedge:
        hedge_container = st.container(border=True)
        with hedge_container:
            c_h1, c_h2 = st.columns([1, 2])
            with c_h1:
                st.markdown("**PORTFOLIO CONFIG**")
                shares = st.number_input("SHARES HELD", value=100, step=10)
                st.markdown("<br>**STRESS TEST MODE**", unsafe_allow_html=True)
                stress_on = st.toggle("SIMULATE MARKET CRASH", value=False)
                if stress_on:
                    sim_price = current_price * 0.85; sim_vol = sigma + 0.20
                    sim_bs = black_scholes(sim_price, strike, ttm, r, sim_vol, opt_type)
                    st.caption(f"📉 Spot: ${sim_price:.2f} | 📈 Vol: {sim_vol:.2%}")
                else: sim_bs = bs_res; sim_price = current_price

            with c_h2:
                port_delta = shares * 1.0
                opt_delta_per_contract = sim_bs['Delta'] * 100
                if abs(opt_delta_per_contract) > 0.001: hedge_contracts_req = -port_delta / opt_delta_per_contract
                else: hedge_contracts_req = 0
                rec_action = "BUY" if hedge_contracts_req > 0 else "SELL"
                color_class = "hedge-warn" if stress_on else "hedge-rec"
                
                st.markdown('<div class="hedge-box">', unsafe_allow_html=True)
                st.markdown(f"**PORTFOLIO EXPOSURE**: <span style='color:white'>${(shares * sim_price):,.0f}</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("##### 🛡️ HEDGE RECOMMENDATION")
                st.markdown(f'<div class="{color_class}">{rec_action} {abs(hedge_contracts_req):.1f} CONTRACTS</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # 2. ROBUSTNESS TEST (New!)
    with tab_robust:
        st.markdown("""
        <div style="padding:10px; background:#15191F; border:1px solid #2C333D; margin-bottom:10px;">
            <span style="color:#1C97F3; font-weight:700;">BACKTESTING CHALLENGE:</span> 
            We simulate selling this option at the start of the chart history and Delta Hedging it daily.
            <br><span style="color:#90A4AE; font-size:12px;">Theory says P&L should be zero. Reality (jumps/gaps) causes leakage.</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("RUN HEDGING SIMULATION"):
            with st.spinner("Simulating Daily Rebalancing..."):
                # Run Backtest
                bt_df = run_hedging_backtest(df, strike, r, sigma, opt_type)
                
                # Plot Results
                fig_bt = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Area 1: Hedging P&L (The Error)
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['PnL'], 
                    mode='lines', name='Cumulative P&L (Hedging Error)',
                    line=dict(color='#FF3D00', width=2),
                    fill='tozeroy'
                ), secondary_y=False)
                
                # Area 2: Stock Price (Context)
                fig_bt.add_trace(go.Scatter(
                    x=bt_df.index, y=bt_df['Spot'],
                    mode='lines', name='Stock Price',
                    line=dict(color='#90A4AE', width=1, dash='dot')
                ), secondary_y=True)
                
                fig_bt.update_layout(
                    height=350, margin=dict(t=30, b=10, l=10, r=10),
                    paper_bgcolor="#1E2329", plot_bgcolor="#1E2329",
                    title="ROBUSTNESS TEST: HEDGING ERROR ACCUMULATION",
                    xaxis=dict(showgrid=True, gridcolor="#2C333D"),
                    yaxis=dict(showgrid=True, gridcolor="#2C333D", title="P&L ($)"),
                    yaxis2=dict(showgrid=False, title="Spot Price"),
                    showlegend=True, legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Final Stats
                final_pnl = bt_df['PnL'].iloc[-1]
                st.metric("TOTAL HEDGING SLIPPAGE", f"${final_pnl:,.2f}", 
                          delta="Gaussian Assumption Failed" if abs(final_pnl) > 50 else "Model Held Up",
                          delta_color="inverse")

    # 3. CONVERGENCE
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

    # 4. VOL SURFACE
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

    # 5. OPTION DESK
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