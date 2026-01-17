import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. CONFIG & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Option Pricing Workstation",
    page_icon="🔵​",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css('style.css')
except FileNotFoundError:
    st.error("style.css not found! Please ensure it exists.")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="Call"):
    if T <= 0 or sigma <= 0:
        return {k: 0.0 for k in ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = -si.norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1)
    theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)
    return {"Price": price, "Delta": delta, "Gamma": gamma, "Vega": vega/100, "Theta": theta/365, "Rho": rho/100}

def crr_binomial_tree(S, K, T, r, sigma, N, option_type="Call"):
    if T <= 0 or sigma <= 0: return 0.0
    dt = T / N; u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; p = (np.exp(r * dt) - d) / (u - d)
    asset_prices = np.zeros(N + 1); asset_prices[0] = S * d**N
    for i in range(1, N + 1): asset_prices[i] = asset_prices[i - 1] * (u / d)
    option_values = np.zeros(N + 1)
    if option_type == "Call": option_values = np.maximum(0, asset_prices - K)
    else: option_values = np.maximum(0, K - asset_prices)
    discount_factor = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = discount_factor * (p * option_values[j + 1] + (1 - p) * option_values[j])
    return option_values[0]

def get_vol_surface_data(ticker_obj, spot_price):
    expirations = ticker_obj.options
    if not expirations: return None, None
    selected_exps = expirations[:6]
    c_strikes, c_tm, c_iv = [], [], []; p_strikes, p_tm, p_iv = [], [], []
    progress = st.progress(0, text="RETRIEVING DATA...")
    for i, exp in enumerate(selected_exps):
        progress.progress((i + 1) / len(selected_exps))
        try:
            chain = ticker_obj.option_chain(exp)
            def process(df):
                T = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
                if T < 0.001: return [], [], []
                mask = (df['impliedVolatility'] > 0.01) & (df['impliedVolatility'] < 2.0) & (df['strike'] > spot_price * 0.6) & (df['strike'] < spot_price * 1.4)
                f = df[mask]
                return f['strike'].tolist(), [T]*len(f), f['impliedVolatility'].tolist()
            cs, ct, civ = process(chain.calls); ps, pt, piv = process(chain.puts)
            c_strikes.extend(cs); c_tm.extend(ct); c_iv.extend(civ)
            p_strikes.extend(ps); p_tm.extend(pt); p_iv.extend(piv)
        except: continue
    progress.empty()
    return (c_strikes, c_tm, c_iv), (p_strikes, p_tm, p_iv)

def plot_3d_surface(strikes, tm, iv, title, color_scale='Blues'):
    if not strikes: return go.Figure()
    x_grid = np.linspace(min(strikes), max(strikes), 40)
    y_grid = np.linspace(min(tm), max(tm), 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = griddata((strikes, tm), iv, (X, Y), method='cubic')
    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y, colorscale=color_scale, opacity=0.9,
        contours_z=dict(show=True, usecolormap=True, project_z=True)
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(family="Roboto", size=14, color="#1C97F3")),
        scene=dict(
            xaxis=dict(title='STRIKE', backgroundcolor="#1E2329", gridcolor="#2C333D", title_font=dict(color="#90A4AE")),
            yaxis=dict(title='TIME (YRS)', backgroundcolor="#1E2329", gridcolor="#2C333D", title_font=dict(color="#90A4AE")),
            zaxis=dict(title='IV', backgroundcolor="#1E2329", gridcolor="#2C333D", title_font=dict(color="#90A4AE")),
        ),
        paper_bgcolor="#1E2329", margin=dict(l=10, r=10, b=10, t=40), height=450
    )
    return fig

# -----------------------------------------------------------------------------
# 3. SIDEBAR & NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; margin-bottom:20px;">
    <h2 style="margin:0; font-family:'Roboto'; font-weight:700; color:#1C97F3; font-style:normal;">PRICING</h2>
    <span style="color:#90A4AE; margin-left:8px; font-size:12px;">WORKSTATION</span>
</div>
""", unsafe_allow_html=True)

index_map = {"S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Dow Jones (DIA)": "DIA", "Russell 2000 (IWM)": "IWM"}
selected_idx = st.sidebar.selectbox("INSTRUMENT", list(index_map.keys()))
ticker = index_map[selected_idx]

st.sidebar.markdown("---")
st.sidebar.markdown("##### PARAMETERS")
spot_manual = st.sidebar.number_input("SPOT PRICE", value=400.0)
use_manual_spot = st.sidebar.checkbox("MANUAL SPOT", value=False)
strike = st.sidebar.number_input("STRIKE", value=400.0)
ttm = st.sidebar.number_input("MATURITY (YRS)", value=0.5)
r = st.sidebar.number_input("RISK-FREE RATE", value=0.045)
opt_type = st.sidebar.selectbox("TYPE", ["Call", "Put"])

st.sidebar.markdown("##### SIMULATION")
vol_manual = st.sidebar.slider("VOLATILITY", 0.01, 1.0, 0.2)
use_manual_vol = st.sidebar.checkbox("MANUAL VOL", value=False)
steps = st.sidebar.slider("TREE STEPS", 10, 500, 50)

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.markdown(f"## {selected_idx} <span style='font-size:16px; color:#1C97F3'>// ANALYTICS</span>", unsafe_allow_html=True)

try:
    with st.spinner("LOADING MARKET DATA..."):
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="1y")
        if df.empty: st.error("NO DATA"); st.stop()
        px = df['Close'] if 'Close' in df.columns else df['Adj Close']
        ret = np.log(px / px.shift(1)).dropna()
        hist_vol = ret.std() * np.sqrt(252)
        S = px.iloc[-1] if not use_manual_spot else spot_manual
        sigma = vol_manual if use_manual_vol else hist_vol
        daily_chg = px.iloc[-1] - px.iloc[-2]
except Exception as e: st.error(e); st.stop()

# METRICS
def render_metric(label, value, delta=None):
    delta_html = ""
    if delta is not None:
        color_class = "delta-pos" if delta >= 0 else "delta-neg"
        delta_html = f'<div class="metric-delta {color_class}">{delta:+.2f}</div>'
    st.markdown(f"""
    <div class="metric-card-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: render_metric("LAST PRICE", f"{S:.2f}", daily_chg)
with c2: render_metric("STRIKE", f"{strike:.2f}")
with c3: render_metric("IMPLIED VOL", f"{sigma:.2%}")
with c4: render_metric("RATE", f"{r:.2%}")

# PRICING
st.markdown("<br>", unsafe_allow_html=True)
bs = black_scholes(S, strike, ttm, r, sigma, opt_type)
crr = crr_binomial_tree(S, strike, ttm, r, sigma, steps, opt_type)

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown('<div class="factset-panel">', unsafe_allow_html=True)
    st.markdown("#### PRICING MODEL")
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; border-bottom:1px solid #2C333D; padding-bottom:8px; margin-bottom:8px;">
        <span style="color:#90A4AE;">Black-Scholes</span>
        <span style="font-weight:700; color:#FFFFFF;">${bs['Price']:.4f}</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:16px;">
        <span style="color:#90A4AE;">Binomial (CRR)</span>
        <span style="font-weight:700; color:#1C97F3;">${crr:.4f}</span>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
        <div><span style="font-size:10px; color:#90A4AE;">DELTA</span><br>{bs['Delta']:.4f}</div>
        <div><span style="font-size:10px; color:#90A4AE;">GAMMA</span><br>{bs['Gamma']:.4f}</div>
        <div><span style="font-size:10px; color:#90A4AE;">THETA</span><br>{bs['Theta']:.4f}</div>
        <div><span style="font-size:10px; color:#90A4AE;">VEGA</span><br>{bs['Vega']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="factset-panel">', unsafe_allow_html=True)
    st.markdown("#### CONVERGENCE")
    step_rng = range(10, 100, 5)
    crr_v = [crr_binomial_tree(S, strike, ttm, r, sigma, n, opt_type) for n in step_rng]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(step_rng), y=crr_v, mode='lines+markers', name='CRR', line=dict(color='#1C97F3', width=2)))
    fig.add_trace(go.Scatter(x=[step_rng[0], step_rng[-1]], y=[bs['Price'], bs['Price']], mode='lines', name='Limit', line=dict(color='#FFFFFF', dash='dash')))
    fig.update_layout(paper_bgcolor="#1E2329", plot_bgcolor="#1E2329", margin=dict(t=30, b=20, l=40, r=20), height=250,
                      xaxis=dict(showgrid=True, gridcolor="#2C333D"), yaxis=dict(showgrid=True, gridcolor="#2C333D"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# VOL SURFACES
if st.checkbox("LOAD VOLATILITY SURFACES"):
    cd, pd_data = get_vol_surface_data(t_obj, S)
    if cd and pd_data:
        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown('<div class="factset-panel">', unsafe_allow_html=True)
            st.plotly_chart(plot_3d_surface(cd[0], cd[1], cd[2], "CALL SKEW", "Plasma"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_r:
            st.markdown('<div class="factset-panel">', unsafe_allow_html=True)
            st.plotly_chart(plot_3d_surface(pd_data[0], pd_data[1], pd_data[2], "PUT SKEW", "Viridis"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# OPTION TABLE (ADDED BACK)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="factset-panel">', unsafe_allow_html=True)
st.markdown("#### OPTION DATA")

if t_obj.options:
    exps = t_obj.options
    # Using columns to organize the selector
    col_sel, col_empty = st.columns([1, 3])
    with col_sel:
        selected_expiry = st.selectbox("EXPIRATION", exps)
    
    try:
        chain = t_obj.option_chain(selected_expiry)
        tab1, tab2 = st.tabs(["CALLS", "PUTS"])
        
        with tab1:
            st.dataframe(chain.calls, use_container_width=True, height=400)
        with tab2:
            st.dataframe(chain.puts, use_container_width=True, height=400)
    except Exception as e:
        st.error(f"Error fetching chain: {e}")
else:
    st.info("No options data available for this ticker.")
st.markdown('</div>', unsafe_allow_html=True)