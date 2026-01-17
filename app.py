import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Index Option Pricing & Vol Surfaces",
    page_icon="📉",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="Call"):
    """Calculates Black-Scholes price and Greeks."""
    if T <= 0 or sigma <= 0:
        return {k: 0.0 for k in ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        delta = si.norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
        delta = -si.norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
        
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1)
    theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)
    
    return {
        "Price": price, "Delta": delta, "Gamma": gamma,
        "Vega": vega / 100, "Theta": theta / 365, "Rho": rho / 100
    }

def crr_binomial_tree(S, K, T, r, sigma, N, option_type="Call"):
    """Calculates Option Price using the CRR Binomial Tree method."""
    if T <= 0 or sigma <= 0: return 0.0

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    asset_prices = np.zeros(N + 1)
    asset_prices[0] = S * d**N
    for i in range(1, N + 1):
        asset_prices[i] = asset_prices[i - 1] * (u / d)
    
    option_values = np.zeros(N + 1)
    if option_type == "Call":
        option_values = np.maximum(0, asset_prices - K)
    else:
        option_values = np.maximum(0, K - asset_prices)
    
    discount_factor = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = discount_factor * (p * option_values[j + 1] + (1 - p) * option_values[j])
            
    return option_values[0]

def get_vol_surface_data(ticker_obj, spot_price):
    """
    Fetches option chains and prepares (X, Y, Z) data for both Calls and Puts.
    """
    expirations = ticker_obj.options
    if not expirations:
        return None, None

    # Limit expirations for performance
    selected_exps = expirations[:8]
    
    # Storage for Call Data
    c_strikes, c_tm, c_iv = [], [], []
    # Storage for Put Data
    p_strikes, p_tm, p_iv = [], [], []
    
    progress_bar = st.progress(0, text="Fetching option chains...")
    
    for i, exp_date_str in enumerate(selected_exps):
        progress_bar.progress((i + 1) / len(selected_exps), text=f"Fetching expiry: {exp_date_str}")
        try:
            chain = ticker_obj.option_chain(exp_date_str)
            calls = chain.calls
            puts = chain.puts
            
            # Time to maturity
            exp_date = pd.to_datetime(exp_date_str)
            today = pd.Timestamp.now().normalize()
            T = (exp_date - today).days / 365.0
            if T < 0.001: continue
            
            # -- Process Calls --
            mask_c = (calls['impliedVolatility'] > 0.01) & (calls['impliedVolatility'] < 2.0) & \
                     (calls['strike'] > spot_price * 0.5) & (calls['strike'] < spot_price * 1.5)
            filtered_c = calls[mask_c]
            c_strikes.extend(filtered_c['strike'].tolist())
            c_tm.extend([T] * len(filtered_c))
            c_iv.extend(filtered_c['impliedVolatility'].tolist())

            # -- Process Puts --
            mask_p = (puts['impliedVolatility'] > 0.01) & (puts['impliedVolatility'] < 2.0) & \
                     (puts['strike'] > spot_price * 0.5) & (puts['strike'] < spot_price * 1.5)
            filtered_p = puts[mask_p]
            p_strikes.extend(filtered_p['strike'].tolist())
            p_tm.extend([T] * len(filtered_p))
            p_iv.extend(filtered_p['impliedVolatility'].tolist())
            
        except Exception:
            continue

    progress_bar.empty()
    
    return (c_strikes, c_tm, c_iv), (p_strikes, p_tm, p_iv)

def plot_3d_surface(strikes, tm, iv, title_text, color_scale='Viridis'):
    """Generates a Plotly 3D Surface figure."""
    if not strikes:
        fig = go.Figure()
        fig.add_annotation(text="No Data Available", showarrow=False)
        return fig

    # Grid Interpolation
    x_grid = np.linspace(min(strikes), max(strikes), 40)
    y_grid = np.linspace(min(tm), max(tm), 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = griddata((strikes, tm), iv, (X, Y), method='cubic')
    
    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y,
        colorscale=color_scale,
        opacity=0.9,
        contours_z=dict(show=True, usecolormap=True, project_z=True)
    )])
    
    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='Strike ($)',
            yaxis_title='Time (Yrs)',
            zaxis_title='Implied Vol'
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        template="plotly_dark",
        height=500
    )
    return fig

# -----------------------------------------------------------------------------
# 3. SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("1. Select Index")

# Dictionary of Famous Indices (using ETF tickers for better data availability)
index_map = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM",
    "Gold (GLD)": "GLD",
    "Silver (SLV)": "SLV"
}

selected_index_name = st.sidebar.selectbox("Choose Instrument", list(index_map.keys()))
ticker_symbol = index_map[selected_index_name]

st.sidebar.markdown("---")
st.sidebar.header("2. Option Params")
spot_price_manual = st.sidebar.number_input("Manual Spot Price ($)", value=400.0)
current_price_input = st.sidebar.checkbox("Use Manual Spot?", value=False)

strike_price = st.sidebar.number_input("Strike Price ($)", value=400.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=0.5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5) / 100
option_type = st.sidebar.selectbox("Option Type (for Greeks)", ["Call", "Put"])

st.sidebar.header("3. Volatility Settings")
manual_vol = st.sidebar.slider("Manual Volatility (%)", 1.0, 100.0, 20.0) / 100
use_manual_vol = st.sidebar.checkbox("Override Hist Vol?", value=False)
steps_N = st.sidebar.slider("Binomial Steps (N)", 10, 500, 50)

# -----------------------------------------------------------------------------
# 4. MAIN LOGIC
# -----------------------------------------------------------------------------
st.title(f"📊 {selected_index_name} Pricing Engine")

# --- A. Data Fetching ---
try:
    with st.spinner(f"Fetching market data for {ticker_symbol}..."):
        ticker_obj = yf.Ticker(ticker_symbol)
        df = ticker_obj.history(period="1y")
        
        if df.empty:
            st.error("No data found. Check your internet.")
            st.stop()

        close_col = 'Close'
        if 'Adj Close' in df.columns: close_col = 'Adj Close'
        df_close = df[close_col]
        
        # Vol Calculation
        log_returns = np.log(df_close / df_close.shift(1)).dropna()
        historical_vol = log_returns.std() * np.sqrt(252)
        
        spot_price = df_close.iloc[-1] if not current_price_input else spot_price_manual
        sigma = manual_vol if use_manual_vol else historical_vol

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- B. Dashboard Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Spot Price", f"${spot_price:.2f}", delta=f"{df_close.iloc[-1]-df_close.iloc[-2]:.2f}")
col2.metric("Strike Price", f"${strike_price:.2f}")
col3.metric("Volatility", f"{sigma:.2%}")
col4.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")

# --- C. Pricing Models ---
st.markdown("### 🧮 Model Pricing & Greeks")
bs_res = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, sigma, option_type)
crr_price = crr_binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, sigma, steps_N, option_type)

c1, c2 = st.columns(2)
c1.info(f"**Black-Scholes:** ${bs_res['Price']:.4f}")
c2.success(f"**Binomial (CRR):** ${crr_price:.4f}")

# Greeks Row
g_cols = st.columns(5)
for i, greek in enumerate(["Delta", "Gamma", "Theta", "Vega", "Rho"]):
    g_cols[i].metric(greek, f"{bs_res[greek]:.4f}")

# --- D. Convergence ---
with st.expander("📉 View Convergence Graph"):
    step_range = range(10, 100, 5)
    crr_vals = [crr_binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, sigma, n, option_type) for n in step_range]
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=list(step_range), y=crr_vals, mode='lines+markers', name='CRR'))
    fig_conv.add_trace(go.Scatter(x=[step_range[0], step_range[-1]], y=[bs_res['Price'], bs_res['Price']], 
                                  mode='lines', line=dict(dash='dash', color='red'), name='BS Limit'))
    fig_conv.update_layout(template="plotly_dark", height=300, margin=dict(t=30,b=20))
    st.plotly_chart(fig_conv, use_container_width=True)

# --- E. Dual Volatility Surfaces ---
st.markdown("---")
st.subheader(f"🌋 Volatility Surfaces: {ticker_symbol}")
st.caption("Comparing Call vs. Put Implied Volatility Skews")

if st.checkbox("Generate 3D Volatility Surfaces", value=False):
    call_data, put_data = get_vol_surface_data(ticker_obj, spot_price)
    
    if call_data and put_data:
        # Unpack data
        c_strikes, c_tm, c_iv = call_data
        p_strikes, p_tm, p_iv = put_data
        
        # Create Columns for Side-by-Side Plots
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Call Option Surface")
            fig_call = plot_3d_surface(c_strikes, c_tm, c_iv, "Call IV Surface", 'Plasma')
            st.plotly_chart(fig_call, use_container_width=True)
            
        with col_right:
            st.markdown("#### Put Option Surface")
            fig_put = plot_3d_surface(p_strikes, p_tm, p_iv, "Put IV Surface", 'Viridis')
            st.plotly_chart(fig_put, use_container_width=True)
    else:
        st.warning("Could not fetch enough option data for surfaces.")

# --- F. Option Chains Table ---
st.markdown("---")
st.markdown("### 📋 Option Chain Data")
selected_expiry = st.selectbox("Select Expiry", ticker_obj.options)
chain = ticker_obj.option_chain(selected_expiry)

t1, t2 = st.tabs(["Calls", "Puts"])
t1.dataframe(chain.calls)
t2.dataframe(chain.puts)