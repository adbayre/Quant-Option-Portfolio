import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as si
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Option Pricing & Convergence",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for "Pretty" UI
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
# 2. MODEL FUNCTIONS
# -----------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="Call"):
    """
    Calculates Black-Scholes price and Greeks.
    """
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
        "Price": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega / 100, # Conventionally shown per 1% vol change
        "Theta": theta / 365, # Conventionally shown per day
        "Rho": rho / 100 # Conventionally shown per 1% rate change
    }

def crr_binomial_tree(S, K, T, r, sigma, N, option_type="Call"):
    """
    Calculates Option Price using the Cox-Ross-Rubinstein (CRR) Binomial Tree method.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity (step N)
    asset_prices = np.zeros(N + 1)
    asset_prices[0] = S * d**N
    for i in range(1, N + 1):
        asset_prices[i] = asset_prices[i - 1] * (u / d)
    
    # Initialize option values at maturity
    option_values = np.zeros(N + 1)
    if option_type == "Call":
        option_values = np.maximum(0, asset_prices - K)
    else:
        option_values = np.maximum(0, K - asset_prices)
    
    # Backward induction
    discount_factor = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = discount_factor * (p * option_values[j + 1] + (1 - p) * option_values[j])
            
    return option_values[0]

# -----------------------------------------------------------------------------
# 3. SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("1. Market Data Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

st.sidebar.header("2. Option Parameters")
current_price_input = st.sidebar.checkbox("Use Manual Spot Price?", value=False)
spot_price_manual = st.sidebar.number_input("Manual Spot Price ($)", value=100.0)

strike_price = st.sidebar.number_input("Strike Price ($)", value=500.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5) / 100
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

st.sidebar.header("3. Volatility Settings")
vol_window = st.sidebar.slider("Historical Vol Window (Days)", 10, 252, 30)
use_manual_vol = st.sidebar.checkbox("Override Historical Vol?", value=False)
manual_vol = st.sidebar.slider("Manual Volatility (%)", 1.0, 100.0, 20.0) / 100

st.sidebar.header("4. Binomial Tree Settings")
steps_N = st.sidebar.slider("Tree Steps (N)", min_value=10, max_value=2000, value=100, step=10)

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

st.title("📊 Quantitative Options Pricing Engine")
st.markdown("Fit **Black-Scholes** and **Binomial (CRR)** models to historical data.")

# --- A. Data Fetching ---
try:
    with st.spinner(f"Fetching data for {ticker}..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        
    if df.empty:
        st.error("No data found. Please check the ticker symbol.")
        st.stop()

    # Calculate Returns and Volatility
    # Use 'Adj Close' if available, otherwise 'Close'
    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    # Handling MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df_close = df[close_col][ticker]
    else:
        df_close = df[close_col]

    df_close = df_close.dropna()
    log_returns = np.log(df_close / df_close.shift(1))
    historical_vol = log_returns.std() * np.sqrt(252)
    
    # Determine which Spot and Vol to use
    spot_price = df_close.iloc[-1] if not current_price_input else spot_price_manual
    sigma = manual_vol if use_manual_vol else historical_vol

    # --- B. Dashboard Top Row (Metrics) ---
    st.subheader("Market Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spot Price ($)", f"{spot_price:.2f}", delta=f"{df_close.iloc[-1] - df_close.iloc[-2]:.2f}")
    with col2:
        st.metric("Strike Price ($)", f"{strike_price:.2f}")
    with col3:
        st.metric("Volatility (σ)", f"{sigma:.2%}", delta=None if use_manual_vol else "Historical")
    with col4:
        st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")

    # --- C. Stock Price Chart (Plotly) ---
    # Handle data format for Plotly
    if isinstance(df.columns, pd.MultiIndex):
         # Flatten for easier plotting if necessary, or just extract specific series
         # For simplicity in yf's new structure, we extract individual series
         idx = df.index
         open_data = df['Open'][ticker]
         high_data = df['High'][ticker]
         low_data = df['Low'][ticker]
         close_data = df['Close'][ticker]
    else:
         idx = df.index
         open_data = df['Open']
         high_data = df['High']
         low_data = df['Low']
         close_data = df['Close']

    fig_candle = go.Figure(data=[go.Candlestick(
        x=idx,
        open=open_data,
        high=high_data,
        low=low_data,
        close=close_data,
        name=ticker
    )])
    fig_candle.update_layout(
        title=f"{ticker} Historical Price Action",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("---")

    # --- D. Pricing & Greeks ---
    st.subheader("Pricing & Greeks Analysis")
    
    # Calculate Models
    bs_res = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, sigma, option_type)
    crr_price = crr_binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, sigma, steps_N, option_type)

    # Display Prices
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.info(f"**Black-Scholes Price:** ${bs_res['Price']:.4f}")
    with p_col2:
        st.success(f"**Binomial (CRR) Price (N={steps_N}):** ${crr_price:.4f}")

    # Display Greeks
    g_cols = st.columns(5)
    greeks_list = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    for i, greek in enumerate(greeks_list):
        with g_cols[i]:
            val = bs_res[greek]
            st.metric(greek, f"{val:.4f}")
    
    st.markdown("---")

    # --- E. Convergence Analysis (Random Walk to Brownian Motion) ---
    st.subheader("Convergence: Random Walk (CRR) → Brownian Motion (BS)")
    st.caption("Visualizing how the discrete binomial tree price oscillates and converges to the continuous Black-Scholes price as the number of time steps (N) increases.")

    if st.button("Run Convergence Simulation (This may take a moment)"):
        with st.spinner("Simulating convergence..."):
            # Simulation parameters
            step_range = range(10, 300, 5) # Calculate for N=10 to N=300
            crr_prices = []
            
            for n in step_range:
                price = crr_binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, sigma, n, option_type)
                crr_prices.append(price)

            # Plotting
            fig_conv = go.Figure()

            # 1. The Oscillating CRR Price (Random Walk Approximation)
            fig_conv.add_trace(go.Scatter(
                x=list(step_range),
                y=crr_prices,
                mode='lines+markers',
                name='CRR Binomial Price',
                line=dict(color='#00CC96', width=2),
                marker=dict(size=4)
            ))

            # 2. The Constant BS Price (Continuous Limit)
            fig_conv.add_trace(go.Scatter(
                x=[step_range[0], step_range[-1]],
                y=[bs_res['Price'], bs_res['Price']],
                mode='lines',
                name='Black-Scholes Price (Limit)',
                line=dict(color='#EF553B', width=3, dash='dash')
            ))

            fig_conv.update_layout(
                title="Convergence of Discrete Tree to Continuous Model",
                xaxis_title="Number of Steps (N)",
                yaxis_title="Option Price ($)",
                template="plotly_dark",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig_conv, use_container_width=True)

            # Theory explanation
            st.info("""
            **Why does this happen?**
            The **CRR Binomial Tree** models price movements as a discrete "Random Walk" (up or down). 
            As we increase the number of steps ($N$) while keeping time ($T$) constant, the time step ($dt$) becomes infinitesimally small.
            According to the **Central Limit Theorem**, this discrete distribution converges to the Log-Normal distribution assumed by **Black-Scholes** (Geometric Brownian Motion).
            The oscillation ("sawtooth" pattern) is due to the discrete placement of nodes relative to the strike price.
            """)

except Exception as e:
    st.error(f"An error occurred: {e}")