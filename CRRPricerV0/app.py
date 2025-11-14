# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# --- CRR Option Pricer Class ---
class CRROptionPricer:
    """Cox-Ross-Rubinstein Binomial Tree Option Pricer"""
    
    def __init__(self, S0, K, T, r, sigma, N, option_type='call', exercise_type='european'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()
        self.exercise_type = exercise_type.lower()
        
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)
        
        self.stock_tree = None
        self.option_tree = None

    def build_stock_tree(self):
        self.stock_tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                self.stock_tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return self.stock_tree
    
    def payoff(self, S):
        if self.option_type == 'call':
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)
    
    def price(self):
        if self.stock_tree is None:
            self.build_stock_tree()
        
        self.option_tree = np.zeros((self.N + 1, self.N + 1))
        self.option_tree[:, self.N] = self.payoff(self.stock_tree[:, self.N])
        
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                continuation = self.discount * (
                    self.q * self.option_tree[j, i + 1] + 
                    (1 - self.q) * self.option_tree[j + 1, i + 1]
                )
                if self.exercise_type == 'american':
                    exercise = self.payoff(self.stock_tree[j, i])
                    self.option_tree[j, i] = np.maximum(continuation, exercise)
                else:
                    self.option_tree[j, i] = continuation
        
        return self.option_tree[0, 0]
    
    def plot_tree(self, tree, title, tree_type):
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor('#0f172b')
        ax.set_facecolor('#0f172b')
        
        ax.set_xlim(-0.5, self.N + 0.5)
        ax.set_ylim(-0.5, self.N + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#e2e8f0')
        
        dx = 1.0
        dy = 1.0
        
        # Draw edges
        for i in range(self.N):
            for j in range(i + 1):
                x = i * dx
                y = (self.N - i) / 2 + j * dy
                ax.plot([x, i+1], [y, (self.N - i - 1)/2 + j*dy], color='#3b82f6', alpha=0.3, linewidth=1.5)
                ax.plot([x, i+1], [y, (self.N - i - 1)/2 + (j+1)*dy], color='#ef4444', alpha=0.3, linewidth=1.5)
        
        # Draw nodes
        for i in range(self.N + 1):
            for j in range(i + 1):
                x = i * dx
                y = (self.N - i) / 2 + j * dy
                value = tree[j, i]
                
                if tree_type == 'option' and self.exercise_type == 'american' and i < self.N:
                    stock_price = self.stock_tree[j, i]
                    intrinsic = self.payoff(stock_price)
                    if abs(value - intrinsic) < 1e-6 and intrinsic > 0:
                        color = '#f87171'
                    else:
                        color = '#60a5fa'
                else:
                    color = '#34d399' if i == self.N else '#60a5fa'
                
                circle = plt.Circle((x, y), 0.15, color=color, ec='#e2e8f0', linewidth=2)
                ax.add_patch(circle)
                ax.text(x, y, f'{value:.2f}', ha='center', va='center', 
                       fontsize=8 if self.N > 8 else 9, fontweight='bold', color='#0f172b')
        
        return fig
    
    def get_greeks(self, epsilon=0.01):
        base_price = self.price()
        
        # Delta
        self.S0 += epsilon
        self.build_stock_tree()
        price_up = self.price()
        self.S0 -= 2 * epsilon
        self.build_stock_tree()
        price_down = self.price()
        self.S0 += epsilon
        delta = (price_up - price_down) / (2 * epsilon)
        gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)
        
        # Vega
        self.sigma += epsilon
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.build_stock_tree()
        price_vega = self.price()
        self.sigma -= epsilon
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        vega = (price_vega - base_price) / epsilon
        
        # Theta
        self.T -= epsilon
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        self.build_stock_tree()
        price_theta = self.price()
        self.T += epsilon
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        theta = (price_theta - base_price) / epsilon
        
        # Rho
        self.r += epsilon
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        self.build_stock_tree()
        price_rho = self.price()
        self.r -= epsilon
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        rho = (price_rho - base_price) / epsilon
        
        self.build_stock_tree()
        self.price()
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}


# --- Black-Scholes Formula ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


# --- Fetch SPY Historical Data ---
def fetch_spy_data(backtest_date=None, lookback_days=252):
    """
    Fetch SPY data from Yahoo Finance
    
    Parameters:
    -----------
    backtest_date : datetime or None
        If provided, fetch data up to this date for backtesting
        If None, fetch current/recent data
    lookback_days : int
        Number of trading days to look back for volatility calculation
    """
    try:
        spy = yf.Ticker("SPY")
        
        if backtest_date:
            # For backtesting: fetch data from lookback period to TODAY
            # This shows how the market evolved AFTER the backtest date
            start_date = backtest_date - timedelta(days=lookback_days * 2)  # Extra buffer for weekends
            end_date = datetime.now()  # Fetch all the way to present
            hist_full = spy.history(start=start_date, end=end_date)
            
            # Convert backtest_date to timezone-aware if hist_full index is timezone-aware
            if hist_full.index.tz is not None:
                if backtest_date.tzinfo is None:
                    backtest_date = backtest_date.replace(tzinfo=hist_full.index.tz)
            
            # Get historical data UP TO backtest date for volatility calculation
            hist_backtest = hist_full[hist_full.index <= backtest_date]
            
            if len(hist_backtest) == 0:
                return None, None, None, None
            
            # Get price at the backtest date
            current_price = hist_backtest['Close'].iloc[-1]
            
            # Calculate volatility from returns UP TO backtest date
            returns = np.log(hist_backtest['Close'] / hist_backtest['Close'].shift(1)).dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return current_price, volatility, hist_full, backtest_date
        else:
            # Current data: fetch last year
            hist = spy.history(period="1y")
            
            if len(hist) == 0:
                return None, None, None, None
            
            current_price = hist['Close'].iloc[-1]
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return current_price, volatility, hist, None
    except Exception as e:
        st.error(f"Error fetching SPY data: {e}")
        return None, None, None, None


# --- Plot SPY Price History ---
def plot_spy_history(hist, selected_date=None, lookback_date=None):
    """Plot SPY price history with optional date markers"""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0f172b')
    ax.set_facecolor('#0f172b')
    
    # Plot full history
    ax.plot(hist.index, hist['Close'], color='#60a5fa', linewidth=1, label='S&P 500 Close Price')
    # Convert dates to timezone-aware if needed
    if hist.index.tz is not None:
        if selected_date and selected_date.tzinfo is None:
            selected_date = selected_date.replace(tzinfo=hist.index.tz)
        if lookback_date and lookback_date.tzinfo is None:
            lookback_date = lookback_date.replace(tzinfo=hist.index.tz)
    # Mark the lookback period start (for volatility calculation)
    if lookback_date and selected_date:
        # Shade the lookback period used for volatility
        ax.axvspan(lookback_date, selected_date, alpha=0.2, color='#34d399', label='Volatility Lookback Period')
    
    # Mark the selected backtest date
    if selected_date:
        ax.axvline(x=selected_date, color='#f87171', linestyle='--', linewidth=1.5, 
                   label=f'Backtest Date: {selected_date.date()}')
        
        # Add annotation
        price_at_date = hist[hist.index <= selected_date]['Close'].iloc[-1]
        ax.annotate(f'${price_at_date:.2f}', 
                   xy=(selected_date, price_at_date),
                   xytext=(10, 10), textcoords='offset points',
                   color='#f87171', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#1d293d', edgecolor='#f87171'))
    
    ax.set_xlabel('Date', fontsize=12, color='#e2e8f0', fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, color='#e2e8f0', fontweight='bold')
    # ax.set_title('S&P 500 Price Evolution (Lookback Period → Present)', fontsize=14, fontweight='bold', color='#e2e8f0')
    ax.tick_params(colors='#e2e8f0', labelsize=10)
    ax.spines['bottom'].set_color('#314158')
    ax.spines['top'].set_color('#314158')
    ax.spines['left'].set_color('#314158')
    ax.spines['right'].set_color('#314158')
    ax.legend(facecolor='#1d293d', edgecolor='#314158', fontsize=10, labelcolor='#e2e8f0')
    ax.grid(True, alpha=0.2, color='#314158', linestyle='--')
    
    return fig


# --- Streamlit App ---
st.set_page_config(
    page_title="CRR Option Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cox-Ross-Rubinstein & Black Scholes Option Pricer")
st.caption("Bayre Adrien | Liu Jack | Hanna Gerguis Alexis | Milcent Marcellin | Jouonang Kapnang Sinthia Vanelle")

# --- Layout: Left (Inputs) | Right (Outputs) ---
left_col, right_col = st.columns([1, 2])

# ========== LEFT COLUMN: MODEL INPUTS ==========
with left_col:
    st.header("Model Parameters")
    
    # Data Source Selection
    st.subheader("Data Source")
    data_source = st.radio("Choose data source:", ["Manual Input", "Current S&P500 Data", "S&P500 Backtesting"], horizontal=False)
    
    if data_source == "Current S&P500 Data":
        if st.button("Load Current S&P500 Data", use_container_width=True):
            with st.spinner("Fetching current S&P500 data..."):
                spy_price, spy_vol, spy_hist, _ = fetch_spy_data()
                if spy_price:
                    st.session_state.spy_price = spy_price
                    st.session_state.spy_vol = spy_vol
                    st.session_state.spy_hist = spy_hist
                    st.session_state.backtest_date = None
                    st.success(f"Fetched! Price: ${spy_price:.2f} | Volatility: {spy_vol*100:.2f}%")
                else:
                    st.error("Failed to fetch S&P500 data.")
        
        S0 = st.session_state.get('spy_price', 500.0)
        sigma_pct = st.session_state.get('spy_vol', 0.15) * 100
        st.info(f"Using Current S&P500: S₀ = ${S0:.2f}, σ = {sigma_pct:.1f}%")
        sigma = sigma_pct / 100
        
    elif data_source == "S&P500 Backtesting":
        st.subheader("Backtesting Date")
        
        # Date picker for backtesting
        min_date = datetime(2010, 1, 1)
        max_date = datetime.now() - timedelta(days=1)
        default_date = datetime(2022, 1, 1)
        
        selected_date = st.date_input(
            "Select historical date:",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
            help="Select a date to backtest option pricing"
        )
        
        lookback_period = st.selectbox(
            "Volatility lookback period:",
            [30, 60, 90, 180, 252],
            index=4,
            help="Number of trading days to calculate historical volatility"
        )
        
        if st.button("Load S&P500 Data for Selected Date", use_container_width=True):
            with st.spinner(f"Fetching S&P500 data for {selected_date}..."):
                backtest_datetime = datetime.combine(selected_date, datetime.min.time())
                spy_price, spy_vol, spy_hist, backtest_dt = fetch_spy_data(backtest_datetime, lookback_period)
                if spy_price:
                    st.session_state.spy_price = spy_price
                    st.session_state.spy_vol = spy_vol
                    st.session_state.spy_hist = spy_hist
                    st.session_state.backtest_date = backtest_datetime
                    st.success(f"Loaded! Price on {selected_date}: ${spy_price:.2f} | Vol: {spy_vol*100:.2f}%")
                else:
                    st.error("Failed to fetch data for selected date.")
        
        S0 = st.session_state.get('spy_price', 500.0)
        sigma_pct = st.session_state.get('spy_vol', 0.15) * 100
        backtest_date = st.session_state.get('backtest_date', None)
        
        if backtest_date:
            st.info(f"Backtesting {backtest_date.date()}: S₀ = ${S0:.2f}, σ = {sigma_pct:.1f}%")
        else:
            st.warning("Please load data for the selected date")
        
        sigma = sigma_pct / 100
        
    else:
        # Manual Input
        S0 = st.number_input("Stock Price (S₀)", min_value=1.0, value=100.0, step=1.0)
        sigma_pct = st.number_input("Volatility (%)", min_value=1.0, value=20.0, step=1.0)
        sigma = sigma_pct / 100
    
    st.divider()
    
    # Other Parameters
    st.subheader("Option Parameters")
    K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    r_pct = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
    r = r_pct / 100
    
    st.divider()
    
    st.subheader("Option Type")
    option_type = st.radio("Type:", ["Call", "Put"], horizontal=True)
    exercise_type = st.radio("Exercise:", ["European", "American"], horizontal=True)
    
    st.divider()
    
    # Convergence Slider
    st.subheader("Convergence Analysis")
    N = st.slider(
        "Number of Steps (N)",
        min_value=2,
        max_value=50,
        value=10,
        step=1,
        help="Adjust to see CRR price converge to Black-Scholes"
    )
    
    st.divider()
    compute_btn = st.button("Compute Option Price", use_container_width=True, type="primary")

# --- Compute Pricing ---
if compute_btn or 'crr_price' not in st.session_state:
    pricer = CRROptionPricer(S0, K, T, r, sigma, int(N), option_type.lower(), exercise_type.lower())
    crr_price = pricer.price()
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type.lower()) if exercise_type == "European" else None
    
    st.session_state.pricer = pricer
    st.session_state.crr_price = crr_price
    st.session_state.bs_price = bs_price
    st.session_state.params = (S0, K, T, r, sigma, option_type, exercise_type, N)
else:
    pricer = st.session_state.pricer
    crr_price = st.session_state.crr_price
    bs_price = st.session_state.bs_price
    S0, K, T, r, sigma, option_type, exercise_type, N = st.session_state.params

# ========== TOP: PRICE & MONEYNESS DISPLAY ==========
st.divider()

intrinsic = max(S0 - K, 0) if option_type == "Call" else max(K - S0, 0)
moneyness = (
    "ATM" if abs(S0 - K) < 1
    else "ITM" if (S0 > K and option_type == "Call") or (S0 < K and option_type == "Put")
    else "OTM"
)

cols = st.columns(5)
cols[0].metric("CRR Price", f"${crr_price:.4f}")
if bs_price:
    cols[1].metric("Black-Scholes", f"${bs_price:.4f}")
    cols[2].metric("Difference", f"${abs(crr_price - bs_price):.4f}")
else:
    cols[1].metric("Black-Scholes", "N/A (American)")
    cols[2].metric("Difference", "N/A")
cols[3].metric("Intrinsic Value", f"${intrinsic:.4f}")
cols[4].metric("Moneyness", moneyness)

st.divider()

# ========== RIGHT COLUMN: VISUALIZATIONS ==========
with right_col:
    
    # --- SPY Price History (if available) ---
    if 'spy_hist' in st.session_state and st.session_state.spy_hist is not None:
        st.header("S&P 500 Price History")
        spy_hist = st.session_state.spy_hist
        backtest_date = st.session_state.get('backtest_date', None)
        
        fig_spy = plot_spy_history(spy_hist, backtest_date)
        st.pyplot(fig_spy)
        plt.close()
        
        st.divider()
    
    # --- Option Value Tree ---
    st.header("Option Value Tree")
    if pricer.option_tree is not None:
        fig_option = pricer.plot_tree(pricer.option_tree, f"{exercise_type} {option_type} - Option Value Tree", "option")
        st.pyplot(fig_option)
        plt.close()
    
    st.divider()
    
    # --- Convergence to Black-Scholes ---
    st.header("CRR Convergence to Black-Scholes")
    
    if exercise_type == "European":
        steps_range = list(range(2, min(201, N + 50), 2))
        crr_prices = []
        
        with st.spinner("Computing convergence..."):
            for n in steps_range:
                temp_pricer = CRROptionPricer(S0, K, T, r, sigma, n, option_type.lower(), 'european')
                crr_prices.append(temp_pricer.price())
        
        # Plot convergence
        fig_conv, ax = plt.subplots(figsize=(12, 6))
        fig_conv.patch.set_facecolor('#0f172b')
        ax.set_facecolor('#0f172b')
        
        ax.plot(steps_range, crr_prices, 'o-', color='#60a5fa', linewidth=2, markersize=5, label='CRR Price', alpha=0.8)
        ax.axhline(y=bs_price, color='#34d399', linestyle='--', linewidth=2.5, label='Black-Scholes Price')
        ax.axvline(x=N, color='#f87171', linestyle=':', linewidth=2.5, alpha=0.8, label=f'Current N = {N}')
        
        ax.set_xlabel('Number of Steps (N)', fontsize=13, color='#e2e8f0', fontweight='bold')
        ax.set_ylabel('Option Price ($)', fontsize=13, color='#e2e8f0', fontweight='bold')
        # ax.set_title('CRR Convergence to Black-Scholes', fontsize=15, fontweight='bold', color='#e2e8f0')
        ax.tick_params(colors='#e2e8f0', labelsize=11)
        ax.spines['bottom'].set_color('#314158')
        ax.spines['top'].set_color('#314158')
        ax.spines['left'].set_color('#314158')
        ax.spines['right'].set_color('#314158')
        ax.legend(facecolor='#1d293d', edgecolor='#314158', fontsize=11, labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#314158', linestyle='--')
        
        st.pyplot(fig_conv)
        plt.close()
    else:
        st.info("ℹConvergence analysis only available for European options")
    
    st.divider()
    
    # --- Greeks ---
    with st.expander("Option Greeks", expanded=True):
        greeks = pricer.get_greeks()
        cols = st.columns(5)
        cols[0].metric("Δ Delta", f"{greeks['Delta']:.4f}")
        cols[1].metric("Γ Gamma", f"{greeks['Gamma']:.4f}")
        cols[2].metric("Θ Theta", f"{greeks['Theta']:.4f}")
        cols[3].metric("ν Vega", f"{greeks['Vega']:.4f}")
        cols[4].metric("ρ Rho", f"{greeks['Rho']:.4f}")

# --- Footer ---
st.divider()
st.markdown('<p style="text-align:center; color:#8A94B0; margin-top:20px;">ESILV Paris | CRR & Black Scholes Models Visualizer v0.1 © 2025</p>', unsafe_allow_html=True)