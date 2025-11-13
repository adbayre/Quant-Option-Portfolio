import streamlit as st
import matplotlib.pyplot as plt
from crr_pricer import CRROptionPricer

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="CRR Option Pricer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Blue Theme Styling ---
st.markdown("""
    <style>
    /* Global layout */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: #0A0F1F;
        color: #EAEAEA;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #3399FF;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Sidebar */
    [data-testid=stSidebar] {
        background-color: #0E1628;
        color: #EAEAEA;
        padding-top: 1.5rem;
    }
    [data-testid=stSidebar] h2, [data-testid=stSidebar] h3 {
        color: #4DA3FF;
    }
    [data-testid=stSidebar] label, [data-testid=stSidebar] span {
        color: #CCCCCC !important;
    }

    /* Inputs */
    input, select, textarea {
        background-color: #1B253A !important;
        color: #EAEAEA !important;
        border: 1px solid #2E3A59 !important;
        border-radius: 6px !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1E3A8A, #2563EB);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0px 0px 6px rgba(37, 99, 235, 0.5);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        box-shadow: 0px 0px 10px rgba(37, 99, 235, 0.8);
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4DA3FF;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        background-color: #111827;
        border-radius: 8px;
        padding: 0.2rem;
    }
    [data-baseweb="tab"] {
        color: #EAEAEA !important;
        font-weight: 600;
    }
    [data-baseweb="tab"]:hover {
        background-color: #1B253A !important;
    }

    /* Divider */
    hr {
        border: 1px solid #1E3A8A;
        margin: 1.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Cox-Ross-Rubinstein Option Pricer")
st.caption("Bayre Adrien | Liu Jack | Hanna Gerguis Alexis | Milcent Marcellin | Jouonang Kapnang Sinthia Vanelle")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Model Parameters")

    # Market variables
    st.markdown("#### Market Variables")
    S0 = st.number_input("Stock Price (S₀)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)

    col1, col2 = st.columns(2)
    with col1:
        T = st.number_input("Maturity (Years)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with col2:
        N = st.number_input("Steps (N)", min_value=2, max_value=200, value=50, step=1)

    col3, col4 = st.columns(2)
    with col3:
        r = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1) / 100
    with col4:
        sigma = st.number_input("Volatility (%)", min_value=0.0, max_value=200.0, value=20.0, step=0.1) / 100

    st.markdown("---")

    # Option type
    st.markdown("#### Option Specification")
    option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    exercise_type = st.radio("Exercise", ["European", "American"], horizontal=True)

    st.markdown("---")
    compute_btn = st.button("Compute Option", use_container_width=True)

# --- Computation ---
if compute_btn or 'pricer' not in st.session_state:
    pricer = CRROptionPricer(S0, K, T, r, sigma, int(N),
                             option_type.lower(), exercise_type.lower())
    price = pricer.price()
    st.session_state.pricer = pricer
    st.session_state.price = price
else:
    pricer = st.session_state.pricer
    price = st.session_state.price

# --- Display ---
st.markdown("## Option Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Option Price", f"${price:.4f}")
with col2:
    intrinsic = max(S0 - K, 0) if option_type == "Call" else max(K - S0, 0)
    st.metric("Intrinsic", f"${intrinsic:.4f}")
with col3:
    st.metric("Time Value", f"${price - intrinsic:.4f}")
with col4:
    moneyness = (
        "ATM" if abs(S0 - K) < 1
        else "ITM" if (S0 > K and option_type == "Call") or (S0 < K and option_type == "Put")
        else "OTM"
    )
    st.metric("Moneyness", moneyness)

st.divider()

# --- Greeks ---
with st.expander("Greeks", expanded=False):
    greeks = pricer.get_greeks()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Δ (Delta)", f"{greeks['Delta']:.4f}")
    with col2: st.metric("Γ (Gamma)", f"{greeks['Gamma']:.4f}")
    with col3: st.metric("Θ (Theta)", f"{greeks['Theta']:.4f}")
    with col4: st.metric("ν (Vega)", f"{greeks['Vega']:.4f}")
    with col5: st.metric("ρ (Rho)", f"{greeks['Rho']:.4f}")

# --- Binomial Trees ---
st.markdown("## Binomial Trees")

tab1, tab2 = st.tabs(["Stock Price Tree", "Option Value Tree"])
with tab1:
    fig_stock = pricer.plot_tree(pricer.stock_tree, "Stock Price Tree", "stock")
    st.pyplot(fig_stock)
    plt.close()

with tab2:
    fig_option = pricer.plot_tree(pricer.option_tree, "Option Value Tree", "option")
    st.pyplot(fig_option)
    plt.close()

# --- Footer ---
st.markdown('<p class="footer">Built with Streamlit | CRR Model Visualizer © 2025</p>', unsafe_allow_html=True)
