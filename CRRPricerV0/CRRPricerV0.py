import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set page config
st.set_page_config(page_title="CRR Option Pricer", layout="wide", initial_sidebar_state="expanded")

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
        
        # Calculate CRR parameters
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)
        
        self.stock_tree = None
        self.option_tree = None
        
    def build_stock_tree(self):
        """Build the stock price tree"""
        self.stock_tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                self.stock_tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return self.stock_tree
    
    def payoff(self, S):
        """Calculate option payoff at maturity"""
        if self.option_type == 'call':
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)
    
    def price(self):
        """Price the option using backward induction"""
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
        """Plot a binomial tree"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.set_xlim(-0.5, self.N + 0.5)
        ax.set_ylim(-0.5, self.N + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        dx = 1.0
        dy = 1.0
        
        # Draw edges
        for i in range(self.N):
            for j in range(i + 1):
                x = i * dx
                y = (self.N - i) / 2 + j * dy
                
                x_next_up = (i + 1) * dx
                y_next_up = (self.N - i - 1) / 2 + j * dy
                ax.plot([x, x_next_up], [y, y_next_up], 'b-', alpha=0.3, linewidth=1.5)
                
                x_next_down = (i + 1) * dx
                y_next_down = (self.N - i - 1) / 2 + (j + 1) * dy
                ax.plot([x, x_next_down], [y, y_next_down], 'r-', alpha=0.3, linewidth=1.5)
        
        # Draw nodes
        for i in range(self.N + 1):
            for j in range(i + 1):
                x = i * dx
                y = (self.N - i) / 2 + j * dy
                value = tree[j, i]
                
                # Determine node color
                if tree_type == 'option' and self.exercise_type == 'american' and i < self.N:
                    stock_price = self.stock_tree[j, i]
                    intrinsic = self.payoff(stock_price)
                    if abs(value - intrinsic) < 1e-6 and intrinsic > 0:
                        color = 'lightcoral'
                    else:
                        color = 'lightblue'
                else:
                    color = 'lightgreen' if i == self.N else 'lightblue'
                
                circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2, zorder=3)
                ax.add_patch(circle)
                
                ax.text(x, y, f'{value:.2f}', ha='center', va='center', 
                       fontsize=9, fontweight='bold', zorder=4)
        
        # Add legend
        if tree_type == 'option':
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Continuation'),
                mpatches.Patch(color='lightgreen', label='Maturity'),
            ]
            if self.exercise_type == 'american':
                legend_elements.append(
                    mpatches.Patch(color='lightcoral', label='Early Exercise')
                )
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        return fig
    
    def get_greeks(self, epsilon=0.01):
        """Calculate option Greeks"""
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
        
        # Gamma
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
        
        # Rebuild
        self.build_stock_tree()
        self.price()
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }


# Streamlit App
def main():
    st.title("🌳 CRR Binomial Option Pricer")
    st.markdown("### Interactive Cox-Ross-Rubinstein Option Pricing Model")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("📊 Option Parameters")
        
        # Market Parameters
        st.subheader("Market Parameters")
        S0 = st.slider("Stock Price (S₀)", 50.0, 200.0, 100.0, 1.0, 
                       help="Current price of the underlying stock")
        K = st.slider("Strike Price (K)", 50.0, 200.0, 100.0, 1.0,
                     help="Exercise price of the option")
        
        col1, col2 = st.columns(2)
        with col1:
            T = st.slider("Time to Maturity (years)", 0.1, 3.0, 1.0, 0.1,
                         help="Time until option expiration")
        with col2:
            N = st.slider("Time Steps", 2, 10, 5, 1,
                         help="Number of steps in binomial tree")
        
        r = st.slider("Risk-free Rate (%)", 0.0, 20.0, 5.0, 0.5,
                     help="Annual risk-free interest rate") / 100
        sigma = st.slider("Volatility (%)", 5.0, 100.0, 20.0, 1.0,
                         help="Annual volatility of stock returns") / 100
        
        st.divider()
        
        # Option Specifications
        st.subheader("Option Specifications")
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
        exercise_type = st.radio("Exercise Type", ["European", "American"], horizontal=True)
        
        st.divider()
        
        # Calculation button
        calculate = st.button("🚀 Calculate Option Price", type="primary", use_container_width=True)
    
    # Main content area
    if calculate or 'pricer' not in st.session_state:
        # Create pricer
        pricer = CRROptionPricer(
            S0=S0, K=K, T=T, r=r, sigma=sigma, N=N,
            option_type=option_type.lower(),
            exercise_type=exercise_type.lower()
        )
        
        # Calculate price
        option_price = pricer.price()
        st.session_state.pricer = pricer
        st.session_state.option_price = option_price
    else:
        pricer = st.session_state.pricer
        option_price = st.session_state.option_price
    
    # Display results in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Option Price", f"${option_price:.4f}", 
                 help="Fair value of the option")
    
    with col2:
        moneyness = "ATM" if abs(S0 - K) < 1 else ("ITM" if (S0 > K and option_type == "Call") or (S0 < K and option_type == "Put") else "OTM")
        st.metric("Moneyness", moneyness,
                 help="At-The-Money, In-The-Money, or Out-of-The-Money")
    
    with col3:
        intrinsic = max(S0 - K, 0) if option_type == "Call" else max(K - S0, 0)
        st.metric("Intrinsic Value", f"${intrinsic:.4f}",
                 help="Value if exercised immediately")
    
    with col4:
        time_value = option_price - intrinsic
        st.metric("Time Value", f"${time_value:.4f}",
                 help="Premium above intrinsic value")
    
    st.divider()
    
    # CRR Parameters
    with st.expander("📐 CRR Model Parameters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Up Factor (u)", f"{pricer.u:.4f}")
        with col2:
            st.metric("Down Factor (d)", f"{pricer.d:.4f}")
        with col3:
            st.metric("Risk-Neutral Prob (q)", f"{pricer.q:.4f}")
        with col4:
            st.metric("Discount Factor", f"{pricer.discount:.4f}")
    
    # Greeks
    with st.expander("📈 Option Greeks", expanded=False):
        with st.spinner("Calculating Greeks..."):
            greeks = pricer.get_greeks()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Delta (Δ)", f"{greeks['Delta']:.4f}",
                     help="Rate of change of option price w.r.t. stock price")
        with col2:
            st.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}",
                     help="Rate of change of delta w.r.t. stock price")
        with col3:
            st.metric("Theta (Θ)", f"{greeks['Theta']:.4f}",
                     help="Rate of change of option price w.r.t. time")
        with col4:
            st.metric("Vega (ν)", f"{greeks['Vega']:.4f}",
                     help="Rate of change of option price w.r.t. volatility")
        with col5:
            st.metric("Rho (ρ)", f"{greeks['Rho']:.4f}",
                     help="Rate of change of option price w.r.t. interest rate")
    
    st.divider()
    
    # Tree Visualizations
    st.header("🌲 Binomial Tree Visualizations")
    
    tab1, tab2 = st.tabs(["📊 Stock Price Tree", "💰 Option Value Tree"])
    
    with tab1:
        st.subheader("Stock Price Evolution")
        fig_stock = pricer.plot_tree(pricer.stock_tree, "Stock Price Tree", "stock")
        st.pyplot(fig_stock)
        plt.close()
    
    with tab2:
        st.subheader("Option Value Evolution")
        fig_option = pricer.plot_tree(pricer.option_tree, "Option Value Tree", "option")
        st.pyplot(fig_option)
        plt.close()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>CRR Binomial Option Pricer | Built with Streamlit | 
        Adjust parameters in the sidebar to see real-time updates</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()