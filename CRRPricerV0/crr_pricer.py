import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches

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
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-0.5, self.N + 0.5)
        ax.set_ylim(-0.5, self.N + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        dx = 1.0
        dy = 1.0
        for i in range(self.N):
            for j in range(i + 1):
                x = i * dx
                y = (self.N - i) / 2 + j * dy
                ax.plot([x, i+1], [y, (self.N - i - 1)/2 + j*dy], 'b-', alpha=0.3)
                ax.plot([x, i+1], [y, (self.N - i - 1)/2 + (j+1)*dy], 'r-', alpha=0.3)
        
        for i in range(self.N + 1):
            for j in range(i + 1):
                x = i * dx
                y = (self.N - i) / 2 + j * dy
                value = tree[j, i]
                color = 'lightgreen' if i == self.N else 'lightblue'
                circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
        
        return fig
    
    def get_greeks(self, epsilon=0.01):
        base_price = self.price()
        # Delta
        self.S0 += epsilon; self.build_stock_tree()
        price_up = self.price()
        self.S0 -= 2 * epsilon; self.build_stock_tree()
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
        self.T -= epsilon; self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt)); self.d = 1 / self.u
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        self.build_stock_tree()
        price_theta = self.price()
        self.T += epsilon; self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt)); self.d = 1 / self.u
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
        self.build_stock_tree(); self.price()
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
