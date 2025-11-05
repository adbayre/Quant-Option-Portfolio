# Load packages
import math
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import numpy as np

### CRR Pricer Class
class CRRPricer:
    def Pricer(OptionType, K, T, S, sigma, r, q, N, Exercise):
        start=time.process_time()
        delta = T/N
        u = math.exp(sigma * math.sqrt(delta))
        d = 1/u
        p = (math.exp((r - q) * delta) - d) / (u - d)
        if Exercise=='E':
            if OptionType=='C':
                # European Call Option
                f = []
                for j in range(N + 1):
                    S_Nj = S * pow(u, j) * pow(d, N - j)
                    f_Nj = max(0, S_Nj - K)
                    f.append(f_Nj)
                for n in reversed(range(N)):
                    f_n = []
                    for j in range(n+1):
                        f_nj = math.exp(-r*delta)*(p*f[j+1]+(1-p)*f[j])
                        f_n.append(f_nj)
                    f = f_n
                option_price = f[0]
            elif OptionType=="P":
                # European Put Option
                f = []
                for j in range(N + 1):
                    S_Nj = S * pow(u, j) * pow(d, N - j)
                    f_Nj = max(0, K - S_Nj)
                    f.append(f_Nj)
                for n in reversed(range(N)):
                    f_n = []
                    for j in range(n+1):
                        f_nj = math.exp(-r*delta)*(p*f[j+1]+(1-p)*f[j])
                        f_n.append(f_nj)
                    f = f_n
                option_price = f[0]
            else:
                print("Option Type not recognized. Use 'C' for Call or 'P' for Put."))
        elif Exercise=='A':
            if OptionType=='C':
                # American Call Option
                f = []
                for j in range(N + 1):
                    S_Nj = S * pow(u, j) * pow(d, N - j)
                    f_Nj = max(0, S_Nj - K)
                    f.append(f_Nj)
                for n in reversed(range(N)):
                    f_n = []
                    for j in range(n+1):
                        S_nj = S * pow(u, j) * pow(d, n - j)
                        exercise_value = max(0, S_nj - K)
                        hold_value = math.exp(-r*delta)*(p*f[j+1]+(1-p)*f[j])
                        f_nj = max(exercise_value, hold_value)
                        f_n.append(f_nj)
                    f = f_n
                option_price = f[0]
            elif OptionType=="P":
                # American Put Option
                f = []
                for j in range(N + 1):
                    S_Nj = S * pow(u, j) * pow(d, N - j)
                    f_Nj = max(0, K - S_Nj)
                    f.append(f_Nj)
                for n in reversed(range(N)):
                    f_n = []
                    for j in range(n+1):
                        S_nj = S * pow(u, j) * pow(d, n - j)
                        exercise_value = max(0, K - S_nj)
                        hold_value = math.exp(-r*delta)*(p*f[j+1]+(1-p)*f[j])
                        f_nj = max(exercise_value, hold_value)
                        f_n.append(f_nj)
                    f = f_n
                option_price = f[0]