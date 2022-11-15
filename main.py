import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from math import *
import scipy.stats as stats
from cProfile import label
import multiprocessing
from tqdm import tqdm
import pickle

# global parameters

S0 = 100 # price at time 0
K = S0 # Strike
R = 0.0001 # interest rate
SIGMA = 0.05  # volatility
T = 5 # Maturity
T_rep = 10**9
X = np.linspace(0, T, T)  # abscisse
Y = np.linspace(0, T_rep, T_rep)
MU_STANDARD = 0  # mean
SIGMA_STANDARD = 1  # standard deviation
STEP = 500

def W(t):
    W_motion = [0]
    for _ in range(1, t):
        W_motion.append(normal(MU_STANDARD, SIGMA_STANDARD) + W_motion[-1])
    return np.array(W_motion)
    
def S(t):
    s = [S0]
    brown_motion = W(t)
    for k in range(1, t):
        s.append(s[-1] * ( 1 + R + SIGMA * (brown_motion[k] - brown_motion[k-1])))
    return np.array(s)

s = S(T)

put = lambda x: max(x - K, 0)

payoff_exp = list(map(put, [sample[-1] for sample in s]))

def Monte_Carlo(M, payoff_list):
    return 1/M * np.sum(payoff_list[:M])

MC = np.array(list(map(Monte_Carlo, tqdm(np.arange(1, T_rep, 1)), [payoff_exp]*(T_rep-1))))

with open('monte_carlo_result.pkl', 'wb') as f:
    pickle.dump(MC, f)
