import numpy as np
from numpy.random import normal
from multiprocessing import Pool
import pickle
import logging

# global parameters
logging.basicConfig(filename="montecarlog.log", encoding="utf-8", level=logging.DEBUG)
S0 = 100  # price at time 0
K = S0  # Strike
R = 0.0001  # interest rate
SIGMA = 0.05  # volatility
T = 100  # Maturity
T_rep = 10**8
X = np.linspace(0, T, T)  # abscisse
Y = np.linspace(0, T_rep, T_rep)
MU_STANDARD = 0  # mean
SIGMA_STANDARD = 1  # standard deviation
STEP = 500


@np.vectorize
def W(t):
    W_motion = normal(MU_STANDARD, SIGMA_STANDARD, size=t)
    W_motion[0] = 0
    for k in range(1, t):
        W_motion[k] += W_motion[k - 1]
    return W_motion


@np.vectorize
def S(t):
    brown_motion = W(t)
    s = 1 + R + SIGMA * (brown_motion[1:] - brown_motion[:-1])
    s = np.insert(s, 0, S0)
    for k in range(1, t):
        s[k] *= s[k - 1]
    return s[-1]


s = S(np.full((T_rep,), T))


@np.vectorize
def put(x):
    return max(x - K, 0)


@np.vectorize
def call(x):
    return max(K - x, 0)


payoff_exp = put(s)



def Monte_Carlo(M):
    return np.mean(payoff_exp[:M])


try:
    with Pool(15) as p:
        MC = p.map(Monte_Carlo, np.arange(1, T_rep))
except:
    logging.warning("Time exceeded")

with open("monte_carlo_result.pkl", "wb") as f:
    pickle.dump(MC, f)
