import numpy as np
import torch
from math import *
import logging

# global parameters
logging.basicConfig(filename="montecarlog.log", encoding="utf-8", level=logging.DEBUG)
S0 = 100  # price at time 0
K = S0  # Strike
R = 0.0001  # interest rate
SIGMA = 0.05  # volatility
T = 100  # Maturity
T_rep = 10**8
X = np.arange(0, T)  # abscisse
Y = np.arange(0, T_rep)
MU_STANDARD = 0  # mean
SIGMA_STANDARD = 1  # standard deviation
STEP = 500
device = torch.device("cpu")

# Brownian path
def W_traj(t):
    W_motion = torch.randn(t).to(device)
    W_motion[0] = 0
    return torch.cumsum(W_motion, dim=0)


# Brownian endpoint
def W(t):
    W_motion = torch.randn(size=(t,)).to(device)
    W_motion[0] = 0
    return torch.sum(W_motion).item()


# Monte Carlo algorithm
def Monte_Carlo(sample):
    t = sample.shape[0]
    sample = torch.cumsum(sample, dim=0)
    return sample / (1 + torch.arange(t))


# Pricing calculus by Euler's path
def S(t: int):
    brown_motion = W_traj(t)
    s = 1 + R + SIGMA * (brown_motion[1:] - brown_motion[:-1])
    S_ini = torch.tensor([S0]).to(device)
    s = torch.cat((S_ini, s))
    return torch.prod(s).item()


# put caluclus
def put(s):
    ReLU = torch.nn.ReLU()
    return ReLU(s - K * np.ones_like(s))


s = torch.zeros(T_rep)
for k in range(T_rep):
    s[k] = S(T)

payoff = put(s)

try:
    MC = Monte_Carlo(payoff).detach().numpy()
except Exception as e:
    logging.warning(e)

with open("monte_carlo_result.pkl", "wb") as f:
    np.save(f, MC)
