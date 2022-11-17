import numpy as np
import torch
import logging

# global parameters
logging.basicConfig(filename="montecarlog.log", encoding="utf-8", level=logging.DEBUG)
S0 = 100  # price at time 0
K = S0  # Strike
R = 0.0001  # interest rate
SIGMA = 0.05  # volatility
T = 100  # Maturity
T_rep = 10**7
X = np.arange(0, T)  # abscisse
Y = np.arange(0, T_rep)
MU_STANDARD = 0  # mean
SIGMA_STANDARD = 1  # standard deviation
STEP = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def W(t):
    W_motion = torch.randn(size=(t,)).to(device)
    W_motion[0] = 0
    S_M = torch.tril(torch.ones((t, t))).to(device)
    return torch.matmul(S_M, W_motion)


def S(t):
    brown_motion = W(t)
    s = 1 + R + SIGMA * (brown_motion[1:] - brown_motion[:-1])
    S_ini = torch.tensor([S0]).to(device)
    s = torch.cat((S_ini, s))
    return torch.prod(s)


s = torch.tensor([S(T) for _ in range(T_rep)])


@np.vectorize
def put(x):
    return torch.max(x - K, 0)


payoff_exp = put(s)


def Monte_Carlo(M):
    return torch.mean(payoff_exp[:M]).item()


MC = np.arange(1, T_rep)
try:
    MC = [Monte_Carlo(k) for k in range(1, T_rep)]
except Exception as e:
    logging.warning(e)

with open("monte_carlo_result.pkl", "wb") as f:
    np.save(f, MC)
