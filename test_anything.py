import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import Collected_runs as cr
import FS3
import sympy
import glob
import time
import scipy as sp
from numpy.polynomial.chebyshev import chebval

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": "cmr10",
        "mathtext.fontset": "cm",
        "font.family": "STIXGeneral",
        "axes.unicode_minus": True,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.top": "on",
        "xtick.major.bottom": "on",
        "ytick.major.left": "on",
        "ytick.major.right": "on",
        "xtick.top": True,
        "ytick.right": True,
    }
)

parameter_dictT = {
    "path_to_data": "/home/daniel/Master_thesis/Data/Leo4DataJ/",
    "alpha": [1.5, 2.0, 2.25, 2.5, 2.75, 3.0],
    "sigma": ["_Tsm050", "_Ts000", "_Ts025", "_Ts050", "_Ts075", "_Ts100"],
    "gRange": [
        np.array([2.425, 2.435]),
        np.array([1.339, 1.348]),
        np.array([1.092, 1.104]),
        np.array([0.914, 0.926]),
        np.array([0.76, 0.785]),
    ],
    "LRange": [[64, 256], [64, 256], [64, 256], [64, 256], [64, 256], [64, 256]],
    "gc": [2.436, 1.34227, 1.097, 0.917, 0.775, 0.66],
    "nu": [0.976, 1.01, 1.05, 1.41, 2.0, 1.8],
    "eta": [0.997, 0.935, 0.8, 0.66, 0.5, 0.4],
    # "chiMax": [9.0, 100.9, 800.4, 100.4, 30000, 50000],
    # "rho_L_power": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # "JRange": [np.array([4.9,5.2]),np.array([2.89,2.90]),np.array([2.15,2.17]),np.array([0,2.18]),np.array([0,2.18])],
    # "J0": [5.1,2.895,2.165,1.89,2.17],
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}

parameter_dictJ = {
    "path_to_data": "/home/daniel/Master_thesis/Data/Leo4DataJ",
    "alpha": np.array([1.5, 2.0, 2.5, 3.0, 3.5]),
    "sigma": ["_Jsm050", "_Js000", "_Js050", "_Js100", "_Js150"],
    "gRange": [
        np.array([9.94, 10.02]),
        np.array([5.77, 5.805]),
        np.array([4.3, 4.38]),
        np.array([3.58, 3.61]),
        np.array([3.17, 3.2]),
    ],
    "LRange": np.array([[32, 128], [32, 128], [32, 128], [32, 128], [32, 128]]),
    "gc": np.array([9.975, 5.802, 4.33, 3.595, 3.1835]),
    "nu": np.array([0.745, 0.736, 0.75, 0.75, 0.723]),
    "eta": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    # "chiMax":  np.array([100000.0, 100000.0, 150000.0, 20000, 8000.0]),
    # "nuJ": [1.0,1.0,1.2,1.0,1.0],
    # "chiMaxJ": [[1.0,126],[10000,10000],[10000,10000],[10000,10000],[10000,10000]]
}


def poly(x, *params):
    f = 0
    for n in range(len(params)):
        f += params[n] * x**n
    return f


def poly_derivative(x, *params):
    f = 0
    for n in range(len(params)):
        f += n * params[n] * x ** (n - 1)
    return f


def log(x, a, b):
    return a * np.log(x + b)


def over(x, a, b, c):
    return a / (x + b) ** 2 + c


def shift(f, L, gc, nu, c, omega):
    def g(x, *args):
        return f((x - gc) * L**nu, *args) * (1 + c * L ** (-omega))

    return g


def gauss_fit(x, mean, std_err):
    return (
        1.0
        / (np.sqrt(2 * np.pi) * std_err)
        * np.exp(-0.5 * ((x - mean) / std_err) ** 2)
    )


def powerlaw(x, a, k):
    return k * x**a


def lambda_k(k, L, alpha, J_perp):
    lamb = 0.0
    for x in np.arange(1, L // 2 + 1):
        lamb += 2 / (x**alpha) * (np.cos(k[0] * x) + np.cos(k[1] * x))
    return 0.5 * (lamb + J_perp)


def epsilon_k(k, L, alpha, J_perp):
    eps = 0.0
    for x in np.arange(1, L // 2 + 1):
        eps += 4 / (x**alpha)
    return 0.5 * (eps + J_perp)


def epsilon_0(L, alpha, J_perp):
    return -1.5 * epsilon_k((0, 0), L, alpha, J_perp)


def energy(k, L, alpha, J_perp):
    eps0 = epsilon_0(L, alpha, J_perp)
    epsk = epsilon_k((0, 0), L, alpha, J_perp)
    lamb = lambda_k(k, L, alpha, J_perp)
    return np.sqrt(epsk**2 - lamb**2) + eps0

x = np.linspace(0,100,101)

i = np.where(x > 50)

print(x[i])