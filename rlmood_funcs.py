import os, sys, itertools, tqdm, logging, warnings, time, math, re, json, math, csv, rpy2, xlsxwriter, random
RST, RED, GRN, BLU, YLW = "\x1b[0m", "\x1b[31m", "\x1b[32m", "\x1b[34m", "\x1b[33m"
NORM, BOLD, ITAL, SUBR, MIXF = "\033[0m", "\u001b[1m", "\033[3m", "\033[4m", "\033[1;3m"
TEST = '\033[48;2;7;23;12;242m'
import pandas as pd
import numpy as np
from datetime import datetime
from os import listdir
from os.path import join
from collections import OrderedDict as odict
from functools import reduce

from scipy import stats
import scipy.optimize as optimize
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind, ttest_ind_from_stats

warnings.simplefilter("ignore")
if not sys.warnoptions: os.environ["PYTHONWARNINGS"] = "ignore" 
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import display, HTML
pd.options.mode.chained_assignment = None

import pingouin as pg
from pingouin import ttest
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp
from statannotations.Annotator import Annotator

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties

def recode(df, var, keep=True):
    df['_'+var] = df[var]
    lvls = np.sort(df[var].unique())
    vals = {v: k for k, v in enumerate(lvls)}
    df = df.replace({var: vals})
    df[var] = df[var].astype(int)
    if not keep: del df['_'+var]
    return df

def var_position(df, varname, position):
    cols = df.columns.tolist()
    cols.insert(position, cols.pop(cols.index(varname)))
    df = df[cols]
    return df

#===============================================================
# RL UTILS
#===============================================================

def sigmoid(beta, dQ):
    p = 1/(1+ np.exp(-beta*dQ))
    cp = np.array([1-p, p])
    return cp

def softmax(beta, Qst):
    p_a_t = np.exp(Qst * beta) / np.sum( np.exp(Qst * beta))
    return p_a_t

def trial_grid(s):
    ntrial, nstate = len(s), len(np.unique(s))
    T = np.zeros((nstate),dtype=int)-1
    grid = np.zeros((ntrial, nstate),dtype=int)
    for t in range(ntrial):
            T[s[t]] = T[s[t]] + 1
            grid[t] = T
    return grid

def get_matrix_bias(S):
    qbias = 0.0
    T = trial_grid(S)
    Nt, Ns = len(S), len(np.unique(S))
    P, H = np.zeros((Nt), dtype=float), np.zeros((Nt+1))
    Q = np.zeros((Ns, 2, int(Nt/Ns)+1), dtype=float)
    return T, Q, P, H

#===============================================================
# RL MODEL
#===============================================================
def model_rw(parms, S, A, R, Q0=None):
    # Rescorla-Wagner Model (RW) (Rescorla, 1972; Sutton & Barto, 2018)
    beta, alpha = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        delta = r_t - Q[s_t, a_t, t]
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

def model_rw2a(parms, S, A, R, Q0=None):
    # Rescorla-Wagner Asymetrical Model (RW2A) (Lefebvre et al., 2017)
    beta, alphap, alphan = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        delta = r_t - Q[s_t, a_t, t]
        alpha = alphap if delta >= 0 else alphan
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

#===============================================================
# RL MODEL GLOBAL REWARD
# ===============================================================
def model_grt(parms, S, A, R, Q0=None):
    # Global Reward Trace Model (GRT) (Wittmann et al., 2020)
    beta, alpha, eta, gamma = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        rp_t = r_t + gamma*H[i]
        delta = rp_t - Q[s_t, a_t, t]
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        H[i+1] = H[i] + eta*(r_t - H[i])
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

def model_arr(parms, S, A, R, Q0=None):
    # Average Reward Rate Model (ARR) (Aberg et al., 2020)
    beta, alpha, eta = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        rp_t = r_t + H[i]
        delta = rp_t - Q[s_t, a_t, t]
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        H[i+1] = H[i] + eta*(r_t - H[i])
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

# =========================================
# MOMENTUM GLOBAL ESTABILITY
# =========================================
def model_pem(parms, S, A, R, Q0=None):
    # Prediction-Error Momentum Model (PEM)
    beta, alpha, eta, gamma = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        rp_t = r_t + gamma*H[i]
        delta = rp_t - Q[s_t, a_t, t]
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        H[i+1] = H[i] + eta*(r_t - Q[s_t, a_t, t] - H[i])
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

# =========================================
# MOMENTUM GLOBAL INESTABILITY
# =========================================
def model_peme(parms, S, A, R, Q0=None):
    # Prediction-Error Momentum Exponential Model (PEMe) (Eldar & Niv, 2015)
    beta, alpha, eta, gamma = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        m = np.tanh(H[i])
        rp_t = r_t * np.power(gamma, m)
        delta = rp_t - Q[s_t, a_t, t]
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        H[i+1] = H[i] + eta*(rp_t - Q[s_t, a_t, t] - H[i])
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

def model_pema(parms, S, A, R, Q0=None):
    # Prediction-Error Momentum Aditive Model (PEMa) (Eldar & Niv, 2015)
    beta, alpha, eta, gamma = parms
    T, Q, P, H = get_matrix_bias(S)
    for i in range(len(S)):
        s_t, t = S[i], T[i, S[i]]
        dQ = Q[s_t, 1, t] - Q[s_t, 0, t]
        cp = sigmoid(beta, dQ)
        a_t, r_t = A[i], R[i]
        rp_t = r_t + gamma*H[i]
        delta = rp_t - Q[s_t, a_t, t]
        Q[s_t, a_t, t+1] = Q[s_t, a_t, t] + alpha*delta
        Q[s_t, 1-a_t, t+1] = Q[s_t, 1-a_t, t]
        H[i+1] = H[i] + eta*(rp_t - Q[s_t, a_t, t] - H[i])
        P[i] = cp[a_t]
    LL = -np.nansum(np.log(P))
    return LL

#===============================================================
# RL MODEL FIT
#===============================================================

def get_subject(data, ksub, stype=False):
    df = data.loc[(data.subj == ksub)].reset_index(drop=True)
    S = df.s if stype else df.ss
    S = S.astype(int).values
    A = df.a.astype(int).values
    R = df.r.astype(float).values
    return S, A, R, df

def set_computational_model(model_target):
    if model_target=='model_rw':
        # Rescorla-Wagner Model (RW) (Rescorla & Wagner, 1972; Sutton & Barto, 1998)
        bound, model = {"beta": [0.001, 10.0], "alpha": [0.0, 1.0], }, model_rw
    elif model_target=='model_rw2a':
        # Rescorla-Wagner Model (RW) (Rescorla & Wagner, 1972; Sutton & Barto, 1998)
        bound, model = {"beta": [0.001, 10.0], "alphap": [0.0, 1.0], "alphan": [0.0, 1.0],}, model_rw2a
    elif model_target=='model_pem':
        bound, model = {"beta": [0.001, 10.0], "alpha": [0.0, 1.0], "eta": [0.0, 1.0], "gamma": [0.001, 10.0], }, model_pem # PAPER ORIGINAL
    elif model_target=='model_pema':
        bound, model = {"beta": [0.001, 10.0], "alpha": [0.0, 1.0], "eta": [0.0, 1.0], "gamma": [0.001, 10.0], }, model_pema
    elif model_target=='model_peme':
        bound, model = {"beta": [0.001, 10.0], "alpha": [0.0, 1.0], "eta": [0.0, 1.0], "gamma": [0.001, 10.0], }, model_peme
    elif model_target=='model_arr':
        # Average Reward Rate Model (ARR) (Aberg et al., 2020)
        bound, model = {"beta": [0.001, 10.0], "alpha": [0.0, 1.0], "eta": [0.0, 1.0], }, model_arr
    elif model_target=='model_grt':
        # Global Reward Trace Model (GRT) (Wittmann et al., 2020)
        bound, model = {"beta": [0.001, 10.0], "alpha": [0.0, 1.0], "eta": [0.0, 1.0], "gamma": [-1.0, 1.0], }, model_grt
    return bound, model

def optimize_global_minima(args, fobj, bnds, seed, disp=False):
    parmsn = list(bnds.keys())
    LB = [v[0] for k,v in bnds.items()]
    UB = [v[1] for k,v in bnds.items()]
    bounds = optimize.Bounds(LB,UB)
    kargs = dict(workers = 1, mutation=(0.5, 1), recombination = 0.6, popsize = 20, seed=seed, init='random', disp=disp)
    opt = optimize.differential_evolution(func = fobj, args = args, bounds = bounds, **kargs,)
    parms = {k:opt.x[i] for i, k in enumerate(parmsn)}
    nll, Np, N = opt.fun, len(parms), len(args[0])
    bic = -2 * -nll + Np * np.log(N)
    aic = 2* Np + 2* nll
    output = {**parms}
    return output, (bic, aic, nll)


