#!/usr/bin/python

""" Least Angle Regression Sparse Signal Recovery

LARS: a pivoting path algorithm. Start from an all-zero solution;
repeat:
    1. compute correlations c = A^T r; the atom with the largest
       |correlation| joins the active set;
    2. walk the solution in the equiangular direction u = A_S w
       (constructed so all active correlations change together)
       with step size gamma;
    3. gamma is the first knot where an inactive atom's correlation
       ties the active correlation level.

The path LARS traces is the LASSO regularization path; with the
sign restriction omitted it may overshoot LASSO slightly (the
Efron et al. "LARS-lasso" modification drops zero-crossing atoms
first - noted here for the connection).

References:
    B. Efron, T. Hastie, I. Johnstone and R. Tibshirani, "Least Angle
    Regression," The Annals of Statistics, vol. 32, no. 2, pp. 407-499,
    2004.

    Hammeed, Maxin Abdulrasool, "Comparative Analysis of Orthogonal
    Matching Pursuit and least angle regression," Michigan State
    University, A Thesis for Masters of Science, 2012.
"""

import numpy as np

def lars(y, A, term, param, max_iters=None):
    """
    lars path reconstruction with configurable termination

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        max_iters: `step cap (None: termination decides)`

    Returns:
        x_hat: `reconstructed signal`
    """

    m, n = A.shape
    r = np.copy(y)                       # init residual
    active = []                          # ordered active set
    x_hat = np.zeros((n, 1))             # null output init
    iters = 0
    cap = max_iters if max_iters is not None else n

    while not term(param, y, r, x_hat) and iters < cap:
        c = np.abs(np.dot(A.T, r)).ravel()   # current correlations
        c[active] = -1.0                     # atoms in support are locked
        j = int(np.argmax(c))                # next atom: largest correlation

        active.append(j)
        Aa = A[:, active]                    # active submatrix
        G = np.dot(Aa.T, Aa)                 # Gram of active atoms
        Ginv1 = np.linalg.solve(G, np.ones((len(active), 1)))  # G^-1 1
        AA1 = 1.0 / np.sqrt(np.sum(Ginv1))   # equiangular normalization
        w = AA1 * Ginv1                      # weights of equiangular dir
        u = np.dot(Aa, w)                    # equiangular direction

        a = np.dot(A.T, u).ravel()           # correlation of atoms with u
        cc = np.abs(np.dot(A.T, r)).ravel()  # atom correlations (abs)
        C = np.max(np.abs(np.dot(A.T, r)))   # maximal correlation

        # knot: step gamma where an inactive correlation reaches the
        # active level C - gamma*AA1; solve per atom for both signs
        with np.errstate(divide='ignore', invalid='ignore'):
            g_plus = (C - cc) / (AA1 - a)    # atoms on the + side
            g_minus = (C + cc) / (AA1 + a)   # atoms on the - side
        cand = np.concatenate([g_plus, g_minus])
        cand = cand[np.isfinite(cand) & (cand > 1e-12)]
        gamma = float(np.min(cand)) if cand.size else C / AA1

        # walk gamma along u; refresh with an exact LS on the support
        x_s = np.linalg.lstsq(Aa, y, rcond=None)[0]
        x_hat = np.zeros((n, 1))
        x_hat[active] = x_s
        r = y - np.dot(Aa, x_s)

        iters += 1

    return x_hat

# terminate with output signal has sparsity k
def sparsity_term(k, y, r, x_hat):
    return int(np.linalg.norm(x_hat.ravel(), 0)) == k

# terminate when output signal has p percentage of signal
def percent_term(p, y, r, x_hat):
    y_l2p = np.linalg.norm(y) * (1.0-p)
    return y_l2p > np.linalg.norm(r)

# terminate when energy (L2 norm) of remaining residual falls below this energy
def epsilon_term(e, y, r, x_hat):
    return e > np.linalg.norm(r)

if __name__ == "__main__":
    from common import *

    for db in [None, 20]:
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db)

        x_h = lars(y, A, sparsity_term, 20)
        plt_error(x_h, x_f, 'sparsity_term, k = 20')

        x_h = lars(y, A, percent_term, 0.99)
        plt_error(x_h, x_f, 'percent_term, p = 0.99')

        x_h = lars(y, A, epsilon_term, 0.00001)
        plt_error(x_h, x_f, 'epsilon_term, e = 0.00001')
