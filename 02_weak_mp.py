#!/usr/bin/python

""" Weak Matching Pursuit Sparse Signal Recovery

References:
    T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE
    Transactions on Signal Processing, vol. 56, no. 6, pp. 2370-2382,
    June 2008.  (defines the weak/gain selection framework)

    R. Gribonval and P. Vandergheynst, "On the convergence of matching
    pursuit," IEEE Transactions on Information Theory, vol. 52, no. 1,
    pp. 172-180, Jan. 2006.
"""

import numpy as np

def weak_mp(y, A, term, param, mu=0.5):
    """
    Instead of the arg-max atom of plain MP, weak MP accepts any
    atom with correlation at least mu times the peak correlation:
        |A_j^T r| >= mu * max_i |A_i^T r|
    Each accepted atom gets its full least-squares coefficient in
    the current support (gain action).

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        mu: `weak selection fraction (in (0,1], small = laxer)`

    Returns:
        x_hat: `reconstructed signal`
    """

    y = np.asarray(y).reshape(-1, 1)     # accept flat or column measurements
    r = np.copy(y)                       # init residual
    support = []                         # list of support loc inds
    mask = np.ones(A.shape[1], dtype=bool)  # atoms not yet in support
    x_hat = np.zeros((A.shape[1], 1))    # null output init

    while not term(param, y, r, x_hat):
        c = np.dot(A.T, r)               # correlation
        cl = np.abs(c).ravel()
        thr = mu * np.max(cl)            # weak acceptance threshold
        fresh = np.nonzero((cl >= thr) & mask)[0]  # weak atoms not yet used
        atom = int(fresh[0]) if fresh.size else \
            int(np.argmax(np.where(mask, cl, -np.inf)))  # fall back: best fresh
        mask[atom] = False
        support.append(atom)                         # grow support
        P = A[:, support]                            # compose the submatrix
        x_s = np.linalg.lstsq(P, y, rcond=None)[0]   # min ||y-P*x||_2
        x_hat[support] = x_s                         # embed coefficients
        r = y - np.dot(P, x_s)                       # update residual

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

    (y, A, x_t, x_f) = gen_test_signal(snr_db=20, k=10, n=200, amps_l=-10, amps_h=10)

    x_h = weak_mp(y, A, sparsity_term, 20, mu=0.9)
    plt_error(x_h, x_f, 'sparsity_term, k = 20, mu = 0.9', alg='weak_mp_sparsity')

    x_h = weak_mp(y, A, percent_term, 0.99, mu=0.9)
    plt_error(x_h, x_f, 'percent_term, p = 0.99, mu = 0.9', alg='weak_mp_percent')

    x_h = weak_mp(y, A, epsilon_term, 0.00001, mu=0.9)
    plt_error(x_h, x_f, 'epsilon_term, mu = 0.9, e = 0.00001', alg='weak_mp_epsilon')
