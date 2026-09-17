#!/usr/bin/python

""" Iterative Re-weighted Least Squares
References:

R. Chartrand and Wotao Yin, "Iteratively reweighted algorithms for compressive sensing," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 3869-3872, doi: 10.1109/ICASSP.2008.4518498.  https://ieeexplore.ieee.org/document/4518498

"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def irls(y, A, term, param, max_iters, delta):
    """
    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function`
        param: `termination parameter` { k: `sparsity` | p: `percent` | e: `epsilon` }
        max_iters: `maximum number of iterations`
        delta: `regularization delta`

    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY

    HISTORY = []
    x_k = np.ones((A.shape[1], 1))
    r = np.ones((A.shape[1], 1))
    i = 0
    while not term(param, y, r, x_k) and i < max_iters:
        HISTORY.append(float(np.linalg.norm(y - np.dot(A, x_k))))
        W_k = np.diagflat(2 / (np.abs(x_k) + delta))
        alpha = np.linalg.inv(delta * W_k + np.dot(A.T,A))
        x_k = np.dot(np.dot(alpha, A.T), y)
        r = x_k - r
        i += 1

    return x_k

# terminate with output signal has sparsity k
def sparsity_term(k, y, r, x_hat):
    return int(np.count_nonzero(x_hat)) == k

# terminate when output signal has p percentage of signal
def percent_term(p, y, r, x_hat):
    y_l2p = np.linalg.norm(y) * (1.0-p)
    return y_l2p > np.linalg.norm(r)

# terminate when energy (L2 norm) of remaining residual falls below this energy
def epsilon_term(e, y, r, x_hat):
    return e > np.linalg.norm(r)

if __name__ == "__main__":
    from common import *

    # IRLS params
    max_iters = 100  # max number of algorithm iterations
    delta = 0.0001   # regularization delta

    for db in [None, 20]:
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db)

        x_h = irls(y, A, sparsity_term, 20, max_iters, delta)
        plt_error(x_h, x_f, 'sparsity_term, k = 20')

        x_h = irls(y, A, percent_term, 0.99, max_iters, delta)
        plt_error(x_h, x_f, 'percent_term, p = 0.99')

        x_h = irls(y, A, epsilon_term, 0.00001, max_iters, delta)
        plt_error(x_h, x_f, 'epsilon_term, k = 0.00001')
