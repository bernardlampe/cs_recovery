#!/usr/bin/python

""" LASSO Reconstruction via FISTA

The LASSO formulation of compressed sensing:
    min_x  0.5 * ||A x - y||_2^2 + lambda * ||x||_1
is solved here with FISTA (accelerated proximal gradient) using the
hand-written soft-thresholding prox. lambda controls the sparsity
of the recovered solution; at the noiseless sweet spot ~ sqrt(2 log n)
times the noise scale the LASSO estimate coincides with basis pursuit.

References:
    R. Tibshirani, "Regression Shrinkage and Selection via the Lasso,"
    Journal of the Royal Statistical Society B, vol. 58, no. 1,
    pp. 267-288, 1996.

    A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems," SIAM Journal on Imaging
    Sciences, vol. 2, no. 1, pp. 183-202, 2009.
"""

import numpy as np


# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def lasso(y, A, term, param, lam=1e-5, max_iters=50000, tol=1e-12):
    """
    LASSO reconstruction via FISTA with soft thresholding

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        lam: `l1 regularization weight`
        max_iters: `FISTA iteration cap`
        tol: `relative change stopping tolerance`

    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY

    from opt import fista, soft_threshold
    x_k, hist = fista(A, y, lam, soft_threshold, max_iters, tol)
    x_hat = x_k
    HISTORY = list(hist)
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

        x_h = lasso(y, A, epsilon_term, 0.00001, lam=1e-5)
        plt_error(x_h, x_f, 'lasso, lam = 0.1')
