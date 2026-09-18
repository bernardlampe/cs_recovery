#!/usr/bin/python

""" Bregman Iterative Reconstruction

Bregman iteration solves the l1 regularized least squares
    min_x  ||A x - y||_2^2 + lam * ||x||_1
by repeatedly solving plain least squares and adding back the
residual - the "Bregman reload" trick:
    y_{i+1} = y + (y - A x_i)
    x_{i+1} = argmin ||A x - y_{i+1}||_2^2 + lam ||x||_1

Each inner solve is one proximal gradient (ISTA) sweep; the outer
loop converges to the same fixed point as basis pursuit - but from
the least-squares side, which is why it handles noisy coupling and
preserves signal energy better than one-shot l1 minimization.

References:
    S. Osher, M. Burger, D. Goldfarb, J. Xu and W. Yin, "An Iterative
    Regularization Method for Total Variation-Based Image
    Restoration," Multiscale Modeling and Simulation, vol. 4, no. 2,
    pp. 460-489, 2005.

    W. Yin, S. Osher, D. Goldfarb and J. Darbon, "Bregman Iterative
    Algorithms for l1-Minimization with Applications to Compressive
    Sensing," SIAM Journal on Imaging Sciences, vol. 1, no. 1,
    pp. 143-168, 2008.
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def bregman(y, A, term, param, lam=0.01, outer=100, inner=30, tol=1e-12):
    """
    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        lam: `l1 regularization weight`
        outer: `number of Bregman reloads`
        inner: `ISTA sweeps per reload`
        tol: `inner FISTA stopping tolerance`
    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY
    HISTORY = []
    from opt import fista, soft_threshold

    m, n = A.shape
    x_hat = np.zeros((n, 1))
    y_b = np.copy(y)                     # Bregman virtual measurement

    for j in range(outer):
        res_norm = float(np.linalg.norm(y - np.dot(A, x_hat)))
        HISTORY.append(res_norm)
        if term(param, y, y - np.dot(A, x_hat), x_hat):
            break
        x_k, _ = fista(A, y_b, lam, soft_threshold, inner, tol, x0=x_hat)
        x_hat = x_k
        y_b = y_b + (y - np.dot(A, x_hat))   # Bregman residual reload

    return x_hat

# terminate when energy (L2 norm) of remaining residual falls below this energy
def epsilon_term(e, y, r, x_hat):
    return e > np.linalg.norm(r)

if __name__ == "__main__":
    from common import *

    (y, A, x_t, x_f) = gen_test_signal(snr_db=20, k=10, n=200, amps_l=-10, amps_h=10)

    x_h = bregman(y, A, epsilon_term, 0.00001, lam=0.01)
    plt_error(x_h, x_f, 'bregman, lam = 0.1', alg='bregman_epsilon')
