#!/usr/bin/python

""" Iterative Hard Thresholding Sparse Signal Recovery

Blumensath-Davies Iterative Hard Thresholding (Norm-IHT / HTP
variant): gradient step on the data-fidelity term followed by
projection onto the set of k-sparse signals (keep k largest).

    x_{i+1} = H_k( x_i + tau * A^T (y - A x_i) )

References:
    T. Blumensath and M. E. Davies, "Iterative Hard Thresholding for
    Compressed Sensing," Applied and Computational Harmonic Analysis,
    vol. 27, no. 3, pp. 265-274, 2009.

    T. Blumensath and M. E. Davies, "Normalized Iterative Hard
    Thresholding: Guaranteed Stability and Performance," IEEE Journal
    of Selected Topics in Signal Processing, vol. 4, no. 2, pp. 298-310,
    2010.
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def iht(y, A, term, param, k, max_iters=300, mu=None):
    """
    iterative hard thresholding with configurable termination criteria

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        k: `sparsity level for the hard threshold`
        max_iters: `maximum iteration count`
        mu: `step size (None -> 3/||A||_2^2 normalized step)`

    Returns:
        x_hat: `reconstructed signal`
    """

    # normalized step (Blumensath-Davies): mu = tau/||A||_2^2, tau ~ (1,3] safe
    if mu is None:
        mu = 3.0 / (np.linalg.norm(A, 2) ** 2)

    global HISTORY
    HISTORY = []

    x = np.zeros((A.shape[1], 1))

    for i in range(max_iters):
        r = y - np.dot(A, x)                   # data fidelity residual
        HISTORY.append(float(np.linalg.norm(r)))
        z = x + mu * np.dot(A.T, r)            # gradient step
        x_new = _hard_threshold(z, k)          # keep k largest

        if np.linalg.norm(x_new - x) < 1e-12:  # converged support
            x = x_new
            break
        x = x_new

    x_hat = x
    return x_hat

def _hard_threshold(z, k):
    """
    project z onto the set of k-sparse vectors (keep k largest |.|)
    """
    out = np.zeros_like(z)
    inds = np.argpartition(np.abs(z).ravel(), -k)[-k:]
    out.ravel()[inds] = z.ravel()[inds]
    return out

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
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db, k=10, n=200, amps_l=-10, amps_h=10)

        x_h = iht(y, A, epsilon_term, 0.00001, 20)
        plt_error(x_h, x_f, 'epsilon_term, k = 20, e = 0.00001', alg='iht_epsilon')
