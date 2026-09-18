#!/usr/bin/python

""" Adaptive Adaptive Direction Method (AADM) Reconstruction

AADM: an adaptive variant of ADMM tailored to compressed sensing.
Each outer iteration solves the split LP/LS pair as in plain ADMM
but adapts the penalty rho to keep the primal and dual residuals
within a band of each other (the standard ADMM balancing heuristic):

    if ||r_p|| > mu * ||r_d||:  rho <- beta * rho
    if ||r_d|| > mu * ||r_p||:  rho <- rho / beta

Adaptive penalty scaling typically halves the iteration count of
fixed-rho ADMM on sparse recovery problems.

References:
    S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein,
    "Distributed Optimization and Statistical Learning via the
    Alternating Direction Method of Multipliers," Foundations and
    Trends in Machine Learning, vol. 3, no. 1, pp. 1-122, 2011.
    (see section 3.4.1: varying the penalty parameter)

    M. Figueiredo and J. Bioucas-Dias, "Restoration of Poissonian
    Images Using Alternating Direction Optimization," IEEE
    Transactions on Image Processing, vol. 20, no. 10, pp. 2752-2766,
    2011.  (ADMM for sparse reconstruction)
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def aadm(y, A, term, param, lam=0.03, rho=1.0, max_iters=50000, tol=1e-11,
         mu=5.0, beta=1.5):
    """
    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        lam: `l1 regularization weight`
        rho: `initial split penalty`
        max_iters: `iteration cap`
        tol: `stop when primal split violation <= tol`
        mu: `residual balancing slack (typical 5-10)`
        beta: `penalty adaptation factor`

    Returns:
        x_hat: `reconstructed signal`
    """

    m, n = A.shape
    x_hat = np.zeros((n, 1))
    z_k = np.zeros((n, 1))
    u_k = np.zeros((n, 1))               # scaled dual variable
    s_k = np.zeros((n, 1))               # dual residual accumulator
    global HISTORY
    HISTORY = []
    r_k = np.copy(y)                     # measurement residual

    i = 0
    while not term(param, y, r_k, x_hat) and i < max_iters:
        # 1. x-update: prox of lam * ||.||_1 at (z + u), step lam/rho
        from opt import soft_threshold
        x_hat = soft_threshold(z_k - u_k, lam / rho)

        # 2. z-update: ridge LS pull toward (x + u) with data fidelity
        step = 1.0 / (np.linalg.norm(A, 2) ** 2 + rho)
        z_new = z_k + step * (np.dot(A.T, y - np.dot(A, z_k)) +
                              rho * (x_hat + u_k - z_k))

        # 3. residuals for adaptation
        r_p = np.linalg.norm(x_hat - z_new)          # primal residual
        r_d = rho * np.linalg.norm(z_new - z_k)      # dual residual (scaled)
        s_k = -rho * (z_new - z_k)

        # 4. adaptive penalty balancing (Boyd 3.4.1)
        if r_p > mu * r_d:
            rho *= beta
        elif r_d > mu * r_p:
            rho /= beta

        # 5. dual ascent and residual refresh
        u_k = u_k + x_hat - z_new
        z_k = z_new
        r_k = y - np.dot(A, z_k)
        HISTORY.append(float(np.linalg.norm(r_k)))

        if r_p <= tol:
            break
        i += 1

    return x_hat

# terminate when energy (L2 norm) of remaining residual falls below this energy
def epsilon_term(e, y, r, x_hat):
    return e > np.linalg.norm(r)

if __name__ == "__main__":
    from common import *

    (y, A, x_t, x_f) = gen_test_signal(snr_db=20, k=10, n=200, amps_l=-10, amps_h=10)

    # adaptive-penalty ADMM needs many iterations; cap so the demo ends
    x_h = aadm(y, A, epsilon_term, 0.00001, lam=0.03, max_iters=8000, tol=1e-9)
    plt_error(x_h, x_f, 'aadm, lam = 0.03', alg='aadm_epsilon')
