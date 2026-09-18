#!/usr/bin/python

""" Gradient Pursuit Sparse Signal Recovery

References:
    T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE
    Transactions on Signal Processing, vol. 56, no. 6, pp. 2370-2382,
    June 2008.
"""

import numpy as np

def gradient_pursuit(y, A, term, param, cg_iters=10, max_iters=200):
    """
    Blumensath-Davies Gradient Pursuit: after greedily detecting a
    support, update the coefficients with a conjugate-gradient descent
    step on that support instead of an exact least-squares solve. One
    CG sweep per iteration keeps the per-iteration cost at the matching
    pursuit level while converging far faster than steepest descent.

    Each iteration:
        1. detect the atom most correlated with the residual
        2. add it to the support
        3. run a few conjugate-gradient (CG) steps for the LS
           solution restricted to the support (Polak-Ribiere)

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        cg_iters: `CG sweeps per iteration (1 = gradient pursuit)`
        max_iters: `hard cap on support size (noisy measurements)`

    Returns:
        x_hat: `reconstructed signal`
    """

    r = np.copy(y)                       # init residual
    lamda = []                           # list of support loc inds
    mask = np.ones(A.shape[1], dtype=bool)  # atoms not yet in support
    x = None                             # support coefficients (len grows)

    def _probe():
        # current estimate for the termination probe
        v = np.zeros((A.shape[1], 1))
        if x is not None:
            v[np.asarray(lamda, dtype=int)] = x
        return v

    while not term(param, y, r, _probe()) and len(lamda) < max_iters:
        c = np.dot(A.T, r)               # correlation
        cl = np.abs(c).ravel()
        ind = int(np.argmax(np.where(mask, cl, -np.inf)))  # best fresh atom
        mask[ind] = False

        lamda.append(ind)                          # update support
        P = A[:, lamda]                            # compose the submatrix

        # conjugate gradient on the support: min ||y - P x||
        x = _cg_on_support(P, y, cg_iters, x)

        x_hat_full = embed(x, lamda, A.shape[1])
        r = y - np.dot(A, x_hat_full)              # update residual

    # embed coefficients in support
    n = A.shape[1]
    x_hat = np.zeros((n, 1))
    x_hat[lamda] = x

    return x_hat


def embed(x, lamda, n):
    """
    scatter support coefficients into an n-vector
    """
    v = np.zeros((n, 1))
    v[np.asarray(lamda, dtype=int)] = x.reshape(-1, 1)
    return v

def _cg_on_support(P, y, iters, x0):
    """
    Polak-Ribiere conjugate gradient restricted to the support

    Solves min_x ||y - P x||_2 whose gradient is P^T(P x - y).
    Seeded with the previous solution, three sweeps suffice in
    practice to nearly exhaust the LS accuracy for small supports.
    """
    H = lambda v: np.dot(P.T, np.dot(P, v))    # support normal operator
    b = np.dot(P.T, y)
    x = np.zeros((P.shape[1], 1))
    if x0 is not None:
        x[:x0.shape[0]] = x0      # warm start: keep previous coefficients
    r_k = b - H(x)                       # CG residual on support
    p_k = np.copy(r_k)
    rr = np.dot(r_k.T, r_k)
    for _ in range(iters):
        alpha = rr / (np.dot(p_k.T, H(p_k)) + 1e-15)
        x = x + alpha * p_k
        r_new = r_k - alpha * H(p_k)
        rrn = np.dot(r_new.T, r_new)
        beta = ((rrn - r_new.T @ r_k) / (rr + 1e-15)).item()  # Polak-Ribiere
        beta = max(beta, 0.0)     # PR+ safeguard keeps p_k a descent dir
        p_k = r_new + beta * p_k
        r_k, rr = r_new, rrn
    return x

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

    x_h = gradient_pursuit(y, A, sparsity_term, 20)
    plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='gradient_pursuit_sparsity')

    x_h = gradient_pursuit(y, A, percent_term, 0.99)
    plt_error(x_h, x_f, 'percent_term, p = 0.99', alg='gradient_pursuit_percent')

    x_h = gradient_pursuit(y, A, epsilon_term, 0.00001)
    plt_error(x_h, x_f, 'epsilon_term, e = 0.00001', alg='gradient_pursuit_epsilon')
