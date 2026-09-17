#!/usr/bin/python

""" Basic FOCUSS algorithm

References:
    I. F. Gorodnitsky and B. D. Rao, "Sparse signal reconstruction from limited data using FOCUSS: a re-weighted minimum norm algorithm," in IEEE Transactions on Signal Processing, vol. 45, no. 3, pp. 600-616, March 1997, doi: 10.1109/78.558475.
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def focuss(y, A, itrs, lam=0.0):
    """
    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        itrs: `number of algorithm itrs`
        lam: `Tikhonov regularization of the pseudo-inverse (0 = plain
             FOCUSS; with noisy measurements a small lam > 0 keeps the
             re-weighted minimum norm from interpolating the noise)`

    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY

    HISTORY = []
    x_k = np.ones((A.shape[1], 1))  # init solution non-zero

    i = 0
    while i < itrs:
        W_pk = np.diagflat(x_k)             # Step 1: W_pk = diag(x_k-1)
        alpha = np.dot(A, W_pk)             # Step 2: q_k = (A*W_pk)^+y
        # regularized pinv: with lam > 0, q = alpha^T (alpha alpha^T +
        # lam I)^-1 y  (ridge-smoothed) instead of the plain alpha^+ y
        # that interpolates the (noisy) measurements exactly
        m = A.shape[0]
        Aa = np.dot(alpha, alpha.T)          # m x m Gram
        if lam > 0:
            q_k = np.dot(alpha.T, np.linalg.solve(
                Aa + lam * np.eye(m), y))
        else:
            q_k = np.dot(np.linalg.pinv(alpha), y)
        x_k = np.dot(W_pk, q_k)             # Step 3: x_k = W_pk * q_k
        HISTORY.append(float(np.linalg.norm(y - np.dot(A, x_k))))

        i+=1

    return x_k



if __name__ == "__main__":
    from common import *

    for db in [None, 20]:
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db, k=10, n=200, amps_l=-10, amps_h=10)

        x_h = focuss(y, A, 500)
        plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='focuss_sparsity')

