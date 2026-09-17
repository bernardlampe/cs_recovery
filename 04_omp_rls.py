import numpy as np


def omp_rls(A, y, term, param, lam=1.0, delta=1e-3, max_iters=None):
    """
    Orthogonal Matching Pursuit with Recursive Least Squares update.

    References:
        D. Zachariah, S. Chatterjee and M. Jansson, "Online Network
        Response Identification with Recursive Least Squares,"
        Signal Processing, vol. 103, pp. 237-246, 2014.
        (RLS support-extension viewpoint)

        Hammeed, Maxin Abdulrasool, "Comparative Analysis of Orthogonal
        Matching Pursuit and least angle regression," Michigan State
        University, A Thesis for Masters of Science, 2012.
        (OMP termination conventions)

    Parameters
    ----------
    A : ndarray (m, n), Measurement matrix.
    y : ndarray (m,), Measurement vector.
    term: `termination function from above`
    param: { k: `sparsity` | p: `percent` | e: `epsilon` }
    lam : float, RLS forgetting factor (default = 1.0, no forgetting).
    delta : float, Small regularization for RLS initialization.
    max_iters : int or None, hard cap on support size (None = n).
        Bounded loops keep noisy-measurement termination robust.

    Returns
    -------
    x_hat : ndarray (n,)
        Reconstructed sparse vector.
    """
    m, n = A.shape
    r = y.copy()
    support = []
    x_hat = np.zeros(n)

    # Initial inverse correlation matrix
    P = (1.0 / delta) * np.eye(m)
    A_support = np.zeros((m, 0))

    while not term(param, y, r, x_hat) and len(support) < (max_iters or n):
        # 1. Find index with maximum correlation
        correlations = A.T @ r
        idx = np.argmax(np.abs(correlations))
        support.append(idx)

        # 2. Update active submatrix
        A_support = A[:, support]

        # 3. Recursive Least Squares update
        # Compute gain vector
        a_new = A[:, idx].reshape(-1, 1)
        g = (P @ a_new) / (lam + a_new.T @ P @ a_new)

        # Update inverse correlation matrix
        P = (1 / lam) * (P - g @ a_new.T @ P)

        # Compute new coefficients via RLS (solves LS incrementally for supported atoms)
        x_support = np.linalg.pinv(A_support) @ y

        # 4. Update r
        r = y - A_support @ x_support

        # 5. reconstruction
        x_hat[support] = x_support.flatten()

    return x_hat

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

    for db in [None, 20]:
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db, k=10, n=200, amps_l=-10, amps_h=10)

        x_h = omp_rls(A, y, sparsity_term, 20)
        plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='omp_rls_sparsity')

        x_h = omp_rls(A, y, percent_term, 0.99)
        plt_error(x_h, x_f, 'percent_term, p = 0.9999', alg='omp_rls_percent')

        x_h = omp_rls(A, y, epsilon_term, 0.00001)
        plt_error(x_h, x_f, 'epsilon_term, k = 0.00001', alg='omp_rls_epsilon')
