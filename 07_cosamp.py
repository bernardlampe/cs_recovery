#!/usr/bin/python

"""
CoSaMP algorithm for sparse signal reconstruction.

References:
    D. Needell and J. A. Tropp, "CoSaMP: Iterative signal recovery
    from incomplete and inaccurate samples," Applied and Computational
    Harmonic Analysis, vol. 26, no. 3, pp. 301-321, 2009.
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None
_prev_support = None                 # prior iteration's pruned support

def cosamp(A, y, k, term, param, max_iters=100):
    """
    CoSaMP algorithm for sparse signal reconstruction.

    Parameters
    ----------
    A : ndarray (m, n), Measurement matrix.
    y : ndarray (m,), Measurement vector.
    k : int, Maximum sparsity
    term: `termination function from above`
    param: { k: `sparsity` | p: `percent` | e: `epsilon` }

    Returns
    -------
    x_hat : ndarray (n,), Reconstructed sparse signal.
    """
    global HISTORY, _prev_support
    HISTORY = []
    _prev_support = None
    y = np.ravel(y)         # accept (m,1) or (m,) measurements
    m, n = A.shape
    r = y.copy()
    x_hat = np.zeros(n)

    while not term(param, y, r, x_hat) and len(HISTORY) < max_iters:
        if HISTORY is not None:
            HISTORY.append(float(np.linalg.norm(r)))
        # Step 1: Proxy signal
        proxy = A.T @ r

        # Step 2: Identify 2k largest components
        omega = np.argsort(np.abs(proxy))[-2*k:]

        # Step 3: Merge supports
        support = np.union1d(np.nonzero(x_hat)[0], omega)

        # Step 4: Solve least squares on merged support
        A_support = A[:, support]
        b_support, _, _, _ = np.linalg.lstsq(A_support, y.ravel(), rcond=None)

        # Step 5: Prune to best k entries
        idx = np.argsort(np.abs(b_support))[-k:]
        support = support[idx]
        x_temp = np.zeros(n)
        x_temp[support] = b_support[idx].ravel()

        # Step 6: Update r
        r = y - A @ x_temp

        # Step 7: Update x_hat estimate (and stabilization bookkeeping)
        _prev_support = frozenset(np.nonzero(x_hat)[0].tolist())
        x_hat = x_temp

    return x_hat

# terminate with output signal support stabilized at sparsity k
def sparsity_term(k, y, r, x_hat):
    global _prev_support
    # CoSaMP emits exactly k nonzeros after the first iteration, so a
    # nonzero-count test can never gate on k (Needell-Tropp: fixed-support
    # iterations stop when the pruned support stops changing).
    sup = frozenset(np.nonzero(x_hat)[0].tolist())
    ok = (len(sup) == k) and (_prev_support == sup)
    _prev_support = sup
    return ok

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

    x_h = cosamp(A, y, 20, sparsity_term, 20)
    plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='cosamp_sparsity')

    x_h = cosamp(A, y, 20, percent_term, 0.99)
    plt_error(x_h, x_f, 'percent_term, p = 0.99', alg='cosamp_percent')

    x_h = cosamp(A, y, 20, epsilon_term, 0.00001)
    plt_error(x_h, x_f, 'epsilon_term, k = 0.00001', alg='cosamp_epsilon')
