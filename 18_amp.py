#!/usr/bin/python

""" Approximate Message Passing (MAP-view) Reconstruction

AMP approximately represents loopy belief propagation on a dense
random linear model with two scalar recursions per iteration:
    x^t = eta(A^T z^{t-1} + x^{t-1}; tau_t)      (denoiser)
    z^t   = y - A x^t + (n/m) * <eta'> z^{t-1}   (Onsager term)
For a sparsity-promoting denoiser eta (soft threshold at the
MAP estimate of i.i.d. Bernoulli-Gaussian coefficients) AMP
reconstructs with state-evolution accuracy in O(mn) per iteration.

The Onsager correction term <eta'> is estimated empirically from
the denoiser sensitivity at the current iterate.

References:
    D. L. Donoho, A. Maleki and A. Montanari, "Message Passing
    Algorithms for Compressed Sensing," Proceedings of the National
    Academy of Sciences, vol. 106, no. 45, pp. 18914-18919, 2009.

    M. Bayati and A. Montanari, "The Dynamics of Message Passing on
    Dense Graphs, with Applications to Compressed Sensing," IEEE
    Transactions on Information Theory, vol. 57, no. 2, pp. 764-785,
    Feb. 2011.  (state evolution analysis)
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def amp(y, A, term, param, lam=1.0, max_iters=300, tol=1e-8):
    """
    approximate message passing reconstruction (soft-threshold eta,
    noise-adaptive threshold per Donoho-Maleki-Montanari)

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        lam: `threshold multiple of the noise-scale estimate`
        max_iters: `iteration cap`
        tol: `stop when residual change <= tol * ||y||`

    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY
    HISTORY = []
    m, n = A.shape
    delta = m / float(n)                 # measurement ratio
    x_k = np.zeros((n, 1))
    z_k = np.copy(y)                     # initial residual is z_0
    r_k = y - np.dot(A, x_k)

    i = 0
    while not term(param, y, r_k, x_k) and i < max_iters:
        w = np.dot(A.T, z_k) + x_k       # pre-denoiser input
        HISTORY.append(float(np.linalg.norm(y - np.dot(A, x_k))))
        tau = np.linalg.norm(z_k) / np.sqrt(m)     # noise-scale estimate
        thr = lam * tau                  # noise-adaptive threshold
        x_new, eta_deriv = _soft_eta(w, thr)

        # Onsager reaction term: sensitivity of eta divided by delta
        onsager = (eta_deriv / delta) * z_k
        z_new = y - np.dot(A, x_new) + onsager

        if np.linalg.norm(z_new - z_k) <= tol * np.linalg.norm(y):
            x_k, z_k, r_k = x_new, z_new, y - np.dot(A, x_new)
            break
        x_k, z_k = x_new, z_new
        r_k = y - np.dot(A, x_k)
        i += 1

    return x_k

def _soft_eta(w, lam):
    """
    MAP denoiser for i.i.d. sparse Gaussians: soft threshold.
    Returns (denoised output, empirical sensitivity eta').
    """
    out = np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)
    eta_deriv = float(np.mean(np.abs(w) > lam))
    return out, eta_deriv

# terminate when energy (L2 norm) of remaining residual falls below this energy
def epsilon_term(e, y, r, x_hat):
    return e > np.linalg.norm(r)

if __name__ == "__main__":
    from common import *

    (y, A, x_t, x_f) = gen_test_signal(snr_db=20, k=10, n=200, amps_l=-10, amps_h=10)

    x_h = amp(y, A, epsilon_term, 0.00001, lam=1.0)
    plt_error(x_h, x_f, 'amp, lam = 1.0', alg='amp_epsilon')
