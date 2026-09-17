"""Proximal operators and thresholding primitives.

These are the building blocks of first-order sparse recovery schemes
(ISTA/FISTA, ADMM, Bregman iteration). Each operator is the proximal
map of a term appearing in a regularized least-squares objective:

    min_x  0.5 * ||A x - y||_2^2  +  lambda * R(x)

where R is e.g. the l1 norm (soft threshold), the l0 pseudo-norm
(hard threshold), or an l2-ball constraint (projection).

References:
    N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
    Trends in Optimization, vol. 1, no. 3, pp. 127-239, 2014.
"""

import numpy as np


def soft_threshold(v, t):
    """
    element-wise soft threshold, the prox of the l1 norm

    A.K.A. shrinkage. Values with |v| <= t map to 0; survivors
    shrink toward 0 by exactly t. This is the proximal operator of
    lambda * ||x||_1 evaluated with lambda = t.

    Parameters:
        v: `input vector or matrix`
        t: `threshold (the effective lambda)`

    Returns:
        prox1: `soft-thresholded copy of v`
    """
    prox1 = np.sign(v) * np.maximum(np.abs(v) - t, 0.0)
    return prox1


def l1_prox(v, t, lam=1.0):
    """
    prox of lam * ||x||_1 (thin convenience wrapper over soft_threshold)

    Parameters:
        v: `input vector or matrix`
        t: `scaled step size`
        lam: `l1 weight`

    Returns:
        prox1: `proxial map applied to v`
    """
    prox1 = soft_threshold(v, lam * t)
    return prox1


def hard_threshold(v, k):
    """
    keep k largest-magnitude entries of v, zero the rest (prox of l0)

    Used by iterative hard thresholding (IHT) and the support
    detection/pruning steps of greedy solvers.

    Parameters:
        v: `input vector`
        k: `target sparsity`

    Returns:
        v_h: `copy of v with only its k largest magnitudes`
    """
    v_h = np.zeros_like(v)
    inds = np.argpartition(np.abs(v).ravel(), -k)[-k:]
    v_h.ravel()[inds] = v.ravel()[inds]
    return v_h


def project_l2_ball(v, radius):
    """
    Euclidean projection of v onto the l2 ball { ||x||_2 <= radius }

    Prox of the indicator of a distance constraint. Constant-scale
    vectors ({||x|| <= r} vs ||v|| <= r) pass through unchanged.

    Parameters:
        v: `input vector`
        radius: `ball radius`

    Returns:
        v_b: `projected vector'
    """
    nrm = np.linalg.norm(v)
    if nrm <= radius:
        v_b = v
    else:
        v_b = (radius / nrm) * v
    return v_b
