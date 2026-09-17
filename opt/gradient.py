"""Hand-written first-order gradient solvers.

Includes plain projected/accelerated gradient descent for smooth
least squares, Armijo backtracking line search, and FISTA with an
optional proximal operator (making it the ISTA/FISTA family, the
workhorses behind LASSO reconstruction).

References:
    A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
    Algorithm for Linear Inverse Problems," SIAM Journal on Imaging
    Sciences, vol. 2, no. 1, pp. 183-202, 2009.

    J. Nocedal and S. J. Wright, "Numerical Optimization," 2nd ed.,
    Springer, 2006, ch. 3 (line search methods).
"""

import numpy as np


def lipschitz_step(A, step=None):
    """
    default step size: 1 / (spectral norm of A)^2 when not supplied

    Parameters:
        A: `sampling matrix`
        step: `user supplied step size or None`

    Returns:
        h: `gradient step size`
    """
    if step is None:
        h = 1.0 / (np.linalg.norm(A, 2) ** 2)
    else:
        h = step
    return h


def armijo_backtracking(step0, f, grad, x, rho=0.5, c=1e-4, max_backtracks=50):
    """
    Armijo backtracking line search: shrink step until f decreases enough

    Sufficient-decrease rule: f(x - h*g) <= f(x) - c*h*||g||_2^2.
    Starting from step0, halve the step until the rule accepts.

    Parameters:
        step0: `initial trial step size`
        f: `objective function taking one argument (x)`
        grad: `gradient at x`
        x: `current point`
        rho: `step shrink factor`
        c: `Armijo sufficient-decrease constant`
        max_backtracks: `halving iteration cap`

    Returns:
        h: `accepted step size`
    """
    h = step0
    fx = f(x)
    gg = np.linalg.norm(grad) ** 2
    for _ in range(max_backtracks):
        if f(x - h * grad) <= fx - c * h * gg:
            break
        h *= rho
    return h


def gradient_descent(A, y, max_iters, tol, step=None, x0=None):
    """
    plain gradient descent on the least-squares objective
        f(x) = 0.5 * ||A x - y||_2^2     grad f(x) = A^T(A x - y)

    The unconstrained foundation the proximal variants build on.

    Parameters:
        A: `sampling matrix`
        y: `compressed samples`
        max_iters: `iteration cap`
        tol: `stop when relative solution change <= tol`
        step: `step size (None -> 1/Lipschitz)`
        x0: `initial guess (None -> zeros)`

    Returns:
        x_k: `iterated solution`
        hist: `history of residual norms ||A x - y||_2`
    """
    h = lipschitz_step(A, step)
    x_k = np.zeros((A.shape[1], 1)) if x0 is None else np.copy(x0)
    hist = []

    for _ in range(max_iters):
        r = y - np.dot(A, x_k)
        hist.append(np.linalg.norm(r))
        x_new = x_k + h * np.dot(A.T, r)
        if np.linalg.norm(x_new - x_k) <= tol * max(np.linalg.norm(x_k), 1e-12):
            x_k = x_new
            break
        x_k = x_new

    return x_k, hist


def fista(A, y, lam, prox, max_iters, tol, step=None, x0=None):
    """
    FISTA accelerated proximal gradient for
        min_x 0.5*||A x - y||_2^2 + lam*R(x)

    ISTA with a Nesterov momentum term; with prox = soft threshold
    this solves LASSO, with hard threshold it solves l0 variants.
    Pass prox=None to get plain accelerated gradient descent.

    Parameters:
        A: `sampling matrix`
        y: `compressed samples`
        lam: `regularization weight passed to prox`
        prox: `proximal operator prox(v, lam*t) or None`
        max_iters: `iteration cap`
        tol: `stop when relative solution change <= tol`
        step: `step size (None -> 1/Lipschitz)`
        x0: `initial guess (None -> zeros)`

    Returns:
        x_k: `iterated solution`
        hist: `history of residual norms ||A x - y||_2`
    """
    h = lipschitz_step(A, step)
    x_k = np.zeros((A.shape[1], 1)) if x0 is None else np.copy(x0)
    z_k = np.copy(x_k)             # momentum point
    t_k = 1.0                      # Nesterov momentum weight
    hist = []

    for _ in range(max_iters):
        r = y - np.dot(A, z_k)
        hist.append(np.linalg.norm(r))
        x_new = z_k + h * np.dot(A.T, r)
        if prox is not None:
            x_new = prox(x_new, lam * h)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        z_k = x_new + ((t_k - 1.0) / t_new) * (x_new - x_k)

        if np.linalg.norm(x_new - x_k) <= tol * max(np.linalg.norm(x_k), 1e-12):
            x_k = x_new
            break
        x_k, t_k = x_new, t_new

    return x_k, hist
