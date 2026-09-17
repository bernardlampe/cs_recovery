"""Hand-written iterative solvers for symmetric positive definite systems.

Conjugate gradient solves normal equations A^T A x = A^T y without
ever forming A^T A explicitly, and the augmented least-squares
solver inverts the Tikhonov-regularized normal equations used by
IRLS/FOCUSS-style reweighting.

References:
    J. R. Shewchuk, "An Introduction to the Conjugate Gradient Method
    Without the Painful Derivation," Carnegie Mellon University, 1994.

    J. Nocedal and S. J. Wright, "Numerical Optimization," 2nd ed.,
    Springer, 2006, ch. 5 (conjugate gradient methods).
"""

import numpy as np


def conjugate_gradient(matvec, b, max_iters, tol, x0=None):
    """
    conjugate gradient for symmetric positive definite systems
        matvec(x) = H x,  solve H x = b

    Never materializes H; only needs products with it. For LASSO/
    Bregman the usual choice is matvec = A^T(A x) on the transformed
    problem, but any SPD operator is legal.

    Parameters:
        matvec: `function computing H @ x`
        b: `right-hand side`
        max_iters: `iteration cap (default n: exact under exact arithmetic)`
        tol: `stop when residual norm <= tol * ||b||`
        x0: `initial guess (None -> zeros)`

    Returns:
        x_k: `conjugate gradient solution`
        hist: `history of residual norms ||b - H x||_2`
    """
    n = b.shape[0]
    x_k = np.zeros((n, 1)) if x0 is None else np.copy(x0)
    r_k = b - matvec(x_k)
    p_k = np.copy(r_k)
    hist = [np.linalg.norm(r_k)]

    for _ in range(min(max_iters, n)):
        rr = np.dot(r_k.T, r_k)
        if rr <= (tol * np.linalg.norm(b)) ** 2:
            break
        alpha = rr / np.dot(p_k.T, matvec(p_k))
        x_k = x_k + alpha * p_k
        r_k = r_k - alpha * matvec(p_k)
        hist.append(np.linalg.norm(r_k))
        p_k = r_k + (np.dot(r_k.T, r_k) / rr) * p_k

    return x_k, hist


def lstsq_normal_equations(A, y, delta=1e-8, use_cg=False, max_iters=None, tol=1e-10):
    """
    solve min_x ||A x - y||_2^2 + delta ||x||_2^2 (Tikhonov)

    The regularized normal equations
        (A^T A + delta I) x = A^T y
    are what reweighted schemes (IRLS, FOCUSS) need at every
    iteration. Delta keeps the system invertible when the support
    matrix becomes rank deficient. Direct solve by default, CG
    optionally to show the matrix-free path.

    Parameters:
        A: `sampling matrix`
        y: `compressed samples`
        delta: `ridge regularization delta`
        use_cg: `solve iteratively with conjugate gradient instead`
        max_iters: `CG iteration cap`
        tol: `CG relative tolerance (direct solve: regularization floor)`

    Returns:
        x: `regularized least-squares solution`
    """
    AtA = np.dot(A.T, A) + delta * np.eye(A.shape[1])
    Aty = np.dot(A.T, y)
    if use_cg:
        x, _ = conjugate_gradient(lambda v: np.dot(AtA, v), Aty,
                                  max_iters or A.shape[1], tol)
    else:
        x = np.linalg.solve(AtA, Aty)
    return x
