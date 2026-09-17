"""Hand-written Alternating Direction Method of Multipliers (ADMM).

Solves composite objectives by splitting variables and coordinating
via a dual (Lagrange multiplier) update. The canonical compressed
sensing split is:

    min_x  ||y - A z||_2^2 + lam * ||x||_1   subject to  z = x

The z-update is a least-squares problem, the x-update is the l1
prox (soft threshold), and the multiplier ascends on the split
violation. Also used by AADM with a second constraint block.

References:
    S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein,
    "Distributed Optimization and Statistical Learning via the
    Alternating Direction Method of Multipliers," Foundations and
    Trends in Machine Learning, vol. 3, no. 1, pp. 1-122, 2011.
"""

import numpy as np


def admm_minimize(A, y, prox, lam=0.1, rho=1.0, max_iters=5000, tol=1e-8):
    """
    two-block ADMM for min_x ||y - A x||_2^2 + lam * R(x)

    With the identity splitting z = x (see module docstring), each
    iteration alternates:
        1. z-update:  min_z ||y - A z||_2^2 + rho/2 ||z - (x+u)||_2^2
                      solved exactly via its normal equations
                      (A^T A + rho I) z = A^T y + rho (x + u)
        2. x-update:  prox of the regularizer at (z + u)
                      x = prox_R^{lam/rho}(z - u)
        3. u-update:  scaled dual ascent u <- u + x - z

    Parameters:
        A: `sampling matrix`
        y: `compressed samples`
        prox: `proximal operator prox(v, t), e.g. opt.l1_prox partial`
        lam: `regularizer weight of R`
        rho: `split penalty parameter`
        max_iters: `iteration cap`
        tol: `stop when primal split violation <= tol`

    Returns:
        x_k: `reconstructed signal (the prox-regularized variable)`
        hist: `history of primal residuals ||z_k - x_k||_2`
    """
    m, n = A.shape

    # factor the z-update normal equations once
    Q = np.dot(A.T, A) + rho * np.eye(n)
    Qinv_Aty = np.linalg.solve(Q, np.dot(A.T, y))
    Qinv = np.linalg.inv(Q)

    x_k = np.zeros((n, 1))
    z_k = np.zeros((n, 1))
    u_k = np.zeros((n, 1))                 # scaled dual variable
    hist = []

    for _ in range(max_iters):
        # 1. z-update: exact solve of the ridge-augmented normal equations
        z_k = Qinv_Aty + rho * np.dot(Qinv, x_k + u_k)

        # 2. x-update: prox of the regularizer
        x_new = prox(z_k - u_k, lam / rho)

        # 3. dual ascent (scaled form)
        u_k = u_k + x_new - z_k

        hist.append(np.linalg.norm(x_new - z_k))
        x_k = x_new

        if hist[-1] <= tol:
            break

    return x_k, hist
