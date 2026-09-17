"""Hand-written dense primal simplex for basis pursuit.

Basis pursuit (BP) recovers sparse signals by solving
    min ||x||_1  subject to  A x = y
exactly. Splitting x = u - v with u, v >= 0 turns it into the
standard-form linear program
    min c^T w  s.t.  M w = y,  w >= 0,   c = 1,
with M = [A, -A], w = [u; v]. A textbook two-phase dense simplex
(the Big-M free variant: phase 1 finds a feasible basis with
artificial variables, phase 2 drives the cost to minimal) solves
it. Every numerical step here is explicit NumPy.

References:
    T. S. Ferguson, "Linear Programming: A Concise Introduction,"
    UCLA course notes, 2006, ch. 3 and 4 (simplex and two-phase).

    S. J. Wright, "Primal-Dual Interior-Point Methods," SIAM, 1997
    (context for polynomial-time alternatives to the simplex).
"""
import numpy as np



def simplex_lp(c, A, b, max_iters=2000, tol=1e-10):
    """
    two-phase primal simplex for standard-form LPs
        min c^T w  s.t.  A w = b, w >= 0

    Phase 1 adds artificial slack columns to find a feasible basis;
    phase 2 pivots on reduced costs to optimality (Bland's rule
    fallback when degeneracy stalls progress).

    Parameters:
        c: `cost vector (len n)`
        A: `constraint matrix (m x n), rows must be consistent`
        b: `right-hand side (len m), assumed non-negative`
        max_iters: `pivot cap`
        tol: `pivot degeneracy tolerance`

    Returns:
        w: `optimal basic solution (len n)`
        obj: `optimal objective c^T w`
    """
    m, n = A.shape
    c = np.asarray(c, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    # standard form requires b >= 0: pre-flip rows with negative rhs
    negrows = b < 0
    A = np.where(negrows.reshape(-1, 1), -A, A)
    b = np.where(negrows, -b, b)

    # phase 1: artificials make the initial basis trivially feasible
    A1 = np.hstack([A, np.eye(m)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis = list(range(n, n + m))           # artificials start basic
    w_basis, basis, _ = _run_simplex(c1, A1, b, basis, max_iters, tol)

    w1 = np.zeros(n + m)
    w1[basis] = w_basis
    artificial_mass = w1[n:].sum()

    # phase 2: drive artificials out, then minimize the real cost
    if artificial_mass > tol:
        raise ValueError("infeasible LP: phase 1 left artificial mass")

    A2 = A1[:, :n]                          # artificials locked at zero
    basis = [j for j in basis if j < n]
    w_basis, basis, _ = _run_simplex(c, A2, b, basis, max_iters, tol)

    w = np.zeros(n)
    w[basis] = w_basis
    return w, float(np.dot(c, w))


def basis_pursuit_dantzig(A, y):
    """
    basis pursuit reconstruction: min ||x||_1 s.t. A x = y

    Assembles the split-variable LP above and solves it with the
    hand-written two-phase simplex.

    Parameters:
        A: `sampling matrix`
        y: `compressed samples`

    Returns:
        x_hat: `reconstructed signal`
    """
    m, n = A.shape
    M = np.hstack([A, -A])
    c = np.ones(2 * n)
    w, _ = simplex_lp(c, M, y.ravel())
    x_hat = (w[:n] - w[n:]).reshape(-1, 1)
    return x_hat


def _run_simplex(c, A, b, basis, max_iters, tol):
    """
    primal simplex given a feasible basis; pivots to optimality

    Parameters:
        c: `cost over all columns`
        A: `constraint matrix`
        b: `right-hand side`
        basis: `current basic column indices`
        max_iters: `pivot cap`
        tol: `degeneracy tolerance`

    Returns:
        x_B: `basic solution values`
        basis: `final basis`
        iters: `pivots taken`
    """
    m, n = A.shape
    iters = 0
    bland = False                           # switch to Bland's rule when degenerate

    while iters < max_iters:
        B = A[:, basis]
        x_B = np.linalg.solve(B, b)
        # dual: c_B^T B^-1 ; reduced costs r_j = c_j - y^T A_j
        y_dual = np.linalg.solve(B.T, c[basis])
        reduced = c - np.dot(A.T, y_dual)

        entering = _pivot_choice(reduced, basis, bland, tol)
        if entering is None:
            break                           # optimal: no improving column

        d = np.linalg.solve(B, A[:, entering])
        # step t >= 0 changes basic values by -t*d; rows with d > 0 block first
        pos = d > tol
        if not np.any(pos):
            raise ValueError("unbounded LP")  # cannot happen for BP

        ratios = np.full(m, np.inf)
        ratios[pos] = x_B[pos] / d[pos]
        leaving_pos = np.argmin(ratios)
        if ratios[leaving_pos] < tol:
            bland = True                    # degenerate stalling: Bland's rule
        basis[leaving_pos] = entering
        iters += 1

    B = A[:, basis]
    x_B = np.linalg.solve(B, b)
    return x_B, basis, iters

def _pivot_choice(reduced, basis, bland, tol):
    """
    entering-variable rule: Dantzig (most negative) or Bland
    """
    improved = [j for j in range(reduced.shape[0]) if j not in basis and reduced[j] < -tol]
    if not improved:
        return None
    if bland:
        return improved[0]              # Bland: smallest improving index
    return min(improved, key=lambda j: reduced[j])  # Dantzig: most negative



