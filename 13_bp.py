#!/usr/bin/python

""" Basis Pursuit via Linear Programming

Basis Pursuit (BP) recovers the sparsest solution of an
underdetermined system exactly:
    min ||x||_1   s.t.   A x = y
Splitting x = u - v, u,v >= 0 turns this into a standard-form LP
solved here by a hand-written two-phase simplex (opt.dantzig).

References:
    S. S. Chen, D. L. Donoho and M. A. Saunders, "Atomic Decomposition
    by Basis Pursuit," SIAM Review, vol. 43, no. 1, pp. 129-159, 2001.

    E. Candès and J. Romberg, "l1-magic: Recovery of Sparse Signals,"
    Caltech Technical Report, 2005 (basis pursuit formulation).
"""

import numpy as np

def bp_linprog(y, A):
    """
    basis pursuit reconstruction via the hand-written simplex LP

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`

    Returns:
        x_hat: `reconstructed signal`
    """

    from opt import basis_pursuit_dantzig
    x_hat = basis_pursuit_dantzig(A, y)
    return x_hat

if __name__ == "__main__":
    from common import *

    for db in [None, 20]:
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db, k=10, n=200, amps_l=-10, amps_h=10)

        x_h = bp_linprog(y, A)
        plt_error(x_h, x_f, 'bp linprog', alg='bp_linprog')
