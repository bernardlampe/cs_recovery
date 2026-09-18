#!/usr/bin/python

"""
Orthogonal Match Pursuit Sparse Signal Recovery

References:
    J. A. Tropp and A. C. Gilbert, "Signal Recovery From Random Measurements
    Via Orthogonal Matching Pursuit," in IEEE Transactions on Information Theory,
    vol. 53, no. 12, pp. 4655-4666, Dec. 2007.

    Hammeed, Maxin Abdulrasool, "Comparative Analysis of Orthogonal Matching Pursuit
    and least angle regression," Michigan State University, A Thesis for Masters of
    Science, 2012.
"""

import numpy as np

def omp(y, A, term, param):
    """
    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }

    Returns:
        x_hat: `reconstructed signal`
    """

    y = np.asarray(y).reshape(-1, 1)     # accept flat or column measurements
    n = A.shape[1]
    x_hat = np.zeros((n, 1))             # null output init (embedded estimate)
    r = y       # init residual
    lamda = []  # list of support loc inds
    phi = []    # list of support vectors
    x = np.array([]) # null output init

    while not term(param, y, r, x_hat):
        c = np.dot(A.T, r)               # correlation
        cl = np.abs(c).ravel()
        cl[lamda] = -np.inf              # residual ⊥ support: never re-pick
        ind = int(np.argmax(cl))         # find best fresh atom

        lamda.append(ind)                # update support
        phi.append(A[:, ind:(ind+1)])    # update support locs

        P = np.concatenate(phi, axis=1)  # compose the submatrix
        x = np.linalg.lstsq(P, y, rcond=None)[0]  # min ||y-P*x||_2
        x_hat[:] = 0
        x_hat[lamda] = x                 # embed coefficients
        r = y - np.dot(P, x)             # update residual


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

    (y, A, x_t, x_f) = gen_test_signal(snr_db=20, k=10, n=200, amps_l=-10, amps_h=10)

    x_h = omp(y, A, sparsity_term, 20)
    plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='omp_sparsity')

    x_h = omp(y, A, percent_term, 0.99)
    plt_error(x_h, x_f, 'percent_term, p = 0.99', alg='omp_percent')

    x_h = omp(y, A, epsilon_term, 0.00001)
    plt_error(x_h, x_f, 'epsilon_term, k = 0.00001', alg='omp_epsilon')
