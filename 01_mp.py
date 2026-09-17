#!/usr/bin/python

""" Match Pursuit Sparse Signal Recovery

References:
    Hammeed, Maxin Abdulrasool, "Comparative Analysis of Orthogonal Matching Pursuit
    and least angle regression," Michigan State University, A Thesis for Masters of
    Science, 2012.
"""

import numpy as np

def mp(y, A, term, param):
    """
    np: match pursuit with configurable termination criteria

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function`
        param: `termination parameter` { k: `sparsity` | p: `percent` | e: `epsilon` }

    Returns:
        x_hat: `reconstructed signal`
    """

    r = np.copy(y)                       # init residual
    x_hat = np.zeros((A.shape[1], 1))    # null output init

    while not term(param, y, r, x_hat):
        c = np.dot(A.T, r)               # correlation
        ind = np.argmax(np.abs(c))       # find abs support of max correlation
        x_hat[ind] += c[ind]             # update ind coefficient
        r = y - np.dot(A, x_hat)         # update residual

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

        x_h = mp(y, A, sparsity_term, 20)
        plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='mp')

        x_h = mp(y, A, percent_term, 0.99)
        plt_error(x_h, x_f, 'percent_term, p = 0.99', alg='mp')

        x_h = mp(y, A, epsilon_term, 0.00001)
        plt_error(x_h, x_f, 'epsilon_term, k = 0.00001', alg='mp')
