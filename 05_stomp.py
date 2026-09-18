#!/usr/bin/python

"""
Stagewise Orthogonal Match Pursuit Sparse Signal Recovery

References:
    D. L. Donoho, Y. Tsaig, I. Drori and J.-L. Starck, "Sparse Solution
    of Underdetermined Linear Equations by Stagewise Orthogonal
    Matching Pursuit," IEEE Transactions on Information Theory,
    vol. 58, no. 2, pp. 1094-1121, Feb. 2012.

    Hammeed, Maxin Abdulrasool, "Comparative Analysis of Orthogonal Matching Pursuit
    and least angle regression," Michigan State University, A Thesis for Masters of
    Science, 2012.

    Angshul Majumdar, "Compressed Sensing for Engineers", CRC Press, 3 Dec 2018.
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None

def stomp(y, A, sigma, N):
    """
    stagewise orthogonal match pursuit with configurable termination criteria

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        sigma: `number of std deviations (2-3)`
        N: `number of iterations`

    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY

    HISTORY = []
    r = np.copy(y)       # init residual
    lamda = set()        # set of support indicies
    x = np.array([])     # null output init
    n = A.shape[1]       # size of solution

    for t in range(N):
        HISTORY.append(float(np.linalg.norm(r)))
        c = np.dot(A.T, r)                  # correlation
        thr = sigma * np.std(r)             # compute number of std devs for threshold
        inds = np.nonzero(np.abs(c) > thr)  # find abs support above threshold
        lamda.update(inds[0].tolist())      # update support

        P = A[:, list(lamda)]                     # compose the submatrix
        x = np.linalg.lstsq(P, y, rcond=None)[0]  # min ||y-P*x||_2
        r = y - np.dot(P, x)                      # update residual

    # embed coefficients in support
    x_hat = np.zeros((n, 1))
    x_hat[list(lamda)] = x

    return x_hat

if __name__ == "__main__":
    from common import *

    (y, A, x_t, x_f) = gen_test_signal(snr_db=20, k=10, n=200, amps_l=-10, amps_h=10)

    x_h = stomp(y, A, 3, 20)
    plt_error(x_h, x_f, 'parameter_term, k = 20', alg='stomp_params')
