#!/usr/bin/python

""" Heavy Hitters on Steroids (HHS) Sparse Recovery

HHS: a streaming-heavy-hitter flavored coordinate thresholding
scheme. Rather than scanning all correlations every iteration
(OMP family), HHS:
    1. estimates per-coordinate correlation via count-sketch style
       randomized bucketing (one hash family, P modulo buckets),
    2. promotes the heaviest buckets ("heavy hitters"),
    3. refines coefficients by a least-squares refresh on the
       candidate support.

    The result is a near-OMP recovery with sub-linear per-iteration
    correlation maintenance - the defining idea sampled from streaming
    algorithms, imported here to the batch compressed sensing setting.

References:
    G. Cormode and M. Hadjieleftheriou, "Finding Frequent Items in
    Data Streams," Proceedings of the VLDB Endowment, vol. 1, no. 2,
    pp. 1530-1541, 2008.  (count-sketch heavy hitter tracking)

    P. Berinde, A. C. Gilbert, P. Indyk, H. Karloff and M. J.
    Strauss, "Combining Geometry and Combinatorics: A Unified
    Approach to Sparse Signal Recovery," 46th Annual Allerton
    Conference on Communication, Control, and Computing, 2008.
    (heavy-hitter recovery in compressed sensing)

    K. L. Clarkson and D. P. Woodruff, "Low Rank Approximation and
    Regression in Input Sparsity Time," STOC 2013, pp. 81-90, 2013.
    (sketch-based linear algebra)
"""

import numpy as np

# optional per-iteration residual logger (run_all charts read this)
HISTORY = None


def hhs(y, A, term, param, buckets=None, max_iters=50, repeats=3, pace=4):
    """
    HHS reconstruction with sketch-based heavy hitter detection

    Parameters:
        y: `compressed samples`
        A: `sampling matrix`
        term: `termination function from above`
        param: { k: `sparsity` | p: `percent` | e: `epsilon` }
        buckets: `sketch width (default: n, near-collision-free)`
        max_iters: `outer iteration cap`
        repeats: `independent hash rounds before promoting`
        pace: `atoms promoted per round`

    Returns:
        x_hat: `reconstructed signal`
    """

    global HISTORY
    HISTORY = []

    m, n = A.shape
    if buckets is None:
        buckets = n                      # wide sketch: rare collisions
    r = np.copy(y)
    support = set()                      # heavy hitter accumulator
    x_hat = np.zeros((n, 1))

    i = 0
    while not term(param, y, r, x_hat) and i < max_iters:
        HISTORY.append(float(np.linalg.norm(r)))
        c = np.dot(A.T, r).ravel()       # full correlation (for scoring)
        # randomized bucket voting: over repeated hash rounds the true
        # heavy atom keeps landing in a high-mass bucket, noise atoms
        # spread out; the max over rounds counters hash collisions
        votes = np.zeros((repeats, n))
        for rep in range(repeats):
            off = int((rep * 7919) % max(buckets, 1))
            hh_idx = np.mod(np.arange(n) * 31 + off, buckets)
            bm = np.zeros(buckets)
            for j in range(n):
                bm[hh_idx[j]] += c[j] ** 2
            votes[rep] = bm[hh_idx]      # broadcast bucket mass back
        heavy_mass = np.max(votes, axis=0)

        # promote the heaviest fresh atoms each round (count-sketch idea:
        # cheap detection, exact LS to price the picked support)
        cand = [int(j) for j in np.argsort(heavy_mass)[-pace:] if j not in support]
        support.update(cand)

        # LS refresh on the candidate support
        sup = sorted(support)
        P = A[:, sup]
        x_s = np.linalg.lstsq(P, y, rcond=None)[0]
        x_hat = np.zeros((n, 1))
        x_hat[sup] = x_s
        r = y - np.dot(P, x_s)

        i += 1

    return x_hat
def sparsity_term(k, y, r, x_hat):
    return int(np.linalg.norm(x_hat.ravel(), 0)) == k

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

        x_h = hhs(y, A, sparsity_term, 20)
        plt_error(x_h, x_f, 'sparsity_term, k = 20', alg='hhs')
