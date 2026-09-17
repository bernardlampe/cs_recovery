import numpy as np

def subspace_pursuit(Phi, y, K, max_iter=100, tol=1e-6):
    """
    Subspace Pursuit (SP) algorithm for sparse recovery.

    References:
        W. Dai and O. Milenkovic, "Subspace Pursuit for Compressive
        Sensing Signal Reconstruction," IEEE Transactions on
        Information Theory, vol. 55, no. 5, pp. 2230-2249, May 2009.

    Parameters
    ----------
    Phi : ndarray of shape (m, n)
        Measurement (sensing) matrix.
    y : ndarray of shape (m,)
        Measurement vector.
    K : int
        Sparsity level (number of non-zero elements to recover).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    tol : float, optional
        Convergence tolerance based on residual norm (default: 1e-6).
        
    Returns
    -------
    x_hat : ndarray of shape (n,)
        Reconstructed sparse signal.
    support : list of int
        Indices of the recovered support.
    residuals : list of float
        History of residual norms.
    """
    m, n = Phi.shape
    residual = y.copy()
    support = set()
    residuals = [np.linalg.norm(residual)]

    for _ in range(max_iter):
        # Step 1: Find indices of largest correlations
        correlations = np.abs(Phi.T @ residual)
        new_indices = set(np.argpartition(correlations, -K)[-K:])
        
        # Step 2: Merge supports
        candidate_support = list(support.union(new_indices))
        
        # Step 3: Solve least-squares problem on candidate support
        Phi_T = Phi[:, candidate_support]
        x_T, _, _, _ = np.linalg.lstsq(Phi_T, y, rcond=None)
        
        # Step 4: Keep only K largest entries
        idx_sorted = np.argsort(np.abs(x_T))[-K:]
        support = {candidate_support[i] for i in idx_sorted}
        
        # Step 5: Update signal estimate and residual
        Phi_S = Phi[:, list(support)]
        x_S, _, _, _ = np.linalg.lstsq(Phi_S, y, rcond=None)
        residual = y - Phi_S @ x_S
        residuals.append(np.linalg.norm(residual))
        
        # Convergence check
        if np.abs(residuals[-2] - residuals[-1]) < tol:
            break

    # Final reconstruction
    x_hat = np.zeros(n)
    x_hat[list(support)] = x_S
    return x_hat, list(support), residuals

if __name__ == "__main__":
    from common import *

    for db in [None, 20]:
        (y, A, x_t, x_f) = gen_test_signal(snr_db=db, k=10, n=200, amps_l=-10, amps_h=10)

        x_h, support, residuals = subspace_pursuit(A, y.ravel(), 20)
        plt_error(x_h, x_f.ravel(), 'sparsity_term, k = 20', alg='sp_sparsity')

