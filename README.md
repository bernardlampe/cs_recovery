# From-Scratch Compressive Sensing, A review of reconstruction algorithms

Every sample is a standalone NumPy script. A sparse signal generated in the
DCT domain runs through the whole course by default, so each technique can be
compared against the previous ones. One shared driver, `run_all.py`, collects
per-iteration residual curves and final errors from every algorithm and writes
the two summary charts.

Example:
```bash
python 03_omp.py                   # run OMP on the standard test signal
./../.venv/Scripts/python run_all.py   # or: run everything, make both charts
```

The numerical toolbox every algorithm leans on lives in `opt/` and is
hand-written too:

  * `opt/prox.py`      - soft/hard thresholding, l2-ball projection (proximal operators)
  * `opt/gradient.py`  - Armijo line search, plain and accelerated (FISTA) proximal gradient
  * `opt/lsqr.py`      - conjugate gradient, regularized normal equations
  * `opt/admm.py`      - two-block ADMM with closed-form z-update
  * `opt/dantzig.py`   - two-phase primal simplex for basis pursuit LPs

Requirements: Python 3, NumPy. Everything else (including plotting and the
DCT transform in `common/common.py`) is hand-rolled in this folder.

---

## Part I - Greedy pursuit (sessions 1–8)

### 1. `01_mp.py` - Matching Pursuit
- **Concepts**: correlation scan, one-atom coefficient update, no backfitting.
- **Update**: each pass picks the atom most correlated with the residual and
  adds its correlation to that atom's coefficient; coefficients of earlier
  atoms are never revised.
- **Key insight**: MP is the cheapest pursuit; it pays for cheapness by letting
  the residual re-correlate with used atoms, so convergence is slow (86 iters
  vs OMP's 10 on the standard problem). The original signal model needs
  m ≈ c·k·log(n/k) samples.

### 2. `02_weak_mp.py` - Weak Matching Pursuit
- **Concepts**: weak acceptance threshold (0 < mu < 1), gain action.
- **Update**: accept ANY atom whose correlation exceeds mu times the peak
  correlation rather than the argmax; convergence is guaranteed for the
  family when mu > 0.5.
- **Key insight**: relaxing argmax to threshold is a cheap-first compromise
  that streaming implementations can use; with mu = 0.9 recovery
  is unchanged on the standard problem.

### 3. `03_omp.py` - Orthogonal Matching Pursuit
- **Concepts**: support set, backfitting least squares on support,
  configurable termination (sparsity/percent/epsilon).
- **Update**: pick the atom most correlated with the residual, then
  re-least-square ALL selected atoms against y, so used atoms lose their
  wrongly-claimed energy.
- **Key insight**: exact LS on the support makes residuals orthogonal to the
  support — guaranteed recovery with enough samples. The termination
  functions introduced here are reused by every later file.

### 4. `04_omp_rls.py` — OMP with Recursive Least Squares updates
- **Concepts**: online inverse-correlation maintenance (P matrix), gain vectors.
- **Update**: instead of recomputing least squares each pass, keeps the inverse
  correlation matrix and applies rank-1 downdates.
- **Key insight**: same support selection as OMP, but the LS solve is amortized
  when atoms stream in one at a time.

### 5. `05_stomp.py` - Stagewise Orthogonal Matching Pursuit
- **Concepts**: threshold selection in batches, sigma-based cutoffs.
- **Update**: pass ALL atoms whose correlation exceeds sigma · std(r);
  sigma in 2..3 is calibrated so each stage adds a sparsity-consistent batch.
- **Key insight**: batch adds trade certainty for speed; recovery
  guarantees are weaker but per-iteration cost drops.

### 6. `06_gradient_pursuit.py` - Gradient Pursuit
- **Concepts**: conjugate-gradient on the support, Polak-Ribiere, seeded warm starts.
- **Update**: after greedy support detection, run cg_iters CG sweeps per
  iteration instead of an exact LS.
- **Key insight**: gradient pursuit converges to the same limit as OMP when
  the subproblem is solved accurately (10-12 CG sweeps), but each sweep is
  cheaper than a full LS: the sweet spot for large supports.

### 7. `07_cosamp.py` - Compressive Sampling Matching Pursuit
- **Concepts**: 2k candidate support, pruning, LS on merged support.
- **Update**: assemble y + proxy candidates into a support of 2k, least-squares
  on that support, prune to the best k.
- **Key insight**: pruning back the support each iteration makes CoSaMP robust
  to initial wrong picks (and to noise on the measurements).

### 8. `08_sp.py` - Subspace Pursuit
- **Concepts**: candidate subspace vs final support distinction, LS twice per iteration.
- **Update**: like CoSaMP but the candidate set is exactly k, and the final
  support is re-LS'd separately from the candidate LS.
- **Key insight**: SP and CoSaMP are practically twins; SP's tighter candidate
  set usually costs a few more iterations but is cheaper per iteration.

---

## Part II — Iterative thresholding & reweighted schemes (sessions 9–11)

### 9. `09_iht.py` - Iterative Hard Thresholding
- **Concepts**: hard-thresholding operator H_k, normalized step size, NP-hard
  l0 minimization relaxation by projection.
- **Update**: x ← H_k(x + tau·Aᵀ(y − Ax)); tau = 3/||A||² is the IHT-safe step.
- **Key insight**: IHT is plain gradient descent where "prox" is projection
  onto the k-sparse set. Slow to contract (288 its on the standard problem)
  but robust and streaming friendly.

### 10. `10_irls.py` - Iteratively Reweighted Least Squares
- **Concepts**: surrogate log-sum penalties, weight update W = diag(2/(|x|+d)).
- **Update**: minimize a weighted-ridge LS per iteration where the weights
  amplify small coefficients (d delta-e.g. 1e-4 keeps the system invertible).
- **Key insight**: reweighting approximates l0 from above; with delta → 0 the
  fixed point enters the true sparse arsenal, sensitive to noise amplitude.

### 11. `11_focuss.py` - FOCal Underdetermined System Solver
- **Concepts**: multiplicative weight updates (W = diag(x_k)), minimum-norm
  reweighting by pseuodoinverse.
- **Update**: x ← Wₚ(AWₚ)⁺y each iteration — the pseudo-inverse restriction
  sharpens the previous estimate.
- **Key insight**: FOCUSS is IRLS's ancestor: same idea of sharpening the
  minimum-norm solution, but multiplicative; converges in very few iterations
  under exact measurements and a good initial guess.

---

## Part III - Convex l1 solvers (sessions 12–14)

### 12. `12_lars.py` - Least Angle Regression
- **Concepts**: equiangular direction, correlated knots, step sizes gamma per section.
- **Update**: grow support by atoms whose correlations tie the active
  correlation, then walk the joint equiangular direction.
- **Key insight**: the whole LASSO path traced in pivots; each step grows
  support by exactly one atom (vs OMP taking one atom per LS).

### 13. `13_bp.py` - Basis Pursuit via linear programming
- **Concepts**: standard-form LP, x = u − v splitting, two-phase primal simplex,
  Bland's rule for degeneracy.
- **Update**: solve min 1ᵀw s.t. [A −A]w = y, w ≥ 0 with the hand-rolled
  simplex in `opt/dantzig.py` (phase 1: feasibility with artificials;
  phase 2: cost descent after locking artificials to 0).
- **Key insight**: BP is the exact l1 program; the correct exact-recovery
  answer (rel. err 10⁻¹²) that every fast solver is approximating. Note the
  normalization step: negative y rows must flip signs first for the
  artificial-basis init to be feasible.

### 14. `14_lasso.py` - LASSO via FISTA
- **Concepts**: proximal soft-thresholding, Nesterov momentum, Lipschitz step.
- **Update**: FISTA with the soft threshold at lam·t; requires a step sized
  at 1/Lipschitz ||A||².
- **Key insight**: the workhorse formulation - LASSO trades off bias vs
  sparsity through lam: too big over-shrinks (0.03 → 10⁻¹ rel err), too small
  → dense solution (1e-6 → 197 nonzeros); the lam sweet-spot for exact
  recovery on this signal is ~1e-5 with a long 21k-iteration tail.

---

## Part IV — Operator-splitting and message passing (sessions 15–18)

### 15. `15_aadm.py` - Adaptive ADMM
- **Concepts**: primal/dual residual balancing, adaptive penalty (Boyd 3.4.1).
- **Update**: x via soft threshold, z via ridge-LS toward (x+u), u dual ascent;
  rho adapts by the mu-band rule to keep both residuals comparable.
- **Key insight**: same l1 formulation as LASSO, but split by constraint rather
  than smoothed; the penalty balancing is what makes it usable without
  hand-tuning rho — but per-iteration behavior is that of the z-step, and on
  this problem it is the slowest solver (3772 its).

### 16. `16_bregman.py` - Bregman Iterative Reconstruction
- **Concepts**: Bregman divergence reload, split l1-least squares, the
  load-bearing "residual feed" trick.
- **Update**: outer loop: x ← argmin ||Ax − y_b||² + lam||x||₁; then
  y_b ← y_b + (y − Ax). Inner solves are 30 ISTA sweeps.
- **Key insight**: Bregman iteration recreates the BP fixed point from the
  least-squares side; the reload loop does the sparsity enforcement instead
  of an exact l1 program. Reaches rel err 10⁻⁸ in ~60 outer iterations.

### 17. `17_hhs.py` - Heavy Hitters on Steroids (sketched greedy)
- **Concepts**: count-sketch bucketing, hash-voting for heavy coordinates,
  LS pricing on the promoted support.
- **Update**: bucket correlations with a random modular hash, vote across
  rounds with max, promote the heaviest pace atoms per round, exact LS on
  support each time.
- **Key insight**: importing the streaming-algorithm toolkit: detection via
  randomized sketching with O(pace·n) work per iteration instead of a full
  correlation scan; an exact single-atom-scan would be O(n) anyway here, but
  the trick pays off when the correlation itself must be maintained streaming.

### 18. `18_amp.py` - Approximate Message Passing
- **Concepts**: Onsager reaction term, denoiser sensitivity, state evolution.
- **Update**: x ← soft(w, lam·tau) with tau estimated from the current residual;
  z ← y − Ax + (eta'/delta)·z (delta = m/n).
- **Key insight**: the Onsager correction makes the scalar z behave Gaussian
  under iteration, and the noise-adaptive threshold tau is what makes AMP work
  without a known noise level. Watch out for divergence at small lam on
  high-delta problems - AMP is exact only in the large-system limit.

---

## The standard problem

`common/common.py::gen_test_signal(k=10, n=200)`:

  * a noiseless (or SNR = 20 dB) sparse DCT signal with |Support| = 10 random
    integer amplitudes,
  * a random Gaussian measurement matrix A normalized to σᵢ·m⁻¹ᐟ² with
    m = 2·k·log(n/k) ≈ 60 rows (well above the k·log(n/k) exact-recovery
    boundary),
  * y = A·x_t.

This is the object each script solves in its `__main__` demo, and exactly the
signal `run_all.py` hands to all 18 solvers.

## Charts

`run_all.py` writes two PNGs into `out/`:

  * `out/convergence.png` — per-iteration residual norm ||y − Ax||₂ on log
    axes; a left panel with the full horizon and a right panel zoomed on the
    first 30 iterations (where the greedy family collapses), so both fast
    and slow solvers are readable.
  * `out/error_chart.png` — final relative l2 error per algorithm, sorted
    best first, log scale.

## Summary lessons

  * Greedy (OMP family): fastest per-iteration convergence on easy problems;
    sensitive to exact sparsity knowledge, degrade gracefully if k is right.
  * Hard thresholding (IHT family): simple loops, weaker local accuracy.
  * Reweighted (IRLS/FOCUSS): few iterations, need regularization legroom
    (delta) and dense initializations.
  * Convex l1 (BP/LASSO/AADM): exact or near-exact, best noise stability;
    cost is iterations/machine time rather than implementation complexity.
  * Sketched/message-passing (HHS/AMP): the modern scaling story; O(n) per
    iteration with statistical rather than exact guarantees.
