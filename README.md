# From-Scratch Compressive Sensing, A review of reconstruction algorithms.

Every algorithm is a standalone script using numpy. A sparse signal generated in the
DCT domain runs through the whole course by default, so each technique can be compared against
the previous ones.

The numerical toolbox every algorithm leans on lives in `opt/`:

  * `opt/prox.py`      - soft/hard thresholding, l2-ball projection (proximal operators)
  * `opt/gradient.py`  - Armijo line search, plain and accelerated (FISTA) proximal gradient
  * `opt/lsqr.py`      - conjugate gradient, regularized normal equations
  * `opt/admm.py`      - two-block ADMM with closed-form z-update
  * `opt/dantzig.py`   - two-phase primal simplex for basis pursuit LPs

---

## Part I - Greedy pursuit

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
  correlation rather than the argmax.
- **Key insight**: relaxing argmax to threshold is a cheap-first compromise
  that streaming implementations can use; the convergence framework holds for
  any mu < 1 (Blumensath-Davies' weak-selection analysis), and with
  mu = 0.9 recovery is unchanged on the standard problem.

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
- **Update**: same support selection as OMP; each accepted atom extends the
  RLS gain computation and rank-1 inverse-correlation update, but the
  coefficient refresh in this reference implementation is still a plain
  pinv least-squares on the active submatrix (full coefficient streaming
  is left as an exercise).
- **Key insight**: the RLS machinery is set up so coefficients can be
  amortized as atoms stream in one at a time rather than re-solved
  globally.

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
  cheaper than a full LS: good for large supports.

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

## Part II — Iterative thresholding & reweighted schemes

### 9. `09_iht.py` - Iterative Hard Thresholding
- **Concepts**: hard-thresholding operator H_k, normalized step size, NP-hard
  l0 minimization relaxation by projection.
- **Update**: x ← H_k(x + tau·Aᵀ(y − Ax)); tau = 3/||A||² is the IHT-safe step.
- **Key insight**: IHT is plain gradient descent where "prox" is projection
  onto the k-sparse set.

### 10. `10_irls.py` - Iteratively Reweighted Least Squares
- **Concepts**: surrogate log-sum penalties, weight update W = diag(2/(|x|+d)).
- **Update**: minimize a weighted-ridge LS per iteration where the weights
  amplify small coefficients (delta e.g. 1e-4 keeps the system invertible).
- **Key insight**: reweighting approximates l0 from above; as delta → 0 the
  fixed point approaches the true sparse solution, at the price of noise sensitivity.

### 11. `11_focuss.py` - FOCal Underdetermined System Solver
- **Concepts**: multiplicative weight updates (W = diag(x_k)), minimum-norm
  reweighting by pseudoinverse.
- **Update**: x ← Wₚ(AWₚ)⁺y each iteration — the pseudo-inverse restriction
  sharpens the previous estimate (Tikhonov-regularized pinv for noisy data).
- **Key insight**: FOCUSS is IRLS's ancestor: same idea of sharpening the
  minimum-norm solution, but multiplicative; converges in very few iterations
  under exact measurements and a good initial guess.

---

## Part III - Convex l1 solvers

### 12. `12_lars.py` - Least Angle Regression
- **Concepts**: equiangular direction, correlation knots, step sizes gamma.
- **Update**: grow support by atoms whose correlations tie the active
  correlation, then walk the joint equiangular direction.
- **Key insight**: the whole LASSO path traced in pivots; each pivot adds
  exactly one atom to the support (like OMP), but the step along the
  equiangular direction is chosen so all active correlations stay tied.

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
  → dense solution (1e-6 → 197 nonzeros); the lam good for exact
  recovery on this signal is ~1e-5 with a long 21k-iteration tail.

---

## Part IV — Operator-splitting and message passing

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
  randomized sketching costs O(repeats·n) per iteration plus the exact LS
  on the small promoted support; the trick pays off when the correlation
  itself must be maintained in a streaming setting.

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

  * a signal with SNR = 20 dB using sparse DCT with |Support| = 10 random integer amplitudes,
  * a random Gaussian measurement matrix A with entries scaled by m^{-1/2},
    m = ceil(2·k·log(n/k)) ≈ 60 rows (well above the k·log(n/k) exact-recovery boundary),
  * y = A·x_t.

This is the object each script solves in its `__main__` demo, and exactly the
signal `run_all.py` hands to all solvers.

## Summary lessons

  * Greedy (OMP family): fastest per-iteration convergence on easy problems;
    sensitive to exact sparsity knowledge, degrade gracefully if sparsity estimates (k) are close.
  * Hard thresholding (IHT family): simple loops, weaker local accuracy.
  * Reweighted (IRLS/FOCUSS): few iterations, need regularization legroom (delta) and dense initializations.
  * Convex l1 (BP/LASSO/AADM): exact or near-exact, best noise stability;
    cost is iterations/machine time rather than implementation complexity.
  * Sketched/message-passing (HHS/AMP): the modern scaling story; O(n) per
    iteration with statistical rather than exact guarantees.

  * Greedy (I): discrete, sequential support growing — cheap steps, locally optimistic, no global optimality promise.
  * Thresholding/reweighting (II): dense continuous iterations — simple formulas, no decisions to undo, weaker local accuracy.
  * Convex ℓ₁ (III): solves one well-posed optimization problem exactly (or its full path) — strongest guarantees, highest
    iteration/machine cost.
 *  Splitting/message passing (IV): decompose the same ℓ₁ problem across variables/operators or infer statistically — the scalable endgame:
    ADMM's modular but slow, Bregman's cheap-and-effective residual trick, HHS's O(n) sketching, AMP's asymptotically exact O(mn) loop.

  * The thread across all four: every part attacks the same problem (exact-sparse recovery from m ≈ k·log(n/k) measurements) by weakening a
    different form of difficulty — Part I removes the combinatorial search, Part II the non-smoothness, Part III the non-convexity, Part IV
    the per-iteration cost.

---

## Insights

### Part I — Greedy pursuit (01–08)

The part-level idea: build the support one (or few) atoms at a time, by correlating with the residual.

| # | Algo | Insight | Exploited concept |
| --- | --- | --- | --- |
| 1 | MP | Cheapest possible step: pick the most-correlated atom, add its correlation value to that coefficient. Old coefficients are never revised. | Correlation scan; error self-corrects only via repeated re-picking (cheap per step) |
| 2 | Weak MP | Accept any atom above μ·max correlation instead of the argmax | Weak/gain selection (Blumensath–Davies): μ>½ guarantees convergence while allowing cheaper/streamed selection |
| 3 | OMP | Re-solve least squares on the whole support each step | Orthogonalization: the residual becomes orthogonal to used atoms → no re-picks, one atom per true component, exact recovery |
| 4 | OMP-RLS | Maintain the inverse correlation of the growing support incrementally (RLS: gain vector, rank-1 update) instead of re-solving | Online rank-1 updates amortize the per-step LS to O(m·s) streams |
| 5 | StOMP | Add all atoms above σ·std(residual) in one stage | Batch selection + statistical cutoff: trades certainty for iteration count |
| 6 | GP | Keep greedy detection but replace exact LS with a few conjugate-gradient sweeps on the support | CG warm-started across iterations ≈ LS accuracy at matching-pursuit cost |
| 7 | CoSaMP | Propose 2k candidates, prune to best k each round | Supporting/pruning: wrong early picks get evicted → provable noise robustness OMP lacks |
| 8 | SP | Same, but candidate set is exactly k and the final support gets a second LS | Subspace refinement: two LS per iteration, cheaper than CoSaMP's 2k merge |

Growth within the part: selection rigor is progressively relaxed (argmax → threshold → batch), while coefficient fidelity is
progressively hardened (drifting → LS → pruned LS → incremental LS).

### Part II — Iterative thresholding & reweighting (09–11)

The part-level idea: no support sets at all — run gradient-like loops on the full vector, sparsity enforced by an operator on each
iterate.

| # | Algo | Insight | Exploited concept |
| --- | --- | --- | --- |
| 9 | IHT | Gradient descent + keep the k largest entries every step | Hard-thresholding operator H_k = Euclidean projection onto the k-sparse set; step τ = 3/‖A‖² keeps it contractive |
| 10 | IRLS | Solve weighted ridge LS where small coefficients get big weights | Smoothed ℓ₀/log-sum surrogate: reweighting approximates ℓ₀ from above; δ→0 sharpens |
| 11 | FOCUSS | Same sharpening, multiplicative form: scale each coordinate by its own magnitude | Minimum-norm/pseudoinverse restriction; ancestor of IRLS, converges in few iterations with a good init |

Contrast with Part I: detection is implicit (the operator, not a greedy choice); per-iteration cost is large (dense n-vector work) but
each iteration is a smooth global improvement — no discrete decisions to regret.

### Part III — Convex ℓ₁ solvers (12–14)

The part-level idea: replace NP-hard ℓ₀ with the convex ℓ₁ program, then solve it well.

| # | Algo | Insight | Exploited concept |
| --- | --- | --- | --- |
| 12 | LARS | Grow support by pivots where an atom's correlation ties the active set; step along the equiangular direction until the next tie | Correlation knots: traces the entire LASSO solution path exactly, in ~k pivots |
| 13 | BP | ℓ₁ minimization is a linear program | Exact reformulation x = u − v, standard-form simplex (two-phase, Bland's rule) — the ground truth every fast solver approximates (1e-12 here) |
| 14 | LASSO | ℓ₁ in the penalized least-squares form, solvable by proximal splitting | FISTA: soft-threshold prox + Nesterov momentum + step 1/‖A‖²; lam explicitly trades bias vs sparsity |

Contrast with Parts I/II: exact global optimum instead of a heuristic trajectory; the price is iterations and tuning (lam), not
cleverness.

### Part IV — Operator splitting & message passing (15–18)

The part-level idea: modern tools — split the problem across operators/variables, or treat inference statistically.

| # | Algo | Insight | Exploited concept |
| --- | --- | --- | --- |
| 15 | AADM | Same ℓ₁/LASSO objective, but split by constraint (minimize ‖Ax−y‖² + ‖z‖₁ s.t. x = z) | ADMM: alternating x-LS / z-soft-threshold / dual ascent, with adaptive ρ balancing primal vs dual residuals — no hand-tuned penalty |
| 16 | Bregman | Reaching the BP fixed point doesn't need an ℓ₁ program | Bregman iteration: inner least-squares solves + add the closed residual back to the data (residual reload); the penalty loop enforces exact support from the LS side. |
| 17 | HHS | Finding the big coefficients doesn't need scanning all of Aᵀr | Streaming sketches: count-sketch hash-bucketing + multi-round voting finds heavy hitters in O(n) per round, then exact LS pricing on the promoted support |
| 18 | AMP | Replace combinatorial inference with scalar message passing | Belief propagation on dense A, decoupled asymptotically: denoiser + Onsager reaction term make errors Gaussian (state evolution); threshold τ adapts without knowing the noise level |

---

## References

Per-algorithm citations, as cited in each algorithm's script:

**`01_mp.py`**
- M. A. Hammeed, "Comparative Analysis of Orthogonal Matching Pursuit and
  Least Angle Regression," M.S. thesis, Michigan State University, 2012.

**`02_weak_mp.py`**
- T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE Transactions on
  Signal Processing, vol. 56, no. 6, pp. 2370-2382, June 2008. (defines the
  weak/gain selection framework)
- R. Gribonval and P. Vandergheynst, "On the convergence of matching
  pursuit," IEEE Transactions on Information Theory, vol. 52, no. 1,
  pp. 172-180, Jan. 2006.

**`03_omp.py`**
- J. A. Tropp and A. C. Gilbert, "Signal Recovery From Random Measurements
  Via Orthogonal Matching Pursuit," IEEE Transactions on Information Theory,
  vol. 53, no. 12, pp. 4655-4666, Dec. 2007.
- M. A. Hammeed, "Comparative Analysis of Orthogonal Matching Pursuit and
  Least Angle Regression," M.S. thesis, Michigan State University, 2012.

**`04_omp_rls.py`**
- D. Zachariah, S. Chatterjee and M. Jansson, "Online Network Response
  Identification with Recursive Least Squares," Signal Processing,
  vol. 103, pp. 237-246, 2014. (RLS support-extension viewpoint)
- M. A. Hammeed, "Comparative Analysis of Orthogonal Matching Pursuit and
  Least Angle Regression," M.S. thesis, Michigan State University, 2012.
  (OMP termination conventions)

**`05_stomp.py`**
- D. L. Donoho, Y. Tsaig, I. Drori and J.-L. Starck, "Sparse Solution of
  Underdetermined Linear Equations by Stagewise Orthogonal Matching
  Pursuit," IEEE Transactions on Information Theory, vol. 58, no. 2,
  pp. 1094-1121, Feb. 2012.
- A. Majumdar, "Compressed Sensing for Engineers," CRC Press, 2018.

**`06_gradient_pursuit.py`**
- T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE Transactions on
  Signal Processing, vol. 56, no. 6, pp. 2370-2382, June 2008.

**`07_cosamp.py`**
- D. Needell and J. A. Tropp, "CoSaMP: Iterative signal recovery from
  incomplete and inaccurate samples," Applied and Computational Harmonic
  Analysis, vol. 26, no. 3, pp. 301-321, 2009.

**`08_sp.py`**
- W. Dai and O. Milenkovic, "Subspace Pursuit for Compressive Sensing
  Signal Reconstruction," IEEE Transactions on Information Theory,
  vol. 55, no. 5, pp. 2230-2249, May 2009.

**`09_iht.py`**
- T. Blumensath and M. E. Davies, "Iterative Hard Thresholding for
  Compressed Sensing," Applied and Computational Harmonic Analysis,
  vol. 27, no. 3, pp. 265-274, 2009.
- T. Blumensath and M. E. Davies, "Normalized Iterative Hard Thresholding:
  Guaranteed Stability and Performance," IEEE Journal of Selected Topics in
  Signal Processing, vol. 4, no. 2, pp. 298-310, 2010.

**`10_irls.py`**
- R. Chartrand and W. Yin, "Iteratively reweighted algorithms for
  compressive sensing," IEEE ICASSP 2008, pp. 3869-3872,
  doi: 10.1109/ICASSP.2008.4518498.

**`11_focuss.py`**
- I. F. Gorodnitsky and B. D. Rao, "Sparse signal reconstruction from
  limited data using FOCUSS: a re-weighted minimum norm algorithm," IEEE
  Transactions on Signal Processing, vol. 45, no. 3, pp. 600-616, March
  1997, doi: 10.1109/78.558475.

**`12_lars.py`**
- B. Efron, T. Hastie, I. Johnstone and R. Tibshirani, "Least Angle
  Regression," The Annals of Statistics, vol. 32, no. 2, pp. 407-499, 2004.
- M. A. Hammeed, "Comparative Analysis of Orthogonal Matching Pursuit and
  Least Angle Regression," M.S. thesis, Michigan State University, 2012.

**`13_bp.py`**
- S. S. Chen, D. L. Donoho and M. A. Saunders, "Atomic Decomposition by
  Basis Pursuit," SIAM Review, vol. 43, no. 1, pp. 129-159, 2001.
- E. Candès and J. Romberg, "l1-magic: Recovery of Sparse Signals," Caltech
  Technical Report, 2005. (basis pursuit formulation)

**`14_lasso.py`**
- R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," Journal
  of the Royal Statistical Society B, vol. 58, no. 1, pp. 267-288, 1996.
- A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
  Algorithm for Linear Inverse Problems," SIAM Journal on Imaging Sciences,
  vol. 2, no. 1, pp. 183-202, 2009.

**`15_aadm.py`**
- S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed
  Optimization and Statistical Learning via the Alternating Direction
  Method of Multipliers," Foundations and Trends in Machine Learning,
  vol. 3, no. 1, pp. 1-122, 2011. (section 3.4.1: varying the penalty parameter)
- M. Figueiredo and J. Bioucas-Dias, "Restoration of Poissonian Images
  Using Alternating Direction Optimization," IEEE Transactions on Image
  Processing, vol. 20, no. 10, pp. 2752-2766, 2011. (ADMM for sparse reconstruction)

**`16_bregman.py`**
- S. Osher, M. Burger, D. Goldfarb, J. Xu and W. Yin, "An Iterative
  Regularization Method for Total Variation-Based Image Restoration,"
  Multiscale Modeling and Simulation, vol. 4, no. 2, pp. 460-489, 2005.
- W. Yin, S. Osher, D. Goldfarb and J. Darbon, "Bregman Iterative
  Algorithms for l1-Minimization with Applications to Compressive Sensing,"
  SIAM Journal on Imaging Sciences, vol. 1, no. 1, pp. 143-168, 2008.

**`17_hhs.py`**
- G. Cormode and M. Hadjieleftheriou, "Finding Frequent Items in Data
  Streams," Proceedings of the VLDB Endowment, vol. 1, no. 2, pp. 1530-1541,
  2008. (count-sketch heavy hitter tracking)
- P. Berinde, A. C. Gilbert, P. Indyk, H. Karloff and M. J. Strauss,
  "Combining Geometry and Combinatorics: A Unified Approach to Sparse
  Signal Recovery," 46th Annual Allerton Conference on Communication,
  Control, and Computing, 2008. (heavy-hitter recovery in compressed sensing)
- K. L. Clarkson and D. P. Woodruff, "Low Rank Approximation and Regression
  in Input Sparsity Time," STOC 2013, pp. 81-90, 2013. (sketch-based linear algebra)

**`18_amp.py`**
- D. L. Donoho, A. Maleki and A. Montanari, "Message Passing Algorithms for
  Compressed Sensing," Proceedings of the National Academy of Sciences,
  vol. 106, no. 45, pp. 18914-18919, 2009.
- M. Bayati and A. Montanari, "The Dynamics of Message Passing on Dense
  Graphs, with Applications to Compressed Sensing," IEEE Transactions on
  Information Theory, vol. 57, no. 2, pp. 764-785, Feb. 2011. (state evolution analysis)

**Toolbox (`opt/`)**
- N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends in
  Optimization, vol. 1, no. 3, pp. 127-239, 2014. (`opt/prox.py`)
- A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
  Algorithm for Linear Inverse Problems," SIAM Journal on Imaging Sciences,
  vol. 2, no. 1, pp. 183-202, 2009. (`opt/gradient.py`)
- J. R. Shewchuk, "An Introduction to the Conjugate Gradient Method Without
  the Painful Derivation," Carnegie Mellon University, 1994. (`opt/lsqr.py`)
- J. Nocedal and S. J. Wright, "Numerical Optimization," 2nd ed., Springer,
  2006. (line search and conjugate gradient; `opt/gradient.py`, `opt/lsqr.py`)
- S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed
  Optimization and Statistical Learning via the Alternating Direction
  Method of Multipliers," Foundations and Trends in Machine Learning,
  vol. 3, no. 1, pp. 1-122, 2011. (`opt/admm.py`)
- T. S. Ferguson, "Linear Programming: A Concise Introduction," UCLA course
  notes, 2006. (`opt/dantzig.py`)
- S. J. Wright, "Primal-Dual Interior-Point Methods," SIAM, 1997.  (`opt/dantzig.py`)
