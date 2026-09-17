#!/usr/bin/python
""" Run all reconstruction algorithms and chart convergence vs error.

Driver for the course: generates a test signal, hands it to every
reconstruction algorithm in the repo, records the per-iteration
residual history of each (convergence) and the final relative l2
error (accuracy), then saves two charts into out/.

Residual history plumbing: each algorithm file exposes a module
level HISTORY list that the algorithm appends its residual norm to
each iteration (see the instrumented copies in instrumented()).

Parallelism: every (algorithm, case) pair is dispatched to a process
pool (max MAX_WORKERS) since the solvers are independent; charts stay
in the parent (matplotlib is not fork-tolerant everywhere).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_WORKERS = 10               # concurrent algorithm evaluations

MODULES = [
    ('01_mp',              'MP'),
    ('02_weak_mp',         'weak MP'),
    ('03_omp',             'OMP'),
    ('04_omp_rls',         'OMP-RLS'),
    ('05_stomp',           'StOMP'),
    ('06_gradient_pursuit','GP'),
    ('07_cosamp',          'CoSaMP'),
    ('08_sp',              'SP'),
    ('09_iht',             'IHT'),
    ('10_irls',            'IRLS'),
    ('11_focuss',          'FOCUSS'),
    ('12_lars',            'LARS'),
    ('13_bp',              'BP-LP'),
    ('14_lasso',           'LASSO'),
    ('15_aadm',            'AADM'),
    ('16_bregman',         'Bregman'),
    ('17_hhs',             'HHS'),
    ('18_amp',             'AMP'),
]

K = 10                 # true sparsity
N = 200                # signal dimension
SEED = 1
SNR_DB = 20            # noise level for the noisy case

CASES = [('clean', None), ('noisy', SNR_DB)]

def run(modname, k, snr_db=None):
    """
    run one algorithm on the standard test signal
    returns (name, hist, rel_err)
    """
    import importlib
    from common.common import gen_test_signal

    np.random.seed(SEED)
    y, A, x_t, x_f = gen_test_signal(k=k, n=N, amps_l=-10, amps_h=10, snr_db=snr_db)
    mod = importlib.import_module(modname)
    name = dict((m, nm) for m, nm in MODULES)[modname]
    hist = []

    # instrumented epsilon_term: captures residual norms per iteration
    def eps_term(e, yy, r, x_hat):
        hist.append(float(np.linalg.norm(r)))
        return float(np.linalg.norm(r)) < e

    # noisy measurements create a residual floor ||n||_2 > 0 that an
    # epsilon-based greedy termination can never reach: such solvers
    # would keep selecting atoms until they perfectly interpolate the
    # noise. In the noisy case cap their support at the true sparsity k.
    greedy = ['01_mp', '02_weak_mp', '03_omp', '04_omp_rls',
              '06_gradient_pursuit', '12_lars']
    e_term = (getattr(mod, 'sparsity_term', eps_term), k) if snr_db else (eps_term, 1e-6)
    term_fn, term_param = e_term if modname in greedy else (eps_term, 1e-6)
    # dispatch: each algorithm with its own calling convention
    calls = {
        '01_mp':                lambda: mod.mp(y, A, term_fn, term_param),
        '02_weak_mp':           lambda: mod.weak_mp(y, A, term_fn, term_param, mu=0.9),
        '03_omp':               lambda: mod.omp(y, A, term_fn, term_param),
        '04_omp_rls':           lambda: mod.omp_rls(A, y, term_fn, term_param),
        '05_stomp':             lambda: mod.stomp(y, A, 3, 100),
        '06_gradient_pursuit':  lambda: mod.gradient_pursuit(y, A, term_fn, term_param),
        '07_cosamp':            lambda: mod.cosamp(A, y.ravel(), k, mod.epsilon_term, 1e-6),
        '08_sp':                lambda: mod.subspace_pursuit(A, y.ravel(), k)[0],
        '09_iht':               lambda: mod.iht(y, A, eps_term, 1e-6, k),
        '10_irls':              lambda: mod.irls(y, A, eps_term, 1e-6, 100, 1e-4),
        '11_focuss':            lambda: mod.focuss(y, A, 50, lam=0.1 if snr_db else 0.0),
        '12_lars':              lambda: mod.lars(y, A, term_fn, term_param),
        '13_bp':                lambda: mod.bp_linprog(y, A),
        '14_lasso':             lambda: mod.lasso(y, A, eps_term, 1e-6),
        '15_aadm':              lambda: mod.aadm(y, A, eps_term, 1e-6),
        '16_bregman':           lambda: mod.bregman(y, A, eps_term, 1e-6),
        '17_hhs':               lambda: mod.hhs(y, A, eps_term, 1e-6),
        '18_amp':               lambda: mod.amp(y, A, eps_term, 1e-6),
    }
    # HHS/SP/BP use their own termination; greedy list uses term_fn above
    x_hat = calls[modname]()

    # HISTORY hook: modules that log their own per-iteration residuals
    if hasattr(mod, 'HISTORY') and mod.HISTORY:
        hist = list(mod.HISTORY)
        mod.HISTORY = []
    # SP returns its own history
    if modname == '08_sp':
        _, _, hist = mod.subspace_pursuit(A, y.ravel(), k)

    x_hat = np.asarray(x_hat).ravel()
    rel = float(np.linalg.norm(x_hat - x_f.ravel()) / np.linalg.norm(x_f.ravel()))
    return (name, hist, rel)

def main():
    # parallelize the independent (algorithm, case) evaluations: 36 jobs
    # (18 algorithms x 2 measurement cases) over MAX_WORKERS processes.
    # The pool keeps charts in the parent; matplotlib is not fork-safe
    # everywhere, so no plotting happens in the children.
    import concurrent.futures as cf
    all_results = {}
    jobs = [(mod, case, snr, K) for case, snr in CASES for mod, _ in MODULES]
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run, mod, k, snr): (case, mod)
                   for mod, case, snr, k in jobs}
        for fut in cf.as_completed(futures):
            case, mod = futures[fut]
            all_results.setdefault(case, []).append((mod,) + fut.result()[1:])

    # restore canonical MODULES ordering + display names for the charts
    order = dict((m, i) for i, (m, _) in enumerate(MODULES))
    display = dict((m, nm) for m, nm in MODULES)
    for case in all_results:
        all_results[case] = [tuple((display[m],) + tuple(rest))
                             for m, *rest in sorted(all_results[case],
                                                    key=lambda t: order[t[0]])]
    for case in all_results:
        all_results[case] = list(all_results[case])
    all_results = dict(all_results)

    clean = dict((name, (hist, err)) for name, hist, err in all_results['clean'])

    # ---- chart 1: convergence, clean and noisy side by side
    colors = plt.cm.tab20(np.linspace(0, 1, len(MODULES)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12),
                             gridspec_kw={'width_ratios': [2, 1]})
    for row, (case, _) in enumerate(CASES):
        ax, axz = axes[row]
        for (name, hist, _), col in zip(all_results[case], colors):
            if hist:
                ax.plot(hist, label=name, color=col, linewidth=1.2, alpha=0.85)
                axz.plot(range(min(len(hist), 30)), hist[:30], color=col,
                         linewidth=1.2, alpha=0.85)
        ax.set_ylabel(r'$\|y - A x\|_2$')
        ax.set_yscale('log')
        ax.set_title(f'full horizon - {case}')
        axz.set_yscale('log')
        axz.set_title(f'zoom: first 30 iterations - {case}')
        axz.grid(True, which='both', alpha=0.3)
        ax.grid(True, which='both', alpha=0.3)
        if row == 0:
            ax.set_xlabel('')
            axz.set_xlabel('')
        else:
            ax.set_xlabel('iteration')
            axz.set_xlabel('iteration (first 30)')
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center',
               ncol=6, fancybox=True, framealpha=0.7, fontsize=8)
    fig.suptitle(f'Convergence rate (n = {N}, k = {K}); '
                 f'clean vs noise at SNR = {SNR_DB} dB')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('out/convergence.png', dpi=110)
    plt.close()

    # ---- chart 2: final relative error; clean drives the ranking
    clean_order = sorted(clean, key=lambda nm: clean[nm][1])   # best first
    clean_errs = [clean[nm][1] for nm in clean_order]
    noisy = dict((name, (hist, err)) for name, hist, err in all_results['noisy'])
    noisy_errs = [noisy[nm][1] for nm in clean_order]

    ypos = np.arange(len(clean_order))
    h = 0.4
    b1 = plt.barh(ypos - h/2, clean_errs, height=h,
                  label='clean', color=plt.cm.viridis(0.2))
    b2 = plt.barh(ypos + h/2, noisy_errs, height=h,
                  label=f'noisy ({SNR_DB} dB)', color=plt.cm.viridis(0.65))
    plt.xscale('log')
    plt.yticks(ypos, clean_order, fontsize=9)
    plt.xlabel('relative l2 reconstruction error')
    plt.title(f'Accuracy (n = {N}, k = {K}) - ranked by clean error')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=2, framealpha=0.9)
    for bars, errs in [(b1, clean_errs), (b2, noisy_errs)]:
        for b, e in zip(bars, errs):
            plt.text(e * 1.12, b.get_y() + b.get_height()/2, f'{e:.1e}',
                     va='center', fontsize=6)
    plt.xlim(right=max(noisy_errs + clean_errs) * 12)
    plt.tight_layout()
    plt.savefig('out/error_chart.png', dpi=110)
    plt.close()

    # ---- console report: statistics from the CLEAN case only
    print(f"{'algorithm':<10} {'iters':>10}  {'rel error (clean)':>18}")
    for name in clean_order:
        hist, err = clean[name]
        print(f"{name:<10} {len(hist):>10}  {err:>18.3e}")
    best = clean_order[0]
    print(f"\nbest (clean): {best}  rel_err = {clean[best][1]:.3e}")

if __name__ == "__main__":
    main()
