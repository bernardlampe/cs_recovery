#!/usr/bin/python
""" Run all reconstruction algorithms and chart convergence vs error.

Driver for the course: generates a test signal, hands it to every
reconstruction algorithm in the repo, records the per-iteration
residual history of each (convergence) and the final relative l2
error (accuracy), then saves two charts into out/.

Residual history plumbing: each algorithm file exposes a module
level HISTORY list that the algorithm appends its residual norm to
each iteration (see the instrumented copies in instrumented()).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

    # dispatch: each algorithm with its own calling convention
    calls = {
        '01_mp':                lambda: mod.mp(y, A, eps_term, 1e-6),
        '02_weak_mp':           lambda: mod.weak_mp(y, A, eps_term, 1e-6, mu=0.9),
        '03_omp':               lambda: mod.omp(y, A, eps_term, 1e-6),
        '04_omp_rls':           lambda: mod.omp_rls(A, y, eps_term, 1e-6),
        '05_stomp':             lambda: mod.stomp(y, A, 3, 100),
        '06_gradient_pursuit':  lambda: mod.gradient_pursuit(y, A, eps_term, 1e-6),
        '07_cosamp':            lambda: mod.cosamp(A, y.ravel(), k, mod.epsilon_term, 1e-6),
        '08_sp':                lambda: mod.subspace_pursuit(A, y.ravel(), k)[0],
        '09_iht':               lambda: mod.iht(y, A, eps_term, 1e-6, k),
        '10_irls':              lambda: mod.irls(y, A, eps_term, 1e-6, 100, 1e-4),
        '11_focuss':            lambda: mod.focuss(y, A, 50),
        '12_lars':              lambda: mod.lars(y, A, eps_term, 1e-6),
        '13_bp':                lambda: mod.bp_linprog(y, A),
        '14_lasso':             lambda: mod.lasso(y, A, eps_term, 1e-6),
        '15_aadm':              lambda: mod.aadm(y, A, eps_term, 1e-6),
        '16_bregman':           lambda: mod.bregman(y, A, eps_term, 1e-6),
        '17_hhs':               lambda: mod.hhs(y, A, eps_term, 1e-6),
        '18_amp':               lambda: mod.amp(y, A, eps_term, 1e-6),
    }
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
    from common.common import gen_test_signal
    import matplotlib.pyplot as plt

    np.random.seed(SEED)
    results = [(name,) + run(mod, K)[1:] for mod, name in MODULES]

    # ---- chart 1: convergence (residual norm vs iteration)
    # dual panel: full view + zoom on the early iterations where the
    # greedy family collapses (their curves hide in a vertical bundle
    # at x ~ 0 in the full view)
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 7),
                                  gridspec_kw={'width_ratios': [2, 1]})
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for (name, hist, _), col in zip(results, colors):
        if hist:
            ax.plot(hist, label=name, color=col, linewidth=1.2, alpha=0.85)
            axz.plot(range(min(len(hist), 30)), hist[:30], color=col, linewidth=1.2, alpha=0.85)
    ax.set_xlabel('iteration')
    ax.set_ylabel(r'$\|y - A x\|_2$')
    ax.set_yscale('log')
    ax.set_title('full horizon')
    axz.set_xlabel('iteration (first 30)')
    axz.set_yscale('log')
    axz.set_title('zoom: first 30 iterations')
    axz.grid(True, which='both', alpha=0.3)
    ax.grid(True, which='both', alpha=0.3)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
               ncol=6, fancybox=True, framealpha=0.7, fontsize=8)
    fig.suptitle(f'Convergence rate (n = {N}, k = {K}, noiseless)')
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('out/convergence.png', dpi=110)
    plt.close()

    # ---- chart 2: final relative error per algorithm
    names = [name for name, _, _ in results]
    errs = [err for _, _, err in results]
    order = np.argsort(errs)                     # best first
    names = [names[i] for i in order]
    errs = [errs[i] for i in order]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), errs, color=plt.cm.viridis(np.linspace(0.05, 0.85, len(names))))
    plt.yscale('log')
    plt.xticks(range(len(names)), names, rotation=35, ha='right', fontsize=9)
    plt.ylabel('relative l2 reconstruction error')
    plt.title(f'Accuracy (n = {N}, k = {K}, noiseless)')
    for b, e in zip(bars, errs):                 # value labels
        plt.text(b.get_x() + b.get_width()/2, e*1.15, f'{e:.1e}',
                 ha='center', va='bottom', fontsize=7, rotation=90)
    plt.tight_layout()
    plt.savefig('out/error_chart.png', dpi=110)
    plt.close()

    print(f"{'algorithm':<10} iters  rel error")
    for name, hist, err in results:
        print(f"{name:<10} {len(hist):>5}  {err:.3e}")

if __name__ == "__main__":
    main()
