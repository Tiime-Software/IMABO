"""Sweep IMABO no-oracle's beta (arm-opening exponent) to match MOSS-AIR.

MOSS-AIR opens ceil(t^0.5) arms; IMABO no-oracle opens ~t^beta.  Setting
IMABO beta=0.5 should reproduce MOSS-AIR's schedule and therefore its
results (both use the identical MOSS index).  We sweep beta in {0.5,0.6,0.7,0.8}
and print, per function, arms-opened + simple/cumulative regret next to the
MOSS-AIR reference.
"""
import sys
import numpy as np
from joblib import Parallel, delayed

from experiments.baselines.ucb_air import MOSSAIR
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from imabo import IMABO

BETAS = [0.5, 0.6, 0.7, 0.8]
FUNCS = ["sin1", "garland", "rastrigin"]


def one_run(function_name, dim, n_iter, seed):
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)
    fn0 = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)  # per-dim max, ~[0,1]
    ss = obj.get_search_space(function_name)

    row = {}
    # MOSS-AIR reference
    opt = MOSSAIR(search_space=ss, beta=1.0, seed=seed)
    cr = 0.0
    for _ in range(n_iter):
        x = opt.suggest(); opt.observe(func(x) / dim); cr += fmax - fn0(x) / dim
    row["MOSS-AIR"] = (len(opt.arms), fmax - fn0(opt.best_config) / dim, cr)
    # IMABO no-oracle at each beta
    for b in BETAS:
        opt = IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=False, beta=b)
        cr = 0.0
        for _ in range(n_iter):
            x = opt.suggest(); opt.observe(func(x) / dim); cr += fmax - fn0(x) / dim
        arms = len(opt.memory.get_current_state().arms)
        row[f"IMABO b={b}"] = (arms, fmax - fn0(opt.best_config) / dim, cr)
    return row


def main():
    dim = 4
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    base_seed = 42
    labels = ["MOSS-AIR"] + [f"IMABO b={b}" for b in BETAS]
    for fn in FUNCS:
        runs = Parallel(n_jobs=8, backend="threading")(
            delayed(one_run)(fn, dim, n_iter, base_seed + r * 1000)
            for r in range(n_runs)
        )
        print(f"[{fn}]")
        for lab in labels:
            arms = np.mean([r[lab][0] for r in runs])
            sr = np.mean([r[lab][1] for r in runs])
            cr = np.mean([r[lab][2] for r in runs])
            print(f"   {lab:14s} arms={arms:6.0f}  simple={sr:.4f}  cumreg={cr:7.1f}")


if __name__ == "__main__":
    main()
