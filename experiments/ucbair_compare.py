"""Compare UCB-AIR against MOSS on the toy functions.

Three algorithms, all on rewards normalised to [0, 1] (min-max via offline
bounds), regret reported in that normalised space:

  * UCB-AIR             -- arm-increasing schedule K(t)=ceil(t^{b/(b+1)}) + UCB1
  * MOSS-AIR            -- the SAME arm-increasing schedule, MOSS index (isolates
                           the index: only UCB1 vs MOSS differs)
  * IMABO (no oracle)   -- the shipped infinite-armed MOSS: uniform reservoir,
                           |M_t| < t^beta arm schedule, MOSS index (the practical
                           "I-MOSS" baseline already in the repo)

Usage: python -m experiments.ucbair_compare [n_iter] [n_runs] [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments.baselines.ucb_air import UCBAIR, MOSSAIR
from experiments.baselines.qrm2 import QRM2
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from imabo import IMABO

ALGOS = ["UCB-AIR", "MOSS-AIR", "QRM2", "IMABO (no oracle)", "IMABO (matched, b=0.5)"]


def reward_bounds(function_name, dim, n_sample=200_000, pad=0.02):
    obj = ObjectiveFunctions(dim=dim, noise_seed=0)
    fn0 = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)
    ss = obj.get_search_space(function_name)
    keys = sorted(ss)
    los = np.array([ss[k]["lower"] for k in keys])
    his = np.array([ss[k]["upper"] for k in keys])
    rng = np.random.default_rng(0)
    X = rng.uniform(los, his, size=(n_sample, len(keys)))
    vals = np.array([fn0(dict(zip(keys, x))) for x in X])
    lo = float(vals.min())
    hi = max(float(vals.max()), fmax * dim)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def make_optimizer(name, ss, seed):
    if name == "UCB-AIR":
        return UCBAIR(search_space=ss, beta=1.0, seed=seed)
    if name == "MOSS-AIR":
        return MOSSAIR(search_space=ss, beta=1.0, seed=seed)
    if name == "QRM2":
        return QRM2(search_space=ss, seed=seed)
    if name == "IMABO (no oracle)":
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=False)
    if name == "IMABO (matched, b=0.5)":
        # beta=0.5 opens ceil(t^0.5) arms == MOSS-AIR's schedule (same MOSS index)
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=False, beta=0.5)
    raise ValueError(name)


def one_run(function_name, dim, n_iter, seed, bounds):
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)          # toy built-in noise
    fn0 = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)
    ss = obj.get_search_space(function_name)
    lo, hi = bounds
    span = hi - lo

    def norm(v):
        return min(1.0, max(0.0, (v - lo) / span))

    fmax_total_norm = norm(fmax * dim)
    out = {}
    for name in ALGOS:
        opt = make_optimizer(name, ss, seed)
        regrets = np.empty(n_iter)
        for i in range(n_iter):
            x = opt.suggest()
            opt.observe(norm(func(x)))                       # normalised reward in [0,1]
            regrets[i] = fmax_total_norm - norm(fn0(x))      # normalised per-round regret
        bx = opt.best_config
        sr = fmax_total_norm - norm(fn0(bx))
        out[name] = {"regrets": regrets, "simple_regret": float(sr)}
    return out


def main():
    dim = 4
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    functions = ["sin1", "garland", "rastrigin"]
    base_seed = 42

    results = {}
    for fn in functions:
        bounds = reward_bounds(fn, dim)
        runs = Parallel(n_jobs=8, backend="threading")(
            delayed(one_run)(fn, dim, n_iter, base_seed + r * 1000, bounds)
            for r in range(n_runs)
        )
        agg = {}
        for name in ALGOS:
            R = np.stack([run[name]["regrets"] for run in runs])
            SR = np.array([run[name]["simple_regret"] for run in runs])
            cum = R.cumsum(axis=1)
            agg[name] = {
                "mean_cum_regret": cum.mean(axis=0).tolist(),
                "std_cum_regret": cum.std(axis=0).tolist(),
                "final_cum_regret_mean": float(cum[:, -1].mean()),
                "final_cum_regret_std": float(cum[:, -1].std()),
                "simple_regret_mean": float(SR.mean()),
                "simple_regret_std": float(SR.std()),
                "simple_regret_all": SR.tolist(),
            }
        results[fn] = agg
        print(f"[{fn}] done")
        for name in ALGOS:
            a = agg[name]
            print(f"   {name:20s} simple={a['simple_regret_mean']:.4f}±{a['simple_regret_std']:.4f}"
                  f"  cumreg={a['final_cum_regret_mean']:.1f}")

    meta = {"dim": dim, "n_iter": n_iter, "n_runs": n_runs, "functions": functions,
            "algos": ALGOS, "results": results}
    outpath = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("ucbair_compare_results.json")
    outpath.write_text(json.dumps(meta))
    print("WROTE", outpath)


if __name__ == "__main__":
    main()
