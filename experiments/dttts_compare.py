"""Compare standard D-TTTS against IMABO (with/without oracle) on toy functions.

Both D-TTTS (Beta/Bernoulli) *and* IMABO's MOSS index assume rewards in [0,1]:
D-TTTS binarizes, and the MOSS confidence bonus ``sqrt(.../n_x)`` is an O(1)
radius calibrated for a unit reward range (the paper's Theorem 1 proof uses
"per-round regret is at most 1" and the minimax bound ``cM*sqrt(NK)``).

``ObjectiveFunctions`` keeps every per-dimension term (``sin1_1d``,
``garland_1d``, ``rastrigin_1d``) roughly in [0,1], so dividing the ``dim``-sum
by ``dim`` gives a reward that is already close to [0,1] -- no offline min-max
sampling needed.  This reward is *not* hard-clamped: additive noise (built-in
or explicit ``sigma``) can push it slightly outside [0,1], which is fine --
IMABO does not enforce the range, and D-TTTS clips internally for its
Bernoulli binarization.  We feed the SAME (dim-normalised) reward to every
algorithm, and report regret in that same space.

Noise model (``--sigma``):
  * ``sigma=None`` (default): use the toy function's own built-in evaluation
    noise (small, ~0.01 per-dimension, dim-invariant after the /dim divide).
  * ``sigma=<float>``: ignore the built-in noise and instead add explicit
    Gaussian noise ``N(0, sigma)`` on the dim-normalised reward.  Use this to
    study noise sensitivity (e.g. the Random-vs-IMABO crossover): at the toy's
    tiny built-in noise pure breadth (Random) is competitive; at sigma>=0.1
    IMABO's re-pulling wins.

Usage: python -m experiments.dttts_compare [n_iter] [n_runs] [out.json] [sigma]
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments.baselines.dttts import DTTTS
from experiments.baselines.random_search import RandomSearch
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from imabo import IMABO


def make_optimizer(name, ss, seed):
    if name == "IMABO (TPE oracle)":
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=True)
    if name == "IMABO (no oracle)":
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=False)
    if name == "D-TTTS":
        return DTTTS(search_space=ss, reward_low=0.0, reward_high=1.0, seed=seed)
    if name == "Random":
        return RandomSearch(search_space=ss, seed=seed)
    raise ValueError(name)


# D-TTTS uses the paper-faithful recommendation rule (argmax posterior
# probability of being optimal), which is its default best_config.  Random
# search is the any-space honest floor.
RUN_ALGOS = ALGOS = [
    "IMABO (TPE oracle)",
    "IMABO (no oracle)",
    "D-TTTS",
    "Random",
]


def one_run(function_name, dim, n_iter, seed, sigma=None):
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)          # built-in noise
    fn0 = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)            # per-dim max, ~[0,1]
    ss = obj.get_search_space(function_name)
    noise_rng = np.random.default_rng(10_000 + seed)

    out = {}
    for name in RUN_ALGOS:
        opt = make_optimizer(name, ss, seed)
        regrets = np.empty(n_iter)
        for i in range(n_iter):
            x = opt.suggest()
            if sigma is None:
                reward = func(x) / dim                       # toy's built-in noise
            else:
                reward = fn0(x) / dim + noise_rng.normal(0.0, sigma)  # explicit noise
            opt.observe(reward)
            # per-round regret, scored on the noiseless f
            regrets[i] = fmax - fn0(x) / dim
        bx = opt.best_config
        sr = fmax - fn0(bx) / dim
        out[name] = {"regrets": regrets, "simple_regret": float(sr)}
    return out


def main():
    dim = 4
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else None  # None => toy built-in noise
    functions = ["sin1", "garland", "rastrigin"]
    base_seed = 42

    results = {}
    for fn in functions:
        runs = Parallel(n_jobs=8, backend="threading")(
            delayed(one_run)(fn, dim, n_iter, base_seed + r * 1000, sigma)
            for r in range(n_runs)
        )
        # aggregate
        agg = {}
        for name in ALGOS:
            R = np.stack([run[name]["regrets"] for run in runs])          # (runs, T)
            SR = np.array([run[name]["simple_regret"] for run in runs])   # (runs,)
            cum = R.cumsum(axis=1)                                        # (runs, T)
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
        print(f"[{fn}] done" + (f" (sigma={sigma})" if sigma is not None else " (built-in noise)"))
        for name in ALGOS:
            a = agg[name]
            print(f"   {name:24s} simple={a['simple_regret_mean']:.4f}±{a['simple_regret_std']:.4f}"
                  f"  cumreg={a['final_cum_regret_mean']:.1f}")

    meta = {"dim": dim, "n_iter": n_iter, "n_runs": n_runs, "sigma": sigma,
            "functions": functions, "algos": ALGOS, "results": results}
    outpath = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("dttts_compare_results.json")
    outpath.write_text(json.dumps(meta))
    print("WROTE", outpath)


if __name__ == "__main__":
    main()
