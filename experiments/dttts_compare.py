"""Compare standard D-TTTS against IMABO (with/without oracle) on toy functions.

Both D-TTTS (Beta/Bernoulli) *and* IMABO's MOSS index assume rewards in [0,1]:
D-TTTS binarizes, and the MOSS confidence bonus ``sqrt(.../n_x)`` is an O(1)
radius calibrated for a unit reward range (the paper's Theorem 1 proof uses
"per-round regret is at most 1" and the minimax bound ``cM*sqrt(NK)``).  The toy
objectives do NOT satisfy this -- they are sums over ``dim`` coordinates
(sin1/garland in ~[0,4], rastrigin in ~[-185,0]) -- so feeding raw rewards
mis-calibrates MOSS (negligible bonus on rastrigin => near-greedy).

We therefore normalise rewards to [0,1] with offline min-max bounds BEFORE
calling observe(), and feed the SAME normalised reward to *every* algorithm.
IMABO validates that its rewards lie in [0,1] (raising otherwise); D-TTTS
receives them already in [0,1] for its Bernoulli binarization.  Regret is
reported in that same normalised [0,1] space (per-round regret in [0,1]),
matching the paper's "normalized cumulative/simple regret".

Noise model (``--sigma``):
  * ``sigma=None`` (default): use the toy function's own built-in evaluation
    noise (small, ~0.01 on the raw scale), normalised into [0,1].  Rewards stay
    in [0,1] so IMABO's range check is left ON.
  * ``sigma=<float>``: ignore the built-in noise and instead add explicit
    Gaussian noise ``N(0, sigma)`` on the [0,1]-rescaled reward.  Additive noise
    can push a reward just outside [0,1], so IMABO's range check is bypassed
    (check_reward_range=False).  Use this to study noise sensitivity (e.g. the
    Random-vs-IMABO crossover): at the toy's tiny built-in noise pure breadth
    (Random) is competitive; at sigma>=0.1 IMABO's re-pulling wins.

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


def reward_bounds(function_name, dim, n_sample=200_000, pad=0.02):
    """Offline (low, high) bounds on the noiseless reward, padded outward.

    Guarantees the theoretical optimum (fmax * dim) is inside [low, high] so the
    normalised reward and regret both lie in [0,1].
    """
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
    hi = max(float(vals.max()), fmax * dim)  # ensure the true optimum is included
    span = hi - lo
    return lo - pad * span, hi + pad * span


def make_optimizer(name, ss, seed, check_range=True):
    # rewards are normalised to [0,1] by one_run() before observe();
    # IMABO validates the [0,1] range, D-TTTS binarizes it.  Random search sees
    # the same normalised reward and needs no [0,1] assumption.  With explicit
    # additive noise (sigma set) the range check is bypassed.
    if name == "IMABO (TPE oracle)":
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=True,
                     check_reward_range=check_range)
    if name == "IMABO (no oracle)":
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=False,
                     check_reward_range=check_range)
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


def one_run(function_name, dim, n_iter, seed, bounds, sigma=None):
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)          # built-in noise
    fn0 = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)
    ss = obj.get_search_space(function_name)
    lo, hi = bounds
    span = hi - lo
    noise_rng = np.random.default_rng(10_000 + seed)

    def norm(v):  # affine map into [0,1], clipped
        return min(1.0, max(0.0, (v - lo) / span))

    fmax_total_norm = norm(fmax * dim)  # normalised optimum (=1 after padding)

    out = {}
    for name in RUN_ALGOS:
        opt = make_optimizer(name, ss, seed, check_range=(sigma is None))
        regrets = np.empty(n_iter)
        for i in range(n_iter):
            x = opt.suggest()
            if sigma is None:
                reward = norm(func(x))                       # toy's built-in noise, normalised
            else:
                reward = norm(fn0(x)) + noise_rng.normal(0.0, sigma)  # explicit [0,1] noise
            opt.observe(reward)
            # normalised per-round regret in [0,1], scored on the noiseless f
            regrets[i] = fmax_total_norm - norm(fn0(x))
        bx = opt.best_config
        sr = fmax_total_norm - norm(fn0(bx))
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
        bounds = reward_bounds(fn, dim)
        runs = Parallel(n_jobs=8, backend="threading")(
            delayed(one_run)(fn, dim, n_iter, base_seed + r * 1000, bounds, sigma)
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
