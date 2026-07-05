import numpy as np
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions


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
