from typing import Optional

import numpy as np


class FiniteProductBenchmark:
    """
    Base class for finite product search spaces:
        X = {0, ..., m-1}^d

    The algorithm observes:
        reward = mean_reward(x) + noise

    Regret should be computed using the noiseless mean_reward(x).

    Note: rewards are *not* clipped, so a noisy observation can fall slightly
    outside [0, 1]. The noiseless ``mean_reward`` is always in [0, 1] with a
    maximum of exactly 1, so ``regret`` stays in [0, 1].
    """

    def __init__(
        self,
        dim: int = 10,
        m: int = 10,
        noise_std: float = 0.05,
        seed: int = 0,
    ):
        self.dim = dim
        self.m = m
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Generic finite product search space.
        # Each coordinate x_i belongs to {0, ..., m-1}.
        self.search_space = {f"x{i}": list(range(m)) for i in range(dim)}

        # All proposed functions below are normalized with maximum 1.
        self.max_value = 1.0

    def get_search_space(self) -> dict[str, dict]:
        """Return the search space in the dict-of-specs format expected by
        optimizers such as :class:`imabo.IMABO`.

        Each coordinate is exposed as a categorical variable, which is the
        faithful modelling for these combinatorial (Hamming / circular / NK)
        landscapes: the optimizer is not told the coordinates carry any ordinal
        structure.
        """
        return {name: {"choices": list(range(self.m))} for name in self.search_space}

    def parse_input(self, x) -> np.ndarray:
        """
        Accepts either:
        - list / tuple / np.ndarray of integers
        - dict like {"x0": 3, "x1": 7, ...}
        """
        if isinstance(x, dict):
            arr = np.array([x[f"x{i}"] for i in range(self.dim)], dtype=int)
        else:
            arr = np.array(x, dtype=int)

        if arr.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {arr.shape[0]}.")

        if np.any(arr < 0) or np.any(arr >= self.m):
            raise ValueError(f"All coordinates must be in {{0, ..., {self.m - 1}}}.")

        return arr

    def mean_reward(self, x) -> float:
        raise NotImplementedError

    def __call__(self, x, noise: bool = True) -> float:
        value = float(self.mean_reward(x))

        if noise and self.noise_std > 0:
            value += self.rng.normal(0.0, self.noise_std)

        return value

    def regret(self, x) -> float:
        """
        Simple instantaneous regret using the noiseless mean.
        """
        return self.max_value - self.mean_reward(x)


class PairwiseInteractionFinite(FiniteProductBenchmark):
    """
    Finite product function with unary terms and pairwise compatibility terms.

    X = {0, ..., m-1}^d

    The maximum is exactly 1, attained at x_star.
    """

    def __init__(
        self,
        dim: int = 10,
        m: int = 10,
        noise_std: float = 0.05,
        seed: int = 0,
        lambda_pair: float = 0.5,
        smoothness: float = 2.0,
        edges: Optional[list[tuple[int, int]]] = None,
    ):
        super().__init__(
            dim=dim,
            m=m,
            noise_std=noise_std,
            seed=seed,
        )

        self.lambda_pair = lambda_pair
        self.smoothness = smoothness

        # Planted optimal configuration.
        self.x_star = self.rng.integers(0, m, size=dim)

        # Default graph: chain interactions.
        if edges is None:
            edges = [(i, i + 1) for i in range(dim - 1)]
        self.edges = edges

        # Pairwise offsets chosen so that x_star satisfies all constraints.
        self.offsets = {
            (i, j): int((self.x_star[j] - self.x_star[i]) % m) for (i, j) in self.edges
        }

    def circular_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = np.abs(a - b)
        return np.minimum(diff, self.m - diff)

    def unary_score(self, x: np.ndarray) -> float:
        """
        Smooth discrete peak around x_star.
        Each coordinate score is in (0, 1], equal to 1 at x_star[j].
        """
        dist = self.circular_distance(x, self.x_star)
        scores = np.exp(-(dist**2) / (2.0 * self.smoothness**2))
        return float(np.mean(scores))

    def pairwise_score(self, x: np.ndarray) -> float:
        """
        Pairwise compatibility score.
        Equal to 1 if every edge satisfies the planted relation.
        """
        if len(self.edges) == 0:
            return 1.0

        good = 0
        for i, j in self.edges:
            offset = self.offsets[(i, j)]
            if x[j] == (x[i] + offset) % self.m:
                good += 1

        return good / len(self.edges)

    def mean_reward(self, x) -> float:
        x = self.parse_input(x)

        unary = self.unary_score(x)
        pairwise = self.pairwise_score(x)

        value = (1.0 - self.lambda_pair) * unary + self.lambda_pair * pairwise
        return float(value)


class HammingPeaksFinite(FiniteProductBenchmark):
    """
    Multi-peak finite product function based on normalized Hamming distance.

    X = {0, ..., m-1}^d

    The maximum is exactly 1, attained at any planted center.
    """

    def __init__(
        self,
        dim: int = 10,
        m: int = 10,
        noise_std: float = 0.05,
        seed: int = 0,
        n_peaks: int = 5,
        gamma: float = 5.0,
    ):
        super().__init__(
            dim=dim,
            m=m,
            noise_std=noise_std,
            seed=seed,
        )

        self.n_peaks = n_peaks
        self.gamma = gamma

        # Hidden peak centers.
        self.centers = self.rng.integers(0, m, size=(n_peaks, dim))

    def normalized_hamming(self, x: np.ndarray, c: np.ndarray) -> float:
        return float(np.mean(x != c))

    def mean_reward(self, x) -> float:
        x = self.parse_input(x)

        values = []
        for c in self.centers:
            h = self.normalized_hamming(x, c)
            values.append(np.exp(-self.gamma * h))

        return float(np.max(values))


class PlantedNKFinite(FiniteProductBenchmark):
    """
    Planted NK-style finite product landscape.

    X = {0, ..., m-1}^d

    Each local term depends on x_j and K neighbors.
    The maximum is exactly 1, attained at x_star.

    K controls interaction difficulty:
        K = 0: additive
        K = 1 or 2: mild interactions
        K >= 3: more rugged combinatorial landscape
    """

    def __init__(
        self,
        dim: int = 10,
        m: int = 10,
        noise_std: float = 0.05,
        seed: int = 0,
        K: int = 2,
        tau: float = 1.5,
    ):
        super().__init__(
            dim=dim,
            m=m,
            noise_std=noise_std,
            seed=seed,
        )

        if K < 0 or K >= dim:
            raise ValueError("K must satisfy 0 <= K < dim.")

        self.K = K
        self.tau = tau

        # Planted optimal configuration.
        self.x_star = self.rng.integers(0, m, size=dim)

        # Cyclic neighborhoods.
        self.neighborhoods = {
            j: [((j + offset) % dim) for offset in range(1, K + 1)] for j in range(dim)
        }

    def local_term(self, x: np.ndarray, j: int) -> float:
        """
        Local interaction term around coordinate j.
        Equal to 1 if x_j and its neighborhood match x_star.
        """
        mismatch = 0

        if x[j] != self.x_star[j]:
            mismatch += 1

        for k in self.neighborhoods[j]:
            if x[k] != self.x_star[k]:
                mismatch += 1

        return float(np.exp(-mismatch / self.tau))

    def mean_reward(self, x) -> float:
        x = self.parse_input(x)

        terms = [self.local_term(x, j) for j in range(self.dim)]
        return float(np.mean(terms))


if __name__ == "__main__":
    # Demo: run IMABO on each finite-product benchmark and plot cumulative regret.
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from experiments.baselines.random_search import RandomSearch
    from imabo import IMABO, IMABOTabFM

    RESULT_DIR = Path(__file__).parent.parent.parent.parent / "results"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    n_iterations = 5000
    dim = 6
    m = 6
    benchmarks = {
        "Hamming Peaks": HammingPeaksFinite(
            dim=dim, m=m, noise_std=0.05, seed=2, n_peaks=3, gamma=1.0
        ),
        "Planted NK": PlantedNKFinite(
            dim=dim, m=m, noise_std=0.05, seed=2, K=1, tau=2.0
        ),
        "Pairwise Interaction": PairwiseInteractionFinite(
            dim=dim, m=m, noise_std=0.05, seed=2, lambda_pair=0.5, smoothness=2.5
        ),
    }

    csv_rows = []

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 5))
    for ax, (label, fn) in zip(axes, benchmarks.items()):
        # HammingPeaksFinite has multiple planted optima (`centers`) instead
        # of a single `x_star`; fall back to the first center for the demo.
        example_optimum = fn.x_star if hasattr(fn, "x_star") else fn.centers[0]

        print(f"\n=== {label} ===")
        print("example optimum:", example_optimum)
        print("theoretical max:", fn.max_value)
        print("mean at optimum:", fn.mean_reward(example_optimum))

        optimizers = {
            "IMABO": IMABO(
                search_space=fn.get_search_space(), seed=0, multivariate=True
            ),
            "IMABO-noTPE": IMABO(
                search_space=fn.get_search_space(),
                seed=0,
                multivariate=True,
                use_tpe=False,
            ),
            "Random Search": RandomSearch(search_space=fn.get_search_space(), seed=0),
            "TabFM-IMABO": IMABOTabFM(
                search_space=fn.get_search_space(),
                seed=0,
            ),
        }

        for opt_label, opt in optimizers.items():
            run_key = f"{label} - {opt_label}"
            seen_configs = set()

            regrets = []
            for _ in tqdm(range(n_iterations), desc=f"{label} / {opt_label}"):
                x = opt.suggest()
                y = fn(x, noise=True)
                opt.observe(y)  # noisy reward
                regrets.append(fn.regret(x))  # noiseless regret, already in [0, 1]

                config_key = tuple(sorted(x.items()))
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    csv_rows.append(
                        {
                            "opt_running": run_key,
                            "configuration": json.dumps(x, sort_keys=True),
                            "noisy_reward": y,
                            "noiseless_reward": fn.mean_reward(x),
                        }
                    )

            best = opt.best_config
            print(f"[{opt_label}] best config:", best)
            print(f"[{opt_label}] simple regret:", fn.regret(best))

            ax.plot(np.cumsum(regrets), label=opt_label)

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(label)
        ax.legend()

    fig.suptitle(
        "Cumulative Regret: IMABO vs. IMABO-noTPE on Finite-Product Benchmarks"
    )
    fig.tight_layout()
    plt.show()

    # csv_path = RESULT_DIR / f"finite_product_history_dim{dim}_m{m}.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.DictWriter(
    #         f,
    #         fieldnames=[
    #             "opt_running",
    #             "configuration",
    #             "noisy_reward",
    #             "noiseless_reward",
    #         ],
    #     )
    #     writer.writeheader()
    #     writer.writerows(csv_rows)
    # print(f"\nSaved {len(csv_rows)} unique (run, config) rows to {csv_path}")
