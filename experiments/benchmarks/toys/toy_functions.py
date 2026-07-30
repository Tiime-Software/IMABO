import numpy as np


# Reuse the EXACT same bimodal Gaussian landscape (local mode at MODE_LO * 1,
# global mode at MODE_HI * 1) that experiments.coordination_barrier_experiment
# builds as its ``family_d`` -- one landscape definition, not a second copy of
# the formula. Imported lazily (only when "gaussian" is actually requested)
# since coordination_barrier_experiment pulls in HierMAB / IMABO / stroquool
# and creates its results directory as an import side effect, none of which
# the rest of this class needs.
def _gaussian_mu_family(dim: int):
    from experiments.coordination_barrier_experiment import make_mu_family

    return make_mu_family(dim)


def _gaussian_mu_star(dim: int) -> float:
    from experiments.coordination_barrier_experiment import _mu_star_family

    return _mu_star_family(dim)


class ObjectiveFunctions:
    def __init__(
        self,
        dim: int = 1,
        noise_std: float = 0.1,
        noise_seed: int = 42,
    ):
        """
        Utility class for objective functions with multidimensional support.

        Args:
            dim: Number of dimensions
            noise_std: Standard deviation of Gaussian noise to add
            noise_seed: Random seed for reproducible noise
        """
        self.dim = dim
        self.noise_std = noise_std
        self.noise_rng = np.random.default_rng(noise_seed)
        # deterministic, landscape-defining cache (keyed by dim); built lazily
        self._gaussian_mu_cache: dict = {}

    def _parse_input(self, x):
        """Parse input to handle both dict and list/array formats."""
        if isinstance(x, dict):
            # Extract values in order: x1, x2, x3, ...
            keys = [f"x{i+1}" for i in range(self.dim)]
            return [x[key] for key in keys]
        elif isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        else:
            # Single scalar value
            return [x]

    # Base functions (1D)
    def sin1_1d(self, x):
        """ICML 2013 paper function: sin(13x)sin(27x)/2 + 0.5"""
        value = np.sin(13 * x) * np.sin(27 * x) / 2.0 + 0.5
        return value

    def garland_1d(self, x):
        """Function used in ICML 2013 paper"""
        value = 4 * x * (1 - x) * (0.75 + 0.25 * (1 - np.sqrt(abs(np.sin(60 * x)))))
        return value

    def quadratic_1d(self, x):
        """Simple quadratic: (x - 0.5)^2"""
        value = -((x - 0.5) ** 2)
        return value

    def rosenbrock_1d(self, x):
        """Modified 1D Rosenbrock-like function"""
        value = -((x - 0.6) ** 2) + 0.5
        return value

    def rastrigin_1d(self, x):
        """1D Rastrigin function"""
        value = -(x**2 - 10 * np.cos(2 * np.pi * x)) + 10
        return value

    # Multidimensional functions
    def sin1(self, x):
        """Sum of sin1_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        return sum(self.sin1_1d(xi) for xi in coords)

    def garland(self, x):
        """Sum of garland_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        return sum(self.garland_1d(xi) for xi in coords)

    def quadratic(self, x):
        """Sum of squared deviations from center (0.5)."""
        coords = self._parse_input(x)[: self.dim]
        return -sum((xi - 0.5) ** 2 for xi in coords)

    def rosenbrock(self, x):
        """Classic Rosenbrock function (modified for maximization)."""
        coords = self._parse_input(x)[: self.dim]
        if len(coords) < 2:
            return self.rosenbrock_1d(coords[0])

        result = 0
        for i in range(len(coords) - 1):
            result -= 100 * (coords[i + 1] - coords[i] ** 2) ** 2 + (1 - coords[i]) ** 2
        return result

    def rastrigin(self, x):
        """N-dimensional Rastrigin function (modified for maximization, scaled to [0,1] per dim)."""
        coords = self._parse_input(x)[: self.dim]
        A = 10
        # Returns sum-scaled reward; divide by dim afterward to get roughly [0, 1]
        result = sum(A * (np.cos(2 * np.pi * xi) - 1) - xi**2 for xi in coords)
        return result / 40 + len(coords)

    def gaussian(self, x):
        """Bimodal, non-separable landscape: the exact ``family_d`` mu from
        experiments.coordination_barrier_experiment (two radial Gaussian
        bumps over the full coordinate vector; local mode at MODE_LO * 1,
        global mode at MODE_HI * 1). Reaching the global mode from the local
        one requires moving every coordinate together. Returns dim * mu(x) so
        this class's harness-wide noiseless/dim normalization recovers mu(x)
        itself."""
        coords = np.asarray(self._parse_input(x)[: self.dim], dtype=float)
        if self.dim not in self._gaussian_mu_cache:
            self._gaussian_mu_cache[self.dim] = _gaussian_mu_family(self.dim)
        return self.dim * self._gaussian_mu_cache[self.dim](coords)

    # Function properties
    def get_theoretical_max(self, function_name):
        """Get theoretical maximum for known functions."""
        if function_name == "gaussian":
            return _gaussian_mu_star(self.dim)
        max_values = {
            "sin1": 0.975599,
            "quadratic": 0.0,
            "rosenbrock": 0.0,
            "garland": 0.997772313413222,
            "rastrigin": 1.0,
        }
        return max_values.get(function_name, None)

    def get_search_space(self, name):
        """Generate search space dictionary for optimizers."""
        if name == "rosenbrock":
            return {
                f"x{i+1}": {"lower": -2.048, "upper": 2.048} for i in range(self.dim)
            }
        elif name == "rastrigin":
            return {f"x{i+1}": {"lower": -5.12, "upper": 5.12} for i in range(self.dim)}
        else:
            # sin1, garland, quadratic -> unit box
            return {f"x{i+1}": {"lower": 0.0, "upper": 1.0} for i in range(self.dim)}

    def get_function_by_name(self, name):
        """Get function by name."""
        functions = {
            "sin1": self.sin1,
            "garland": self.garland,
            "quadratic": self.quadratic,
            "rosenbrock": self.rosenbrock,
            "rastrigin": self.rastrigin,
            "gaussian": self.gaussian,
        }
        return functions.get(name)
