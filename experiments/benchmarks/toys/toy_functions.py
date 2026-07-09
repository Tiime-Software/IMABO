import numpy as np
from functools import partial


class ObjectiveFunctions:
    def __init__(self, dim: int = 1, noise_std: float = 0.01, noise_seed: int = 42):
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

    def _add_noise(self, value):
        """Add Gaussian noise on the summed-reward scale.

        ``noise_std`` is expressed on the per-dimension scale (the scale used
        after dividing the sum by ``dim``), so it is scaled by ``dim`` here --
        after that later division the effective noise is ``N(0, noise_std)``,
        independent of ``dim``.
        """
        if self.noise_std > 0:
            return value + self.noise_rng.normal(0, self.noise_std * self.dim)
        return value

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
        """1D Rastrigin function, rescaled to roughly [0, 1] like sin1_1d/garland_1d."""
        A = 10
        value = (A * (np.cos(2 * np.pi * x) - 1) - x**2) / 40 + 1
        return value

    # Multidimensional functions
    def sin1(self, x, noise=True):
        """Sum of sin1_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        result = sum(self.sin1_1d(xi) for xi in coords)
        return self._add_noise(result) if noise else result

    def garland(self, x, noise=True):
        """Sum of garland_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        result = sum(self.garland_1d(xi) for xi in coords)
        return self._add_noise(result) if noise else result

    def quadratic(self, x, noise=True):
        """Sum of squared deviations from center (0.5)."""
        coords = self._parse_input(x)[: self.dim]
        result = -sum((xi - 0.5) ** 2 for xi in coords)
        return self._add_noise(result) if noise else result

    def rosenbrock(self, x, noise=True):
        """Classic Rosenbrock function (modified for maximization)."""
        coords = self._parse_input(x)[: self.dim]
        if len(coords) < 2:
            result = self.rosenbrock_1d(coords[0])
            return self._add_noise(result) if noise else result

        result = 0
        for i in range(len(coords) - 1):
            result -= 100 * (coords[i + 1] - coords[i] ** 2) ** 2 + (1 - coords[i]) ** 2
        return self._add_noise(result) if noise else result

    def rastrigin(self, x, noise=True):
        """Sum of rastrigin_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        result = sum(self.rastrigin_1d(xi) for xi in coords)
        return self._add_noise(result) if noise else result

    # Function properties
    def get_theoretical_max(self, function_name):
        """Get theoretical maximum for known functions."""
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
            return {f"x{i+1}": {"lower": 0.0, "upper": 1.0} for i in range(self.dim)}

    def get_function_by_name(self, name, **kwargs):
        """Get function by name."""
        functions = {
            "sin1": partial(self.sin1, **kwargs),
            "garland": partial(self.garland, **kwargs),
            "quadratic": partial(self.quadratic, **kwargs),
            "rosenbrock": partial(self.rosenbrock, **kwargs),
            "rastrigin": partial(self.rastrigin, **kwargs),
        }
        return functions.get(name)
