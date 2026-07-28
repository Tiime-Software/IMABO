import numpy as np
from functools import partial


class ObjectiveFunctions:
    def __init__(
        self,
        dim: int = 1,
        noise_std: float = 0.1,
        noise_seed: int = 42,
        kappa: float = 50.0,
        rot_seed: int = 0,
    ):
        """
        Utility class for objective functions with multidimensional support.

        Args:
            dim: Number of dimensions
            noise_std: Standard deviation of Gaussian noise to add
            noise_seed: Random seed for reproducible noise
            kappa: Condition number for the non-separable ``ellipsoid_rot`` /
                ``rastrigin_rot`` objectives (ratio of largest to smallest
                eigenvalue of the coupling matrix). Higher = more
                ill-conditioned, so coordinate-descent methods zig-zag more.
            rot_seed: Seed for the FIXED rotation matrix and optimum center of
                the rotated objectives. Deliberately independent of
                ``noise_seed`` so that the landscape (and its known maximum) is
                identical across the different runs of a benchmark, while only
                the observation noise varies run to run.
        """
        self.dim = dim
        self.noise_std = noise_std
        self.noise_rng = np.random.default_rng(noise_seed)
        self.kappa = kappa
        self.rot_seed = rot_seed
        # deterministic, landscape-defining caches (keyed by dim); built lazily
        self._rot_cache: dict = {}
        self._center_cache: dict = {}
        self._norm_cache: dict = {}

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

    def _add_noise(self, value, fn_name: str):
        """Add Gaussian noise on the summed-reward scale.

        ``noise_std`` (sigma) is expressed on the *per-dimension* scale used by
        the regret (``value / dim``). The objective returns a sum over ``dim``
        dimensions, so a per-dim std of sigma corresponds to a std of
        ``sigma * dim`` on the sum; after the regret's ``/dim`` normalization the
        effective noise is exactly ``N(0, sigma)`` — dim-invariant and matching
        the regret scale.
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
        """1D Rastrigin function"""
        value = -(x**2 - 10 * np.cos(2 * np.pi * x)) + 10
        return value

    # Multidimensional functions
    def sin1(self, x, noise=True):
        """Sum of sin1_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        result = sum(self.sin1_1d(xi) for xi in coords)
        return self._add_noise(result, "sin1") if noise else result

    def garland(self, x, noise=True):
        """Sum of garland_1d over all dimensions."""
        coords = self._parse_input(x)[: self.dim]
        result = sum(self.garland_1d(xi) for xi in coords)
        return self._add_noise(result, "garland") if noise else result

    def quadratic(self, x, noise=True):
        """Sum of squared deviations from center (0.5)."""
        coords = self._parse_input(x)[: self.dim]
        result = -sum((xi - 0.5) ** 2 for xi in coords)
        return self._add_noise(result, "quadratic") if noise else result

    def rosenbrock(self, x, noise=True):
        """Classic Rosenbrock function (modified for maximization)."""
        coords = self._parse_input(x)[: self.dim]
        if len(coords) < 2:
            result = self.rosenbrock_1d(coords[0])
            return self._add_noise(result, "rosenbrock") if noise else result

        result = 0
        for i in range(len(coords) - 1):
            result -= 100 * (coords[i + 1] - coords[i] ** 2) ** 2 + (1 - coords[i]) ** 2
        return self._add_noise(result, "rosenbrock") if noise else result

    def rastrigin(self, x, noise=True):
        """N-dimensional Rastrigin function (modified for maximization, scaled to [0,1] per dim)."""
        coords = self._parse_input(x)[: self.dim]
        A = 10
        # Returns sum-scaled reward; divide by dim afterward to get roughly [0, 1]
        result = sum(A * (np.cos(2 * np.pi * xi) - 1) - xi**2 for xi in coords)
        scaled = result / 40 + len(coords)
        return self._add_noise(scaled, "rastrigin") if noise else scaled

    # ------------------------------------------------------------------ #
    # Non-separable (coupled) objectives
    # ------------------------------------------------------------------ #
    # The classic toy suite (sin1, garland, rastrigin) is additively
    # separable: f(x) = sum_i g(x_i), so the optimal value of each coordinate
    # is independent of the others. That is precisely the assumption a
    # coordinate-wise grid method (e.g. Hier-UCB) is built on, so those
    # objectives flatter grid/coordinate-descent baselines. The two objectives
    # below break separability by evaluating the base function on a ROTATED set
    # of coordinates z = R (x - c): every z_i mixes all x_j, so the best value
    # of one hyper-parameter depends on the others. This is the regime that
    # distinguishes a joint model (IMABO's multivariate TPE) from per-axis
    # coordinate search.

    def _rotation(self, dim: int) -> np.ndarray:
        """Fixed orthogonal rotation matrix for ``dim`` (cached, landscape-defining)."""
        if dim not in self._rot_cache:
            rng = np.random.default_rng(self.rot_seed)
            Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
            self._rot_cache[dim] = Q
        return self._rot_cache[dim]

    def _center(self, name: str, dim: int, lo: float, hi: float) -> np.ndarray:
        """Off-grid optimum location inside the box (cached, deterministic).

        Placed away from the box center and off any symmetric linspace grid, so
        a discretized baseline has a nonzero regret floor no budget removes.
        """
        key = (name, dim, lo, hi)
        if key not in self._center_cache:
            rng = np.random.default_rng(self.rot_seed + 101)
            # interior fractions in ~[0.28, 0.72], irrational-ish to dodge grids
            frac = 0.28 + 0.44 * rng.random(dim)
            self._center_cache[key] = lo + (hi - lo) * frac
        return self._center_cache[key]

    def _coupling_matrix(self, dim: int) -> tuple[np.ndarray, float]:
        """Ill-conditioned SPD coupling matrix M = R^T diag(d) R and its lambda_max.

        Eigenvalues are geometrically spaced from 1 to ``kappa`` so the
        condition number is exactly ``kappa`` (controls coordinate-descent
        zig-zag cost); ``lambda_max`` is returned for reward normalization.
        """
        if dim not in self._norm_cache:
            R = self._rotation(dim)
            eig = np.geomspace(1.0, self.kappa, dim) if dim > 1 else np.array([1.0])
            M = R.T @ np.diag(eig) @ R
            self._norm_cache[dim] = (M, float(eig.max()))
        return self._norm_cache[dim]

    def ellipsoid_rot(self, x, noise=True):
        """Smooth, unimodal, NON-separable rotated ill-conditioned ellipsoid.

        f(x) = dim * (1 - q),  q = (x-c)^T M (x-c) / norm  in [0, 1]

        where M = R^T diag(geomspace(1, kappa)) R couples all coordinates and
        the optimum sits at the off-grid center ``c``. Returns the sum-scaled
        reward (per-dim max 1.0 after the harness's ``/dim``). The single-mode
        landscape isolates the effect of COUPLING alone (no multimodality),
        making it the clean control against the separable ``quadratic``.
        """
        lo, hi = 0.0, 1.0
        coords = np.asarray(self._parse_input(x)[: self.dim], dtype=float)
        c = self._center("ellipsoid_rot", self.dim, lo, hi)
        M, lam_max = self._coupling_matrix(self.dim)
        v = coords - c
        # Upper bound on v^T M v over the box: lam_max * max ||x - c||^2.
        max_sq = np.sum(np.maximum((c - lo) ** 2, (hi - c) ** 2))
        norm = lam_max * max_sq
        q = float(v @ M @ v) / norm
        result = self.dim * (1.0 - q)
        return self._add_noise(result, "ellipsoid_rot") if noise else result

    def rastrigin_rot(self, x, noise=True):
        """Multimodal AND non-separable: Rastrigin on rotated coordinates.

        z = R (x - c); f = sum_i [A (cos(2*pi*z_i) - 1) - z_i^2], with the
        global optimum at the off-grid center ``c`` (z = 0). The quadratic bowl
        is rotation-invariant, but the cosine ripple is rotated, so the local
        optima form a lattice that is NOT axis-aligned — a coordinate-wise
        search cannot decompose it. This is the hardest toy: it stresses
        coupling and multimodality together (cf. CEC rotated Rastrigin).
        """
        lo, hi = -5.12, 5.12
        A = 10
        coords = np.asarray(self._parse_input(x)[: self.dim], dtype=float)
        c = self._center("rastrigin_rot", self.dim, lo, hi)
        R = self._rotation(self.dim)
        z = R @ (coords - c)
        result = float(np.sum(A * (np.cos(2 * np.pi * z) - 1) - z**2))
        scaled = result / 40 + self.dim
        return self._add_noise(scaled, "rastrigin_rot") if noise else scaled

    # Function properties
    def get_theoretical_max(self, function_name):
        """Get theoretical maximum for known functions."""
        max_values = {
            "sin1": 0.975599,
            "quadratic": 0.0,
            "rosenbrock": 0.0,
            "garland": 0.997772313413222,
            "rastrigin": 1.0,
            # Both non-separable objectives are built so the per-dim reward
            # (sum over dim, then /dim as the harness does) has max exactly 1
            # at the off-grid optimum center c (q=0 / z=0).
            "ellipsoid_rot": 1.0,
            "rastrigin_rot": 1.0,
        }
        return max_values.get(function_name, None)

    def get_search_space(self, name):
        """Generate search space dictionary for optimizers."""
        if name == "rosenbrock":
            return {
                f"x{i+1}": {"lower": -2.048, "upper": 2.048} for i in range(self.dim)
            }
        elif name in ("rastrigin", "rastrigin_rot"):
            return {f"x{i+1}": {"lower": -5.12, "upper": 5.12} for i in range(self.dim)}
        else:
            # sin1, garland, quadratic, ellipsoid_rot -> unit box
            return {f"x{i+1}": {"lower": 0.0, "upper": 1.0} for i in range(self.dim)}

    def get_function_by_name(self, name, **kwargs):
        """Get function by name."""
        functions = {
            "sin1": partial(self.sin1, **kwargs),
            "garland": partial(self.garland, **kwargs),
            "quadratic": partial(self.quadratic, **kwargs),
            "rosenbrock": partial(self.rosenbrock, **kwargs),
            "rastrigin": partial(self.rastrigin, **kwargs),
            "ellipsoid_rot": partial(self.ellipsoid_rot, **kwargs),
            "rastrigin_rot": partial(self.rastrigin_rot, **kwargs),
        }
        return functions.get(name)
