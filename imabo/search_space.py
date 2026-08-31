from __future__ import annotations

import math
import random
from typing import Any, Callable

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.types import ArmConfig, ArmKey

SpaceFunction = Callable[["Trial"], None]


def _draw(distribution: BaseDistribution, rng: random.Random) -> Any:
    """Draw one value from a parameter's own marginal of ``P0``."""
    if isinstance(distribution, FloatDistribution):
        if distribution.step is not None:
            # A stepped axis is a grid: draw one of its points uniformly.
            n = int(round((distribution.high - distribution.low) / distribution.step))
            return distribution.low + distribution.step * rng.randint(0, n)
        if distribution.log:
            return math.exp(
                rng.uniform(math.log(distribution.low), math.log(distribution.high))
            )
        return rng.uniform(distribution.low, distribution.high)
    if isinstance(distribution, IntDistribution):
        if distribution.step != 1:
            return rng.randrange(
                distribution.low, distribution.high + 1, distribution.step
            )
        # Integers are drawn uniformly even on a log-scaled axis, matching the
        # sampler this replaces.
        return rng.randint(distribution.low, distribution.high)
    if isinstance(distribution, CategoricalDistribution):
        return rng.choice(distribution.choices)
    raise TypeError(f"cannot sample {type(distribution).__name__}")


def _type_of(distribution: BaseDistribution) -> str:
    if isinstance(distribution, IntDistribution):
        return "integer"
    if isinstance(distribution, FloatDistribution):
        return "continuous"
    return "categorical"


class Trial:
    """What a space function is handed, one per configuration drawn.

    Each ``suggest_*`` call does two things at once: it *declares* a parameter -- its
    name, type and bounds -- and it *returns* a value drawn for it. Declaring is what
    lets :class:`SearchSpace` recover the same per-parameter distributions a dict spec
    would have given, so the TPE and mutation oracles need no notion of where the space
    came from.

    Modelled on Optuna's ``trial``, and used the same way: the space is never described
    anywhere, it is discovered as the function runs.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng
        self.distributions: dict[str, BaseDistribution] = {}
        self.params: ArmConfig = {}

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        return self._suggest(name, FloatDistribution(low, high, log=log, step=step))

    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int:
        return self._suggest(name, IntDistribution(low, high, log=log, step=step))

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self._suggest(name, CategoricalDistribution(choices))

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        declared = self.distributions.get(name)
        if declared is not None and declared != distribution:
            raise ValueError(
                f"{name!r} was suggested twice with different bounds in one call"
            )
        self.distributions[name] = distribution
        value = _draw(distribution, self._rng)
        self.params[name] = value
        return value


class SearchSpace:
    """A mixed continuous / integer / categorical space, and how to sample it.

    Everything that depends on the *shape* of the space lives here, so that policies and
    oracles never branch on a parameter's type. :meth:`sample` is the paper's baseline
    distribution ``P0``: each parameter drawn independently and uniformly, log-uniformly
    when declared on a logarithmic scale.

    A space is declared either as one dict per parameter::

        SearchSpace({
            "learning_rate": {"lower": 1e-5, "upper": 1.0, "log": True},
            "n_layers": {"lower": 1, "upper": 8, "int": True},
            "dropout": {"lower": 0.0, "upper": 0.5},
            "optimizer": {"choices": ["adam", "sgd", "adamw"]},
        })

    or as a function in Optuna's style, which asks a :class:`Trial` for its parameters::

        def space(trial):
            trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
            trial.suggest_int("n_layers", 1, 8)
            trial.suggest_float("dropout", 0.0, 0.5)
            trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"])

        SearchSpace(space)

    The two describe the same space, with three things worth knowing about the function
    form:

    - It is called once on construction to discover the parameters, so it must be free of
      side effects.
    - It must declare the same parameters with the same bounds on every call. A parameter
      that exists only on some branch, or whose range depends on another parameter, is a
      conditional space, which is not supported yet and is refused with an explanation.
    - Its return value is not used: the configuration is the set of parameters it
      suggested. Returning one is an error rather than a silent surprise.

    The order of the ``suggest_*`` calls is the order in which the generator is consumed,
    so a dict and a function describing the same space give the same space but not the
    same seeded sequence unless the calls happen to be in sorted-name order.
    """

    def __init__(self, space: dict[str, dict[str, Any]] | SpaceFunction):
        self.spec = space
        self.function: SpaceFunction | None = space if callable(space) else None
        if self.function is not None:
            # A dedicated generator: discovering the space must never consume draws
            # from the run's own stream.
            probe = Trial(random.Random(0))
            self._call(probe)
            self.distributions = probe.distributions
        else:
            self.distributions = _distributions_from_spec(space)

        # `names` is sorted -- it fixes the order of an arm key. `distributions` keeps
        # the declaration order instead, because Optuna's multivariate Parzen estimator
        # iterates it and its sampling would consume the RNG in a different order.
        self.names: list[str] = sorted(self.distributions)
        self.types: dict[str, str] = {
            name: _type_of(distribution)
            for name, distribution in self.distributions.items()
        }

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"SearchSpace({len(self.names)} parameters: {', '.join(self.names)})"

    def is_categorical(self, name: str) -> bool:
        return self.types[name] == "categorical"

    def sample(self, rng: random.Random) -> ArmConfig:
        """Draw a configuration from the baseline distribution ``P0``."""
        if self.function is None:
            return {name: self.sample_value(name, rng) for name in self.names}
        trial = Trial(rng)
        self._call(trial)
        if trial.distributions != self.distributions:
            raise ValueError(
                "the space function declared different parameters than it did on the "
                "first call. A parameter that exists only on some branch is a "
                "conditional space, which is not supported yet: every call must "
                f"declare {sorted(self.distributions)} with the same bounds."
            )
        return trial.params

    def sample_value(self, name: str, rng: random.Random) -> Any:
        """Draw one parameter from its own marginal of ``P0``."""
        return _draw(self.distributions[name], rng)

    def encode(self, config: ArmConfig) -> ArmKey:
        """The hashable identity of a configuration: its values in sorted-name order."""
        return tuple(config[name] for name in self.names)

    def decode(self, key: ArmKey) -> ArmConfig:
        """Inverse of :meth:`encode`."""
        return dict(zip(self.names, key))

    def _call(self, trial: Trial) -> None:
        returned = self.function(trial)
        if returned is not None:
            raise TypeError(
                "a space function must not return anything: the configuration is the "
                "set of parameters it suggested. Assign the suggested values to local "
                "variables if you need them, and drop the return."
            )
        if not trial.distributions:
            raise ValueError("the space function suggested no parameter")


def _distributions_from_spec(
    spec: dict[str, dict[str, Any]],
) -> dict[str, BaseDistribution]:
    distributions: dict[str, BaseDistribution] = {}
    for name, parameter in spec.items():
        lower, upper = parameter.get("lower"), parameter.get("upper")
        if lower is not None and upper is not None:
            log = parameter.get("log", False)
            if parameter.get("int", False):
                distributions[name] = IntDistribution(lower, upper, log=log)
            else:
                distributions[name] = FloatDistribution(lower, upper, log=log)
        elif parameter.get("choices"):
            distributions[name] = CategoricalDistribution(parameter["choices"])
        else:
            raise ValueError(f"invalid specification for {name!r}: {parameter!r}")
    return distributions
