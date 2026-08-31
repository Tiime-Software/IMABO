"""The baselines the paper compares IMABO against, usable on their own.

Each one drives the same loop as :class:`imabo.IMABO` -- ``suggest()``, then
``observe(reward)``, with ``best_config`` for the configuration to report -- so a baseline
drops into any harness that already drives the optimizer.

They do not all accept the same search spaces. The reservoir baselines take anything
:class:`imabo.SearchSpace` takes -- a dict, a suggestion function, or a ready-made space.
The tree-search generators (``stroquool``, ``stosoo``, ``hoo_t``) work on ``[0, 1]**d``
instead and are driven through :class:`TimedOptimizer`, as the paper's experiments do.
"""

from imabo.baselines.hier_mab import HierMAB
from imabo.baselines.optuna_bandit import OptunaBandit
from imabo.baselines.qrm2 import QRM2
from imabo.baselines.random_search import RandomSearch
from imabo.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from imabo.baselines.ucb_air import MOSSAIR, UCBAIR

__all__ = [
    "MOSSAIR",
    "UCBAIR",
    "HierMAB",
    "OptunaBandit",
    "QRM2",
    "RandomSearch",
    "TimedOptimizer",
    "hoo_t",
    "stosoo",
    "stroquool",
]
