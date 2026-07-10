"""IMABO: Infinite Multi-Armed Bandits with Oracles."""

from imabo.memory import (
    ArmStats,
    CurrentState,
    InMemoryStorage,
    Memory,
    config_to_key,
    key_to_config,
)
from imabo.moss import (
    kl_divergence,
    kl_ucb,
    moss_anytime,
    ucb,
    ucb_siri,
)
from imabo.optimizer import IMABO, FiniteIMABO
from imabo.tpe import default_gamma, default_weights, hyperopt_default_gamma
from imabo.types import ArmConfig, ArmKey

try:
    # TabFMIMABO pulls in optional deps (tabfm, pandas, huggingface_hub,
    # safetensors) not required by the core package; degrade gracefully.
    from imabo.tabfm_optimizer import TabFMIMABO

    _TABFM_AVAILABLE = True
except ImportError:
    _TABFM_AVAILABLE = False

__all__ = [
    # Optimizers
    "IMABO",
    "FiniteIMABO",
    # Memory
    "InMemoryStorage",
    "Memory",
    "ArmStats",
    "CurrentState",
    "config_to_key",
    "key_to_config",
    # Bandit indices
    "moss_anytime",
    "ucb",
    "kl_ucb",
    "kl_divergence",
    "ucb_siri",
    # TPE helpers
    "default_gamma",
    "hyperopt_default_gamma",
    "default_weights",
    # Types
    "ArmKey",
    "ArmConfig",
]

if _TABFM_AVAILABLE:
    __all__.append("TabFMIMABO")
