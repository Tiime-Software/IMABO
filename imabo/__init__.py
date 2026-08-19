"""IMABO: Infinite Multi-Armed Bandits with Oracles."""

from imabo.imabo import IMABO
from imabo.memory import (
    ArmStats,
    CoordCredits,
    CurrentState,
    Decision,
    InMemoryStorage,
    Memory,
    config_to_key,
    key_to_config,
)
from imabo.oracle import Oracle
from imabo.oracles import (
    IMOSSTPE,
    KLUCB,
    CandidatePool,
    IMOSSMutateKLTPE,
    IMOSSRandom,
    IMOSSTabFM,
    IMOSSTabPFN,
    MutateKLTPEOracle,
    RandomOracle,
    TabFMOracle,
    TabPFNOracle,
    TPEOracle,
    load_tabfm,
    load_tabpfn,
)
from imabo.oracles.parzen import (
    adaptive_categorical_distance_func,
    default_gamma,
    default_weights,
    hyperopt_default_gamma,
    numeric_l1_distance,
    split_good_bad,
    tpe_suggest,
)
from imabo.policies import IMOSS, BudgetedUCB, anytime_moss_index
from imabo.policy import AllocationPolicy
from imabo.search_space import SearchSpace, Trial
from imabo.types import ArmConfig, ArmKey

__all__ = [
    "Decision",
    "CoordCredits",
    # The framework
    "IMABO",
    "AllocationPolicy",
    "Oracle",
    "SearchSpace",
    "Trial",
    "Memory",
    "InMemoryStorage",
    "ArmStats",
    "CurrentState",
    "ArmKey",
    "ArmConfig",
    "config_to_key",
    "key_to_config",
    # Policies
    "IMOSS",
    "BudgetedUCB",
    "anytime_moss_index",
    # Oracles
    "RandomOracle",
    "TPEOracle",
    "MutateKLTPEOracle",
    "TabPFNOracle",
    "TabFMOracle",
    "CandidatePool",
    "KLUCB",
    "load_tabpfn",
    "load_tabfm",
    # The paper's algorithms, by name
    "IMOSSRandom",
    "IMOSSTPE",
    "IMOSSMutateKLTPE",
    "IMOSSTabPFN",
    "IMOSSTabFM",
    # TPE helpers
    "default_gamma",
    "hyperopt_default_gamma",
    "default_weights",
    "adaptive_categorical_distance_func",
    "numeric_l1_distance",
    "split_good_bad",
    "tpe_suggest",
]
