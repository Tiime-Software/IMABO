"""Benchmark configurations for HPO experiments."""

from typing import Any, Dict


def create_toy_function_config(name: str, dim: int):
    """Create configuration for toy functions."""
    return {
        "name": name.capitalize(),
        "dim": dim,
        "description": f"{name.capitalize()} toy function ({dim}D)",
        "sample_metric": "function_value",
        "avg_metric": "function_value",
        "param_specs": {
            f"x{i+1}": {"lower": 0.0, "upper": 1.0, "log": False} for i in range(dim)
        },
    }


BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "svm": {
        "name": "SVM",
        "fidelity": {"subsample": 0.5},
        "description": "Support Vector Machine benchmark",
        "sample_metric": "sample_acc",
        "avg_metric": "val_acc",
        "test_metric": "test_acc",
        "param_specs": {
            "C": {"lower": 2**-10, "upper": 2**10, "log": True},
            "gamma": {"lower": 2**-10, "upper": 2**10, "log": True},
        },
    },
    "nn": {
        "name": "Neural Network",
        "fidelity": {"iter": 100, "subsample": 0.8},
        "description": "Multi-layer Perceptron benchmark",
        "sample_metric": "sample_acc",
        "avg_metric": "val_acc",
        "test_metric": "test_acc",
        "param_specs": {
            "depth": {"lower": 1, "upper": 3, "log": False, "int": True},
            "width": {"lower": 16, "upper": 1024, "log": True, "int": True},
            "batch_size": {"lower": 4, "upper": 256, "log": True, "int": True},
            "alpha": {"lower": 1e-8, "upper": 1.0, "log": True},
            "learning_rate_init": {"lower": 1e-5, "upper": 1.0, "log": True},
        },
    },
    "rf": {
        "name": "Random Forest",
        "fidelity": {"n_estimators": 100, "subsample": 0.8},
        "description": "Random Forest benchmark",
        "sample_metric": "sample_acc",
        "avg_metric": "val_acc",
        "test_metric": "test_acc",
        "param_specs": {
            "max_depth": {"lower": 1, "upper": 50, "log": True, "int": True},
            "min_samples_split": {"lower": 2, "upper": 128, "log": True, "int": True},
            "max_features": {"lower": 0.0, "upper": 1.0, "log": False},
            "min_samples_leaf": {"lower": 1, "upper": 20, "log": False, "int": True},
        },
    },
    "xgboost": {
        "name": "XGBoost",
        "fidelity": {"n_estimators": 100, "subsample": 0.8},
        "description": "XGBoost benchmark",
        "sample_metric": "sample_acc",
        "avg_metric": "val_acc",
        "test_metric": "test_acc",
        "param_specs": {
            "eta": {"lower": 2**-10, "upper": 1.0, "log": True},
            "max_depth": {"lower": 1, "upper": 50, "log": True, "int": True},
            "colsample_bytree": {"lower": 0.1, "upper": 1.0, "log": False},
            "reg_lambda": {"lower": 2**-10, "upper": 2**10, "log": True},
        },
    },
    "pybnn": {
        "name": "PyBNN",
        # "fidelity": {"n_units_1": 100, "n_units_2": 100},
        "description": "PyBNN benchmark",
        "sample_metric": "function_value_sample",
        "avg_metric": "function_value",
        "param_specs": {
            "l_rate": {"lower": 1e-6, "upper": 1e-1, "log": True},
            "burn_in": {"lower": 0.0, "upper": 0.8, "log": False},
            "n_units_1": {"lower": 16, "upper": 512, "log": True, "int": True},
            "n_units_2": {"lower": 16, "upper": 512, "log": True, "int": True},
            "mdecay": {"lower": 0.0, "upper": 1.0, "log": False},
        },
    },
    "histgb": {
        "name": "Histogram-based Gradient Boosting",
        "fidelity": {"n_estimators": 100, "subsample": 0.8},
        "description": "Histogram-based Gradient Boosting benchmark",
        "sample_metric": "sample_acc",
        "avg_metric": "val_acc",
        "test_metric": "test_acc",
        "param_specs": {
            "max_depth": {"lower": 6, "upper": 30, "log": True, "int": True},
            "max_leaf_nodes": {"lower": 2, "upper": 64, "log": True, "int": True},
            "learning_rate": {"lower": 2**-10, "upper": 1.0, "log": True},
            "l2_regularization": {"lower": 2**-10, "upper": 2**10, "log": True},
        },
    },
    "lr": {
        "name": "Logistic Regression",
        "fidelity": {"iter": 1000, "subsample": 1.0},
        "description": "Logistic Regression benchmark",
        "sample_metric": "sample_acc",
        "avg_metric": "val_acc",
        "test_metric": "test_acc",
        "param_specs": {
            "alpha": {"lower": 1e-5, "upper": 1.0, "log": True},
            "eta0": {"lower": 1e-5, "upper": 1.0, "log": True},
        },
    },
}
