import numpy as np
import pytest

from imabo import (
    IMABO,
    FiniteIMABO,
    InMemoryStorage,
    config_to_key,
    key_to_config,
    kl_ucb,
    moss_anytime,
    ucb,
    ucb_siri,
)


SIMPLE_SEARCH_SPACE = {
    "x1": {"lower": 0.0, "upper": 1.0},
    "x2": {"lower": 0.0, "upper": 1.0},
}


class TestIMABO:
    def test_initialization(self):
        opt = IMABO(search_space=SIMPLE_SEARCH_SPACE, n_startup_trials=5)
        state = opt.memory.get_current_state()
        assert len(state.arms) == 5

    def test_suggest_returns_valid_config(self):
        opt = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=42)
        config = opt.suggest()
        assert "x1" in config
        assert "x2" in config
        assert 0.0 <= config["x1"] <= 1.0
        assert 0.0 <= config["x2"] <= 1.0

    def test_suggest_observe_cycle(self):
        opt = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=42, n_startup_trials=3)
        for _ in range(20):
            config = opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_observe_without_suggest_raises(self):
        opt = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=42)
        with pytest.raises(RuntimeError):
            opt.observe(1.0)

    def test_observe_out_of_range_raises(self):
        opt = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=42, n_startup_trials=3)
        opt.suggest()
        with pytest.raises(ValueError):
            opt.observe(3.5)
        opt2 = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=42, n_startup_trials=3)
        opt2.suggest()
        with pytest.raises(ValueError):
            opt2.observe(-0.1)

    def test_observe_in_range_ok_and_bypass(self):
        opt = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=42, n_startup_trials=3)
        opt.suggest()
        opt.observe(0.7)  # in [0,1]: no error
        # explicit opt-out disables the check
        opt2 = IMABO(
            search_space=SIMPLE_SEARCH_SPACE, seed=42, n_startup_trials=3,
            check_reward_range=False,
        )
        opt2.suggest()
        opt2.observe(3.5)  # bypassed

    def test_best_config_maximizes(self):
        opt = IMABO(
            search_space={"x1": {"lower": 0.0, "upper": 1.0}},
            seed=42,
            n_startup_trials=5,
        )
        for i in range(50):
            config = opt.suggest()
            reward = 1.0 - abs(config["x1"] - 0.7)  # in [0,1], argmax at x1=0.7
            opt.observe(reward)
        best = opt.best_config
        assert best is not None
        assert abs(best["x1"] - 0.7) < 0.4

    def test_log_scale_parameters(self):
        space = {"lr": {"lower": 1e-5, "upper": 1.0, "log": True}}
        opt = IMABO(search_space=space, seed=42, n_startup_trials=3)
        config = opt.suggest()
        assert 1e-5 <= config["lr"] <= 1.0

    def test_integer_parameters(self):
        space = {"n": {"lower": 1, "upper": 100, "int": True}}
        opt = IMABO(search_space=space, seed=42, n_startup_trials=3)
        config = opt.suggest()
        assert isinstance(config["n"], int)
        assert 1 <= config["n"] <= 100

    def test_categorical_parameters(self):
        space = {"model": {"choices": ["a", "b", "c"]}}
        opt = IMABO(search_space=space, seed=42, n_startup_trials=3)
        config = opt.suggest()
        assert config["model"] in ["a", "b", "c"]

    def test_mixed_search_space(self):
        space = {
            "lr": {"lower": 1e-4, "upper": 1.0, "log": True},
            "n_layers": {"lower": 1, "upper": 10, "int": True},
            "activation": {"choices": ["relu", "tanh"]},
        }
        opt = IMABO(search_space=space, seed=42, n_startup_trials=5)
        for _ in range(30):
            config = opt.suggest()
            assert 1e-4 <= config["lr"] <= 1.0
            assert isinstance(config["n_layers"], int)
            assert config["activation"] in ["relu", "tanh"]
            opt.observe(np.random.random())

    def test_reproducibility(self):
        configs_a = []
        opt_a = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=123, n_startup_trials=3)
        for _ in range(10):
            configs_a.append(opt_a.suggest())
            opt_a.observe(0.5)

        configs_b = []
        opt_b = IMABO(search_space=SIMPLE_SEARCH_SPACE, seed=123, n_startup_trials=3)
        for _ in range(10):
            configs_b.append(opt_b.suggest())
            opt_b.observe(0.5)

        for a, b in zip(configs_a, configs_b):
            assert a == b

    def test_delayed_strategy(self):
        opt = IMABO(
            search_space=SIMPLE_SEARCH_SPACE,
            seed=42,
            switch_strategy="delayed",
            n_startup_trials=3,
        )
        for _ in range(20):
            config = opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None


class TestInMemoryStorage:
    def test_observe_updates_mean(self):
        storage = InMemoryStorage(param_names=["x1"])
        config = {"x1": 0.5}
        storage.observe(config, 1.0)
        storage.observe(config, 0.0)
        state = storage.get_current_state()
        key = (0.5,)
        assert abs(state.arms[key].mean_reward - 0.5) < 1e-10
        assert state.arms[key].nb_rewarded == 2

    def test_pull_arm_increments_pending(self):
        storage = InMemoryStorage(param_names=["x1"])
        key = (0.5,)
        storage.pull_arm(key)
        state = storage.get_current_state()
        assert state.arms[key].nb_pending == 1
        assert state.nb_steps == 1

    def test_reward_frequency(self):
        storage = InMemoryStorage(param_names=["x1"])
        for i in range(200):
            storage.pull_arm((float(i),))
        for i in range(100):
            storage.observe({"x1": float(i)}, 1.0)
        freq = storage.get_reward_frequency()
        assert 0.4 < freq < 0.6


class TestKeyHelpers:
    def test_config_to_key_roundtrip(self):
        names = ["b", "a"]
        config = {"a": 1, "b": 2}
        key = config_to_key(config, names)
        # Key is ordered by sorted param names: ("a", "b") -> (1, 2)
        assert key == (1, 2)
        assert key_to_config(key, names) == config


class TestBanditIndices:
    def test_moss_anytime_beta(self):
        score = moss_anytime(
            mean_reward=0.5,
            n_arms=5,
            step_counter=100,
            nb_rewarded_arm=10,
        )
        assert score >= 0.5  # bonus is non-negative

    def test_ucb_bonus_direction(self):
        up = ucb(mean=0.5, nb_rewarded_arm=4, total_pulls=100, bonus_type="ucb")
        down = ucb(mean=0.5, nb_rewarded_arm=4, total_pulls=100, bonus_type="lcb")
        assert up >= 0.5 >= down

    def test_kl_ucb_in_range(self):
        score = kl_ucb(mean=0.5, pulls=10, t=100)
        assert 0.5 <= score <= 1.0

    def test_ucb_siri_bonus_direction(self):
        up = ucb_siri(0.5, 4, 100, bonus_type="ucb")
        down = ucb_siri(0.5, 4, 100, bonus_type="lcb")
        assert up >= 0.5 >= down


class TestFiniteIMABO:
    def test_suggest_observe_cycle(self):
        opt = FiniteIMABO(
            search_space=SIMPLE_SEARCH_SPACE,
            seed=42,
            total_budget=100,
            n_startup_trials=3,
        )
        for _ in range(40):
            opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_kl_ucb_pull_strategy(self):
        opt = FiniteIMABO(
            search_space=SIMPLE_SEARCH_SPACE,
            seed=42,
            total_budget=100,
            n_startup_trials=3,
            pull_strategy="kl_ucb",
        )
        for _ in range(40):
            opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_best_config_maximizes(self):
        opt = FiniteIMABO(
            search_space={"x1": {"lower": 0.0, "upper": 1.0}},
            seed=42,
            total_budget=200,
            n_startup_trials=5,
        )
        for _ in range(80):
            config = opt.suggest()
            reward = 1.0 - abs(config["x1"] - 0.7)
            opt.observe(reward)
        best = opt.best_config
        assert best is not None
        assert abs(best["x1"] - 0.7) < 0.5
