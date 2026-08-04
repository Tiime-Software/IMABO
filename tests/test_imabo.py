import numpy as np
import pytest

from imabo import (
    IMABO,
    ArmStats,
    FiniteIMABO,
    IMABOCoordUCB,
    IMABOTabPFN,
    InMemoryStorage,
    config_to_key,
    key_to_config,
    kl_ucb,
    moss_anytime,
    ucb,
    ucb_siri,
)
from imabo.moss import KLUCB1


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


class TestWinningOracles:
    """The two tuned explore oracles (see winning_configs.pdf)."""

    SPACE = {f"x{i}": {"choices": list(range(6))} for i in range(4)}

    def _run(self, opt, n=600, seed=3):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            config = opt.suggest()
            opt.observe(float(rng.random() < 0.1 + 0.06 * sum(config.values())))
        return opt

    def test_klucb1_forces_each_choice_once_then_ranks(self):
        # select() does not reserve: an uncredited choice keeps being returned
        # until its vote actually arrives, which is what the oracle relies on
        # (the credit for a proposal lands several rounds after the selection).
        b = KLUCB1(3)
        assert [b.select() for _ in range(3)] == [0, 0, 0]
        for i, r in ((0, 1.0), (1, 0.0), (2, 1.0)):
            assert b.select() == i
            b.update(i, r)
        idx = b.indices()
        assert np.isfinite(idx).all()
        assert idx[1] < idx[0]  # the losing choice ranks last

    def test_klucb1_revise_keeps_the_vote_count(self):
        b = KLUCB1(2)
        b.update(0, 0.3)
        b.revise(0, 0.5)  # same vote, sharpened value
        assert b.n[0] == 1.0
        assert b.means[0] == pytest.approx(0.8)

    def test_coord_ucb_mutates_one_coordinate_from_the_incumbent(self):
        opt = self._run(IMABOCoordUCB(search_space=self.SPACE, seed=1, beta=0.5))
        state = opt.memory.get_current_state()
        assert len(state.arms) > 5
        # Every registered decision is a single-coordinate move off its parent.
        for decisions in opt._pending_choice.values():
            for d in decisions:
                child = key_to_config(d.arm_key, opt.param_names)
                parent = key_to_config(d.parent_key, opt.param_names)
                differing = [p for p in opt.param_names if child[p] != parent[p]]
                assert differing == [opt.param_names[d.coord]]

    def test_coord_ucb_credits_the_arm_mean_not_the_first_pull(self):
        opt = self._run(IMABOCoordUCB(search_space=self.SPACE, seed=2, beta=0.5))
        opt._refresh_credits()
        for decisions in opt._pending_choice.values():
            for d in decisions:
                if d.credited is None:
                    continue
                assert d.credited == pytest.approx(opt._arm_mean(d.arm_key))
        # One vote per decision, however many rewards the arm collected.
        assert opt.coord_bandit.n.sum() == pytest.approx(
            sum(
                1
                for ds in opt._pending_choice.values()
                for d in ds
                if d.credited is not None
            )
        )

    def test_coord_ucb_is_reproducible(self):
        a = self._run(IMABOCoordUCB(search_space=self.SPACE, seed=7, beta=0.5))
        b = self._run(IMABOCoordUCB(search_space=self.SPACE, seed=7, beta=0.5))
        assert a.best_config == b.best_config
        assert np.allclose(a.coord_bandit.sum, b.coord_bandit.sum)

    def test_tabpfn_mutation_pool_composition(self):
        opt = IMABOTabPFN(
            search_space=self.SPACE,
            seed=0,
            candidate_source="mutation",
            candidate_uniform_frac=0.1,
            tabpfn_model={},
        )
        stats = ArmStats(mean_reward=0.8, nb_rewarded=5, nb_pending=0)
        parent = {f"x{i}": 3 for i in range(4)}
        rewarded = [(config_to_key(parent, opt.param_names), stats)]
        pool = opt._sample_candidates(rewarded)
        assert len(pool) == opt.n_candidates
        # 10% uniform, the rest one coordinate off the incumbent.
        mutants = pool[10:]
        assert len(mutants) == 90
        for m in mutants:
            assert sum(m[p] != parent[p] for p in opt.param_names) == 1

    def test_tabpfn_uniform_pool_is_the_default(self):
        opt = IMABOTabPFN(search_space=self.SPACE, seed=0, tabpfn_model={})
        assert opt.candidate_source == "uniform"
        assert len(opt._sample_candidates([])) == opt.n_candidates

    def test_mutation_scale_is_a_no_op_on_categorical_axes(self):
        import random

        from imabo.mutation import local_value_sampler, mutate_value

        opt = IMABOTabPFN(search_space=self.SPACE, seed=0, tabpfn_model={})
        scaled = [
            local_value_sampler(opt.distributions, random.Random(11), 0.1)("x0", 3)
            for _ in range(50)
        ]
        plain = [
            mutate_value("x0", 3, opt.distributions, random.Random(11))
            for _ in range(50)
        ]
        assert scaled == plain

    def test_local_step_stays_in_range_and_is_local(self):
        import math
        import random

        from imabo.mutation import local_value_sampler

        space = {"lr": {"lower": 1e-5, "upper": 1.0, "log": True}}
        opt = IMABOCoordUCB(search_space=space, seed=0, beta=0.5)
        f = local_value_sampler(opt.distributions, random.Random(0), 0.1)
        d = opt.distributions["lr"]
        width = math.log(d.high) - math.log(d.low)
        near = 0
        for current in (d.low, d.high, 1e-3):
            for _ in range(2000):
                v = f("lr", current)
                assert d.low <= v <= d.high  # boundary clamp, original space
                near += abs(math.log(v) - math.log(current)) / width < 0.1
        # A uniform redraw would land within 10% only ~19% of the time.
        assert near / 6000 > 0.5


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
