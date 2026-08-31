import math
import random

import numpy as np
import pytest
from frozendict import frozendict

from imabo import (
    IMABO,
    IMOSS,
    KLUCB,
    MOSSAIR,
    QRM2,
    UCBAIR,
    AllocationPolicy,
    ArmStats,
    BudgetedUCB,
    CandidatePool,
    CurrentState,
    HierMAB,
    InMemoryStorage,
    Memory,
    MutateKLTPEOracle,
    OptunaBandit,
    Oracle,
    RandomOracle,
    RandomSearch,
    SearchSpace,
    TabPFNOracle,
    TimedOptimizer,
    TPEOracle,
    Trial,
    anytime_moss_index,
    config_to_key,
    key_to_config,
    stosoo,
)
from imabo.oracles.mutate_kl_tpe_oracle import kl_ucb
from imabo.oracles.tpe_oracle import lcb
from imabo.policies.budgeted_ucb import ucb

SIMPLE_SEARCH_SPACE = {
    "x1": {"lower": 0.0, "upper": 1.0},
    "x2": {"lower": 0.0, "upper": 1.0},
}


def imoss_tpe(space, seed=42, n_warmup=10, **policy):
    """The default pairing, which most of these tests only need to run."""
    return IMABO(space, IMOSS(n_warmup=n_warmup, **policy), TPEOracle(), seed=seed)


class TestIMABO:
    def test_initialization(self):
        opt = imoss_tpe(SIMPLE_SEARCH_SPACE, n_warmup=5)
        assert len(opt.state.arms) == 5

    def test_suggest_returns_valid_config(self):
        config = imoss_tpe(SIMPLE_SEARCH_SPACE).suggest()
        assert "x1" in config
        assert "x2" in config
        assert 0.0 <= config["x1"] <= 1.0
        assert 0.0 <= config["x2"] <= 1.0

    def test_suggest_observe_cycle(self):
        opt = imoss_tpe(SIMPLE_SEARCH_SPACE, n_warmup=3)
        for _ in range(20):
            opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_observe_without_suggest_raises(self):
        opt = imoss_tpe(SIMPLE_SEARCH_SPACE)
        with pytest.raises(RuntimeError):
            opt.observe(1.0)

    def test_observe_accepts_an_explicit_config(self):
        # Feedback that arrives out of order is credited by configuration, not by
        # "whatever was suggested last".
        opt = imoss_tpe(SIMPLE_SEARCH_SPACE, n_warmup=3)
        first = opt.suggest()
        opt.suggest()
        opt.observe(1.0, config=first)
        assert opt.state.arms[opt.space.encode(first)].nb_rewarded == 1

    @pytest.mark.parametrize("reward", [0.7, 3.5, -0.1])
    def test_observe_accepts_any_reward(self, reward):
        # observe() does not validate or clip the reward range: callers are
        # responsible for normalising into [0,1] (MOSS's regret bound assumes
        # it), but an out-of-range value is recorded rather than rejected.
        opt = imoss_tpe(SIMPLE_SEARCH_SPACE, n_warmup=3)
        opt.suggest()
        opt.observe(reward)
        assert any(arm.nb_rewarded > 0 for arm in opt.state.arms.values())

    def test_best_config_maximizes(self):
        opt = imoss_tpe({"x1": {"lower": 0.0, "upper": 1.0}}, n_warmup=5)
        for _ in range(50):
            config = opt.suggest()
            opt.observe(1.0 - abs(config["x1"] - 0.7))  # in [0,1], argmax at 0.7
        best = opt.best_config
        assert best is not None
        assert abs(best["x1"] - 0.7) < 0.4

    def test_log_scale_parameters(self):
        space = {"lr": {"lower": 1e-5, "upper": 1.0, "log": True}}
        config = imoss_tpe(space, n_warmup=3).suggest()
        assert 1e-5 <= config["lr"] <= 1.0

    def test_integer_parameters(self):
        space = {"n": {"lower": 1, "upper": 100, "int": True}}
        config = imoss_tpe(space, n_warmup=3).suggest()
        assert isinstance(config["n"], int)
        assert 1 <= config["n"] <= 100

    def test_categorical_parameters(self):
        space = {"model": {"choices": ["a", "b", "c"]}}
        config = imoss_tpe(space, n_warmup=3).suggest()
        assert config["model"] in ["a", "b", "c"]

    def test_mixed_search_space(self):
        space = {
            "lr": {"lower": 1e-4, "upper": 1.0, "log": True},
            "n_layers": {"lower": 1, "upper": 10, "int": True},
            "activation": {"choices": ["relu", "tanh"]},
        }
        opt = imoss_tpe(space, n_warmup=5)
        for _ in range(30):
            config = opt.suggest()
            assert 1e-4 <= config["lr"] <= 1.0
            assert isinstance(config["n_layers"], int)
            assert config["activation"] in ["relu", "tanh"]
            opt.observe(np.random.random())

    def test_reproducibility(self):
        def run():
            opt = imoss_tpe(SIMPLE_SEARCH_SPACE, seed=123, n_warmup=3)
            configs = []
            for _ in range(10):
                configs.append(opt.suggest())
                opt.observe(0.5)
            return configs

        assert run() == run()

    def test_delayed_policy(self):
        opt = IMABO(
            SIMPLE_SEARCH_SPACE,
            IMOSS(n_warmup=3, delayed=True),
            TPEOracle(),
            seed=42,
        )
        for _ in range(20):
            opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_run_drives_the_loop(self):
        opt = imoss_tpe({"x1": {"lower": 0.0, "upper": 1.0}}, n_warmup=5)
        best = opt.run(lambda c: 1.0 - abs(c["x1"] - 0.7), n_rounds=50)
        assert best is not None
        assert abs(best["x1"] - 0.7) < 0.4


class TestDefaults:
    """`IMABO(search_space)` alone, and the three ways to give it a space."""

    def test_defaults_are_imoss_tpe(self):
        opt = IMABO(SIMPLE_SEARCH_SPACE, seed=0)
        assert isinstance(opt.policy, IMOSS)
        assert isinstance(opt.oracle, TPEOracle)
        assert opt.policy.beta == 0.5  # the paper's value

    def test_either_component_can_be_given_alone(self):
        assert isinstance(
            IMABO(SIMPLE_SEARCH_SPACE, IMOSS(beta=0.7), seed=0).oracle, TPEOracle
        )
        assert isinstance(
            IMABO(SIMPLE_SEARCH_SPACE, oracle=RandomOracle(), seed=0).policy, IMOSS
        )

    def test_defaults_match_the_explicit_pairing(self):
        def run(opt):
            configs = []
            for _ in range(20):
                configs.append(opt.suggest())
                opt.observe(0.5)
            return configs

        assert run(IMABO(SIMPLE_SEARCH_SPACE, seed=3)) == run(
            IMABO(SIMPLE_SEARCH_SPACE, IMOSS(), TPEOracle(), seed=3)
        )

    # The dict form, on the three shapes of space the paper's experiments use.
    SPACES = {
        "grid": {f"x{i}": {"choices": list(range(6))} for i in range(4)},
        "mixed": {
            "num_layers": {"choices": [1, 2, 3, 4, 5]},
            "batch_size": {"lower": 16, "upper": 512, "int": True, "log": True},
            "learning_rate": {"lower": 1e-4, "upper": 1e-1, "log": True},
            "momentum": {"lower": 0.1, "upper": 0.99},
            "max_dropout": {"lower": 0.0, "upper": 1.0},
        },
        "hotpotqa": {
            "top_k": {"lower": 1, "upper": 10, "int": True},
            "temperature": {"lower": 0.0, "upper": 1.0},
            "prompt_template": {"choices": ["few_shot", "zero_shot", "naive"]},
            "model": {"choices": ["a", "b", "c", "d", "e", "f"]},
        },
    }

    @pytest.mark.parametrize("name", sorted(SPACES))
    def test_a_dict_space_needs_nothing_else(self, name):
        spec = self.SPACES[name]
        opt = IMABO(spec, seed=1)
        for _ in range(80):
            config = opt.suggest()
            assert set(config) == set(spec)
            opt.observe(0.5)
        assert opt.best_config is not None

    @pytest.mark.parametrize("name", sorted(SPACES))
    def test_a_dict_and_a_prebuilt_SearchSpace_are_interchangeable(self, name):
        spec = self.SPACES[name]

        def run(space):
            opt = IMABO(space, seed=1)
            return [opt.suggest() for _ in range(5)]

        assert run(spec) == run(SearchSpace(spec))


class TestSearchSpace:
    def test_names_are_sorted_but_distributions_keep_declaration_order(self):
        # The key order is sorted; the distribution order is the declaration order,
        # because Optuna's multivariate Parzen estimator iterates it and its sampling
        # would otherwise consume the RNG differently.
        space = SearchSpace({"b": {"lower": 0.0, "upper": 1.0}, "a": {"choices": [1, 2]}})
        assert space.names == ["a", "b"]
        assert list(space.distributions) == ["b", "a"]

    def test_encode_decode_roundtrip(self):
        space = SearchSpace(SIMPLE_SEARCH_SPACE)
        config = {"x1": 0.25, "x2": 0.75}
        assert space.encode(config) == (0.25, 0.75)
        assert space.decode(space.encode(config)) == config

    def test_rejects_an_invalid_parameter(self):
        with pytest.raises(ValueError):
            SearchSpace({"oops": {}})

    def test_sample_stays_in_range(self):
        space = SearchSpace(
            {
                "lr": {"lower": 1e-5, "upper": 1.0, "log": True},
                "n": {"lower": 2, "upper": 9, "int": True},
                "k": {"choices": ["a", "b"]},
            }
        )
        rng = random.Random(0)
        for _ in range(200):
            config = space.sample(rng)
            assert 1e-5 <= config["lr"] <= 1.0
            assert isinstance(config["n"], int) and 2 <= config["n"] <= 9
            assert config["k"] in ("a", "b")


class TestSpaceFunction:
    """Declaring the space as a function, in Optuna's style."""

    SPEC = {
        "lr": {"lower": 1e-4, "upper": 1e-1, "log": True},
        "depth": {"lower": 1, "upper": 8, "int": True},
        "model": {"choices": ["a", "b"]},
    }

    @staticmethod
    def fn(trial):
        trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        trial.suggest_int("depth", 1, 8)
        trial.suggest_categorical("model", ["a", "b"])

    def test_it_describes_the_same_space_as_the_dict(self):
        # The oracles read `names`, `distributions` and `types`; a function must give
        # them exactly what the dict gives, or TPE and mutation would behave differently.
        from_dict = SearchSpace(self.SPEC)
        from_fn = SearchSpace(self.fn)
        assert from_fn.names == from_dict.names
        assert from_fn.distributions == from_dict.distributions
        assert from_fn.types == from_dict.types

    def test_call_order_is_draw_order(self):
        # The two modes describe the same space but do not consume the generator in the
        # same order: a dict is drawn in sorted-name order, a function in the order it
        # calls suggest_*. Written in sorted-name order, the two coincide exactly.
        def sorted_order(trial):
            trial.suggest_int("depth", 1, 8)
            trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            trial.suggest_categorical("model", ["a", "b"])

        def draws(space):
            rng = random.Random(7)
            return [space.sample(rng) for _ in range(25)]

        assert draws(SearchSpace(sorted_order)) == draws(SearchSpace(self.SPEC))
        assert draws(SearchSpace(self.fn)) != draws(SearchSpace(self.SPEC))

    def test_suggested_values_are_returned_to_the_function(self):
        seen = {}

        def fn(trial):
            seen["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            seen["model"] = trial.suggest_categorical("model", ["a", "b"])

        config = SearchSpace(fn).sample(random.Random(0))
        assert seen == config

    def test_returning_a_config_is_rejected(self):
        # The trap this closes: returning a dict with an extra fixed field, which would
        # otherwise be dropped without a word.
        def fn(trial):
            return {"lr": trial.suggest_float("lr", 1e-4, 1e-1), "retrieval": "dense"}

        with pytest.raises(TypeError, match="must not return"):
            SearchSpace(fn)

    def test_an_empty_space_is_rejected(self):
        with pytest.raises(ValueError, match="no parameter"):
            SearchSpace(lambda trial: None)

    def test_the_same_name_twice_with_different_bounds_is_rejected(self):
        def fn(trial):
            trial.suggest_float("lr", 0.0, 1.0)
            trial.suggest_float("lr", 0.0, 2.0)

        with pytest.raises(ValueError, match="different bounds"):
            SearchSpace(fn)

    def test_a_conditional_space_is_rejected_with_a_clear_message(self):
        # Not supported yet. It must fail with an explanation rather than a KeyError
        # somewhere inside encode().
        def fn(trial):
            if trial.suggest_categorical("model", ["small", "large"]) == "large":
                trial.suggest_float("experts", 1.0, 8.0)
            trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        space = SearchSpace(fn)
        rng = random.Random(0)
        with pytest.raises(ValueError, match="conditional space"):
            for _ in range(50):
                space.sample(rng)

    def test_varying_bounds_across_calls_are_rejected(self):
        # Same names, but a branch-dependent range. Also conditional: TPE would fit one
        # branch's bounds and propose outside the other's.
        def fn(trial):
            model = trial.suggest_categorical("model", ["small", "large"])
            trial.suggest_float("lr", 1e-5, 1e-2 if model == "large" else 1e-1, log=True)

        space = SearchSpace(fn)
        rng = random.Random(0)
        with pytest.raises(ValueError, match="conditional space"):
            for _ in range(50):
                space.sample(rng)

    def test_discovery_does_not_consume_the_run_rng(self):
        # Constructing the space calls the function once to discover the parameters. If
        # that call drew from the caller's generator, the first sample would start part
        # way into the stream and every seeded run would shift.
        from imabo.search_space import Trial

        space = SearchSpace(self.fn)
        first = space.sample(random.Random(9))

        untouched = Trial(random.Random(9))
        self.fn(untouched)
        assert first == untouched.params

    def test_it_drives_a_full_optimization(self):
        opt = IMABO(self.fn, IMOSS(beta=0.5, n_warmup=5), TPEOracle(), seed=0)
        for _ in range(60):
            config = opt.suggest()
            assert 1e-4 <= config["lr"] <= 1e-1
            assert isinstance(config["depth"], int)
            assert config["model"] in ("a", "b")
            opt.observe(1.0 - abs(config["depth"] - 6) / 8)
        best = opt.best_config
        assert best is not None and abs(best["depth"] - 6) <= 3

    @pytest.mark.parametrize(
        "oracle",
        [RandomOracle, TPEOracle, MutateKLTPEOracle],
        ids=["random", "tpe", "mutate"],
    )
    def test_every_model_free_oracle_accepts_a_function_space(self, oracle):
        opt = IMABO(self.fn, IMOSS(beta=0.5, n_warmup=4), oracle(), seed=1)
        for _ in range(80):
            opt.suggest()
            opt.observe(0.5)
        assert opt.best_config is not None


class TestOptunaAlignment:
    """`Trial` mirrors Optuna's, so a copied objective works unchanged."""

    def test_step_is_accepted_like_optuna(self):
        from optuna.distributions import FloatDistribution, IntDistribution

        def space(trial):
            trial.suggest_float("grid", 0.0, 1.0, step=0.25)
            trial.suggest_int("even", 2, 10, step=2)

        built = SearchSpace(space)
        assert built.distributions["grid"] == FloatDistribution(0.0, 1.0, step=0.25)
        assert built.distributions["even"] == IntDistribution(2, 10, step=2)

        rng = random.Random(0)
        drawn = [built.sample(rng) for _ in range(200)]
        assert {c["grid"] for c in drawn} <= {0.0, 0.25, 0.5, 0.75, 1.0}
        assert {c["even"] for c in drawn} <= {2, 4, 6, 8, 10}

    def test_the_log_step_conflict_is_optunas_own(self):
        def space(trial):
            trial.suggest_float("x", 1e-3, 1.0, step=0.1, log=True)

        with pytest.raises(ValueError, match="step"):
            SearchSpace(space)

    def test_trial_uses_optunas_attribute_names(self):
        trial = Trial(random.Random(0))
        trial.suggest_float("x", 0.0, 1.0)
        assert set(trial.params) == {"x"}
        assert set(trial.distributions) == {"x"}


class TestInMemoryStorage:
    SPACE = SearchSpace({"x1": {"lower": 0.0, "upper": 1.0}})

    def test_observe_updates_mean(self):
        storage = InMemoryStorage(param_names=["x1"])
        config = {"x1": 0.5}
        storage.observe(config, 1.0)
        storage.observe(config, 0.0)
        state = storage.get_current_state()
        key = (0.5,)
        assert abs(state.arms[key].mean_reward - 0.5) < 1e-10
        assert state.arms[key].nb_rewarded == 2

    def test_pull_increments_pending(self):
        storage = InMemoryStorage(param_names=["x1"])
        key = (0.5,)
        storage.pull_arm(key)
        state = storage.get_current_state()
        assert state.arms[key].nb_pending == 1
        assert state.nb_steps == 1

    def test_reward_rate(self):
        storage = InMemoryStorage(param_names=["x1"])
        for i in range(200):
            storage.pull_arm((float(i),))
        for i in range(100):
            storage.observe({"x1": float(i)}, 1.0)
        assert 0.4 < storage.get_reward_frequency() < 0.6

    def test_reward_rate_is_one_until_there_is_evidence(self):
        storage = InMemoryStorage(param_names=["x1"])
        for i in range(10):
            storage.pull_arm((float(i),))
        assert storage.get_reward_frequency() == 1.0

    def test_rewarded_arms_keep_admission_order(self):
        # The policy and the oracles break ties on the first arm admitted, so this
        # must not be sorted.
        storage = InMemoryStorage(param_names=["x1"])
        storage.observe({"x1": 0.9}, 0.9)
        storage.observe({"x1": 0.1}, 0.2)
        state = storage.get_current_state()
        assert [key for key, _ in IMABO.rewarded_arms(state)] == [(0.9,), (0.1,)]


class TestKeyHelpers:
    def test_config_to_key_roundtrip(self):
        names = ["b", "a"]
        key = config_to_key({"a": 1, "b": 2}, names)
        assert key == (1, 2)  # ordered by sorted name, not by `names`
        assert key_to_config(key, names) == {"a": 1, "b": 2}


class TestBanditIndices:
    def test_anytime_moss_index_adds_a_bonus(self):
        assert anytime_moss_index(0.5, n_pulls=10, n_arms=5, t=100) >= 0.5

    def test_anytime_moss_index_bonus_shrinks_with_pulls(self):
        few = anytime_moss_index(0.5, n_pulls=2, n_arms=5, t=1000)
        many = anytime_moss_index(0.5, n_pulls=200, n_arms=5, t=1000)
        assert few > many >= 0.5

    def test_lcb_is_pessimistic(self):
        assert lcb(0.5, n_pulls=4, t=100) <= 0.5

    def test_kl_ucb_in_range(self):
        assert 0.5 <= kl_ucb(mean=0.5, pulls=10, t=100) <= 1.0

    def test_ucb_is_optimistic(self):
        assert ucb(0.5, n_pulls=4, budget=100) >= 0.5


class TestPaperOracles:
    """The oracles of Sections 4.1.3 to 4.1.5."""

    SPACE = {f"x{i}": {"choices": list(range(6))} for i in range(4)}

    def _run(self, opt, n=600, seed=3):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            config = opt.suggest()
            opt.observe(float(rng.random() < 0.1 + 0.06 * sum(config.values())))
        return opt

    def _mutate_oracle(self, seed, beta=0.5):
        oracle = MutateKLTPEOracle()
        return IMABO(self.SPACE, IMOSS(beta=beta), oracle, seed=seed), oracle

    def test_klucb_forces_each_choice_once_then_ranks(self):
        # select() does not reserve: an uncredited choice keeps being returned
        # until its vote actually arrives, which is what the oracle relies on
        # (the credit for a proposal lands several rounds after the selection).
        b = KLUCB(3)
        assert [b.select() for _ in range(3)] == [0, 0, 0]
        for i, r in ((0, 1.0), (1, 0.0), (2, 1.0)):
            assert b.select() == i
            b.update(i, r)
        idx = b.indices()
        assert np.isfinite(idx).all()
        assert idx[1] < idx[0]  # the losing choice ranks last

    def test_klucb_revise_keeps_the_vote_count(self):
        b = KLUCB(2)
        b.update(0, 0.3)
        b.revise(0, 0.5)  # same vote, sharpened value
        assert b.n[0] == 1.0
        assert b.means[0] == pytest.approx(0.8)

    def test_mutate_changes_one_coordinate_of_the_best_arm(self):
        opt, oracle = self._mutate_oracle(seed=1)
        self._run(opt)
        assert len(opt.state.arms) > 5
        # Every registered decision is a single-coordinate move off its parent.
        names = opt.space.names
        for d in opt.memory.get_decisions():
            child = opt.space.decode(d.arm_key)
            parent = opt.space.decode(d.parent_key)
            differing = [p for p in names if child[p] != parent[p]]
            assert differing == [names[d.coord]]

    def test_mutate_credits_the_arm_mean_not_the_first_pull(self):
        opt, oracle = self._mutate_oracle(seed=2)
        self._run(opt)
        bandit = oracle._bandit()
        oracle._refresh_credits(bandit)
        credited = [d for d in opt.memory.get_decisions() if d.credited is not None]
        for d in credited:
            assert d.credited == pytest.approx(oracle._arm_mean(d.arm_key))
        # One vote per decision, however many rewards the arm collected.
        assert bandit.n.sum() == pytest.approx(len(credited))

    def test_mutate_is_reproducible(self):
        opt_a, oracle_a = self._mutate_oracle(seed=7)
        opt_b, oracle_b = self._mutate_oracle(seed=7)
        self._run(opt_a)
        self._run(opt_b)
        assert opt_a.best_config == opt_b.best_config
        assert np.allclose(
            opt_a.memory.get_coord_credits().total,
            opt_b.memory.get_coord_credits().total,
        )

    def test_candidate_pool_composition(self):
        space = SearchSpace(self.SPACE)
        pool = CandidatePool()
        parent = {f"x{i}": 3 for i in range(4)}
        rewarded = [(space.encode(parent), ArmStats(mean_reward=0.8, nb_rewarded=5))]

        candidates = pool.build(rewarded, space, random.Random(0))
        assert len(candidates) == pool.n
        # 10% uniform, the rest one coordinate off the best arm.
        mutants = candidates[10:]
        assert len(mutants) == 90
        for m in mutants:
            assert sum(m[p] != parent[p] for p in space.names) == 1

    def test_uniform_pool_still_available(self):
        space = SearchSpace(self.SPACE)
        pool = CandidatePool(source="uniform")
        candidates = pool.build([], space, random.Random(0))
        assert len(candidates) == pool.n

    def test_pool_drops_duplicates_and_open_arms(self):
        space = SearchSpace(self.SPACE)
        storage = InMemoryStorage(param_names=space.names)
        open_arm = {f"x{i}": 0 for i in range(4)}
        storage.set(space.encode(open_arm), ArmStats())
        duplicate = {f"x{i}": 1 for i in range(4)}
        survivors = CandidatePool().drop_duplicates_and_open(
            [open_arm, duplicate, dict(duplicate)],
            storage.get_current_state(),
            space,
            random.Random(0),
        )
        assert survivors == [duplicate]

    def test_defaults_are_the_tuned_configuration(self):
        # Constructing either oracle with no options must give the configuration
        # winning_configs.pdf specifies -- that is the point of this branch.
        t = TabPFNOracle(model={})
        assert (t.pool.source, t.pool.uniform_frac) == ("mutation", 0.1)
        assert (t.pool.scale, t.refit_every, t.quantile) == (0.1, 1, 0.975)
        assert t.acquisition == "quantile" and t.pool.filter_open is True
        c = MutateKLTPEOracle()
        assert (c.min_arms, c.n_candidates) == (10, 24)
        assert IMOSS().beta == 0.5
        opt = IMABO(self.SPACE, IMOSS(), c, seed=0)
        assert isinstance(c._bandit(), KLUCB)   # rebuilt from the memory, not held
        assert opt.policy.beta == 0.5

    def test_mutation_scale_is_a_no_op_on_categorical_axes(self):
        from imabo.oracles.candidate_pool import local_value_sampler, mutate_value

        space = SearchSpace(self.SPACE)
        scaled = [
            local_value_sampler(space.distributions, random.Random(11), 0.1)("x0", 3)
            for _ in range(50)
        ]
        plain = [
            mutate_value("x0", 3, space.distributions, random.Random(11))
            for _ in range(50)
        ]
        assert scaled == plain

    def test_local_step_stays_in_range_and_is_local(self):
        from imabo.oracles.candidate_pool import local_value_sampler

        space = SearchSpace({"lr": {"lower": 1e-5, "upper": 1.0, "log": True}})
        sample = local_value_sampler(space.distributions, random.Random(0), 0.1)
        d = space.distributions["lr"]
        width = math.log(d.high) - math.log(d.low)
        near = 0
        for current in (d.low, d.high, 1e-3):
            for _ in range(2000):
                v = sample("lr", current)
                assert d.low <= v <= d.high  # boundary clamp, original space
                near += abs(math.log(v) - math.log(current)) / width < 0.1
        # A uniform redraw would land within 10% only ~19% of the time.
        assert near / 6000 > 0.5


class TestBudgetedUCB:
    def test_suggest_observe_cycle(self):
        opt = IMABO(
            SIMPLE_SEARCH_SPACE,
            BudgetedUCB(budget=100, n_warmup=3),
            TPEOracle(),
            seed=42,
        )
        for _ in range(40):
            opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_kl_ucb_index(self):
        opt = IMABO(
            SIMPLE_SEARCH_SPACE,
            BudgetedUCB(budget=100, n_warmup=3, index="kl_ucb"),
            TPEOracle(),
            seed=42,
        )
        for _ in range(40):
            opt.suggest()
            opt.observe(np.random.random())
        assert opt.best_config is not None

    def test_best_config_maximizes(self):
        opt = IMABO(
            {"x1": {"lower": 0.0, "upper": 1.0}},
            BudgetedUCB(budget=200, n_warmup=5),
            TPEOracle(),
            seed=42,
        )
        for _ in range(80):
            config = opt.suggest()
            opt.observe(1.0 - abs(config["x1"] - 0.7))
        best = opt.best_config
        assert best is not None
        assert abs(best["x1"] - 0.7) < 0.5

    def test_reports_the_most_served_arm(self):
        # Unlike IMOSS, which reports the best mean.
        opt = IMABO(
            SIMPLE_SEARCH_SPACE,
            BudgetedUCB(budget=100, n_warmup=2),
            RandomOracle(),
            seed=0,
        )
        for _ in range(60):
            opt.suggest()
            opt.observe(0.5)
        state = opt.state
        served = max(IMABO.rewarded_arms(state), key=lambda a: a[1].nb_rewarded)[0]
        assert opt.best_config == opt.space.decode(served)


class TestWritingYourOwn:
    """The framework's two extension points, exercised end to end."""

    def test_a_custom_oracle_is_five_lines(self):
        class GridOracle(Oracle):
            """Admit the arm nearest a coarse grid point not yet opened."""

            def suggest(self, state, rewarded_arms, score):
                for step in (0.25, 0.5, 0.75):
                    config = {"x1": step, "x2": step}
                    if self.space.encode(config) not in state.arms:
                        return config
                return self.space.sample(self.rng)

        opt = IMABO(SIMPLE_SEARCH_SPACE, IMOSS(n_warmup=2), GridOracle(), seed=0)
        for _ in range(40):
            config = opt.suggest()
            opt.observe(1.0 - abs(config["x1"] - 0.5))
        assert opt.state.arms[(0.25, 0.25)].nb_rewarded > 0

    def test_a_custom_policy_only_needs_expand_and_select(self):
        class RoundRobin(AllocationPolicy):
            """Open five arms, then serve them in turn."""

            def setup(self, space, rng, memory):
                super().setup(space, rng, memory)
                for _ in range(5):
                    memory.set(space.encode(space.sample(rng)), ArmStats())

            def expand(self, state, rewarded_arms):
                return len(state.arms) < 8

            def select(self, state, rewarded_arms):
                return list(state.arms)[state.nb_steps % len(state.arms)]

        opt = IMABO(SIMPLE_SEARCH_SPACE, RoundRobin(), RandomOracle(), seed=0)
        for _ in range(40):
            opt.suggest()
            opt.observe(0.5)
        assert len(opt.state.arms) == 8
        # The ABC's default reporting rule: the best observed mean.
        assert opt.best_config is not None

    def test_propose_shows_what_the_oracle_would_admit(self):
        opt = imoss_tpe(SIMPLE_SEARCH_SPACE, n_warmup=3)
        for _ in range(30):
            opt.suggest()
            opt.observe(0.5)
        before = opt.state.nb_steps
        proposed = opt.propose()
        assert set(proposed) == {"x1", "x2"}
        # Diagnostics only: proposing serves nothing and opens no arm.
        assert opt.state.nb_steps == before

    def test_a_custom_memory_backend_plugs_in(self):
        class CountingMemory(InMemoryStorage):
            def __init__(self, param_names):
                super().__init__(param_names)
                self.pulls = 0

            def pull_arm(self, key):
                self.pulls += 1
                super().pull_arm(key)

        space = SearchSpace(SIMPLE_SEARCH_SPACE)
        memory = CountingMemory(param_names=space.names)
        opt = IMABO(space, IMOSS(n_warmup=3), TPEOracle(), seed=0, memory=memory)
        for _ in range(20):
            opt.suggest()
            opt.observe(0.5)
        assert memory.pulls == 20


class TestArmStats:
    def test_pending_is_not_decremented_when_a_reward_never_arrives(self):
        # n_rewards + n_pending is not the pull count: a censored pull stays pending
        # forever, which is what the delay-aware rules of Appendix C.1 read.
        space = SearchSpace(SIMPLE_SEARCH_SPACE)
        storage = InMemoryStorage(param_names=space.names)
        key = space.encode({"x1": 0.5, "x2": 0.5})
        storage.pull_arm(key)
        storage.pull_arm(key)
        storage.observe({"x1": 0.5, "x2": 0.5}, 1.0)
        arm = storage.get_current_state().arms[key]
        assert (arm.nb_rewarded, arm.nb_pending) == (1, 1)
        assert arm == ArmStats(mean_reward=1.0, nb_rewarded=1, nb_pending=1)


class TestMemoryBackedOracles:
    """The oracles keep nothing of their own: the memory carries it all."""

    SPACE = {"lr": {"lower": 0.0, "upper": 1.0}, "model": {"choices": ["a", "b"]}}

    @pytest.mark.parametrize(
        "oracle",
        [
            RandomOracle(),
            TPEOracle(),
            MutateKLTPEOracle(min_arms=3),
            TabPFNOracle(model={}, fit_granularity="pull"),
        ],
    )
    def test_no_oracle_holds_mutable_state(self, oracle):
        opt = IMABO(self.SPACE, IMOSS(beta=0.8, n_warmup=3), oracle, seed=1)
        before = {k: repr(v) for k, v in vars(oracle).items() if k != "rng"}
        for _ in range(60):
            opt.suggest()
            opt.observe(0.5)
        after = {k: repr(v) for k, v in vars(oracle).items() if k != "rng"}
        assert after == before

    def _bare_memory(self):
        """A backend that implements the `Memory` contract and nothing more."""

        class BareMemory(Memory):
            def __init__(self):
                self.arms = {}

            def set(self, key, stats):
                self.arms[key] = stats

            def get_reward_frequency(self):
                return 1.0

            def increment_step_counter(self):
                pass

            def get_current_state(self):
                return CurrentState(nb_steps=0, arms=frozendict(self.arms))

            def pull_arm(self, arm_key):
                pass

            def observe(self, config, reward):
                pass

        return BareMemory()

    def test_mutate_names_what_the_memory_is_missing(self):
        with pytest.raises(TypeError, match="get_decisions"):
            IMABO(self.SPACE, IMOSS(), MutateKLTPEOracle(), memory=self._bare_memory())

    def test_tabpfn_pull_mode_needs_the_reward_log(self):
        with pytest.raises(TypeError, match="get_rewards"):
            IMABO(
                self.SPACE,
                IMOSS(),
                TabPFNOracle(model={}, fit_granularity="pull"),
                memory=self._bare_memory(),
            )

    def test_a_bare_memory_still_runs_the_stateless_oracles(self):
        """The `Memory` contract alone is enough for Random and TPE."""
        for oracle in (RandomOracle(), TPEOracle()):
            opt = IMABO(self.SPACE, IMOSS(n_warmup=3), oracle, memory=self._bare_memory())
            assert opt.suggest() is not None

    def test_arm_mean_matches_the_memory_reward_log(self):
        opt = IMABO(self.SPACE, IMOSS(beta=0.8, n_warmup=3), MutateKLTPEOracle(min_arms=3), seed=3)
        rng = random.Random(0)
        for _ in range(80):
            opt.suggest()
            opt.observe(rng.random())
        for key, stats in opt.state.arms.items():
            if not stats.nb_rewarded:
                continue
            rewards = opt.memory.get_rewards(key)
            assert len(rewards) == stats.nb_rewarded
            assert opt.oracle._arm_mean(key) == pytest.approx(stats.mean_reward)

    def test_a_restarted_run_keeps_the_oracle_bookkeeping(self):
        """The point of the design: crash-safety with no oracle-specific API."""
        disk = {}

        class PersistentMemory(InMemoryStorage):
            def __init__(self, param_names):
                super().__init__(param_names)
                self.memory = disk.setdefault("arms", {})
                self._rewards = disk.setdefault("rewards", {})
                self._decisions = disk.setdefault("decisions", [])
                self._coord_credits = disk.get("credits")
                self.step_counter = disk.get("steps", 0)
                self._total_rewarded = sum(a.nb_rewarded for a in self.memory.values())
                self._total_pending = sum(a.nb_pending for a in self.memory.values())

            def save_decisions(self, decisions):
                super().save_decisions(decisions)
                disk["decisions"] = self._decisions

            def save_coord_credits(self, credits):
                super().save_coord_credits(credits)
                disk["credits"] = credits

            def increment_step_counter(self):
                super().increment_step_counter()
                disk["steps"] = self.step_counter

        def build():
            return IMABO(
                self.SPACE,
                IMOSS(beta=0.8, n_warmup=3),
                MutateKLTPEOracle(min_arms=3),
                seed=42,
                memory=PersistentMemory(sorted(self.SPACE)),
            )

        opt = build()
        for i in range(120):
            if i and i % 30 == 0:
                opt = build()                       # the process died and came back
            config = opt.suggest()
            opt.observe(0.9 if config["model"] == "b" else 0.2)

        credits = opt.memory.get_coord_credits()
        assert len(opt.memory.get_decisions()) > 20, "decisions did not survive"
        assert credits is not None and credits.t > 20, "credits did not survive"
        assert opt.best_config["model"] == "b", "the run did not converge across restarts"


class TestBaselines:
    """The paper's baselines, exposed from the library and driven like IMABO."""

    FLAT = {
        "a": {"lower": 0.0, "upper": 1.0},
        "b": {"choices": [1, 2, 3]},
        "c": {"lower": 1, "upper": 9, "int": True},
    }
    @staticmethod
    def function(trial):
        trial.suggest_float("a", 0.0, 1.0)
        trial.suggest_categorical("b", [1, 2, 3])
        trial.suggest_int("c", 1, 9)

    ALL = [
        ("RandomSearch", lambda s: RandomSearch(s, seed=1)),
        ("QRM2", lambda s: QRM2(s, seed=1)),
        ("UCBAIR", lambda s: UCBAIR(s, seed=1)),
        ("MOSSAIR", lambda s: MOSSAIR(s, seed=1)),
        ("HierMAB", lambda s: HierMAB(s, n_points=6, seed=1)),
        ("OptunaBandit", lambda s: OptunaBandit(s, seed=1)),
    ]

    def _drive(self, optimizer, n=25):
        configs = []
        for _ in range(n):
            if getattr(optimizer, "done", False):
                break
            configs.append(optimizer.suggest())
            optimizer.observe(0.5)
        return configs

    @pytest.mark.parametrize("name,make", ALL)
    def test_every_baseline_drives_like_imabo(self, name, make):
        """suggest() / observe(reward) / best_config -- the same loop as IMABO."""
        optimizer = make(self.FLAT)
        configs = self._drive(optimizer)
        assert configs and all(isinstance(c, dict) for c in configs)
        assert hasattr(optimizer, "best_config")

    @pytest.mark.parametrize("name,make", ALL)
    def test_a_prebuilt_search_space_is_accepted(self, name, make):
        assert self._drive(make(SearchSpace(self.FLAT)))

    @pytest.mark.parametrize("name,make", ALL)
    def test_a_suggestion_function_is_accepted(self, name, make):
        assert self._drive(make(self.function))

    def test_the_tree_search_generators_are_exported(self):
        """They work on [0, 1]**d, driven through TimedOptimizer as the experiments do."""
        random.seed(0)
        optimizer = TimedOptimizer(stosoo, 60, 2)
        served = 0
        while not optimizer.done and served < 60:
            x = optimizer.suggest()
            assert len(x) == 2 and all(0.0 <= v <= 1.0 for v in x)
            optimizer.observe(x, 0.5)
            served += 1
        assert served > 0

