from collections import Counter

import numpy as np
import pytest
from frozendict import frozendict

from imabo import (
    EXP3,
    IMABO,
    ArmStats,
    CurrentState,
    FiniteIMABO,
    IMABOCoordUCB,
    IMABOTabPFN,
    UCB1,
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


class TestNonStationaryBandits:
    """Discounted UCB1 and EXP3 -- the two answers to a coordinate bandit whose
    reward decays as its coordinate gets optimised (see imabo.coord_ucb)."""

    def test_discount_one_is_a_no_op(self):
        plain, disc = UCB1(3, bonus="kl"), UCB1(3, bonus="kl", discount=1.0)
        for i, r in enumerate([0.9, 0.1, 0.5, 0.8, 0.2, 0.7]):
            plain.update(i % 3, r)
            disc.update(i % 3, r)
        assert np.allclose(plain.indices(), disc.indices())

    def test_discount_forgets_old_rewards(self):
        b = UCB1(2, discount=0.5)
        b.update(0, 1.0)  # arm 0 was great, long ago
        for _ in range(10):
            b.update(1, 0.5)
        # 1.0 * 0.5**10 of weight left on arm 0, so its mean is untouched but its
        # effective count has collapsed -- which is what re-opens it to selection.
        assert b.means[0] == pytest.approx(1.0)
        assert b.n[0] < 1e-2
        assert b.n[1] > 1.0

    def test_discounted_revise_matches_knowing_the_value_upfront(self):
        late, upfront = UCB1(2, discount=0.9), UCB1(2, discount=0.9)
        epoch = late.update(0, 0.3)
        upfront.update(0, 0.8)
        for _ in range(5):
            late.update(1, 0.1)
            upfront.update(1, 0.1)
        late.revise(0, 0.8 - 0.3, epoch)
        assert late.sum[0] == pytest.approx(upfront.sum[0])

    def test_exp3_probabilities_are_a_distribution(self):
        b = EXP3(4, rng=np.random.default_rng(0))
        for _ in range(50):
            b.update(b.select(), 0.5)
        p = b.probabilities()
        assert p.min() > 0.0
        assert p.sum() == pytest.approx(1.0)

    def test_exp3_concentrates_on_the_better_choice(self):
        b = EXP3(2, rng=np.random.default_rng(0))
        rng = np.random.default_rng(1)
        for _ in range(400):
            i = b.select()
            b.update(i, float(rng.random() < (0.9 if i == 0 else 0.1)))
        assert b.probabilities()[0] > 0.7

    def test_exp3_tracks_a_switch_only_with_mixing(self):
        # Arm 0 pays for the first half, arm 1 for the second: the setting the
        # coordinate bandit is actually in. Plain EXP3 chases the best FIXED arm.
        def run(mixing):
            b = EXP3(2, mixing=mixing, rng=np.random.default_rng(0))
            for step in range(1200):
                i = b.select()
                good = 0 if step < 600 else 1
                b.update(i, 1.0 if i == good else 0.0)
            return b.probabilities()[1]

        assert run(0.05) > run(0.0)

    def test_exp3_rejects_the_improvement_credit(self):
        with pytest.raises(ValueError, match="exp3"):
            IMABOCoordUCB(
                search_space={"x": {"choices": [0, 1, 2]}},
                bandit_rule="exp3",
                credit_rule="improvement",
            )

    def test_per_parent_scope_keeps_one_bandit_per_parent(self):
        opt = IMABOCoordUCB(
            search_space={f"x{i}": {"choices": list(range(5))} for i in range(3)},
            seed=1,
            beta=0.5,
            parent_rule="best",
            credit_rule="arm_mean",
            bandit_bonus="kl",
            coord_bandit_scope="parent",
        )
        rng = np.random.default_rng(2)
        for _ in range(800):
            config = opt.suggest()
            opt.observe(float(rng.random() < 0.1 + 0.06 * sum(config.values())))
        # More than one parent was mutated, each got its own bandit, and each
        # bandit's votes are its own rather than the run's total.
        assert len(opt.coord_bandits) > 1
        assert None not in opt.coord_bandits
        for key, bandit in opt.coord_bandits.items():
            assert key in opt.memory.get_current_state().arms
            assert bandit.n.size == len(opt.param_names)

    def test_per_parent_credit_routes_to_the_bandit_that_chose(self):
        # A vote revised long after it was cast must land on its OWN parent's
        # bandit, not on whichever parent happens to be current.
        opt = IMABOCoordUCB(
            search_space={f"x{i}": {"choices": [0, 1, 2]} for i in range(2)},
            seed=3,
            beta=0.5,
            parent_rule="best",
            credit_rule="arm_mean",
            coord_bandit_scope="parent",
        )
        rng = np.random.default_rng(4)
        for _ in range(600):
            config = opt.suggest()
            opt.observe(float(rng.random() < 0.2 + 0.2 * sum(config.values())))
        for decisions in opt._pending_choice.values():
            for decision in decisions:
                if decision.credited_high is None:
                    continue
                bandit = opt.coord_bandits[decision.parent_key]
                assert bandit.n[decision.coord] > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bandit_rule": "ucb", "bandit_discount": 0.95},
            {"coord_bandit_scope": "parent"},
            {"bandit_rule": "exp3", "coord_bandit_scope": "parent"},
            {"bandit_rule": "exp3"},
            {"bandit_rule": "exp3", "exp3_mixing": 0.01, "exp3_feedback": "reward"},
            {"bandit_rule": "exp3", "value_rule": "ucb"},
        ],
    )
    def test_coord_ucb_runs_and_stays_reproducible(self, kwargs):
        def run():
            opt = IMABOCoordUCB(
                search_space={f"x{i}": {"choices": list(range(5))} for i in range(3)},
                seed=7,
                beta=0.5,
                parent_rule="best",
                credit_rule="arm_mean",
                **kwargs,
            )
            rng = np.random.default_rng(3)
            out = []
            for _ in range(400):
                config = opt.suggest()
                reward = float(rng.random() < 0.1 + 0.06 * sum(config.values()))
                opt.observe(reward)
                out.append(reward)
            return out, opt

        first, opt = run()
        second, _ = run()
        assert first == second  # the stochastic selection rule is still seeded
        assert opt.coord_bandit.epoch > 0
        assert len(opt.memory.get_current_state().arms) > 1


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


MIXED_SEARCH_SPACE = {
    "lr": {"lower": 1e-5, "upper": 1.0, "log": True},
    "depth": {"lower": 1, "upper": 5, "int": True},
    "kind": {"choices": ["a", "b", "c"]},
}


class TestTabPFNCandidatePool:
    """The mutation candidate pool of IMABOTabPFN (candidate_source="mutation").

    Only the pool machinery is exercised -- constructing the optimizer never
    imports tabpfn (the checkpoint is loaded lazily on the first fit), so these
    run without the `experiments` extra.
    """

    @staticmethod
    def _population(opt, rewards):
        return [
            (config_to_key(config, opt.param_names), ArmStats(mean_reward=r, nb_rewarded=3))
            for config, r in rewards
        ]

    def _opt(self, **kwargs):
        kwargs.setdefault("candidate_source", "mutation")
        return IMABOTabPFN(search_space=MIXED_SEARCH_SPACE, seed=0, **kwargs)

    def test_invalid_options_rejected(self):
        for bad in (
            {"candidate_source": "evolution"},
            {"candidate_uniform_frac": 1.5},
            {"candidate_temperature": 0.0},
        ):
            with pytest.raises(ValueError):
                IMABOTabPFN(search_space=MIXED_SEARCH_SPACE, **bad)

    def test_parent_probabilities_scale_invariant(self):
        opt = self._opt()
        scores = np.array([0.1, 0.2, 0.3, 0.9])
        # T = temperature * std(scores), so a positive affine rescaling of the
        # rewards leaves the distribution untouched.
        assert np.allclose(
            opt._parent_probabilities(scores),
            opt._parent_probabilities(scores * 100 + 7),
        )

    def test_parent_probabilities_temperature_limits(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9])
        greedy = self._opt(candidate_temperature=1e-3)._parent_probabilities(scores)
        flat = self._opt(candidate_temperature=1e3)._parent_probabilities(scores)
        assert greedy.argmax() == 3 and greedy[3] == pytest.approx(1.0)
        assert flat == pytest.approx(np.full(4, 0.25), abs=0.01)

    def test_parent_probabilities_degenerate_population(self):
        opt = self._opt()
        assert opt._parent_probabilities(np.full(4, 0.5)) == pytest.approx(
            np.full(4, 0.25)
        )
        assert opt._parent_probabilities(np.array([0.5])) == pytest.approx([1.0])

    def test_mutation_changes_exactly_one_parameter(self):
        opt = self._opt()
        parent = {"lr": 0.01, "depth": 3, "kind": "b"}
        mutated = Counter()
        for _ in range(300):
            child = opt._mutate_config(parent)
            differing = [k for k in parent if child[k] != parent[k]]
            assert len(differing) == 1
            mutated[differing[0]] += 1
        # Every coordinate is a mutation target (uniform choice among them).
        assert set(mutated) == set(parent)

    def test_discrete_mutation_excludes_parent_value(self):
        opt = self._opt()
        parent = {"lr": 0.01, "depth": 3, "kind": "b"}
        children = [opt._mutate_config(parent) for _ in range(600)]
        # Only the children whose mutated coordinate IS this one -- the others
        # inherit the parent's value untouched.
        depths = {c["depth"] for c in children if c["depth"] != parent["depth"]}
        kinds = {c["kind"] for c in children if c["kind"] != parent["kind"]}
        assert depths == {1, 2, 4, 5}  # the int domain minus the parent's 3
        assert kinds == {"a", "c"}

    def test_pool_mixes_uniform_draws_and_mutants(self):
        opt = self._opt(n_candidates=200, candidate_uniform_frac=0.25)
        population = self._population(
            opt,
            [
                ({"lr": 0.5, "depth": 5, "kind": "a"}, 0.9),
                ({"lr": 1e-4, "depth": 1, "kind": "b"}, 0.2),
            ],
        )
        pool = opt._sample_candidates(population)
        assert len(pool) == 200
        parents = [key_to_config(k, opt.param_names) for k, _ in population]
        distances = [
            min(
                sum(candidate[p] != parent[p] for p in opt.param_names)
                for parent in parents
            )
            for candidate in pool
        ]
        # The 150 mutants sit exactly one coordinate from their parent. A
        # uniform draw is at distance >= 2 unless it happens to match a parent's
        # depth AND kind (~13% of draws with two parents), so most of the 50
        # uniform ones are further out.
        assert sum(d == 1 for d in distances) >= 150
        assert sum(d >= 2 for d in distances) >= 30

    def _state(self, population):
        return CurrentState(nb_steps=500, arms=frozendict(dict(population)))

    def test_tpe_sources_need_a_state_and_two_arms(self):
        # Without the state there is no good/bad split to fit densities on, so
        # both TPE-based sources degrade to the plain uniform pool.
        for source in ("mutation_tpe", "tpe"):
            opt = self._opt(candidate_source=source, n_candidates=40)
            population = self._population(
                opt,
                [
                    ({"lr": 0.5, "depth": 5, "kind": "a"}, 0.9),
                    ({"lr": 1e-4, "depth": 1, "kind": "b"}, 0.2),
                ],
            )
            assert len(opt._sample_candidates(population)) == 40
            one_arm = population[:1]
            assert len(opt._sample_candidates(one_arm, self._state(one_arm))) == 40

    def test_mutation_tpe_pool_is_one_coordinate_off_a_parent(self):
        opt = self._opt(
            candidate_source="mutation_tpe", n_candidates=100, candidate_uniform_frac=0.1
        )
        population = self._population(
            opt,
            [
                ({"lr": 0.5, "depth": 5, "kind": "a"}, 0.9),
                ({"lr": 1e-4, "depth": 1, "kind": "b"}, 0.2),
                ({"lr": 1e-2, "depth": 3, "kind": "c"}, 0.5),
            ],
        )
        pool = opt._sample_candidates(population, self._state(population), 0, 9)
        parents = [key_to_config(k, opt.param_names) for k, _ in population]
        distances = [
            min(
                sum(candidate[p] != parent[p] for p in opt.param_names)
                for parent in parents
            )
            for candidate in pool
        ]
        assert len(pool) == 100
        # Same 90/10 split as "mutation": only the value of the mutated
        # coordinate changes (drawn by TPE instead of uniformly).
        assert sum(d == 1 for d in distances) >= 90

    def test_tpe_pool_is_not_parent_anchored(self):
        opt = self._opt(candidate_source="tpe", n_candidates=100)
        population = self._population(
            opt,
            [
                ({"lr": 0.5, "depth": 5, "kind": "a"}, 0.9),
                ({"lr": 1e-4, "depth": 1, "kind": "b"}, 0.2),
                ({"lr": 1e-2, "depth": 3, "kind": "c"}, 0.5),
            ],
        )
        pool = opt._sample_candidates(population, self._state(population), 0, 9)
        parents = [key_to_config(k, opt.param_names) for k, _ in population]
        distances = [
            min(
                sum(candidate[p] != parent[p] for p in opt.param_names)
                for parent in parents
            )
            for candidate in pool
        ]
        assert len(pool) == 100
        # Joint draws from l, not perturbations of one arm: most candidates
        # differ from every parent in more than a single coordinate.
        assert sum(d >= 2 for d in distances) > len(pool) // 2

    def test_uniform_source_and_empty_population_fall_back_to_uniform(self):
        uniform = IMABOTabPFN(
            search_space=MIXED_SEARCH_SPACE, seed=0, n_candidates=50
        )
        population = self._population(uniform, [({"lr": 0.5, "depth": 5, "kind": "a"}, 0.9)])
        assert len(uniform._sample_candidates(population)) == 50
        # No rewarded arm to mutate yet -> uniform pool, whatever the source.
        assert len(self._opt(n_candidates=50)._sample_candidates([])) == 50


class TestIMABOCoordUCB:
    """The surrogate-free coordinate-bandit oracle (imabo/coord_ucb.py)."""

    def test_invalid_temperature_rejected(self):
        with pytest.raises(ValueError):
            IMABOCoordUCB(search_space=MIXED_SEARCH_SPACE, temperature=0.0)

    def test_optimizes_a_separable_objective(self):
        opt = IMABOCoordUCB(
            search_space={
                "x1": {"choices": [0.0, 0.25, 0.5, 0.75, 1.0]},
                "x2": {"choices": [0.0, 0.25, 0.5, 0.75, 1.0]},
            },
            seed=0,
            beta=0.5,
            min_arms_for_mutation=5,
        )
        for _ in range(400):
            config = opt.suggest()
            opt.observe(1.0 - (abs(config["x1"] - 0.75) + abs(config["x2"] - 0.25)) / 2)
        assert opt.best_config == {"x1": 0.75, "x2": 0.25}

    def test_coordinate_bandit_credited_once_per_proposed_arm(self):
        opt = IMABOCoordUCB(
            search_space=MIXED_SEARCH_SPACE,
            seed=0,
            beta=0.5,
            min_arms_for_mutation=5,
        )
        for _ in range(300):
            opt.suggest()
            opt.observe(0.5)
        bandit = opt.coord_bandit
        # One update per mutant that has since been rewarded, and every reward
        # was 0.5, so each coordinate's running mean must be exactly that.
        assert bandit.n.sum() > 0
        assert bandit.n.size == len(opt.param_names)
        for i in range(bandit.n.size):
            if bandit.n[i]:
                assert bandit.means[i] == pytest.approx(0.5)
        # Nothing is credited twice: repeated exploit pulls of the same arm are
        # not new coordinate decisions.
        assert bandit.n.sum() <= opt.memory.get_current_state().nb_steps

    def test_proposals_are_single_coordinate_mutations_of_a_known_arm(self):
        opt = IMABOCoordUCB(
            search_space=MIXED_SEARCH_SPACE, seed=0, beta=0.5, min_arms_for_mutation=5
        )
        for _ in range(120):
            opt.suggest()
            opt.observe(float(np.random.random()))

        state = opt.memory.get_current_state()
        rewarded = opt.get_rewarded_arms(state)
        known = [key_to_config(k, opt.param_names) for k, _ in rewarded]
        for _ in range(30):
            proposal = opt.suggest_new(state, rewarded)
            distance = min(
                sum(proposal[p] != arm[p] for p in opt.param_names) for arm in known
            )
            assert distance == 1

    def test_json_serializable_configs(self):
        # Proposed values come back through Optuna's external repr, which yields
        # np.float64 for float params; the experiment scripts json.dump configs.
        import json

        opt = IMABOCoordUCB(
            search_space={"lr": {"lower": 1e-5, "upper": 1.0, "log": True}},
            seed=0,
            beta=0.5,
            min_arms_for_mutation=5,
        )
        for _ in range(120):
            opt.suggest()
            opt.observe(float(np.random.random()))
        json.dumps(opt.best_config)
