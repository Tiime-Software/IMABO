"""Per-iteration distribution of the arm each oracle would suggest.

One file runs the RF-tabular-bandit experiment for every method, including both
foundation-model oracles: Google TabFM (``IMOSS-TabFM``) and Prior-Labs
TabPFN-3 (``IMOSS-TabPFN``). All runs are checkpointed per seed and resumable,
so reruns skip finished seeds.

Where the proposals come from is the axis this comparison varies. Plain
``IMOSS-TabPFN`` ranks a *uniform* candidate pool, which in a large space is the
binding constraint -- no pool member starts out near a good arm. The variants
change only how candidates are generated, never how they are scored:

    IMOSS-TabPFN-coord      10% uniform + mutants of the BEST arm so far (one
                            coordinate resampled uniformly)
    ...-coord-softmax       same, but each mutant's parent is sampled from the
                            population by softmax(mean reward)
    ...-coord-TPE-softmax   softmax parent, and the mutated coordinate's value
                            comes from a univariate TPE instead of a uniform draw
    IMOSS-TabPFN-TPE        no parent: the pool is drawn from TPE's own proposal
                            density, so TabPFN's acquisition replaces TPE's
                            expected improvement as the selection rule
    IMOSS-coordUCB-TPE      no surrogate at all: coordinate by Hier-MAB's own UCB1
                            bandit, value by univariate TPE. The plain name mutates
                            the best arm so far; -softmax draws the parent from the
                            population, -lastprop mutates this oracle's previous
                            proposal, -moss mutates whatever the exploit phase
                            would pull now

One more baseline sits outside that family: ``Hier-TPE`` is Hier-MAB with its
low-level per-axis value bandit replaced by a univariate TPE (no IMOSS switching
rule, no proposal oracle -- see experiments/baselines/hier_tpe.py).

Reproduce (each command resumable; the surrogate-free baselines are shared by
both foundation-model figures):

    # run everything (all algorithms x all benchmarks), then plot both figures:
    python -m experiments.rf_arm_distribution_experiment

    # run a single method:
    python -m experiments.rf_arm_distribution_experiment --algorithm IMOSS-TabPFN

    # a proposal variant at the explore-heavy switching exponent (its own
    # '..._beta0.8' files, so beta=0.5 results are never clobbered):
    python -m experiments.rf_arm_distribution_experiment \
        --algorithm IMOSS-TabPFN-coord --beta 0.8

    # TabPFN per-pull variant (one row per pull, no per-arm averaging, KV-cache);
    # written to distinct '..._pull' files/plots so it never clobbers per-arm:
    python -m experiments.rf_arm_distribution_experiment \
        --algorithm IMOSS-TabPFN --fit-granularity pull

    # only (re)plot a foundation model's figure from existing result JSONs, with
    # every proposal variant drawn at BOTH switching exponents in one panel:
    python -m experiments.rf_arm_distribution_experiment \
        --plot-only --foundation tabpfn --coord-betas 0.5 0.8

The TabPFN arm needs the experiment extra (``pip install -e ".[experiments]"``).
"""

import argparse
import copy
import json
import time
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.hier_tpe import HierTPE
from experiments.baselines.random_search import RandomSearch
from experiments.baselines.ucb_air import UCBAIR
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from imabo import IMABO
from imabo.coord_ucb import IMABOCoordUCB
from imabo.memory import config_to_key
from imabo.optimizer import IMABOTabFM, load_tabfm
from imabo.tabpfn_optimizer import IMABOTabPFN, load_tabpfn

# Pick up TABPFN_TOKEN (PriorLabs license/API key) and friends from .env, as
# the HotpotQA experiment already does for OPENROUTER_API_KEY.
load_dotenv()

RESULT_DIR = Path(__file__).parent.parent / "results" / "hpo_finite_arm_distribution"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5
# Exploration floor of IMOSS-TPE-eps0.1 (encoded in its name).
EPS_GREEDY = 0.1
N_SHADOW = 10
N_JOBS = 8


def _silence_known_warnings() -> None:
    """Mute the noisy, harmless ``FutureWarning`` TabPFN's internal
    ``ColumnTransformer`` raises on every fit (about ``force_int_remainder_cols``
    in scikit-learn >=1.6) -- one block per TabPFN fit, saying nothing about the
    experiment. Set as a process-global filter (safe under the joblib threading
    backend); re-applied after ``load_tabpfn`` since importing tabpfn/sklearn can
    reset the warnings registry. A no-op for the non-TabPFN algorithms.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"The format of the columns of the 'remainder' transformer",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"sklearn\.compose\._column_transformer",
    )


_silence_known_warnings()


class Algorithm(Enum):
    IMOSS_TPE = "IMOSS-TPE"
    # TPE oracle with an exploration floor: EPS_GREEDY of the explore steps skip
    # the oracle and draw a uniform random config (IMABO's `eps_greedy`). TPE
    # imitates the density of good observations, so it has no term that grows for
    # an unsampled region; this supplies one unconditionally.
    IMOSS_TPE_EPS = "IMOSS-TPE-eps0.1"
    # TPE with a univariate Parzen estimator (multivariate=False): one
    # independent density per hyperparameter, i.e. a factored proposal.
    IMOSS_TPE_UNI = "IMOSS-TPE-univ"
    IMOSS = "IMOSS-Random"
    RANDOM = "Random Search"
    IMOSS_TABFM = "IMOSS-TabFM"
    IMOSS_TABPFN = "IMOSS-TabPFN"
    # Same TabPFN oracle, but ranking an evolutionary candidate pool instead of
    # a uniform one: 10% uniform, the rest single-coordinate mutations of the
    # config served last (IMABOTabPFN's candidate_source="mutation").
    IMOSS_TABPFN_COORD = "IMOSS-TabPFN-coord"
    # ... with the parent instead sampled from the whole rewarded population by
    # softmax(mean reward), so one pool spans several arms.
    IMOSS_TABPFN_COORD_SOFTMAX = "IMOSS-TabPFN-coord-softmax"
    # Same mutation pool, but the mutated coordinate's value comes from a
    # univariate TPE instead of a uniform draw (candidate_source="mutation_tpe").
    # Mutation pool with the univariate-TPE value, at both parent rules. The
    # plain name is the best-arm parent, as everywhere else.
    IMOSS_TABPFN_COORD_TPE = "IMOSS-TabPFN-coord-TPE"
    # ... with each mutant taking a RANDOM draw from the coordinate's TPE density
    # instead of its EI-argmax. The argmax is deterministic, so it hands every
    # mutant the same value and collapses a 90-mutant pool to 4 distinct configs;
    # sampling restores pool diversity (22 of the 25 possible neighbours).
    IMOSS_TABPFN_COORD_TPE_SAMPLED = "IMOSS-TabPFN-coord-TPE-sampled"
    IMOSS_TABPFN_COORD_TPE_SOFTMAX = "IMOSS-TabPFN-coord-TPE-softmax"
    # No parent, no mutation: the whole pool is drawn from TPE's own proposal
    # density and TabPFN's acquisition replaces TPE's expected improvement as
    # the rule that picks the winner (candidate_source="tpe").
    IMOSS_TABPFN_TPE = "IMOSS-TabPFN-TPE"
    # Half mutation pool, half joint multivariate TPE, deduplicated and topped up
    # with uniform draws to a full 100 distinct candidates. The two halves have
    # opposite failure modes -- the mutants saturate the incumbent's
    # neighbourhood, the TPE draws concentrate where l already is -- so the mix
    # asks whether TabPFN ranks better when handed both at once. The "-newarm"
    # one additionally drops already-open arms before scoring; the plain one lets
    # the acquisition re-propose a known arm, which is what it prefers to do.
    IMOSS_TABPFN_MIX = "IMOSS-TabPFN-mix"
    IMOSS_TABPFN_MIX_NEWARM = "IMOSS-TabPFN-mix-newarm"
    # Mostly-mutation instead of half-and-half: 10 uniform / 10 joint TPE / 80
    # mutants. The 50/50 mix lost to the pure mutation pool when neither filters
    # open arms (+28.0 +- 23.8), so this asks whether a small joint-TPE injection
    # -- enough to put something more than one coordinate from the parent in
    # front of the surrogate -- helps without diluting the neighbourhood.
    IMOSS_TABPFN_MUT80 = "IMOSS-TabPFN-mut80"
    IMOSS_TABPFN_MUT80_NEWARM = "IMOSS-TabPFN-mut80-newarm"
    # refit_every ablation on the 90-mutant + 10-uniform pool. The default 10
    # means TabPFN is fit only ~7 times in a 5000-round run and 89% of proposals
    # come off a stale shortlist, chosen by a model that never saw the arms
    # opened earlier in the same batch. r1 refits at every explore step.
    IMOSS_TABPFN_COORD_R1 = "IMOSS-TabPFN-coord-r1"
    IMOSS_TABPFN_COORD_R3 = "IMOSS-TabPFN-coord-r3"
    # The same three refit depths without the open-arm filter. The archived
    # pre-fix run at 464.3 is NOT a matched control for these -- it predates the
    # dedup step too -- so refit=10 is re-run here with dedup on, filter off.
    IMOSS_TABPFN_COORD_NOFILTER = "IMOSS-TabPFN-coord-nofilter"
    IMOSS_TABPFN_COORD_R1_NOFILTER = "IMOSS-TabPFN-coord-r1-nofilter"
    IMOSS_TABPFN_COORD_R3_NOFILTER = "IMOSS-TabPFN-coord-r3-nofilter"
    # The pools worth re-testing now that refit_every=1 makes the surrogate
    # actually adaptive: every earlier pool comparison ran at refit_every=10,
    # where TabPFN was fit ~7 times a run and could barely tell them apart.
    IMOSS_TABPFN_MIX_R1 = "IMOSS-TabPFN-mix-r1"
    IMOSS_TABPFN_MUT80_R1 = "IMOSS-TabPFN-mut80-r1"
    # ... plus the untested one that attacks saturation at its root: a
    # single-coordinate pool can only reach 25 configs on this grid, so 90 draws
    # collapse to ~23 distinct. A geometric mutation size reaches 224 at k=2
    # alone, giving the surrogate a pool that does not run out.
    IMOSS_TABPFN_KGEOM_R1 = "IMOSS-TabPFN-kgeom-r1"
    # The mix pool topped up with mutants of the runner-up arms instead of
    # uniform draws. Measured on mix-r1, the uniform top-up is 33% of the pool
    # and wins 13% of the argmaxes -- on predictive variance, since uniform
    # points are far from the training table and the 0.99 quantile rewards
    # unfamiliarity. This keeps the pool full without that unbudgeted leak.
    IMOSS_TABPFN_MIXRU_R1 = "IMOSS-TabPFN-mixRU-r1"
    # Acquisition sweep at refit_every=1. The earlier sweep ran at refit_every=10
    # -- the stale regime -- and found 0.841/0.975/0.99 indistinguishable here. On
    # the 2-D continuous boxes, dropping 0.99 -> 0.841 was worth -71 points and
    # closed the entire gap to IMOSS-mutate-KLxTPE, because the quantile rewards
    # predictive variance and so picks the least familiar candidate. On this grid
    # every candidate sits in one 25-config neighbourhood, so the variance term
    # should be near-constant and this should change little -- which is the point:
    # it tests that explanation rather than assuming it.
    IMOSS_TABPFN_R1_Q841 = "IMOSS-TabPFN-coord-r1-q0.841"
    IMOSS_TABPFN_R1_Q975 = "IMOSS-TabPFN-coord-r1-q0.975"
    IMOSS_TABPFN_R1_Q90 = "IMOSS-TabPFN-coord-r1-q0.9"
    # No TabPFN at all: coordinate by Hier-MAB's own UCB1 bandit, value by
    # univariate TPE (imabo.coord_ucb.IMABOCoordUCB); parent as above.
    IMOSS_COORD_UCB_TPE = "IMOSS-coordUCB-TPE"
    IMOSS_COORD_UCB_TPE_SOFTMAX = "IMOSS-coordUCB-TPE-softmax"
    # The same UCB1 coordinate bandit with NO model or bandit on the value: a
    # uniform draw over the axis. Ablates the low level -- is choosing the value
    # deliberately worth anything once the coordinate is chosen deliberately?
    IMOSS_COORD_UCB_RANDOM = "IMOSS-coordUCB-random"
    # The null for both deliberate choices: mutate the best arm at a UNIFORM
    # coordinate with a UNIFORM value -- no bandit, no density, no surrogate.
    IMOSS_MUTATE_RANDOM = "IMOSS-mutate-random"
    # ... and with only the value chosen deliberately, to separate the two.
    IMOSS_MUTATE_TPE = "IMOSS-mutate-TPE"
    # Same rule, never re-proposing a known arm, and mutating more than one
    # coordinate: exactly 2, or 1 plus a geometric tail. Multi-coordinate is the
    # one thing no rule here has tried, and it is what the separability assumption
    # behind coordinate-wise search forbids.
    IMOSS_MUTATE_TPE_NEW = "IMOSS-mutate-TPE-newarm"
    IMOSS_MUTATE_TPE_K2 = "IMOSS-mutate-TPE-k2"
    IMOSS_MUTATE_TPE_KGEOM = "IMOSS-mutate-TPE-kgeom"
    # The two bandit levels crossed, with a KL-UCB (variance-adapted) width and
    # the arm-mean credit that width needs. Named <coordinate rule>x<value rule>;
    # the two cells with no bandit at either level are IMOSS-mutate-TPE
    # (random x TPE) and IMOSS-mutate-random (random x random).
    IMOSS_MUTATE_KLXKL = "IMOSS-mutate-KLxKL"
    # Both levels KL-UCB (the value bandit's arms are the axis' own categorical
    # values, so nothing is discretised on this grid), plus forced novelty found by
    # masking the values that would reproduce a known arm.
    IMOSS_MUTATE_KLXKL_NEWARM = "IMOSS-mutate-KLxKL-newarm"
    IMOSS_MUTATE_KLXTPE = "IMOSS-mutate-KLxTPE"
    # The same cell at the textbook Hoeffding width, which is what isolates the
    # confidence width from the credit rule: IMOSS-coordUCB-TPE is also Hoeffding
    # but credits first_pull, so KLxTPE vs that one confounds the two.
    IMOSS_MUTATE_UCBXTPE = "IMOSS-mutate-UCBxTPE"
    # ... and the fourth cell of that 2x2 (KL width, first-pull credit), so the
    # width and the credit rule have an identified interaction and not just two
    # main effects each measured at a single level of the other.
    IMOSS_MUTATE_KLXTPE_FIRSTPULL = "IMOSS-mutate-KLxTPE-firstpull"
    # One coordinate bandit PER PARENT instead of one for the run: rather than
    # forgetting a reward that drifts, condition on the thing it depends on.
    IMOSS_MUTATE_KLXTPE_PERPARENT = "IMOSS-mutate-KLxTPE-perparent"
    # The same, at the Hoeffding width: per-parent conditioning splits the votes,
    # and the two widths respond to a thin count very differently -- KL narrows on
    # the Bernoulli variance while Hoeffding does not, so a fresh bandit stays in
    # round-robin far longer under Hoeffding.
    IMOSS_MUTATE_UCBXTPE_PERPARENT = "IMOSS-mutate-UCBxTPE-perparent"
    # The coordinate bandit's reward is NOT stationary: the credit of mutating a
    # coordinate is measured against the current incumbent, and collapses once the
    # incumbent already holds a good value on that axis, so a stationary estimator
    # keeps re-picking an axis that paid off long ago. Three answers, all in the
    # same (parent=best, value=TPE, credit=arm_mean) cell as IMOSS-mutate-KLxTPE:
    # discounted KL-UCB at two memory lengths (effective sample size 1/(1-gamma),
    # against ~1000 votes per run), and EXP3 / EXP3.S, which assume no
    # stationarity at all -- EXP3.S being the variant that tracks a moving best.
    IMOSS_MUTATE_DKLXTPE_95 = "IMOSS-mutate-dKLxTPE-0.95"
    IMOSS_MUTATE_DKLXTPE_99 = "IMOSS-mutate-dKLxTPE-0.99"
    IMOSS_MUTATE_EXP3XTPE = "IMOSS-mutate-EXP3xTPE"
    IMOSS_MUTATE_EXP3SXTPE = "IMOSS-mutate-EXP3SxTPE"
    # ... and the textbook reward-weighted EXP3, whose importance-weighted estimate
    # divides an accuracy near 0.9 by p instead of a loss near 0.1.
    IMOSS_MUTATE_EXP3RXTPE = "IMOSS-mutate-EXP3rxTPE"
    IMOSS_MUTATE_KLXRAND = "IMOSS-mutate-KLxrand"
    IMOSS_MUTATE_RANDXKL = "IMOSS-mutate-randxKL"
    # The coordinate bandit credited with the IMPROVEMENT over the parent
    # (mean(child) - mean(parent), both live), at two confidence widths: textbook
    # UCB1, and a KL-UCB adapted to such credits by rescaling the arm means onto
    # [0, 1] by their own spread (they are negative and tightly clustered, so
    # plain KL-UCB does not apply).
    IMOSS_MUTATE_UCBXTPE_IMPROVE = "IMOSS-mutate-UCBxTPE-improve"
    IMOSS_MUTATE_KLXTPE_IMPROVE = "IMOSS-mutate-KLxTPE-improve"
    # ... plus one extra bandit arm that proposes a whole multivariate-TPE config
    # instead of a mutation, so the bandit itself decides how often to jump
    # globally rather than hill-climb.
    IMOSS_MUTATE_KLXTPE_GLOBAL = "IMOSS-mutate-KLxTPE-global"
    # Value taken as a random draw from l instead of its EI-argmax.
    IMOSS_MUTATE_KLXTPE_SAMPLED = "IMOSS-mutate-KLxTPE-sampled"
    # Every explore step must open a NEW arm, found by masking the axis values that
    # would reproduce a known arm rather than by resampling.
    IMOSS_MUTATE_KLXTPE_NEWARM = "IMOSS-mutate-KLxTPE-newarm"
    # Coordinate bandit ranking coordinates by CONTRIBUTION (credit = child mean
    # minus parent mean, both live), at an exploration weight scaled down to the
    # size of that signal, and required to open a new arm every explore step.
    IMOSS_COORD_UCB_TPE_CONTRIB = "IMOSS-coordUCB-TPE-contrib"
    # The two parent rules that are NOT the best-arm hill-climb the plain names
    # use: mutate this oracle's own previous proposal (a walk), or the arm the
    # exploit phase would pull right now (which agrees with the best arm only
    # ~58% of the time, since MOSS adds an exploration bonus).
    IMOSS_COORD_UCB_TPE_NEW = "IMOSS-coordUCB-TPE-newarm"
    IMOSS_COORD_UCB_TPE_LASTPROP = "IMOSS-coordUCB-TPE-lastprop"
    IMOSS_COORD_UCB_TPE_MOSS = "IMOSS-coordUCB-TPE-moss"
    # Hier-MAB's whole proposal rule (UCB1 axis + UCB1 value on the axis' own
    # value set) as an IMOSS explore oracle. The base version mutates the config
    # served last (a local walk, Hier-MAB's incumbent analogue); the -softmax
    # version instead samples the parent from the rewarded population.
    IMOSS_HIER_MAB = "IMOSS-Hier-MAB"
    IMOSS_HIER_MAB_SOFTMAX = "IMOSS-Hier-MAB-softmax"
    # Same rule, but each decision's vote carries a better estimate of what it
    # produced than Hier-MAB's single first pull: the arm's running mean over
    # every reward the exploit phase collects for it (median 50 on this
    # benchmark), or that mean minus the parent's -- "did this coordinate pay".
    IMOSS_HIER_MAB_MEAN = "IMOSS-Hier-MAB-mean"
    IMOSS_HIER_MAB_IMPROVE = "IMOSS-Hier-MAB-improve"
    IMOSS_HIER_MAB_CONTRIB = "IMOSS-Hier-MAB-contrib"
    # require_new_arm ON ITS OWN, so the novelty requirement can be attributed
    # separately from the "-contrib" bundle (which also changes the credit and
    # the exploration weight).
    IMOSS_HIER_MAB_NEW = "IMOSS-Hier-MAB-newarm"
    IMOSS_HIER_MAB_LASTPROP = "IMOSS-Hier-MAB-lastprop"
    IMOSS_HIER_MAB_MOSS = "IMOSS-Hier-MAB-moss"
    # Hier-MAB with its low-level per-axis value bandit replaced by a univariate
    # TPE (experiments/baselines/hier_tpe.py). Not an IMOSS method: no
    # explore/exploit switching rule, no proposal-oracle probe.
    HIER_TPE = "Hier-TPE"
    UCB_AIR = "UCB-AIR"


# The TabPFN-oracle arms: everything below that reads a TabPFN-only option
# (acquisition/quantile/fit_granularity) or logs the surrogate diagnostics
# applies to all of them.
TABPFN_ALGORITHMS = (
    Algorithm.IMOSS_TABPFN,
    Algorithm.IMOSS_TABPFN_MIX,
    Algorithm.IMOSS_TABPFN_MIX_NEWARM,
    Algorithm.IMOSS_TABPFN_MUT80,
    Algorithm.IMOSS_TABPFN_MUT80_NEWARM,
    Algorithm.IMOSS_TABPFN_COORD_R1,
    Algorithm.IMOSS_TABPFN_COORD_R3,
    Algorithm.IMOSS_TABPFN_COORD_NOFILTER,
    Algorithm.IMOSS_TABPFN_COORD_R1_NOFILTER,
    Algorithm.IMOSS_TABPFN_COORD_R3_NOFILTER,
    Algorithm.IMOSS_TABPFN_MIX_R1,
    Algorithm.IMOSS_TABPFN_MUT80_R1,
    Algorithm.IMOSS_TABPFN_KGEOM_R1,
    Algorithm.IMOSS_TABPFN_MIXRU_R1,
    Algorithm.IMOSS_TABPFN_R1_Q841,
    Algorithm.IMOSS_TABPFN_R1_Q975,
    Algorithm.IMOSS_TABPFN_R1_Q90,
    Algorithm.IMOSS_TABPFN_COORD,
    Algorithm.IMOSS_TABPFN_COORD_SOFTMAX,
    Algorithm.IMOSS_TABPFN_COORD_TPE,
    Algorithm.IMOSS_TABPFN_COORD_TPE_SAMPLED,
    Algorithm.IMOSS_TABPFN_COORD_TPE_SOFTMAX,
    Algorithm.IMOSS_TABPFN_TPE,
)

# The proposal-pool variants of this comparison, i.e. the arms that read the
# `candidate_*` options. IMOSS-coordUCB-TPE is here too: it uses the same
# softmax parent selection (`candidate_temperature`), just no candidate pool.
POOL_VARIANT_ALGORITHMS = (
    Algorithm.IMOSS_TABPFN_MIX,
    Algorithm.IMOSS_TABPFN_MIX_NEWARM,
    Algorithm.IMOSS_TABPFN_MUT80,
    Algorithm.IMOSS_TABPFN_MUT80_NEWARM,
    Algorithm.IMOSS_TABPFN_COORD_R1,
    Algorithm.IMOSS_TABPFN_COORD_R3,
    Algorithm.IMOSS_TABPFN_COORD_NOFILTER,
    Algorithm.IMOSS_TABPFN_COORD_R1_NOFILTER,
    Algorithm.IMOSS_TABPFN_COORD_R3_NOFILTER,
    Algorithm.IMOSS_TABPFN_MIX_R1,
    Algorithm.IMOSS_TABPFN_MUT80_R1,
    Algorithm.IMOSS_TABPFN_KGEOM_R1,
    Algorithm.IMOSS_TABPFN_MIXRU_R1,
    Algorithm.IMOSS_TABPFN_R1_Q841,
    Algorithm.IMOSS_TABPFN_R1_Q975,
    Algorithm.IMOSS_TABPFN_R1_Q90,
    Algorithm.IMOSS_MUTATE_KLXKL_NEWARM,
    Algorithm.IMOSS_TABPFN_COORD,
    Algorithm.IMOSS_TABPFN_COORD_SOFTMAX,
    Algorithm.IMOSS_TABPFN_COORD_TPE,
    Algorithm.IMOSS_TABPFN_COORD_TPE_SAMPLED,
    Algorithm.IMOSS_TABPFN_COORD_TPE_SOFTMAX,
    Algorithm.IMOSS_TABPFN_TPE,
    Algorithm.IMOSS_COORD_UCB_TPE,
    Algorithm.IMOSS_COORD_UCB_TPE_SOFTMAX,
    Algorithm.IMOSS_COORD_UCB_RANDOM,
    Algorithm.IMOSS_COORD_UCB_TPE_CONTRIB,
    Algorithm.IMOSS_COORD_UCB_TPE_LASTPROP,
    Algorithm.IMOSS_COORD_UCB_TPE_MOSS,
    Algorithm.IMOSS_COORD_UCB_TPE_NEW,
    Algorithm.IMOSS_MUTATE_RANDOM,
    Algorithm.IMOSS_MUTATE_TPE,
    Algorithm.IMOSS_MUTATE_TPE_NEW,
    Algorithm.IMOSS_MUTATE_TPE_K2,
    Algorithm.IMOSS_MUTATE_TPE_KGEOM,
    Algorithm.IMOSS_MUTATE_KLXKL,
    Algorithm.IMOSS_MUTATE_KLXTPE,
    Algorithm.IMOSS_MUTATE_UCBXTPE,
    Algorithm.IMOSS_MUTATE_KLXTPE_FIRSTPULL,
    Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT,
    Algorithm.IMOSS_MUTATE_UCBXTPE_PERPARENT,
    Algorithm.IMOSS_MUTATE_DKLXTPE_95,
    Algorithm.IMOSS_MUTATE_DKLXTPE_99,
    Algorithm.IMOSS_MUTATE_EXP3XTPE,
    Algorithm.IMOSS_MUTATE_EXP3SXTPE,
    Algorithm.IMOSS_MUTATE_EXP3RXTPE,
    Algorithm.IMOSS_MUTATE_KLXRAND,
    Algorithm.IMOSS_MUTATE_RANDXKL,
    Algorithm.IMOSS_MUTATE_UCBXTPE_IMPROVE,
    Algorithm.IMOSS_MUTATE_KLXTPE_IMPROVE,
    Algorithm.IMOSS_MUTATE_KLXTPE_GLOBAL,
    Algorithm.IMOSS_MUTATE_KLXTPE_SAMPLED,
    Algorithm.IMOSS_MUTATE_KLXTPE_NEWARM,
    Algorithm.IMOSS_HIER_MAB,
    Algorithm.IMOSS_HIER_MAB_SOFTMAX,
    Algorithm.IMOSS_HIER_MAB_MEAN,
    Algorithm.IMOSS_HIER_MAB_IMPROVE,
    Algorithm.IMOSS_HIER_MAB_CONTRIB,
    Algorithm.IMOSS_HIER_MAB_LASTPROP,
    Algorithm.IMOSS_HIER_MAB_MOSS,
    Algorithm.IMOSS_HIER_MAB_NEW,
)

# The surrogate-free mutation oracles, and the (parent_rule, value_rule) each one
# hands to IMABOCoordUCB.
# credit_rule per arm, for the ones that deviate from Hier-MAB's own
# first-pull crediting (see IMABOCoordUCB's module docstring).
CREDIT_RULES = {
    Algorithm.IMOSS_MUTATE_KLXKL_NEWARM: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXKL: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXTPE: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT: "arm_mean",
    Algorithm.IMOSS_MUTATE_UCBXTPE_PERPARENT: "arm_mean",
    Algorithm.IMOSS_MUTATE_UCBXTPE: "arm_mean",
    Algorithm.IMOSS_MUTATE_DKLXTPE_95: "arm_mean",
    Algorithm.IMOSS_MUTATE_DKLXTPE_99: "arm_mean",
    Algorithm.IMOSS_MUTATE_EXP3XTPE: "arm_mean",
    Algorithm.IMOSS_MUTATE_EXP3SXTPE: "arm_mean",
    Algorithm.IMOSS_MUTATE_EXP3RXTPE: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXTPE_GLOBAL: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXTPE_SAMPLED: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXTPE_NEWARM: "arm_mean",
    Algorithm.IMOSS_MUTATE_KLXRAND: "arm_mean",
    Algorithm.IMOSS_MUTATE_RANDXKL: "arm_mean",
    Algorithm.IMOSS_MUTATE_UCBXTPE_IMPROVE: "improvement",
    Algorithm.IMOSS_MUTATE_KLXTPE_IMPROVE: "improvement",
    Algorithm.IMOSS_HIER_MAB_MEAN: "arm_mean",
    Algorithm.IMOSS_HIER_MAB_IMPROVE: "improvement",
    Algorithm.IMOSS_HIER_MAB_CONTRIB: "improvement",
    Algorithm.IMOSS_COORD_UCB_TPE_CONTRIB: "improvement",
}

# The "-contrib" arms: rank coordinates by contribution, and make that ranking
# actually bite. The improvement credits measured on this grid span ~0.07-0.09
# across coordinates over ~60 decisions each, while UCB1's default bonus at that
# count is ~0.43 -- so alpha has to come down by ~an order of magnitude for the
# means to decide anything (bonus ~ sqrt(alpha) * 0.43).
# Arms that require every explore step to open a new arm. The "-contrib" ones
# also change the credit and the exploration weight; the "-newarm" ones change
# nothing else, which is what isolates the novelty requirement.
NEW_ARM_ALGORITHMS = (
    Algorithm.IMOSS_MUTATE_KLXKL_NEWARM,
    Algorithm.IMOSS_MUTATE_KLXTPE_NEWARM,
    Algorithm.IMOSS_HIER_MAB_NEW,
    Algorithm.IMOSS_COORD_UCB_TPE_NEW,
    Algorithm.IMOSS_MUTATE_TPE_NEW,
    Algorithm.IMOSS_MUTATE_TPE_K2,
    Algorithm.IMOSS_MUTATE_TPE_KGEOM,
)

# Arms whose coordinate is drawn uniformly instead of by the UCB1 bandit.
# Per-vote decay of the coordinate/value bandits' counts (discounted UCB), where
# it is not 1.0 (no discounting).
# Arms running one coordinate bandit per parent instead of one for the run.
PER_PARENT_ALGORITHMS = (
    Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT,
    Algorithm.IMOSS_MUTATE_UCBXTPE_PERPARENT,
)

BANDIT_DISCOUNTS = {
    Algorithm.IMOSS_MUTATE_DKLXTPE_95: 0.95,
    Algorithm.IMOSS_MUTATE_DKLXTPE_99: 0.99,
}

# Arms whose bandits are EXP3 rather than UCB1, and the EXP3.S mixing rate each
# one runs at (0 = plain EXP3, which tracks a fixed best choice only).
EXP3_ALGORITHMS = {
    Algorithm.IMOSS_MUTATE_EXP3XTPE: 0.0,
    Algorithm.IMOSS_MUTATE_EXP3SXTPE: 0.01,
    Algorithm.IMOSS_MUTATE_EXP3RXTPE: 0.0,
}

RANDOM_COORD_ALGORITHMS = (
    Algorithm.IMOSS_MUTATE_RANDOM,
    Algorithm.IMOSS_MUTATE_TPE,
    Algorithm.IMOSS_MUTATE_TPE_NEW,
    Algorithm.IMOSS_MUTATE_TPE_K2,
    Algorithm.IMOSS_MUTATE_TPE_KGEOM,
    Algorithm.IMOSS_MUTATE_RANDXKL,
)

# Arms whose bandits use the KL-UCB width instead of Hoeffding. They need credits
# in [0, 1], hence the arm-mean credit rule that goes with them.
# Arms whose bandits use the spread-rescaled KL-UCB width, the one that admits
# the (negative, tightly clustered) improvement credit.
KL_SCALED_ALGORITHMS = (Algorithm.IMOSS_MUTATE_KLXTPE_IMPROVE,)

# Arms that add the global multivariate-TPE proposal to the coordinate bandit.
GLOBAL_TPE_ALGORITHMS = (Algorithm.IMOSS_MUTATE_KLXTPE_GLOBAL,)

KL_BANDIT_ALGORITHMS = (
    Algorithm.IMOSS_MUTATE_KLXKL_NEWARM,
    Algorithm.IMOSS_MUTATE_KLXTPE_GLOBAL,
    Algorithm.IMOSS_MUTATE_KLXTPE_SAMPLED,
    Algorithm.IMOSS_MUTATE_KLXTPE_NEWARM,
    Algorithm.IMOSS_MUTATE_KLXKL,
    Algorithm.IMOSS_MUTATE_KLXTPE,
    Algorithm.IMOSS_MUTATE_KLXTPE_FIRSTPULL,
    Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT,
    Algorithm.IMOSS_MUTATE_DKLXTPE_95,
    Algorithm.IMOSS_MUTATE_DKLXTPE_99,
    Algorithm.IMOSS_MUTATE_KLXRAND,
    Algorithm.IMOSS_MUTATE_RANDXKL,
)

# How many coordinates one mutation changes, where it is not the default 1.
MUTATION_SIZES = {
    Algorithm.IMOSS_MUTATE_TPE_K2: 2,
    Algorithm.IMOSS_MUTATE_TPE_KGEOM: "geometric",
}

CONTRIB_ALGORITHMS = (
    Algorithm.IMOSS_COORD_UCB_TPE_CONTRIB,
    Algorithm.IMOSS_HIER_MAB_CONTRIB,
)
CONTRIB_COORD_ALPHA = 0.05

COORD_UCB_RULES = {
    Algorithm.IMOSS_MUTATE_KLXKL_NEWARM: ("best", "ucb"),
    Algorithm.IMOSS_COORD_UCB_TPE: ("best", "tpe"),
    Algorithm.IMOSS_COORD_UCB_TPE_SOFTMAX: ("softmax", "tpe"),
    Algorithm.IMOSS_COORD_UCB_RANDOM: ("best", "random"),
    Algorithm.IMOSS_COORD_UCB_TPE_CONTRIB: ("best", "tpe"),
    Algorithm.IMOSS_COORD_UCB_TPE_LASTPROP: ("last_proposal", "tpe"),
    Algorithm.IMOSS_COORD_UCB_TPE_MOSS: ("moss", "tpe"),
    Algorithm.IMOSS_HIER_MAB: ("best", "ucb"),
    Algorithm.IMOSS_HIER_MAB_SOFTMAX: ("softmax", "ucb"),
    Algorithm.IMOSS_HIER_MAB_MEAN: ("best", "ucb"),
    Algorithm.IMOSS_HIER_MAB_IMPROVE: ("best", "ucb"),
    Algorithm.IMOSS_HIER_MAB_CONTRIB: ("best", "ucb"),
    Algorithm.IMOSS_HIER_MAB_LASTPROP: ("last_proposal", "ucb"),
    Algorithm.IMOSS_HIER_MAB_MOSS: ("moss", "ucb"),
    Algorithm.IMOSS_HIER_MAB_NEW: ("best", "ucb"),
    Algorithm.IMOSS_COORD_UCB_TPE_NEW: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_RANDOM: ("best", "random"),
    Algorithm.IMOSS_MUTATE_TPE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_TPE_NEW: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_TPE_K2: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_TPE_KGEOM: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXKL: ("best", "ucb"),
    Algorithm.IMOSS_MUTATE_KLXTPE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_UCBXTPE_PERPARENT: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXTPE_FIRSTPULL: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_UCBXTPE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_DKLXTPE_95: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_DKLXTPE_99: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_EXP3XTPE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_EXP3SXTPE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_EXP3RXTPE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXRAND: ("best", "random"),
    Algorithm.IMOSS_MUTATE_RANDXKL: ("best", "ucb"),
    Algorithm.IMOSS_MUTATE_UCBXTPE_IMPROVE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXTPE_IMPROVE: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXTPE_GLOBAL: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXTPE_SAMPLED: ("best", "tpe"),
    Algorithm.IMOSS_MUTATE_KLXTPE_NEWARM: ("best", "tpe"),
}

# Arms with no IMOSS switching exponent: their single stored configuration is
# valid at every beta, so `beta` must not suffix their slug (it would fork a
# byte-identical checkpoint under a second name). UCB-AIR does take a `beta`, but
# it is its own reservoir-tail exponent, pinned to the paper's value in
# build_optimizer -- unrelated to the explore/exploit switch.
BETA_FREE_ALGORITHMS = (Algorithm.UCB_AIR, Algorithm.HIER_TPE)

# Only these are drawn at every ``coord_betas`` entry in the figure; every other
# series appears at the figure's own ``beta`` alone. The switching exponent is a
# property of the run, not of the method, so which betas are worth showing per
# series is an editorial choice -- these two are the ones whose beta comparison
# the figure is meant to carry.
EXTRA_BETA_ALGORITHMS = (
    Algorithm.IMOSS_HIER_MAB,
    Algorithm.IMOSS_HIER_MAB_SOFTMAX,
    Algorithm.IMOSS_HIER_MAB_MEAN,
    Algorithm.IMOSS_HIER_MAB_IMPROVE,
    Algorithm.IMOSS_HIER_MAB_CONTRIB,
    Algorithm.IMOSS_HIER_MAB_LASTPROP,
    Algorithm.IMOSS_HIER_MAB_MOSS,
    Algorithm.IMOSS_HIER_MAB_NEW,
)

# Arms whose mutants take a random draw from the coordinate's TPE density rather
# than its (deterministic) EI-argmax.
TPE_SAMPLED_ALGORITHMS = (Algorithm.IMOSS_TABPFN_COORD_TPE_SAMPLED,)

# IMABOTabPFN's candidate_source per TabPFN arm (see its module docstring).
# TabPFN shortlist depth: one fit ranks the pool and its top `refit_every`
# candidates are served over that many explore steps without refitting. Default 10.
TABPFN_REFIT_EVERY = {
    Algorithm.IMOSS_TABPFN_MIX_R1: 1,
    Algorithm.IMOSS_TABPFN_MUT80_R1: 1,
    Algorithm.IMOSS_TABPFN_KGEOM_R1: 1,
    Algorithm.IMOSS_TABPFN_MIXRU_R1: 1,
    Algorithm.IMOSS_TABPFN_R1_Q841: 1,
    Algorithm.IMOSS_TABPFN_R1_Q975: 1,
    Algorithm.IMOSS_TABPFN_R1_Q90: 1,
    Algorithm.IMOSS_TABPFN_COORD_R1: 1,
    Algorithm.IMOSS_TABPFN_COORD_R1_NOFILTER: 1,
    Algorithm.IMOSS_TABPFN_COORD_R3: 3,
    Algorithm.IMOSS_TABPFN_COORD_R3_NOFILTER: 3,
}

# Arms that let the acquisition re-propose an already-open arm.
TABPFN_NO_FILTER = (
    Algorithm.IMOSS_TABPFN_MIX,
    Algorithm.IMOSS_TABPFN_MUT80,
    Algorithm.IMOSS_TABPFN_COORD_NOFILTER,
    Algorithm.IMOSS_TABPFN_COORD_R1_NOFILTER,
    Algorithm.IMOSS_TABPFN_COORD_R3_NOFILTER,
)

# Per-arm acquisition quantile, overriding the CLI --quantile.
TABPFN_QUANTILES = {
    Algorithm.IMOSS_TABPFN_R1_Q841: 0.841,
    Algorithm.IMOSS_TABPFN_R1_Q975: 0.975,
    Algorithm.IMOSS_TABPFN_R1_Q90: 0.9,
}

# Arms whose mix pool is topped up with runner-up mutants, not uniform draws.
TABPFN_TOPUP = {Algorithm.IMOSS_TABPFN_MIXRU_R1: "runner_up"}

# How many coordinates one TabPFN mutant changes, where it is not the default 1.
TABPFN_MUTATION_SIZES = {Algorithm.IMOSS_TABPFN_KGEOM_R1: "geometric"}

# Share of a mutation pool drawn from the joint multivariate TPE instead.
TABPFN_TPE_FRACS = {
    Algorithm.IMOSS_TABPFN_MUT80: 0.1,
    Algorithm.IMOSS_TABPFN_MUT80_NEWARM: 0.1,
    Algorithm.IMOSS_TABPFN_MUT80_R1: 0.1,
}

CANDIDATE_SOURCES = {
    Algorithm.IMOSS_TABPFN: "uniform",
    Algorithm.IMOSS_TABPFN_COORD: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_SOFTMAX: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_TPE: "mutation_tpe",
    Algorithm.IMOSS_TABPFN_COORD_TPE_SAMPLED: "mutation_tpe",
    Algorithm.IMOSS_TABPFN_COORD_TPE_SOFTMAX: "mutation_tpe",
    Algorithm.IMOSS_TABPFN_TPE: "tpe",
    Algorithm.IMOSS_TABPFN_MIX: "mix",
    Algorithm.IMOSS_TABPFN_MIX_NEWARM: "mix",
    Algorithm.IMOSS_TABPFN_MUT80: "mutation",
    Algorithm.IMOSS_TABPFN_MUT80_NEWARM: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_R1: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_R3: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_NOFILTER: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_R1_NOFILTER: "mutation",
    Algorithm.IMOSS_TABPFN_COORD_R3_NOFILTER: "mutation",
    Algorithm.IMOSS_TABPFN_MIX_R1: "mix",
    Algorithm.IMOSS_TABPFN_MUT80_R1: "mutation",
    Algorithm.IMOSS_TABPFN_KGEOM_R1: "mutation",
    Algorithm.IMOSS_TABPFN_MIXRU_R1: "mix",
    Algorithm.IMOSS_TABPFN_R1_Q841: "mutation",
    Algorithm.IMOSS_TABPFN_R1_Q975: "mutation",
    Algorithm.IMOSS_TABPFN_R1_Q90: "mutation",
}

# Which config a mutation pool builds its mutants from (IMABOTabPFN's
# parent_rule). The plain names are the local rule -- mutate the config served
# last -- and the "-softmax" ones sample a parent from the population, the same
# naming split as the IMOSS-Hier-MAB pair.
TABPFN_PARENT_RULES = {
    Algorithm.IMOSS_TABPFN_COORD: "best",
    Algorithm.IMOSS_TABPFN_COORD_SOFTMAX: "softmax",
    Algorithm.IMOSS_TABPFN_COORD_TPE: "best",
    Algorithm.IMOSS_TABPFN_COORD_TPE_SAMPLED: "best",
    Algorithm.IMOSS_TABPFN_COORD_TPE_SOFTMAX: "softmax",
    Algorithm.IMOSS_TABPFN_MIX: "best",
    Algorithm.IMOSS_TABPFN_MIX_NEWARM: "best",
    Algorithm.IMOSS_TABPFN_MUT80: "best",
    Algorithm.IMOSS_TABPFN_MUT80_NEWARM: "best",
    Algorithm.IMOSS_TABPFN_COORD_R1: "best",
    Algorithm.IMOSS_TABPFN_COORD_R3: "best",
    Algorithm.IMOSS_TABPFN_COORD_NOFILTER: "best",
    Algorithm.IMOSS_TABPFN_COORD_R1_NOFILTER: "best",
    Algorithm.IMOSS_TABPFN_COORD_R3_NOFILTER: "best",
    Algorithm.IMOSS_TABPFN_MIX_R1: "best",
    Algorithm.IMOSS_TABPFN_MUT80_R1: "best",
    Algorithm.IMOSS_TABPFN_KGEOM_R1: "best",
    Algorithm.IMOSS_TABPFN_MIXRU_R1: "best",
    Algorithm.IMOSS_TABPFN_R1_Q841: "best",
    Algorithm.IMOSS_TABPFN_R1_Q975: "best",
    Algorithm.IMOSS_TABPFN_R1_Q90: "best",
}


def algo_slug(
    algorithm: Algorithm,
    fit_granularity: str = "arm",
    acquisition: str = "quantile",
    quantile: float = 0.99,
    beta: float = BETA,
    candidate_uniform_frac: float = 0.1,
    candidate_temperature: float = 1.0,
) -> str:
    """Filesystem-safe label, used for per-algorithm result filenames.

    IMOSS-TabPFN's default configuration -- quantile acquisition at the
    0.99 level -- keeps the plain ``imoss_tabpfn`` slug. Non-default variants
    get distinct suffixes -- ``_pull`` (per-pull tables), ``_q<q>`` (quantile
    at a non-default level), ``_ucb_q<q>`` (moment UCB at the
    normality-equivalent kappa), ``_f<frac>``/``_T<temp>`` (a non-default
    mutation-pool uniform share / parent-selection temperature, IMOSS-TabPFN-coord
    only) -- so their result JSONs (and plotted series) live alongside, and never
    clobber, the default's. The surrogate-free arms are unaffected by these
    options, so they keep their usual slugs and are reused across all variants
    and both foundation models.

    ``beta`` (the IMOSS explore/exploit switching exponent) applies to every
    IMOSS arm, so a non-default value suffixes the slug of *any* algorithm
    (``_beta0.8``): runs at different betas are different configurations and
    must never share a checkpoint file.
    """
    slug = algorithm.value.lower().replace(" ", "_").replace("-", "_")
    if algorithm == Algorithm.IMOSS_TABPFN and fit_granularity == "pull":
        slug += "_pull"
    if algorithm in TABPFN_ALGORITHMS:
        if acquisition == "ucb":
            slug += f"_ucb_q{quantile:g}"
        elif quantile != 0.99:
            slug += f"_q{quantile:g}"
    if algorithm in POOL_VARIANT_ALGORITHMS:
        if candidate_uniform_frac != 0.1:
            slug += f"_f{candidate_uniform_frac:g}"
        if candidate_temperature != 1.0:
            slug += f"_T{candidate_temperature:g}"
    if beta != BETA and algorithm not in BETA_FREE_ALGORITHMS:
        slug += f"_beta{beta:g}"
    return slug


def algo_label(algorithm: Algorithm, beta: float = BETA) -> str:
    """Display label for a series: the algorithm name, plus a ``-beta<b>`` tag
    when it was run at a non-default switching exponent.

    Matches the slug's ``_beta<b>`` suffix, so the plotting code's slug ->
    label mapping stays a mechanical rename (see :func:`make_plots`).
    """
    if beta == BETA or algorithm in BETA_FREE_ALGORITHMS:
        return algorithm.value
    return f"{algorithm.value}-beta{beta:g}"


def benchmark_tag(bm_id: int, noise: bool) -> str:
    """Filename prefix -- keeps different benchmarks (bm_id) and the noiseless
    ablation's files from ever colliding with (or overwriting) each other."""
    return f"rf{bm_id}" if noise else f"rf{bm_id}noiseless"


def build_optimizer(
    algorithm: Algorithm,
    search_space: dict[str, Any],
    seed: int,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    beta: float = BETA,
):
    if algorithm == Algorithm.IMOSS_TPE:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            beta=beta,
        )
    elif algorithm == Algorithm.IMOSS_TPE_EPS:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            beta=beta,
            eps_greedy=EPS_GREEDY,
        )
    elif algorithm == Algorithm.IMOSS_TPE_UNI:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=False,
            beta=beta,
        )
    elif algorithm == Algorithm.IMOSS:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            use_tpe=False,
            beta=beta,
        )
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=search_space, seed=seed)
    elif algorithm == Algorithm.IMOSS_TABFM:
        model = tabfm_model if tabfm_model is not None else load_tabfm()
        return IMABOTabFM(
            search_space=search_space,
            seed=seed,
            tabfm_model=model,
            beta=beta,
        )
    elif algorithm in TABPFN_ALGORITHMS:
        model = tabpfn_model if tabpfn_model is not None else load_tabpfn()
        return IMABOTabPFN(
            search_space=search_space,
            seed=seed,
            tabpfn_model=model,
            beta=beta,
            candidate_source=CANDIDATE_SOURCES[algorithm],
            parent_rule=TABPFN_PARENT_RULES.get(algorithm, "softmax"),
            filter_open_candidates=algorithm not in TABPFN_NO_FILTER,
            candidate_tpe_frac=TABPFN_TPE_FRACS.get(algorithm, 0.0),
            refit_every=TABPFN_REFIT_EVERY.get(algorithm, 10),
            mutation_size=TABPFN_MUTATION_SIZES.get(algorithm, 1),
            candidate_topup=TABPFN_TOPUP.get(algorithm, "uniform"),
            tpe_value_pick=(
                "sample" if algorithm in TPE_SAMPLED_ALGORITHMS else "ei_argmax"
            ),
        )
    elif algorithm in COORD_UCB_RULES:
        parent_rule, value_rule = COORD_UCB_RULES[algorithm]
        return IMABOCoordUCB(
            search_space=search_space,
            seed=seed,
            beta=beta,
            parent_rule=parent_rule,
            value_rule=value_rule,
            credit_rule=CREDIT_RULES.get(algorithm, "first_pull"),
            require_new_arm=algorithm in CONTRIB_ALGORITHMS + NEW_ARM_ALGORITHMS,
            coord_rule="random" if algorithm in RANDOM_COORD_ALGORITHMS else "ucb",
            mutation_size=MUTATION_SIZES.get(algorithm, 1),
            bandit_bonus=(
                "kl"
                if algorithm in KL_BANDIT_ALGORITHMS
                else "kl_scaled"
                if algorithm in KL_SCALED_ALGORITHMS
                else "hoeffding"
            ),
            global_tpe_arm=algorithm in GLOBAL_TPE_ALGORITHMS,
            tpe_value_pick=(
                "sample"
                if algorithm is Algorithm.IMOSS_MUTATE_KLXTPE_SAMPLED
                else "ei_argmax"
            ),
            coord_alpha=(
                CONTRIB_COORD_ALPHA if algorithm in CONTRIB_ALGORITHMS else 1.0
            ),
            bandit_rule="exp3" if algorithm in EXP3_ALGORITHMS else "ucb",
            bandit_discount=BANDIT_DISCOUNTS.get(algorithm, 1.0),
            exp3_mixing=EXP3_ALGORITHMS.get(algorithm, 0.0),
            exp3_feedback=(
                "reward"
                if algorithm is Algorithm.IMOSS_MUTATE_EXP3RXTPE
                else "loss"
            ),
            coord_bandit_scope=(
                "parent"
                if algorithm in PER_PARENT_ALGORITHMS
                else "global"
            ),
        )
    elif algorithm == Algorithm.HIER_TPE:
        return HierTPE(search_space=search_space, seed=seed)
    elif algorithm == Algorithm.UCB_AIR:
        # NOTE: UCB-AIR's `beta` is its reservoir tail exponent (see
        # baselines/ucb_air.py), NOT the IMOSS switching exponent, so it stays
        # pinned at the value the paper runs it with rather than tracking the
        # `beta` argument above.
        return UCBAIR(
            search_space=search_space,
            seed=seed,
            beta=BETA,
        )


# The oracle-proposal shadow probe (see _oracle_propose) always calls the real
# oracle (bypassing the cheap MOSS exploit branch), which is expensive for
# IMOSS-TabFM -- one TabFM fit+predict per probed iteration, no cross-
# iteration caching possible since N_SHADOW already equals TabFM's own
# refit_every. Probing every iteration costs ~10h/run; we only need this
# signal at a resolution that supports the cumulative-regret figure, not a
# per-iteration trace, so it's sampled every ORACLE_PROBE_EVERY iterations.
ORACLE_PROBE_EVERY = 100


def _oracle_propose(shadow: Any) -> Any:
    """The oracle's own raw proposal, decoupled from the optimizer's current
    explore/exploit phase.

    `IMABO.suggest()` (imabo/optimizer.py) only consults the oracle --
    uniform random for plain IMOSS, the TPE Parzen estimator for IMOSS-TPE, the
    TabFM surrogate for IMOSS-TabFM -- while in its "explore" phase; once
    `len(arms) >= t**beta` it switches to `suggest_existing`, a MOSS/UCB index
    lookup over already-pulled arms (imabo/moss.py). That mixture is what the
    *algorithm* suggests, not what the oracle itself would propose. Calling
    the oracle path directly here, every iteration regardless of phase, is
    what isolates "the distribution of the oracle every time it suggests an
    arm" from the exploit-driven convergence measured previously.

    Bypassing suggest() also means memory.pull_arm() -- and therefore
    step_counter/nb_pending -- is never touched (imabo/optimizer.py:163-164),
    so this is even safer against state leakage than the previous
    shadow.suggest() probe.
    """
    if not getattr(shadow, "use_tpe", False):
        return shadow.generate_random_config()
    state = shadow.memory.get_current_state()
    rewarded_arms = shadow.get_rewarded_arms(state)
    nb_pending_total = sum(s.nb_pending for s in state.arms.values())
    nb_rewarded_total = sum(s.nb_rewarded for s in state.arms.values())
    return shadow.suggest_new(state, rewarded_arms, nb_pending_total, nb_rewarded_total)


def _shadow_copy(opt: Any) -> Any:
    """Deep-copy an optimizer's mutable state for a disposable "what would it
    suggest right now" probe, without deep-copying (or corrupting) heavy
    read-only attributes shared across instances.

    IMABOTabFM's `_tabfm_model` is a single frozen pretrained model reused
    across every run/budget (see run_experiment below); deep-copying its
    weights on every one of thousands of iterations would be
    both unnecessary (it's never mutated) and far too slow. Pre-seeding the
    deepcopy memo with its id makes copy.deepcopy skip it and reuse the same
    reference in the copy instead. IMABOTabPFN's `_tabpfn_model` (a small shared
    settings dict) is skipped the same way, for parity.
    """
    memo = {}
    for attr in ("_tabfm_model", "_tabpfn_model"):
        model = getattr(opt, attr, None)
        if model is not None:
            memo[id(model)] = model
    return copy.deepcopy(opt, memo)


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    n_shadow: int = N_SHADOW,
    oracle_probe_every: int = ORACLE_PROBE_EVERY,
    fit_granularity: str = "arm",
    max_num_rows: int | None = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
    beta: float = BETA,
    candidate_uniform_frac: float = 0.1,
    candidate_temperature: float = 1.0,
) -> dict:
    """Run one seed, recording every `oracle_probe_every` iterations the
    mean/std of the true reward of `n_shadow` independent draws from the
    oracle's raw proposal at that state (see _shadow_copy, _oracle_propose),
    alongside the real-trajectory fields (regrets, simple regret, suggestion
    counts).

    The surrogate diagnostics (the ``tabfm_*`` fields) fire for either
    foundation-model oracle -- IMOSS-TabFM or IMOSS-TabPFN -- and keep the
    ``tabfm_*`` names verbatim so the plotting code reads both unchanged.
    ``fit_granularity`` / ``max_num_rows`` / ``acquisition`` / ``quantile``
    only affect IMOSS-TabPFN (see :class:`imabo.tabpfn_optimizer.IMABOTabPFN`):
    ``"arm"`` (default) fits one row per arm at its mean reward, ``"pull"``
    one row per individual pull (no averaging, KV-cache on); ``acquisition``
    is the ``quantile`` level of the predictive distribution (``"quantile"``,
    default) or the moment UCB at the normality-equivalent kappa (``"ucb"``)
    -- ``quantile`` is the single exploration knob for both.
    ``candidate_uniform_frac`` / ``candidate_temperature`` only affect
    IMOSS-TabPFN-coord (its mutation candidate pool: the uniform share, and the
    parent-selection temperature in units of the population's score
    dispersion). ``beta`` is the IMOSS switching exponent, shared by every
    IMOSS arm.
    """
    if algorithm == Algorithm.IMOSS_TABFM:
        opt = IMABOTabFM(
            search_space=bench.get_search_space(),
            seed=seed,
            tabfm_model=tabfm_model,
            beta=beta,
            suggest_method="max",
            n_estimators=4,
        )
    elif algorithm in TABPFN_ALGORITHMS:
        # Per-pull tables grow to O(#pulls) rows, so lift the default in-context
        # row cap well above the per-arm default (200) unless told otherwise.
        if max_num_rows is None:
            effective_max_rows = 10000 if fit_granularity == "pull" else 200
        else:
            effective_max_rows = max_num_rows
        opt = IMABOTabPFN(
            search_space=bench.get_search_space(),
            seed=seed,
            tabpfn_model=tabpfn_model,
            beta=beta,
            n_estimators=4,
            fit_granularity=fit_granularity,
            max_num_rows=effective_max_rows,
            acquisition=acquisition,
            quantile=TABPFN_QUANTILES.get(algorithm, quantile),
            candidate_source=CANDIDATE_SOURCES[algorithm],
            candidate_uniform_frac=candidate_uniform_frac,
            candidate_temperature=candidate_temperature,
            parent_rule=TABPFN_PARENT_RULES.get(algorithm, "softmax"),
            filter_open_candidates=algorithm not in TABPFN_NO_FILTER,
            candidate_tpe_frac=TABPFN_TPE_FRACS.get(algorithm, 0.0),
            refit_every=TABPFN_REFIT_EVERY.get(algorithm, 10),
            mutation_size=TABPFN_MUTATION_SIZES.get(algorithm, 1),
            candidate_topup=TABPFN_TOPUP.get(algorithm, "uniform"),
            tpe_value_pick=(
                "sample" if algorithm in TPE_SAMPLED_ALGORITHMS else "ei_argmax"
            ),
        )
    elif algorithm in COORD_UCB_RULES:
        parent_rule, value_rule = COORD_UCB_RULES[algorithm]
        opt = IMABOCoordUCB(
            search_space=bench.get_search_space(),
            seed=seed,
            beta=beta,
            temperature=candidate_temperature,
            parent_rule=parent_rule,
            value_rule=value_rule,
            credit_rule=CREDIT_RULES.get(algorithm, "first_pull"),
            require_new_arm=algorithm in CONTRIB_ALGORITHMS + NEW_ARM_ALGORITHMS,
            coord_rule="random" if algorithm in RANDOM_COORD_ALGORITHMS else "ucb",
            mutation_size=MUTATION_SIZES.get(algorithm, 1),
            bandit_bonus=(
                "kl"
                if algorithm in KL_BANDIT_ALGORITHMS
                else "kl_scaled"
                if algorithm in KL_SCALED_ALGORITHMS
                else "hoeffding"
            ),
            global_tpe_arm=algorithm in GLOBAL_TPE_ALGORITHMS,
            tpe_value_pick=(
                "sample"
                if algorithm is Algorithm.IMOSS_MUTATE_KLXTPE_SAMPLED
                else "ei_argmax"
            ),
            coord_alpha=(
                CONTRIB_COORD_ALPHA if algorithm in CONTRIB_ALGORITHMS else 1.0
            ),
            bandit_rule="exp3" if algorithm in EXP3_ALGORITHMS else "ucb",
            bandit_discount=BANDIT_DISCOUNTS.get(algorithm, 1.0),
            exp3_mixing=EXP3_ALGORITHMS.get(algorithm, 0.0),
            exp3_feedback=(
                "reward"
                if algorithm is Algorithm.IMOSS_MUTATE_EXP3RXTPE
                else "loss"
            ),
            coord_bandit_scope=(
                "parent"
                if algorithm in PER_PARENT_ALGORITHMS
                else "global"
            ),
        )
    else:
        opt = build_optimizer(
            algorithm,
            bench.get_search_space(),
            seed,
            tabfm_model,
            tabpfn_model,
            beta=beta,
        )
    param_names = sorted(bench.get_search_space().keys())

    # Both foundation-model oracles log the surrogate diagnostics below (under
    # the shared ``tabfm_*`` field names).
    is_surrogate = algorithm == Algorithm.IMOSS_TABFM or algorithm in TABPFN_ALGORITHMS

    # UCB-AIR (experiments/baselines/ucb_air.py) has no oracle/exploit split --
    # it's kept in this experiment for the shared cumulative-regret comparison
    # only, so it's exempt from the oracle-proposal shadow probe below.
    has_oracle = hasattr(opt, "generate_random_config")

    regrets = []
    simple_regret_trace = []
    shadow_probe_iterations = []
    shadow_reward_mean = []
    shadow_reward_std = []
    tabfm_suggestion_probe_iterations = []
    tabfm_suggestion_predicted_rewards = []
    tabfm_suggestion_predicted_max_rewards = []
    tabfm_suggestion_true_rewards = []
    tabfm_train_rewards = []
    tabfm_candidate_probe_iterations = []
    tabfm_candidate_configs = []
    tabfm_candidate_predicted_rewards = []
    tabfm_candidate_true_rewards = []
    suggestion_counts: Counter = Counter()
    for i in tqdm(range(n_iterations), desc=algorithm.value, leave=False):
        if has_oracle and n_shadow > 0 and i % oracle_probe_every == 0:
            # _oracle_propose() never calls memory.pull_arm()
            shadow = _shadow_copy(opt)

            # Piggyback on the same N_SHADOW draws to also get TabFM's own
            # predicted value for each one (see IMABOTabFM.on_suggestion) --
            # collecting onto `shadow` only, never `opt`, so the real
            # trajectory stays uninstrumented. `on_suggestion` doesn't fire
            # when suggest_new falls back to a random config (not enough
            # rewarded arms yet), which is all-or-nothing across these 10
            # draws since none of them mutate `shadow`'s state.
            probe_preds: list[tuple[Any, float, float]] = []
            if hasattr(shadow, "on_suggestion"):
                shadow.on_suggestion = (
                    lambda config, mean_pred, max_pred: probe_preds.append(
                        (config, mean_pred, max_pred)
                    )
                )

            # The full candidate pool (all n_candidates), captured on the one
            # real TabFM fit among these draws -- for a pool-wide MSE (TabFM's
            # accuracy across the candidate space), not just at its picks.
            probe_pool: list[tuple[Any, float]] = []
            if hasattr(shadow, "on_candidates_scored"):
                shadow.on_candidates_scored = lambda cands, preds: probe_pool.extend(
                    zip(cands, preds)
                )

            shadow_configs = [_oracle_propose(shadow) for _ in range(n_shadow)]
            shadow_rewards = [bench.mean_reward(c) for c in shadow_configs]
            shadow_probe_iterations.append(i)
            shadow_reward_mean.append(float(np.mean(shadow_rewards)))
            shadow_reward_std.append(float(np.std(shadow_rewards)))

            if probe_preds:
                # Raw predicted/true reward pairs, not a pre-computed metric:
                # lets any per-draw metric (squared error, signed bias,
                # anything dreamed up later) be derived downstream from
                # these two lists without rerunning the experiment.
                tabfm_suggestion_probe_iterations.append(i)
                tabfm_suggestion_predicted_rewards.append(
                    [mean_pred for _, mean_pred, _ in probe_preds]
                )
                tabfm_suggestion_predicted_max_rewards.append(
                    [max_pred for _, _, max_pred in probe_preds]
                )
                tabfm_suggestion_true_rewards.append(
                    [bench.mean_reward(config) for config, _, _ in probe_preds]
                )

                # The reward labels TabFM actually fit on at this probe (one
                # per distinct rewarded arm, the exact set _fit_surrogate
                # uses -- see _oracle_propose). Logged raw and aligned 1:1
                # with the suggestion probes above, to test whether the
                # predicted-reward collapse tracks a downward drift in this
                # training-label distribution as random exploration keeps
                # adding mostly-mediocre arms.
                shadow_state = shadow.memory.get_current_state()
                shadow_rewarded_arms = shadow.get_rewarded_arms(shadow_state)
                tabfm_train_rewards.append(
                    [float(stats.mean_reward) for _, stats in shadow_rewarded_arms]
                )

            if probe_pool:
                # Config + predicted (reward units) + true reward for every
                # candidate in the scored pool -- logged raw so a pool-wide
                # MSE, or a per-(depth,features)-cell MSE map of where TabFM's
                # surrogate is inaccurate, is derivable downstream without
                # rerunning. Configs stored as key lists (param_names order).
                tabfm_candidate_probe_iterations.append(i)
                tabfm_candidate_configs.append(
                    [[config[p] for p in param_names] for config, _ in probe_pool]
                )
                tabfm_candidate_predicted_rewards.append(
                    [float(pred) for _, pred in probe_pool]
                )
                tabfm_candidate_true_rewards.append(
                    [bench.mean_reward(config) for config, _ in probe_pool]
                )

        x = opt.suggest()
        y = bench(x, noise=True)
        opt.observe(y)
        regrets.append(bench.regret(x))
        suggestion_counts[config_to_key(x, param_names)] += 1
        incumbent = opt.best_config
        simple_regret_trace.append(
            bench.regret(incumbent) if incumbent is not None else bench.max_value
        )

    best = opt.best_config
    simple_regret = bench.regret(best) if best is not None else bench.max_value
    best_reward = bench.mean_reward(best) if best is not None else None

    best_key = config_to_key(best, param_names) if best is not None else None
    most_suggested_key, most_suggested_count = (
        suggestion_counts.most_common(1)[0] if suggestion_counts else (None, 0)
    )
    return {
        "regrets": regrets,
        "simple_regret_trace": simple_regret_trace,
        "simple_regrets": simple_regret,
        "shadow_probe_iterations": shadow_probe_iterations if has_oracle else None,
        "shadow_reward_mean": shadow_reward_mean if has_oracle else None,
        "shadow_reward_std": shadow_reward_std if has_oracle else None,
        "tabfm_suggestion_probe_iterations": (
            tabfm_suggestion_probe_iterations if is_surrogate else None
        ),
        "tabfm_suggestion_predicted_rewards": (
            tabfm_suggestion_predicted_rewards if is_surrogate else None
        ),
        "tabfm_suggestion_predicted_max_rewards": (
            tabfm_suggestion_predicted_max_rewards if is_surrogate else None
        ),
        "tabfm_suggestion_true_rewards": (
            tabfm_suggestion_true_rewards if is_surrogate else None
        ),
        "tabfm_train_rewards": (tabfm_train_rewards if is_surrogate else None),
        "tabfm_candidate_probe_iterations": (
            tabfm_candidate_probe_iterations if is_surrogate else None
        ),
        "tabfm_candidate_configs": (
            tabfm_candidate_configs if is_surrogate else None
        ),
        "tabfm_candidate_predicted_rewards": (
            tabfm_candidate_predicted_rewards if is_surrogate else None
        ),
        "tabfm_candidate_true_rewards": (
            tabfm_candidate_true_rewards if is_surrogate else None
        ),
        "best_config": best,
        "best_reward": best_reward,
        "best_config_suggestions": (
            suggestion_counts[best_key] if best_key is not None else 0
        ),
        "most_suggested_count": most_suggested_count,
        "is_best_most_suggested": best_key is not None
        and best_key == most_suggested_key,
        # Full real-trajectory pull distribution over the search space (every
        # opt.suggest(), explore + exploit), keyed by config. Tuples aren't
        # JSON keys, so stored as [config_key_list, count] pairs. The 2D
        # pulls-overlay plot reconstructs (param -> value) via param_names.
        "suggestion_counts": [
            [list(key), count] for key, count in suggestion_counts.items()
        ],
        "param_names": param_names,
    }


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 10,
    base_seed: int = 42,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    n_jobs: int = N_JOBS,
    fit_granularity: str = "arm",
    max_num_rows: int | None = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
    beta: float = BETA,
    candidate_uniform_frac: float = 0.1,
    candidate_temperature: float = 1.0,
    n_shadow: int = N_SHADOW,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm, checkpointed per
    run."""
    slug = algo_slug(
        algorithm,
        fit_granularity,
        acquisition,
        quantile,
        beta,
        candidate_uniform_frac,
        candidate_temperature,
    )
    stem = f"{benchmark_tag(bench.bm_id, True)}_{slug}_{n_iterations}iters"

    all_results: list[dict | None] = [None] * n_runs
    pending = []
    for i in range(n_runs):
        run_path = RESULT_DIR / f"{stem}_run{i}.json"
        if run_path.exists():
            with open(run_path) as f:
                all_results[i] = json.load(f)
            tqdm.write(f"--- {stem}_run{i} already complete, skipping ---")
        else:
            pending.append(i)

    def _one_run(i: int) -> dict:
        seed = base_seed + i
        local_bench = copy.copy(bench)
        local_bench.reset_noise(seed)
        result = run_single_experiment(
            local_bench,
            n_iterations,
            algorithm,
            seed=seed,
            tabfm_model=tabfm_model,
            tabpfn_model=tabpfn_model,
            fit_granularity=fit_granularity,
            max_num_rows=max_num_rows,
            acquisition=acquisition,
            quantile=TABPFN_QUANTILES.get(algorithm, quantile),
            beta=beta,
            candidate_uniform_frac=candidate_uniform_frac,
            candidate_temperature=candidate_temperature,
            n_shadow=n_shadow,
        )
        with open(RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
            json.dump(result, f)
        return result

    if pending:
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
            delayed(_one_run)(i) for i in pending
        )
        for i, result in zip(pending, results):
            all_results[i] = result

    return all_results


def run_experiment(
    bench,
    n_runs,
    base_seed,
    n_iter,
    algorithm: Algorithm,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    n_jobs: int = N_JOBS,
    fit_granularity: str = "arm",
    max_num_rows: int | None = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
    beta: float = BETA,
    candidate_uniform_frac: float = 0.1,
    candidate_temperature: float = 1.0,
    n_shadow: int = N_SHADOW,
) -> None:
    # Load/warm up the surrogate model once (both loaders are memoized, so
    # passing a preloaded model in from the caller avoids re-warming per bench).
    if algorithm == Algorithm.IMOSS_TABFM and tabfm_model is None:
        tabfm_model = load_tabfm()
        print("Loaded TabFM model (once, reused across all runs/budgets).")
    if algorithm in TABPFN_ALGORITHMS and tabpfn_model is None:
        tabpfn_model = load_tabpfn()
        _silence_known_warnings()  # importing tabpfn/sklearn can reset filters
        print("TabPFN-3 ready (checkpoint cached; reused across all runs/budgets).")

    gran = ""
    if algorithm in TABPFN_ALGORITHMS:
        acq = (
            f"quantile q={quantile:g}"
            if acquisition == "quantile"
            else f"ucb q={quantile:g}"
        )
        gran = f" [{fit_granularity}, {acq}]"
    if algorithm in POOL_VARIANT_ALGORITHMS:
        if algorithm in CANDIDATE_SOURCES:
            how = (
                f"{CANDIDATE_SOURCES[algorithm]} pool: {candidate_uniform_frac:g} uniform"
                f", parent={TABPFN_PARENT_RULES.get(algorithm, '-')}"
            )
        else:
            parent_rule, value_rule = COORD_UCB_RULES[algorithm]
            coord = "random" if algorithm in RANDOM_COORD_ALGORITHMS else "UCB1"
            how = (
                f"parent={parent_rule}, {coord} coordinate, {value_rule} value, "
                f"credit={CREDIT_RULES.get(algorithm, 'first_pull')}"
                + (
                    f", coord_alpha={CONTRIB_COORD_ALPHA:g}, new arm required"
                    if algorithm in CONTRIB_ALGORITHMS
                    else ""
                )
            )
        gran = gran.rstrip("]") + (
            f"{',' if gran else ' ['} {how}, T={candidate_temperature:g}*sd]"
        )
    print(f"\n{algorithm.value}{gran} (beta={beta:g}): T={n_iter}, {n_runs} runs...")
    run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        tabfm_model=tabfm_model,
        tabpfn_model=tabpfn_model,
        n_jobs=n_jobs,
        fit_granularity=fit_granularity,
        max_num_rows=max_num_rows,
        acquisition=acquisition,
        quantile=quantile,
        beta=beta,
        candidate_uniform_frac=candidate_uniform_frac,
        candidate_temperature=candidate_temperature,
        n_shadow=n_shadow,
    )


# The IMOSS proposal-oracle family for the oracle-distribution shadow probe,
# plus UCB-AIR kept only for the shared cumulative-regret grid
# (run_single_experiment skips the shadow probe for it -- see has_oracle). Both
# foundation-model oracles are included so a single run reproduces both figures.
_DEFAULT_ALGORITHMS = [
    Algorithm.IMOSS,
    Algorithm.IMOSS_TPE,
    Algorithm.IMOSS_TABFM,
    Algorithm.IMOSS_TABPFN,
    Algorithm.IMOSS_TABPFN_COORD,
    Algorithm.UCB_AIR,
]


def _register_series(slug: str, label: str, after: str | None = None) -> None:
    """Teach the plotting code about a variant run's series.

    Every loader in the plot module globs result files and maps the filename's
    slug through ``_PRETTY_LABELS``, then filters/orders the result by
    ``_CANONICAL_ORDER`` (anything absent is silently dropped, so an
    unregistered variant would just not appear). This registers both, inserting
    the label right after ``after`` so the legend keeps a sensible order
    (e.g. a beta-tagged series next to its default-beta sibling).
    """
    from experiments.utils.plots.plot_configs import _CANONICAL_ORDER, _PRETTY_LABELS

    _PRETTY_LABELS[slug] = label
    if label not in _CANONICAL_ORDER:
        idx = (
            _CANONICAL_ORDER.index(after) + 1
            if after in _CANONICAL_ORDER
            else len(_CANONICAL_ORDER)
        )
        _CANONICAL_ORDER.insert(idx, label)


def make_plots(
    benchmarks,
    n_iterations,
    foundation: str = "tabpfn",
    fit_granularity: str = "arm",
    save_fig: bool = True,
    acquisition: str = "quantile",
    quantile: float = 0.99,
    beta: float = BETA,
    coord_betas: list[float] | None = None,
    candidate_uniform_frac: float = 0.1,
    candidate_temperature: float = 1.0,
    series_filter: list[str] | None = None,
    out_tag: str | None = None,
    average: bool = False,
) -> None:
    """Draw the paper's RF figures for one foundation-model oracle: the static
    reward-landscape structure grid (benchmark-only), the combined
    cumulative-regret + oracle-proposal-quality grid, and the surrogate
    suggested-config MSE grid.

    ``foundation`` selects the foundation-model series ("tabfm" or "tabpfn");
    the three surrogate-free baselines are shared by both figures. For "tabpfn"
    with ``fit_granularity="pull"``, ``acquisition="quantile"`` and/or a
    non-default ``quantile``, the variant series and suffixed filenames
    (``..._pull``/``..._q0.975``/``..._ucb_q0.99``) are used, so they sit next
    to (never overwrite) the default per-arm q=0.99 quantile ones.

    For "tabpfn" the figure also carries the mutation-pool oracle
    (IMOSS-TabPFN-coord) next to the uniform-pool one. ``beta`` selects which
    runs are read for every beta-dependent (IMOSS) series: at a non-default
    beta their beta-tagged slugs/labels are used and the filenames get a
    ``_beta<b>`` suffix, so each switching exponent gets its own figure.
    UCB-AIR and Hier-MAB are drawn from their single stored configuration
    either way -- neither takes the IMOSS switching exponent (UCB-AIR's own
    ``beta`` is an unrelated reservoir-tail exponent).

    ``series_filter`` keeps only the named series (display labels, e.g.
    ``["IMOSS-Hier-MAB", "IMOSS-Hier-MAB-beta0.8", "Hier-MAB"]``) in the
    regret/oracle grid, and ``out_tag`` names its file
    (``..._regret_and_oracle_grid_tabpfn_<out_tag>``). Everything is still
    registered first, so any subset of the run configurations can be drawn
    together; use it when the full comparison has more series than a panel can
    carry legibly. The landscape and surrogate-MSE figures are skipped when a
    filter is given -- they are not about this comparison.

    ``coord_betas`` overrides that for the mutation-pool oracle alone: one
    IMOSS-TabPFN-coord series per listed switching exponent, drawn in the *same*
    panels (e.g. ``[0.5, 0.8]`` -- explore-light vs explore-heavy) while every
    other series stays at ``beta``. Extra betas are tagged in the legend and
    appended to the output filename (``_coordbeta0.8``), so the single-beta
    figure is never overwritten. Defaults to ``[beta]``.
    """
    coord_betas = [beta] if coord_betas is None else list(coord_betas)

    # Head-less: make the plotting helpers' trailing ``plt.show()`` a no-op so
    # every PDF is written without a GUI (an interactive backend would block).
    import matplotlib

    matplotlib.use("Agg")
    from experiments.utils.plots.rf_arm_distribution_plot import (
        _load_suggestion_mse_traces,
        _plot_suggestion_metric_grid,
        plot_regret_and_oracle_grid,
    )

    # `plot_landscape_structure_grid` was split in two (heatmap grid + reward
    # CDF) by commit 32344ca without updating this caller, so this import used to
    # raise ImportError and make_plots could not run at all.
    from experiments.utils.plots.rf_landscape_plot import (
        plot_landscape_heatmap_grid,
        plot_landscape_reward_cdf,
    )

    is_tabpfn = foundation == "tabpfn"
    is_pull = is_tabpfn and fit_granularity == "pull"
    fm_label = (
        "IMOSS-TabPFN-pull"
        if is_pull
        else "IMOSS-TabPFN"
        if is_tabpfn
        else "IMOSS-TabFM"
    )
    suffix = "_pull" if is_pull else ""
    if is_tabpfn and acquisition == "ucb":
        variant = f"ucb_q{quantile:g}"
    elif is_tabpfn and acquisition == "quantile" and quantile != 0.99:
        variant = f"q{quantile:g}"
    else:
        variant = None
    if variant is not None:
        fm_label += f"-{variant}"
        suffix += f"_{variant}"
        # Register the variant's slug -> display label so every loader that
        # globs result files (they map slugs through _PRETTY_LABELS) picks the
        # variant series up under a readable name.
        _register_series(
            algo_slug(Algorithm.IMOSS_TABPFN, fit_granularity, acquisition, quantile),
            fm_label,
            after="IMOSS-TabPFN",
        )

    # Beta-tagged series: at a non-default switching exponent every IMOSS arm is
    # a different configuration, stored under (and plotted as) its own
    # `_beta<b>`-suffixed slug/label -- see algo_slug/algo_label.
    if beta != BETA:
        base_algo = Algorithm.IMOSS_TABPFN if is_tabpfn else Algorithm.IMOSS_TABFM
        beta_tag = f"-beta{beta:g}"
        fm_label += beta_tag
        suffix += f"_beta{beta:g}"
        _register_series(
            algo_slug(base_algo, fit_granularity, acquisition, quantile, beta),
            fm_label,
            after=base_algo.value,
        )
    else:
        beta_tag = ""

    # Every proposal-pool variant, once per requested switching exponent. Several
    # betas of one variant land in the same panel, so the extra ones need their
    # own style: all betas of a variant normalize to its single canonical identity
    # (the shared style table is keyed by method, not by run config), which would
    # otherwise draw them as one indistinguishable line. Convention here: colour
    # = variant, marker + dashed line = the non-default beta -- what this repo
    # already does for same-method variants in one panel (Hier-MAB grid sizes,
    # IMOSS-TPE-univ).
    from experiments.utils.plots.plot_configs import algorithm_style
    from experiments.utils.plots.rf_arm_distribution_plot import (
        _SERIES_LINESTYLE,
        _SERIES_STYLE,
    )

    _BETA_MARKERS = {
        Algorithm.IMOSS_TABPFN_COORD: "P",
        Algorithm.IMOSS_TABPFN_COORD_SOFTMAX: "1",
        Algorithm.IMOSS_TABPFN_COORD_TPE_SOFTMAX: "d",
        Algorithm.IMOSS_TABPFN_TPE: ">",
        Algorithm.IMOSS_COORD_UCB_TPE: "8",
        Algorithm.IMOSS_COORD_UCB_TPE_SOFTMAX: "2",
        Algorithm.IMOSS_COORD_UCB_RANDOM: "|",
        Algorithm.IMOSS_HIER_MAB: "*",
        Algorithm.IMOSS_HIER_MAB_SOFTMAX: "p",
        Algorithm.IMOSS_HIER_MAB_MEAN: "3",
        Algorithm.IMOSS_HIER_MAB_IMPROVE: "4",
        Algorithm.IMOSS_HIER_MAB_CONTRIB: "+",
        Algorithm.IMOSS_COORD_UCB_TPE_CONTRIB: "x",
        Algorithm.IMOSS_COORD_UCB_TPE_LASTPROP: "<",
        Algorithm.IMOSS_COORD_UCB_TPE_MOSS: ">",
        Algorithm.IMOSS_HIER_MAB_LASTPROP: "^",
        Algorithm.IMOSS_HIER_MAB_MOSS: "v",
    }

    variant_labels = []
    for algorithm in POOL_VARIANT_ALGORITHMS if is_tabpfn and not is_pull else []:
        betas = coord_betas if algorithm in EXTRA_BETA_ALGORITHMS else [beta]
        for pool_beta in betas:
            label = algo_label(algorithm, pool_beta)
            if variant is not None and algorithm in TABPFN_ALGORITHMS:
                tag = f"-beta{pool_beta:g}" if pool_beta != BETA else ""
                label = f"{algorithm.value}-{variant}{tag}"
            _register_series(
                algo_slug(
                    algorithm,
                    fit_granularity,
                    acquisition,
                    quantile,
                    pool_beta,
                    candidate_uniform_frac,
                    candidate_temperature,
                ),
                label,
                after=algorithm.value,
            )
            if pool_beta != beta:
                color, _ = algorithm_style(algorithm.value)
                _SERIES_STYLE[label] = (color, _BETA_MARKERS[algorithm])
                _SERIES_LINESTYLE[label] = "--"
            variant_labels.append(label)
    for pool_beta in coord_betas:
        if pool_beta != beta:
            suffix += f"_coordbeta{pool_beta:g}"

    if beta != BETA:
        for algorithm in (Algorithm.IMOSS, Algorithm.IMOSS_TPE, Algorithm.IMOSS_TPE_EPS):
            _register_series(
                algo_slug(algorithm, beta=beta),
                algo_label(algorithm, beta),
                after=algorithm.value,
            )
    imoss_baselines = [
        algo_label(Algorithm.IMOSS, beta),
        algo_label(Algorithm.IMOSS_TPE, beta),
        algo_label(Algorithm.IMOSS_TPE_EPS, beta),
    ]
    surrogates = [fm_label] + variant_labels
    # Hier-MAB's per-run JSONs are produced by factored_baseline_experiment.py
    # (same directory, filename scheme, and seed pairing); it has no proposal
    # oracle, so like UCB-AIR it appears only in the regret row. IMOSS-TPE-univ
    # (multivariate=False) has stored runs but is not part of the paper figure.
    # Neither UCB-AIR nor Hier-MAB takes the switching exponent, so their single
    # stored configuration is reused at every beta (see the docstring).
    regret_algos = imoss_baselines + surrogates + ["UCB-AIR", "Hier-MAB", "Hier-TPE"]
    oracle_algos = imoss_baselines + surrogates

    if series_filter is not None:
        keep = set(series_filter)
        missing = keep - set(regret_algos) - set(oracle_algos)
        if missing:
            raise ValueError(
                f"series_filter names series this figure does not register: "
                f"{sorted(missing)}. Registered: {sorted(set(regret_algos))}"
            )
        regret_algos = [a for a in regret_algos if a in keep]
        oracle_algos = [a for a in oracle_algos if a in keep]
        suffix = f"_{out_tag}" if out_tag else "_filtered"
    else:
        print("Generating RF reward-landscape structure figures (benchmark-only)...")
        bm_ids = tuple(int(tag.removeprefix("rf")) for tag in benchmarks)
        plot_landscape_heatmap_grid(bm_ids=bm_ids, save_fig=save_fig)
        plot_landscape_reward_cdf(bm_ids=bm_ids, save_fig=save_fig)

    print(f"Generating combined regret + oracle-proposal-quality grid ({fm_label})...")
    plot_regret_and_oracle_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        regret_algorithms=regret_algos,
        oracle_algorithms=oracle_algos,
        out_name=f"regret_and_oracle_grid_{foundation}{suffix}{'_avg' if average else ''}",
        average=average,
    )

    if series_filter is not None:
        return

    print(f"Generating {foundation} suggested-config MSE grid...")
    _plot_suggestion_metric_grid(
        _load_suggestion_mse_traces,
        f"{'TabPFN' if is_tabpfn else 'TabFM'} Suggested-Config MSE",
        f"{foundation}_suggestion_mse_grid{suffix}",
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        log_scale=True,
        # Only this figure's foundation-model series: the loader also picks
        # up the acquisition-sweep variant runs, which must not be drawn.
        algorithms=surrogates,
    )


def score_table(
    benchmarks,
    n_iterations: int,
    beta: float = BETA,
    coord_betas: list[float] | None = None,
    save_csv: bool = True,
) -> str:
    """Markdown table of every candidate's score, one row per method.

    Per benchmark: mean +- across-seed standard deviation of the cumulative
    regret at ``n_iterations``, and the same divided by ``n_iterations`` (the
    average regret per pull, comparable across horizons). The final column totals
    the cumulative regret over the benchmarks *per seed* -- seed ``base_seed + i``
    is the same run configuration on every benchmark, so the sum is taken within
    a seed and only over seeds present on all of them, which gives the total a
    genuine across-seed spread instead of one summed from three separate means.
    Rows are ordered by that total; a method missing a benchmark has no total and
    is listed after, ordered by mean rank.

    Reads the per-run JSONs directly, so the seed count is whatever is on disk
    and is reported per row rather than assumed uniform.
    """
    import csv as _csv
    import json as _json
    import re as _re

    from experiments.utils.plots.plot_configs import _ordered
    from experiments.utils.plots.rf_arm_distribution_plot import _load_shadow_traces

    coord_betas = [beta] if coord_betas is None else list(coord_betas)
    for algorithm in POOL_VARIANT_ALGORITHMS:
        for pool_beta in coord_betas if algorithm in EXTRA_BETA_ALGORITHMS else [beta]:
            _register_series(
                algo_slug(algorithm, beta=pool_beta),
                algo_label(algorithm, pool_beta),
                after=algorithm.value,
            )
    from experiments.utils.plots.plot_configs import _PRETTY_LABELS

    # label -> benchmark -> {run index: cumulative regret}. Run indices are kept
    # (not positions) so the per-seed totals below line up across benchmarks even
    # when a method has a different number of seeds on each.
    cum: dict[str, dict[str, dict[int, float]]] = {}
    oracle: dict[str, dict[str, float]] = {}
    for bench in benchmarks:
        pattern = _re.compile(rf"{bench}_(.+)_{n_iterations}iters_run(\d+)\.json$")
        for path in sorted(RESULT_DIR.glob(f"{bench}_*_{n_iterations}iters_run*.json")):
            match = pattern.match(path.name)
            if not match:
                continue
            slug, run = match.group(1), int(match.group(2))
            label = _PRETTY_LABELS.get(slug)
            if label is None:
                continue
            with open(path) as f:
                data = _json.load(f)
            cum.setdefault(label, {}).setdefault(bench, {})[run] = float(
                np.sum(data["regrets"])
            )
        shadow, _ = _load_shadow_traces(bench, n_iterations)
        for label, (_, means, _sd) in shadow.items():
            oracle.setdefault(label, {})[bench] = float(np.mean(means[:, -10:]))

    labels = _ordered(cum)

    def totals(label: str) -> np.ndarray | None:
        per_bench = cum[label]
        if len(per_bench) < len(benchmarks):
            return None
        shared = set.intersection(*(set(per_bench[b]) for b in benchmarks))
        if not shared:
            return None
        return np.array(
            [sum(per_bench[b][run] for b in benchmarks) for run in sorted(shared)]
        )

    ranked = {
        b: sorted(
            (lab for lab in labels if b in cum[lab]),
            key=lambda lab: float(np.mean(list(cum[lab][b].values()))),
        )
        for b in benchmarks
    }

    def sort_key(label: str):
        t = totals(label)
        if t is not None:
            return (0, float(t.mean()))
        rs = [ranked[b].index(label) for b in benchmarks if label in ranked[b]]
        return (1, sum(rs) / len(rs) if rs else float("inf"))

    names = [_bench_name(b) for b in benchmarks]
    lines = [
        "| method | n | " + " | ".join(f"{n} avg / cum" for n in names) + " | total cum |",
        "|" + "---|" * (len(names) + 3),
    ]
    for label in sorted(labels, key=sort_key):
        parts, ns = [], []
        for bench in benchmarks:
            runs = cum[label].get(bench)
            if not runs:
                parts.append("-")
                continue
            values = np.array(list(runs.values()))
            ns.append(values.size)
            sd = values.std(ddof=1) if values.size > 1 else 0.0
            parts.append(
                f"{values.mean() / n_iterations:.4f} / {values.mean():.1f}±{sd:.0f}"
            )
        t = totals(label)
        total = (
            f"**{t.mean():.1f}**±{t.std(ddof=1) if t.size > 1 else 0.0:.0f}"
            if t is not None
            else "-"
        )
        n_str = str(min(ns)) if len(set(ns)) == 1 else f"{min(ns)}-{max(ns)}"
        lines.append(f"| {label} | {n_str} | " + " | ".join(parts) + f" | {total} |")

    table = "\n".join(lines)
    if save_csv:
        out = RESULT_DIR / f"scores_{n_iterations}iters.csv"
        with open(out, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(
                ["method", "benchmark", "n_seeds", "cum_regret_mean",
                 "cum_regret_sd", "avg_regret", "oracle_quality_last10"]
            )
            for label in sorted(labels, key=sort_key):
                for bench in benchmarks:
                    runs = cum[label].get(bench)
                    if not runs:
                        continue
                    values = np.array(list(runs.values()))
                    sd = values.std(ddof=1) if values.size > 1 else 0.0
                    q = oracle.get(label, {}).get(bench)
                    w.writerow([
                        label, _bench_name(bench), values.size,
                        f"{values.mean():.4f}", f"{sd:.4f}",
                        f"{values.mean() / n_iterations:.6f}",
                        "" if q is None else f"{q:.4f}",
                    ])
                t = totals(label)
                if t is not None:
                    w.writerow([
                        label, "TOTAL", t.size, f"{t.mean():.4f}",
                        f"{t.std(ddof=1) if t.size > 1 else 0.0:.4f}",
                        f"{t.mean() / (n_iterations * len(benchmarks)):.6f}", "",
                    ])
        print(f"Saved to {out}")
    return table


def _bench_name(tag: str) -> str:
    from experiments.utils.plots.plot_configs import _bench_title

    return _bench_title(tag)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--algorithm",
        default="all",
        choices=["all"] + [a.value for a in Algorithm],
        help="method to run (default: 'all' -- every algorithm x benchmark)",
    )
    p.add_argument("--n-runs", type=int, default=10, help="independent seeds per algorithm")
    p.add_argument("--n-iter", type=int, default=5000, help="iterations (T) per run")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=N_JOBS, help="parallel run workers")
    p.add_argument(
        "--benchmarks",
        type=int,
        nargs="+",
        default=[146822, 31, 167120],
        help="OpenML task ids: segment=146822, credit-g=31, numerai28.6=167120",
    )
    p.add_argument(
        "--fit-granularity",
        choices=["arm", "pull"],
        default="arm",
        help=(
            "IMOSS-TabPFN training-table granularity: 'arm' (default, paper) "
            "fits one row per arm at its mean reward; 'pull' fits one row per "
            "individual pull (no averaging) with TabPFN's KV-cache. 'pull' "
            "results go to distinct '..._pull' files/plots."
        ),
    )
    p.add_argument(
        "--max-num-rows",
        type=int,
        default=None,
        help="TabPFN in-context row cap (default: 200 for arm, 10000 for pull).",
    )
    p.add_argument(
        "--acquisition",
        choices=["ucb", "quantile"],
        default="quantile",
        help=(
            "IMOSS-TabPFN acquisition: 'quantile' (default) ranks candidates "
            "on the --quantile level of TabPFN's predictive distribution "
            "(the GIT-BO quantile form of UCB); 'ucb' ranks on "
            "mean + kappa*std at kappa = Phi^-1(--quantile). 'ucb' results "
            "go to distinct '..._ucb_q<q>' files/plots."
        ),
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help=(
            "IMOSS-TabPFN exploration level in (0,1) (default 0.99, the "
            "best/most seed-stable setting in the acquisition comparison): "
            "the level ranked on by the quantile acquisition, or converted "
            "to the UCB weight via kappa = Phi^-1(quantile) for 'ucb'. "
            "Non-default values go to distinct '..._q<q>' files/plots."
        ),
    )
    p.add_argument(
        "--beta",
        type=float,
        default=BETA,
        help=(
            f"IMOSS explore/exploit switching exponent (default {BETA:g}): explore "
            "while |arms| < t**beta. Applies to every IMOSS arm (Random/TPE/TabFM/"
            "TabPFN); runs at a non-default beta go to distinct "
            "'..._beta<b>' files/plots. UCB-AIR and Hier-MAB are unaffected."
        ),
    )
    p.add_argument(
        "--coord-betas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "plotting only: switching exponents to draw IMOSS-TabPFN-coord at, "
            f"all in the same panels (e.g. '--coord-betas {BETA:g} 0.8'). Every "
            "other series stays at --beta; the extra betas are tagged in the "
            "legend and appended to the filename. Default: just --beta."
        ),
    )
    p.add_argument(
        "--figure-series",
        nargs="+",
        default=None,
        help=(
            "plotting only: draw just these series (display labels, e.g. "
            "'IMOSS-Hier-MAB' 'IMOSS-Hier-MAB-beta0.8' 'Hier-MAB'). Use for a "
            "focused comparison when the full figure has too many series to read; "
            "name the output with --figure-tag."
        ),
    )
    p.add_argument(
        "--table",
        action="store_true",
        help=(
            "print a markdown score table (average and cumulative regret, oracle "
            "proposal quality) for every candidate and write it as CSV, then exit."
        ),
    )
    p.add_argument(
        "--average-regret",
        action="store_true",
        help=(
            "plot the running average regret (cumulative / round) instead of the "
            "cumulative total, which separates near-parallel curves. Written to a "
            "distinct '..._avg' file."
        ),
    )
    p.add_argument(
        "--figure-tag",
        default=None,
        help="plotting only: filename suffix for a --figure-series figure.",
    )
    p.add_argument(
        "--candidate-uniform-frac",
        type=float,
        default=0.1,
        help=(
            "IMOSS-TabPFN-coord only: fraction of the candidate pool still drawn "
            "uniformly at random (default 0.1); the rest are single-coordinate "
            "mutations of population arms. Non-default values go to distinct "
            "'..._f<frac>' files/plots."
        ),
    )
    p.add_argument(
        "--candidate-temperature",
        type=float,
        default=1.0,
        help=(
            "IMOSS-TabPFN-coord only: parent-selection softmax temperature in "
            "units of the population's own reward dispersion, T = temp * "
            "std(mean_rewards) (default 1.0 = one e-fold of selection weight per "
            "standard deviation; -> 0 greedy, large -> uniform). Non-default "
            "values go to distinct '..._T<temp>' files/plots."
        ),
    )
    p.add_argument(
        "--foundation",
        choices=["tabfm", "tabpfn"],
        default=None,
        help=(
            "which foundation-model figure to (re)plot; default: inferred from "
            "--algorithm, or both when --algorithm all."
        ),
    )
    p.add_argument(
        "--plot", action="store_true", help="after running, draw the paper figure(s)"
    )
    p.add_argument(
        "--plot-only", action="store_true", help="skip running, only (re)plot"
    )
    p.add_argument(
        "--n-shadow",
        type=int,
        default=N_SHADOW,
        help=(
            "oracle-quality probe draws per probe point; 0 disables the probe "
            "entirely. The probe calls the real oracle on a DEEP COPY of the "
            "optimizer, so the trajectory (and hence `regrets`) is identical "
            "either way -- but for a surrogate oracle it dominates the run: at "
            "5000 iterations it makes ~50 TabPFN fits against the search's "
            "~10-20. Disable it when only the regret is needed; the shadow "
            "trace it produces feeds the arm-distribution figure alone."
        ),
    )
    p.add_argument("--no-plot", action="store_true", help="run but skip plotting")
    p.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke test: T=60, 2 runs, credit-g only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.table:
        bench_tags = tuple(f"rf{bm_id}" for bm_id in args.benchmarks)
        print(
            score_table(
                bench_tags,
                args.n_iter,
                beta=args.beta,
                coord_betas=args.coord_betas,
            )
        )
        raise SystemExit(0)

    if args.quick:
        n_runs, n_iter, bm_ids = 2, 60, [31]
    else:
        n_runs, n_iter, bm_ids = args.n_runs, args.n_iter, args.benchmarks

    bench_tags = tuple(f"rf{bm_id}" for bm_id in bm_ids)
    algorithms = (
        list(_DEFAULT_ALGORITHMS)
        if args.algorithm == "all"
        else [Algorithm(args.algorithm)]
    )

    if not args.plot_only:
        # Warm up each needed surrogate model once, reused across all benchmarks.
        tabfm_model = load_tabfm() if Algorithm.IMOSS_TABFM in algorithms else None
        if tabfm_model is not None:
            print("Loaded TabFM model (once, reused across all runs/budgets).")
        tabpfn_model = None
        if any(a in TABPFN_ALGORITHMS for a in algorithms):
            tabpfn_model = load_tabpfn()
            _silence_known_warnings()
            print("TabPFN-3 ready (checkpoint cached; reused across all runs/budgets).")

        total_tasks = len(bm_ids) * len(algorithms)
        start = time.time()
        with tqdm(total=total_tasks, desc="benchmark x algorithm", unit="task") as bar:
            for bm_id in bm_ids:
                bench = RFTabularFiniteBenchmark(bm_id=bm_id)
                print(
                    f"\nRF tabular finite benchmark (OpenML task {bm_id}): "
                    f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}"
                )
                for algorithm in algorithms:
                    run_experiment(
                        bench,
                        n_runs,
                        args.base_seed,
                        n_iter,
                        algorithm,
                        tabfm_model=tabfm_model,
                        tabpfn_model=tabpfn_model,
                        n_jobs=args.n_jobs,
                        fit_granularity=args.fit_granularity,
                        max_num_rows=args.max_num_rows,
                        acquisition=args.acquisition,
                        quantile=args.quantile,
                        beta=args.beta,
                        candidate_uniform_frac=args.candidate_uniform_frac,
                        candidate_temperature=args.candidate_temperature,
                        n_shadow=args.n_shadow,
                    )
                    bar.update(1)
                    done, total = bar.n, bar.total
                    elapsed = time.time() - start
                    eta = elapsed / done * (total - done) if done else 0.0
                    bar.set_postfix_str(f"elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m")

    # Plot when asked (--plot/--plot-only) or by default after an "all" run.
    want_plot = not args.no_plot and (
        args.plot or args.plot_only or args.algorithm == "all"
    )
    if want_plot:
        if args.foundation is not None:
            foundations = [args.foundation]
        elif args.algorithm == "all":
            foundations = ["tabfm", "tabpfn"]
        elif args.algorithm == Algorithm.IMOSS_TABFM.value:
            foundations = ["tabfm"]
        elif args.algorithm in (a.value for a in TABPFN_ALGORITHMS):
            foundations = ["tabpfn"]
        else:
            foundations = []  # a surrogate-free run defines no figure on its own
        for foundation in foundations:
            make_plots(
                bench_tags,
                n_iter,
                foundation=foundation,
                fit_granularity=args.fit_granularity,
                save_fig=True,
                acquisition=args.acquisition,
                quantile=args.quantile,
                beta=args.beta,
                coord_betas=args.coord_betas,
                candidate_uniform_frac=args.candidate_uniform_frac,
                candidate_temperature=args.candidate_temperature,
                series_filter=args.figure_series,
                out_tag=args.figure_tag,
                average=args.average_regret,
            )
