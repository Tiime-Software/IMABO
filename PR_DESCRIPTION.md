# Add D-TTTS / Random / UCB-AIR / QRM2 baselines, dim-normalised toy rewards, and O(1) MOSS speedup

## Summary

This PR adds five fair, any-search-space baselines to the toy comparison
(D-TTTS, Random search, UCB-AIR, MOSS-AIR and QRM2), fixes the toy objectives
so their rewards are naturally close to `[0,1]` (matching MOSS's unit-range
calibration) instead of validating/enforcing that range inside IMABO, and
removes an `O(K²)`-per-step bottleneck in the in-memory storage. All existing
tests pass (22 total).

It also includes a small analysis result: IMABO-no-oracle is exactly the
classical infinite-armed MOSS strategy (UCB-AIR with the MOSS index), and its
`beta` knob is the arm-opening exponent — setting `beta=0.5` reproduces the
canonical `K(t)=ceil(t^{1/2})` schedule.

## Motivation

Two correctness issues surfaced while extending the toy comparison:

1. **IMABO's reward-scale assumption was silent.** MOSS's confidence bonus
   `sqrt((1+α)/2 · log(t/(K·n_x)) / n_x)` is an `O(1)` radius calibrated for a
   **roughly unit reward range**. The toy objectives are sums over `dim`
   coordinates (sin1/garland in ~[0, 4], and rastrigin's old raw formula in
   ~[-185, 20]), so feeding raw rewards mis-calibrates exploration — a
   negligible bonus on rastrigin made IMABO near-greedy. We first tried fixing
   this by validating the range inside IMABO (`observe()` raising on
   out-of-range rewards), but that pushed a modelling problem onto a runtime
   check. We instead fix it at the source: every toy function's per-dimension
   term (`sin1_1d`, `garland_1d`, and now `rastrigin_1d`) is rescaled to land
   roughly in `[0,1]`, so dividing the `dim`-sum by `dim` gives a reward close
   to `[0,1]` by construction. This does **not** need to be exact — additive
   noise can still push a reward slightly outside `[0,1]`, and that's fine:
   MOSS's bonus only needs a roughly-unit range, not a hard bound. `imabo/optimizer.py`
   ends up with **no net change** from this PR — no reward-range validation
   was added.
   As a side effect, this also fixes a latent bug: `rastrigin`'s theoretical
   max was hard-coded to `0.0`, which was wrong for its old raw scale (the
   true per-dimension max was ~20); it's now `1.0`, correct for the rescaled
   function.

2. **`get_reward_frequency()` was `O(K)` and called per-arm per-step**, i.e.
   `O(K²)` per step / `O(T·K²)` total. On a T=3000 run this dominated wall time
   (a 20-seed sweep took ~46 min).

## Changes

### `imabo/memory.py` — O(1) reward frequency (speedup)
- `InMemoryStorage` maintains running totals `_total_rewarded` / `_total_pending`,
  updated incrementally in `set`/`pull_arm`/`observe`.
- `get_reward_frequency()` is now `O(1)`. Verified to reproduce the previous
  value exactly (max abs diff `0.0` over a 1500-step run).
- Minor correctness tightening in `observe()`: `nb_pending` is only decremented
  when positive, keeping the pending total exact.

### `experiments/benchmarks/toys/toy_functions.py` — rescale rastrigin, dim-scale noise
- `rastrigin_1d` is now `(10·(cos(2πx) − 1) − x²) / 40 + 1`, landing roughly in
  `[0,1]` per dimension — algebraically the same Rastrigin surface, just
  affinely rescaled, matching what `sin1_1d`/`garland_1d` already did.
  `get_theoretical_max("rastrigin")` is corrected from `0.0` to `1.0` to match.
- `_add_noise` now scales `noise_std` by `dim`, so after a caller divides the
  `dim`-sum reward by `dim`, the effective noise stays `N(0, noise_std)` —
  independent of `dim` — instead of shrinking as `dim` grows.

### `experiments/baselines/dttts.py` — D-TTTS baseline (new)
Faithful implementation of Dynamic Top-Two Thompson Sampling
(Shang, Kaufmann & Valko 2019). Same `suggest()/observe()/best_config`
interface as `IMABO`. Rewards normalised to [0,1] then Agrawal–Goyal
binarized; a single `Beta(t−k, 1)` pseudo-arm collapses all unopened arms.
Provides both recommendation rules: `best_config` = argmax posterior
probability of being optimal (the paper's rule, Monte-Carlo over the Beta
posteriors); `best_config_mean` = argmax posterior mean.

### `experiments/baselines/random_search.py` — Random baseline (new)
Fully any-space random search (uniform / log-uniform / integer / categorical),
recommending the best single observed reward. The honest floor that isolates
what the bandit (MOSS) and oracle (TPE) layers actually buy.

### `experiments/baselines/ucb_air.py` — UCB-AIR + MOSS-AIR baselines (new)
Two infinite-armed baselines in the same generator interface, both using the
**Arm-Increasing Rule** (Wang, Audibert & Munos 2008): keep an active set that
grows as `K(t) = ceil(t^{β/(β+1)})` (saturating at `ceil(√t)` for β≥1), drawing
fresh reservoir arms uniformly as `t` grows; a never-pulled active arm has an
infinite index so it is tried once immediately.
- `UCBAIR` — the paper's variance-aware **UCB-V** index (their eq. 3):
  `Bᵢ = μ̂ᵢ + √(2·Vᵢ·Eₜ/nᵢ) + 3·Eₜ/nᵢ`, where `Vᵢ` is the empirical reward
  variance of arm `i` and `Eₜ = log t` is the exploration sequence. `_Arm` now
  tracks `sumsq` alongside `sum`/`n` to compute `Vᵢ`. The variance term lets a
  near-optimal, low-variance arm's bonus shrink well below plain UCB1's
  `√(2 log t / n)` — the regime the paper calls variance "crucial" for.
- `MOSSAIR` — the *same* schedule with the repo's `moss_anytime` index instead,
  isolating the effect of the index choice from the arm-opening rate.

### `experiments/baselines/qrm2.py` — QRM2 baseline (new)
QRM2 (Roy Chaudhuri & Kalyanakrishnan, *Quantile-Regret Minimisation in
Infinitely Many-Armed Bandits*, UAI 2018, Algorithm 2): a **parameter-free**
doubling wrapper around fixed-horizon MOSS. In phase `r` it sets horizon
`t_r = 2^r`, grows the arm pool to `n_r = ceil(t_r^0.347)` fresh reservoir arms,
and runs `MOSS(K_r, t_r)` with **restarted** within-phase statistics (the pool
accumulates; the MOSS stats and horizon reset each phase). The exponent 0.347
opens far fewer arms than the AIR baselines' effective 0.5 (~t^0.35 vs ~t^0.5) —
it targets *quantile*-regret without knowing the quantile fraction, trading
breadth for depth. Uses the same `moss_anytime` index as MOSS-AIR.

### `experiments/ucbair_compare.py` — UCB-AIR comparison harness (new)
Compares UCB-AIR, MOSS-AIR, IMABO-no-oracle (β=0.8, the shipped I-MOSS), and
IMABO-no-oracle (β=0.5, schedule-matched to MOSS-AIR) on the toy functions.
Every algorithm observes `func(x) / dim` — close to `[0,1]` by construction
thanks to the toy-function rescale above, no offline min-max sampling needed.

### `experiments/imoss_beta_sweep.py` — β-matching study (new)
Sweeps IMABO-no-oracle's `beta` against MOSS-AIR to show they are the same
algorithm once the arm-opening exponent matches.

### `experiments/imoss_beta_tradeoff.py` — β trade-off trajectories (new)
Sweeps `beta ∈ {0.4,…,0.8}` and plots, per β, the **trajectory** through the
(average per-round regret = cum/t, simple regret) plane as t grows — not just
the endpoint — overlaying UCB-AIR and MOSS-AIR (fixed `ceil(√t)` schedule). Run
at T=10000, 12 seeds. `plot()` regenerates the figure from the saved JSON.

### `experiments/dttts_compare.py` — comparison harness (new)
Compares IMABO (±TPE), D-TTTS, and Random on the toy functions, feeding every
algorithm the same `dim`-normalised reward and reporting regret in that same
space. One `--sigma` argument selects the noise model:
- **default (no sigma):** the toy function's own built-in noise
  (dim-invariant ~0.01 after the `/dim` divide).
- **`sigma=<float>`:** explicit Gaussian noise on the dim-normalised reward,
  which can push it slightly outside `[0,1]` — used for the noise-sensitivity
  / Random-vs-IMABO crossover study.

## Impact on existing experiments

**`experiments/toy_experiment.py` is behaviour-changing, but no longer
validation-changing** (nothing raises, since IMABO never gained a hard check):
- It still feeds `func(x)` directly to IMABO/StoSOO/HOO-T/Stroquool without
  dividing by `dim` — that's unchanged and **not fixed in this PR** (still
  flagged for a follow-up so those numbers move onto a `[0,1]`-per-round scale).
- `rastrigin`'s values themselves changed: they're now on the same rescaled,
  bounded-per-dimension surface as `sin1`/`garland` instead of the old raw
  `~[-185, 20]` scale, and `get_theoretical_max("rastrigin")` is corrected
  from the wrong `0.0` to the correct `1.0`. This means `toy_experiment.py`'s
  rastrigin regret numbers change (for the better — they were computed against
  a wrong theoretical max before).
- Built-in noise on all three functions is now scaled by `dim` internally
  (`noise_std * dim` before the sum), so at `dim=4` the raw noise added to the
  observed reward is `N(0, 0.04)` instead of `N(0, 0.01)` — dim-invariant once
  a caller divides by `dim`, as the new comparison harnesses do; unchanged in
  effect for callers (like the current `toy_experiment.py`) that don't.
- Any code already dividing rewards by `dim` before use (all four new harnesses)
  is unaffected in scale and just runs faster (`O(1)` reward frequency).

**Performance:** single-run T=1200 went 7.2s → 0.30s (~24×); a full 3-function
× 20-seed × T=3000 sweep went ~46 min → ~3.7 min. Results are numerically
identical to the pre-speedup version (the fix is behaviour-preserving).

**Reproducibility:** all 22 tests pass. `imabo/optimizer.py` has zero net
diff from before this PR — the new baselines and harnesses are purely
additive, and the toy-function rescale lives entirely in
`experiments/benchmarks/toys/toy_functions.py`.

## Key finding from the new comparison

Measured on the full sweep (T=3000, 20 seeds, dim=4, reward = `func(x)/dim`,
naturally close to `[0,1]`):

- **At the toy's tiny built-in noise:** Random has the best (lowest) simple
  regret on sin1 (0.077 vs IMABO+TPE's 0.106) and rastrigin (0.068 vs 0.098) —
  breadth beats re-pulling when one evaluation is already near-exact — but
  IMABO+TPE already wins on garland even at this noise level (0.127 vs
  Random's 0.145). D-TTTS is last on simple regret on all three (0.27–0.33).
  On cumulative regret IMABO (with or without TPE) is best on all three, and
  Random is worst on all three (it never exploits).
- **At σ=0.1 (explicit noise):** IMABO+TPE is the outright best on *both*
  metrics on all three functions (simple regret 0.090–0.143, cumulative
  922–977); D-TTTS is last on simple regret (0.26–0.31), and Random is last on
  cumulative regret (1373–1388).

D-TTTS's general weakness on simple regret traces to its Bernoulli
binarization (which discards information MOSS's continuous mean keeps) and
its `Beta(t−k,1)` pseudo-arm driving a breadth-runaway that under-samples each
arm.

## Key finding from the UCB-AIR comparison

On the same toy setup (T=3000, 20 seeds, reward = `func(x)/dim`):

- **Index effect (UCB-AIR vs MOSS-AIR, identical arm schedule):** the two draw
  the same seed-locked reservoir and recommend the same arm, so their *simple*
  regret is identical — but MOSS-AIR's tighter, minimax-calibrated bonus
  concentrates pulls on the leader far faster, cutting *cumulative* regret by
  ~33–42% (sin1 1286→745 = 42%, garland 1287→859 = 33%, rastrigin 1285→767 =
  40%). The index choice barely moves *which* arm you recommend at equal
  budget; it strongly moves the regret paid getting there. Interestingly,
  UCB-V's variance term doesn't help here: the toy's noise variance is tiny,
  so the variance-scaled term shrinks fast, but UCB-V's *other* term
  (`3·Eₜ/nᵢ`, a range/bias correction independent of variance) is large for
  the typical `n` reached at T=3000–10000 and ends up driving *more*
  exploration than plain UCB1 would — UCB-AIR's cumulative regret is a few
  percent *higher* under UCB-V than it was under UCB1 in this low-noise regime
  (e.g. sin1 1217→1286 at T=3000).
- **Schedule effect (arm-opening rate):** opening more arms (higher exponent)
  monotonically lowers simple regret; cumulative regret is *not* monotonic in
  β (see the β trade-off below — it is U-shaped, minimised at an intermediate
  β). IMABO's default β=0.8 opens 605 arms at T=3000 (best simple regret, but
  not the lowest cumulative regret); MOSS-AIR opens 55.
- **IMABO-no-oracle *is* infinite-armed MOSS.** MOSS-AIR and IMABO-no-oracle
  share the identical MOSS index and uniform reservoir; they differ only in the
  arm-opening exponent. Setting `IMABO(use_tpe=False, beta=0.5)` reproduces
  MOSS-AIR's `ceil(√t)` schedule and matches its results to within ~2–6% on
  every function (residual = independent reservoir draw + IMABO's
  `n_min_rewarded` undersampling guard and pending-pull bookkeeping).
- **The β trade-off (T=10000):** simple regret falls monotonically as β rises
  (more arms opened → better identification: rastrigin 0.202→0.086 over
  β=0.4→0.8). Average per-round regret (cum/t) is **not** monotonic — it is
  U-shaped on sin1 and rastrigin, dipping to a minimum near β=0.6 (rastrigin
  0.246→0.213→0.292 over β=0.4→0.6→0.8) before rising; on garland the minimum
  is already at β=0.4. Intuition: too few arms wastes pulls re-confirming a
  mediocre leader, too many pays breadth cost, so an intermediate β minimises
  per-round regret. Below the knee both regrets fall together (a Pareto
  improvement), and only past it does the classic explore/exploit trade-off
  set in. UCB-AIR and MOSS-AIR give identical simple regret on every function
  (same schedule, same recommended arm), but UCB-AIR's UCB-V bonus lands it at
  much higher per-round regret (~0.39–0.41 vs MOSS-AIR's ~0.21–0.25) for the
  same reason as above — the range/bias term dominates the shrunk variance
  term at this noise level; MOSS-AIR sits on the I-MOSS β≈0.5–0.6 curve,
  confirming they coincide.
- **QRM2 is depth-heavy on these reservoirs.** Its `ceil(t^0.347)` schedule
  opens only 23 arms at T=10000 (vs MOSS-AIR 100, I-MOSS β=0.8 1585), so it
  pays the *highest* simple regret of the family (e.g. rastrigin 0.228 vs
  I-MOSS β=0.8's 0.086) while keeping average per-round regret moderate
  (~0.28–0.30, between MOSS-AIR's ~0.21–0.25 and UCB-AIR's ~0.39–0.41). This is
  expected: QRM2 minimises *quantile* regret parameter-free, and on dense toy
  reservoirs where many draws are near-optimal, opening more arms (higher β)
  is what helps raw simple regret. QRM2's advantage would show on reservoirs
  where near-optimal arms are rare (small β-regularity), which these toys are
  not.
