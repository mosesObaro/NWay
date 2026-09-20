# Model Design

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

Every quantitative statement here is measured over **21,545 matches** from the
seven target leagues, seasons 2017/18–2025/26, by
`research/feasibility_study.py`.

---

## 1. Feature engineering strategy

### 1.1 The as-of contract

Feature code never touches the database directly. It receives a
`FeatureContext(as_of, fixture, config)` and reads through a repository that
**requires** `as_of` and filters `knowledge_time <= as_of`. There is no
unfiltered accessor to reach for, which is the point: leakage should require
deliberate effort, not merely inattention.

Each feature is declared, not just implemented:

```python
FeatureSpec(
    key="home.gf_per_match.last5.home_only",
    entity="HOME_TEAM",
    window=Window(kind="MATCHES", size=5, venue="HOME"),
    inputs=["team_match_stats.goals"],
    min_observations=3,               # below this -> NULL, reason INSUFFICIENT_HISTORY
    max_staleness_hours=96,           # above this -> NULL, reason STALE
    competition_scope="SAME_COMPETITION",
    requires_completed_fixtures=True,
)
```

The registry is hashed into `feature_version`. Changing a window silently is
impossible; it produces a new version, and old predictions keep pointing at the
definition that produced them.

### 1.2 Team form features

Computed for home and away sides, over windows of **last 3 / 5 / 10 matches and
season-to-date**, each in **all / home-only / away-only** variants:

goals for, goals against, shots, shots on target, shot conversion,
shots-on-target ratio, corners for/against, cards, clean-sheet rate, BTTS rate,
failed-to-score rate, points per match, first-half and second-half goals for
and against, `sxg_proxy` for and against.

That is a large grid, and **most of it will not survive**. The brief is right to
say windows must be validated rather than assumed. The selection procedure is:
fit the goal model with each window family in isolation on 2017/18–2022/23,
score on 2023/24 by log loss, and keep only families that improve on the
simpler alternative by more than the bootstrap standard error. The expectation
from the literature and from the measured stability of these leagues is that a
**5–10 match window with exponential time decay** wins and that the 3-match
window is noise; the procedure is what decides, not the expectation.

**Venue splits are kept despite halving the sample**, because the measured home
effect is large and non-constant — home teams take 5.42 corners per match
against 4.42 away, and the home-win rate moved from 46.1% (2017/18) to 40.2%
(2020/21) to 43.7% (2025/26).

### 1.3 Opponent adjustment

Raw form conflates a team's quality with its fixture list. Two mechanisms, in
increasing order of cost:

1. **Latent attack/defence ratings** from the Dixon–Coles fit itself
   (§3). Because those parameters are estimated jointly across every match, they
   are already opponent-adjusted; they are exported as features
   (`home.attack_rating`, `away.defence_rating`, and their difference) for the
   corners, cards and context models.
2. **Strength of schedule**: the mean opponent rating faced in the window, and
   an adjusted-goals feature
   `adj_gf = Σ(goals_for_i / expected_goals_against_opponent_i) / n`.

Both are as-of quantities: ratings are refit only on matches with
`knowledge_time <= as_of`, and a model artefact records its
`train_window_end`, which a check asserts is ≤ every `as_of` that uses it.

### 1.4 Match context features

Objective and measurable only — nothing resembling "motivation":

- `home_advantage` — league-and-season specific, estimated inside the model.
- `rest_days_home`, `rest_days_away`, and their difference.
- `matches_last_14_days` per team — fixture congestion.
- `is_uefa_midweek_before` / `after` — a domestic fixture bracketed by a
  European tie.
- `competition_id`, `stage`, `leg`, `aggregate_deficit` for two-legged ties.
- `season_progress` (0–1), `is_first_5_matchweeks`.
- `is_promoted_team` — newly promoted sides have no prior-season history in the
  competition, and this flag is what stops the model treating a NULL history as
  average.
- `travel_km` (great-circle between venues) — **UEFA only**, where the range is
  meaningful; within a domestic league it is mostly noise.
- Weather at kickoff from the **forecast** available at `as_of`
  [Recommended]: temperature, wind speed, precipitation probability.

### 1.5 Player features — [Future]

Specified in [DATA_SOURCES.md §9](DATA_SOURCES.md) as blocked. The intended
shape, when a data source exists: goals/assists/shots/SoT per 90, expected
minutes from a starts-and-substitutions model, penalty and set-piece
responsibility, recent minutes, and opponent defensive strength. Nothing is
built until minutes can be estimated, because the scoring probability is
dominated by the minutes distribution.

---

## 2. Model inventory

| Model | Family | Predicts | Status | Why it exists |
|---|---|---|---|---|
| `baseline_uniform` | constant | 1X2 | **[MVP]** | Floor; log loss 1.099 |
| `baseline_base_rate` | league/season frequency | 1X2, O/U, BTTS | **[MVP]** | The bar to beat: 1.071 |
| `baseline_form` | recent points/goals | 1X2 | **[MVP]** | Tests whether form alone carries signal |
| `baseline_naive_goals` | mean GF/GA, independent Poisson | all goal markets | **[MVP]** | Isolates the value of Dixon–Coles' additions |
| `goals_dixon_coles` | time-weighted MLE, low-score correction | **goal matrix** | **[MVP]** | Primary model; derives every goal/result market |
| `goals_gbm_residual` | gradient boosting on DC residuals | λ adjustment | **[Recommended]** | Captures context (rest, congestion) DC cannot express |
| `corners_nb` | negative binomial GLM | corner counts | **[Recommended]** | Measured overdispersion 1.18 rules out Poisson |
| `cards_nb` | negative binomial GLM | card counts | **[Recommended]** | Overdispersion 1.34, plus large league effects |
| `halves_dc` | DC with half-specific rates | HT markets | **[Recommended]** | HT data is 100% complete |
| `players_*` | — | — | **[Future]** | Blocked on data |

### Why Dixon–Coles rather than a classifier

A three-class classifier over {H, D, A} throws away the goal count, cannot
produce Over 2.5 or BTTS without separate models, and leaves those models free
to disagree with each other. Modelling goals first gives one coherent joint
distribution from which **every** goal-derived market falls out by summation,
so P(Over 2.5) ≤ P(Over 1.5) holds by construction rather than by luck.

### Why Poisson for goals but negative binomial for corners and cards

Measured variance-to-mean ratios:

| Quantity | Mean | Variance | var/mean | Verdict |
|---|---|---|---|---|
| Total goals | 2.81 | 2.81 | **1.00** | Poisson-consistent |
| Home goals | 1.55 | 1.73 | 1.12 | Poisson + team heterogeneity |
| Away goals | 1.26 | 1.40 | 1.11 | Poisson + team heterogeneity |
| Total corners | 9.87 | 11.65 | **1.18** | Overdispersed |
| Home corners | 5.42 | 8.94 | **1.65** | Strongly overdispersed |
| Total cards | 4.27 | 5.71 | **1.34** | Overdispersed |

The goals ratio of 1.00 is a genuinely strong result: the marginal distribution
of total goals is Poisson to two decimal places, and the mild 1.11–1.12 on the
per-side counts is what varying λ across matches produces from *conditionally*
Poisson data. Corners and cards are a different story, and fitting them with
Poisson would understate the variance — which matters precisely at the tails
where over/under lines sit.

---

## 3. The goal model

### 3.1 Specification

For fixture *k* between home *i* and away *j*:

```
log λ_home = μ_c + home_adv_c + attack_i + defence_j + γ·context_k
log λ_away = μ_c +               attack_j + defence_i + γ·context_k
```

with `μ_c` and `home_adv_c` per competition (goal rates differ materially:
Bundesliga 3.12 goals/match against La Liga 2.58), a sum-to-zero constraint on
attack and defence, and the Dixon–Coles low-score dependence correction τ
applied to the 0-0, 1-0, 0-1 and 1-1 cells, where independent Poisson is known
to misfit.

**Time decay.** Matches are weighted `w = exp(-ξ · Δt_days)`, with ξ fitted by
maximising held-out log likelihood rather than assumed. A half-life somewhere
around 6–12 months is the usual answer; the measured collapse and recovery of
home advantage between 2019/20 and 2022/23 is exactly the phenomenon ξ has to
track.

**Shrinkage.** Attack and defence are regularised toward the competition mean
with an L2 penalty tuned by held-out likelihood. Without it, a promoted team
with four matches played gets an extreme rating and the model produces
overconfident nonsense in August — which is also when the fixture calendar is
dense and the notifier is most active.

**COVID seasons.** 2019/20 is truncated (Ligue 1 279 matches, Eredivisie 232)
and 2020/21 was played in empty stadiums, where home advantage fell to +0.155
goals against +0.356 the season before. These are not discarded — that would
throw away 4,700 matches — but the home-advantage term carries a
`crowd_present` indicator so the anomaly is absorbed by a parameter instead of
contaminating every team rating.

### 3.2 From λ to a goal matrix

```
P(H=h, A=a) = τ(h,a) · Poisson(h; λ_home) · Poisson(a; λ_away)
```

evaluated on a 0–10 × 0–10 grid (covering >99.99% of mass) and renormalised.
That matrix is stored as the run's `lambda_home`/`lambda_away` plus the τ
parameters — enough to regenerate the whole matrix on demand.

### 3.3 Markets derived from the matrix, not modelled separately

| Market | Derivation |
|---|---|
| Home / Draw / Away | Σ below, on, above the diagonal |
| Double chance, draw-no-bet | Sums and renormalisation of the above |
| Over/Under 0.5, 1.5, 2.5, 3.5 | Σ over `h + a > line` |
| BTTS | `1 − P(h=0) − P(a=0) + P(0,0)` |
| Team to score / clean sheet | Marginal sums |
| Team total goals, goal ranges | Marginal / block sums |
| Goal margin | Σ over `h − a = m` |
| Expected goals (home/away/total) | λ_home, λ_away, λ_home + λ_away |

**Sanity invariants asserted after every prediction run**: the matrix sums to
1 ± 1e-9; the 1X2 probabilities sum to 1; over-lines are monotonically
non-increasing in the line; `P(BTTS) ≤ P(Over 1.5)`; expected goals equals the
matrix mean. A violation fails the run rather than emitting a quiet
inconsistency.

### 3.4 Half-time markets [Recommended]

Halves are **not** λ/2. Measured across 21,544 matches: **1.244 goals in the
first half, 1.563 in the second** — a 2H/1H ratio of 1.256. The half model
fits separate rates with a shared team strength and a per-competition half
split, then derives HT 1X2 and first/second-half over-under from two smaller
matrices.

### 3.5 The `sxg_proxy`

No free, terms-compliant xG source exists ([DATA_SOURCES.md §8](DATA_SOURCES.md)).
The proxy is an explicit, fitted, documented stand-in:

```
sxg_proxy = a · shots_on_target + b · (shots − shots_on_target)
```

with `a`, `b` fitted per competition by Poisson regression of goals on the two
shot counts. It is a **shot-volume-and-accuracy estimate**, not a shot-quality
model: it cannot tell a tap-in from a thirty-yard effort, which is the whole
point of real xG. Its documented use is as a *form-smoothing* feature — a team
that has generated many good shots while scoring few is likelier to score next
time — not as a claim about chance quality. The validation is honest: report
the correlation between `sxg_proxy` and realised goals, and state it alongside
the published correlation for commercial xG so the gap is visible. Any feature
derived from it carries the `sxg_` prefix.

---

## 4. Corners and cards [Recommended]

Separate models, because corners and cards are not deducible from the goal
distribution and are driven by different mechanisms (territorial dominance and
playing style; referee strictness and match tension).

**Corners** — negative binomial GLM per side:
`log E[corners] = μ_c + home_corner_adv + corner_attack_i + corner_defence_j + β·(shot dominance)`,
with a fitted dispersion parameter. Totals come from the convolution of the two
sides' distributions; the small positive dependence between them (both rise in
open games) is handled with a shared match-level intensity term.

**Cards** — negative binomial with **mandatory per-competition intercepts**.
The measured spread makes the reason unmissable:

| League | Mean cards/match | Referee identity available |
|---|---|---|
| Primeira Liga | 5.39 | no |
| La Liga | 5.08 | no |
| Serie A | 4.57 | no |
| Ligue 1 | 4.01 | no |
| Bundesliga | 3.92 | no |
| Premier League | 3.61 | **yes (100%)** |
| Eredivisie | 3.18 | no |

Primeira Liga issues 70% more cards than the Eredivisie. A pooled intercept
would be wrong in both directions simultaneously. **Referee identity is
available only for the Premier League**, so a referee random effect exists in
the Premier League card model alone, and the other six leagues carry an
explicit `referee_unknown` flag rather than an imputed average.

**Per-league eligibility gate.** Corners and cards markets are enabled per
league only where the training sample clears the configured minimum
(`config/markets.yaml`). Eredivisie and Primeira Liga have statistics only from
2017/18 — roughly 2,750 matches each against ~9,900 for the Premier League — so
they enter later, and only if held-out calibration passes.

---

## 5. Where machine learning fits

The GBM is a **residual model**, not a replacement: it predicts the correction
to log λ from context features the Dixon–Coles structure cannot express (rest
differential, congestion, European hangover, promoted-team status, stage and
leg, weather). This keeps the coherent goal distribution while letting a
flexible learner add what it can.

It ships only if it beats plain Dixon–Coles on held-out log likelihood by more
than the bootstrap standard error, in a walk-forward test. With ~2,400 matches
per season across seven leagues, ~21,500 rows total and perhaps 40 usable
features, a GBM is at the edge of what the data supports — it will overfit
happily if allowed.

**No neural networks.** The dataset is tens of thousands of rows with a strong,
well-understood parametric structure. Deep learning has nothing to add here, and
adding it would trade interpretability and calibration for nothing measurable.

---

## 6. Probability calibration

A model can rank well and still be miscalibrated, and this system's output *is*
the probability, so calibration is a first-class stage rather than a
post-processing nicety.

- **Method**: isotonic regression where there are ≥ 1,000 settled samples for
  the (market, competition) cell; Platt scaling between 300 and 1,000; identity
  below 300, with the market **not eligible for recommendation** until it
  clears the sample floor. Beta calibration is evaluated as an alternative for
  the extreme-probability markets, where isotonic's step functions behave
  poorly near 0 and 1 — which is precisely where Over 1.5 lives.
- **Temporal fitting, always.** The calibrator is fitted on a window that ends
  before the prediction it adjusts. Fitting a calibrator on data that includes
  the target season is a subtle and very common form of leakage.
- **Pooling rule**: fit per (market, competition) when the sample allows,
  otherwise per market with a competition offset. Seven leagues split fourteen
  ways is how a calibrator ends up fitting noise.
- **Stored**: every calibrator is a versioned artefact with its fit window,
  sample size, and ECE before and after. `prediction.calibrated_probability` is
  what the recommendation engine reads; `raw_probability` is retained so the
  calibrator's effect stays auditable.

**Measured reference.** De-vigged closing odds on Over 2.5 across 21,544
matches give an ECE of **0.0132** with the largest bin deviation at +0.040 —
that is what a well-calibrated market looks like, and it is the standard the
models are held to.

---

## 7. Validation

**Walk-forward, expanding window.** No random splits anywhere.

```
train 2017/18–2021/22  →  calibrate 2022/23  →  test 2023/24
train 2017/18–2022/23  →  calibrate 2023/24  →  test 2024/25
train 2017/18–2023/24  →  calibrate 2024/25  →  test 2025/26
```

Within a season the walk-forward is finer: refit at a configurable cadence
(default fortnightly) using only matches completed before the refit date, so
that an in-season prediction is made by a model that could have existed then.

The **calibration fold is distinct from both training and test**. Sharing the
calibration and test folds inflates every calibration metric reported.

---

## 8. Evaluation metrics

Accuracy is not reported as a headline anywhere. For a market with a 77% base
rate, always predicting "yes" scores 77% accuracy and is worthless.

**Classification** — log loss (primary), Brier, reliability diagram, ECE
(10 equal-count bins), ROC-AUC for discrimination, and hit rate *per
probability bucket* with Wilson intervals.

**Counts** — MAE, RMSE, Poisson (or negative-binomial) deviance, mean
log-likelihood per match.

**Calibration** — ECE, maximum calibration error, reliability diagram with
per-bin sample counts, and a Spiegelhalter Z-test for whether observed
miscalibration is distinguishable from chance.

**Recommendation-level** — for every selected prediction: predicted
probability, outcome, hit rate by confidence bucket, and calibration restricted
to *recommended* predictions, broken out by market, competition, season and
model version, each with `n` and a confidence interval.

### Benchmarks the models must clear

Measured on 21,544 matches, seven leagues, 2017/18–2025/26:

| Model | 1X2 log loss | Multiclass Brier |
|---|---|---|
| Uniform (⅓ each) | 1.0986 | 0.6667 |
| **Pooled base rate — must beat** | **1.0711** | **0.6481** |
| Target for `goals_dixon_coles` | **≤ 1.00** | ≤ 0.61 |
| De-vigged closing odds (ceiling) | 0.9563 | 0.5670 |

The market number is a ceiling, not a target: closing odds aggregate lineup
news, money and information this system will not have. A model landing between
1.00 and 1.02 is doing real work; one at 1.07 has learned nothing beyond the
base rate; one below 0.95 on held-out data is a leakage bug, not a
breakthrough — and the leakage suite should be run before celebrating.

---

## 9. Model registry and promotion

A model is promoted to `is_active` only when, in a walk-forward test: it beats
the incumbent on log loss by more than the bootstrap standard error; its ECE is
≤ 0.03 on held-out data; no competition degrades by more than a configured
tolerance (an average gain that hides a collapsed Eredivisie is not a gain); and
the leakage suite passes. Promotion is recorded, never silent, and the previous
version stays available so that historical predictions remain reproducible.
