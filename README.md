# NWay — Football Match Prediction, Recommendation & Validation

A rolling football prediction service. It estimates **probabilities** for
match outcomes across seven European leagues and the Champions League, selects
7–20 high-quality recommendations from a rolling 72-hour fixture window, emails
them about half an hour before the earliest match in the batch, and validates
every prediction against the real result.

It is driven by fixture timestamps, not gameweeks, and it stays silent far more
often than it speaks.

```bash
make install && .venv/bin/pip install -e .
.venv/bin/nway ingest --history --seasons 2017/18..2025/26
.venv/bin/nway train --through 2026-06-30
.venv/bin/nway demo          # see the whole thing work, no credentials needed
```

Full instructions: [docs/SETUP.md](docs/SETUP.md).

## What it does, measured

Trained on **21,545 matches** across nine seasons of the seven target leagues.
Walk-forward backtest over the full 2024/25 season (2,364 matches, 11 refits):

| Model | 1X2 log loss | Brier |
|---|---|---|
| Uniform (⅓ each) | 1.0986 | 0.6667 |
| Pooled base rate — the bar to beat | 1.0765 | 0.6481 |
| **This system** | **0.9808** | **0.5845** |
| De-vigged bookmaker closing odds (ceiling) | 0.9563 | 0.5670 |

Per-market calibration from the same backtest — predicted against actual, over
2,364 matches each:

| Market | Predicted | Actual | ECE |
|---|---|---|---|
| Home win | 0.433 | 0.426 | 0.019 |
| Over 1.5 | 0.762 | 0.777 | 0.038 |
| Over 2.5 | 0.524 | 0.531 | 0.046 |
| BTTS | 0.517 | 0.544 | 0.049 |
| Double chance 1X | 0.677 | 0.677 | 0.025 |

The backtest also simulates the notification rules: over that season it would
have sent **61 batches**, stayed silent for **22 windows** with too few
qualifying predictions, and found **19 windows with no fixtures at all**.

## Which competitions it can actually predict

Measured on the held-out 2025/26 season — the model's training window ended
before it — against each league's own base rate:

| League | n | Log loss | Its base rate | Gain | Over 1.5 ECE |
|---|---|---|---|---|---|
| Primeira Liga | 306 | 0.9257 | 1.0834 | **+0.158** | 0.039 |
| Bundesliga | 306 | 0.9783 | 1.0704 | **+0.092** | 0.079 |
| Eredivisie | 306 | 0.9882 | 1.0711 | **+0.083** | 0.062 |
| Serie A | 380 | 1.0082 | 1.0851 | **+0.077** | 0.091 |
| La Liga | 380 | 0.9856 | 1.0464 | **+0.061** | 0.066 |
| Ligue 1 | 306 | 1.0074 | 1.0616 | +0.054 | 0.072 |
| Premier League | 380 | 1.0369 | 1.0793 | +0.042 | 0.062 |
| **Champions League** | 187 | **0.9290** | 1.0189 | **+0.090** | 0.074 |

All eight beat their own baseline. The ordering is worth reading: the
**Premier League is the hardest** of them, and the **Primeira Liga the
easiest** — a league with three dominant clubs and a long tail is far more
predictable than the most scrutinised competition in the sport. The Champions
League scores well for the same reason in reverse: an enormous quality spread,
from Real Madrid to the qualifying-round entrants, makes team ratings highly
informative.

The Champions League has **no CSV source**, so its training history comes from
football-data.org instead — goals only, which is all the goal model needs:

```bash
nway ingest --org-history --competitions champions_league
```

Fitting it separately rather than pooling the domestic leagues matters: it runs
at a **home advantage of +0.264 against the Premier League's +0.117**. About 55
of 126 upcoming fixtures clear the data gates; clubs from outside the seven
configured leagues carry thin histories and are filtered out rather than
guessed at.

## How it decides what to send

Most ticks send nothing. Measured on the 2024/25 calendar, a rolling 72-hour
window across the seven leagues was **completely empty 15.7% of the time** — in
runs of seven to eight days during FIFA international windows — and only 68% of
windows held seven matches at all.

Probability floors are **per market**, because a flat floor is incoherent:
Over 1.5 comes in 77.1% of the time and a home win 43.8%, so one number would
mean opposite things.

```
floor(market) = max(0.65, base_rate + 0.25 × (1 − base_rate))
```

Ranking is **not** by raw probability. Each candidate is scored by the 10th
percentile of a Beta posterior over its settled history, so a well-evidenced
82% outranks a thinly-evidenced 85%, and thin samples are penalised
automatically.

Diversity caps stop a monoculture: unconstrained, selection produces batches
that are **81% double chance**. Those are genuinely the highest probabilities
available and, as a product, close to worthless — one goal-model error would
sink the whole batch. The caps cost about **8 percentage points** of send
frequency, which is recorded rather than hidden.

## Architecture

```
football-data.org ──┐                      (fixtures, UTC kickoffs, results)
football-data.co.uk ┼─▶ raw_response ──▶ entity resolution ──▶ bitemporal store
openfootball ───────┘   (replayable)       (canonical ids)      (event_time +
                                                                knowledge_time)
                                                    │
                                    as-of feature store (107 declared features)
                                                    │
                                    Dixon–Coles ──▶ goal matrix ──▶ every market
                                                    │
                                              calibration (isotonic / Platt)
                                                    │
                                     recommendation engine ──▶ planner ──▶ Resend
                                                    │
                                     settlement ──▶ evaluation ──▶ monitoring
```

Every fact carries **`event_time`** (when it happened) and **`knowledge_time`**
(when this system could first have known it). A prediction made at `as_of` may
read a row only when `knowledge_time <= as_of`. This is what makes the backtest
honest, and it is enforced in four places: the repository API, a database CHECK
constraint, a model-artefact guard, and the leakage test suite.

Markets are all derived from **one** goal matrix, so `P(Over 2.5) > P(Over 1.5)`
is impossible by construction rather than by luck.

## Commands

```bash
nway check                                # is everything configured?
nway schedule --write-state               # when should the next run be?
nway init-db                              # create the schema
nway ingest --history --seasons 2017/18..2025/26
nway ingest                               # live fixtures and results
nway train --through 2026-06-30
nway backtest --from 2024-08-01 --to 2025-05-31
nway tick [--dry-run] [--as-of ...]       # one scheduler cycle
nway demo                                 # replay a matchday, no credentials
nway report --window 90
nway explain --prediction-id 1234         # why, from stored rows alone
nway status
```

## Tests

```bash
make test           # 154 tests
make test-leakage   # temporal leakage only
```

The leakage suite poisons the database with absurd post-`as_of` data and
asserts that no feature changes. It includes a **canary** — a deliberately
leaky feature the harness must catch — so the suite cannot decay into one that
passes because it stopped checking.

Architecture rules are tests too: the recommendation engine may not import the
model package, nothing outside `clock.py` may read wall time, and no scheduling
query may filter on `matchday`.

## Documentation

| Document | Contents |
|---|---|
| [SETUP.md](docs/SETUP.md) | Installation, credentials, scheduling |
| [AUTOMATION.md](docs/AUTOMATION.md) | The self-scheduling GitHub Actions workflow |
| [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Architecture, components, principles |
| [DATA_SOURCES.md](docs/DATA_SOURCES.md) | Source-by-source assessment with measured coverage and licensing |
| [DATA_MODEL.md](docs/DATA_MODEL.md) | Bitemporal schema and entity resolution |
| [MODEL_DESIGN.md](docs/MODEL_DESIGN.md) | Features, model families, calibration, metrics |
| [PREDICTION_PIPELINE.md](docs/PREDICTION_PIPELINE.md) | The as-of contract and leakage prevention |
| [PREDICTION_LIFECYCLE.md](docs/PREDICTION_LIFECYCLE.md) | Discovery through post-match validation |
| [BACKTESTING.md](docs/BACKTESTING.md) | Historical simulation methodology |
| [NOTIFICATION_ARCHITECTURE.md](docs/NOTIFICATION_ARCHITECTURE.md) | 72h window, 7–20 rule, batching, suppression |
| [SCHEDULING_DESIGN.md](docs/SCHEDULING_DESIGN.md) | The idempotent tick |
| [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) | What is tested, and what deliberately is not |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phased roadmap |
| [RISK_AND_LIMITATIONS.md](docs/RISK_AND_LIMITATIONS.md) | Where this will be wrong |

## Evidence

Every quantitative claim above is reproducible with stdlib Python:

```bash
python3 research/feasibility_study.py --download
python3 research/threshold_study.py
```

Recorded output is in [research/evidence/](research/evidence/).

## Known gaps, stated plainly

- **No true xG.** FBref's terms prohibit building tools on scraped data, and
  Understat's `robots.txt` disallows all crawlers. The system uses a
  shot-based proxy named `sxg_proxy` everywhere — never `xg` — which knows
  about shot volume and accuracy but nothing about shot quality.
- **No player markets.** No free source supplies lineups, expected minutes or
  injuries at the required volume, so they are not generated at all.
- **No Europa or Conference League.** Both sit behind paid tiers on every free
  source checked. Adding them is a YAML edit once a plan is in place.
- **No Nations League or Women's Champions League.** Registered and disabled.
  The Nations League is a paid tier *and* international football, which the
  club-fitted model cannot predict; the Women's Champions League has no source
  on football-data.org, football-data.co.uk, openfootball or StatsBomb open
  data. Both would need their own trained model, not just a configuration
  change — see [DATA_SOURCES.md §7b](docs/DATA_SOURCES.md).
- **Corners and cards are modelled but not enabled.** The data supports them
  (measured overdispersion 1.18 and 1.34, so negative binomial rather than
  Poisson), but they are gated per league until the training sample clears the
  configured floor.

## Not a betting product

Outputs are probability estimates with quantified uncertainty. Every email
carries an explicit notice that a stated 85% estimate is expected to be wrong
roughly one time in seven. Nothing here is advice. See
[RISK_AND_LIMITATIONS.md](docs/RISK_AND_LIMITATIONS.md).
