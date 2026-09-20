# Backtesting

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

A backtest that uses today's knowledge of history is not a backtest. This
document defines how the system reconstructs what it *would have known* at each
historical moment.

---

## 1. Principle

For a simulated prediction at historical time `T`, the only data visible is
rows with `knowledge_time <= T`. Not `event_time <= T` — **`knowledge_time`**.
A match played before `T` whose statistics only reached us afterwards is
invisible, exactly as it was at the time.

## 2. Same code path

The backtester does not reimplement the pipeline. It sets the injected clock
and calls the production functions:

```python
for T in simulated_tick_times(start, end, interval_minutes=30):
    with clock.frozen_at(T):
        fixtures = discover_upcoming(horizon_hours=72)
        for f in due_for_prediction(fixtures, T):
            predict(f, as_of=T)
        candidates = score_candidates(as_of=T)
        decision  = should_send_notification(now=T, ...)
        if decision.should_send:
            record_simulated_batch(decision)
```

Any divergence between backtest and production is then a bug in one shared
implementation, not a discrepancy between two.

## 3. Reconstructing knowledge times

The awkward part: `knowledge_time` is only recorded prospectively, and a
backtest over 2018–2024 predates the system.

Historical rows are therefore assigned **conservative reconstructed knowledge
times**, flagged `knowledge_time_is_estimated = 1`:

| Data | Reconstructed knowledge time | Basis |
|---|---|---|
| Match result | kickoff + 2h | Results are public immediately |
| Match statistics (football-data.co.uk) | kickoff + **72h** | Measured publication lag: the site updates ~twice weekly and showed a 3-day-old timestamp when checked |
| Fixture schedule | 30 days before kickoff | Fixture lists are published well in advance |
| Weather | forecast issued 24h before kickoff | Open-Meteo historical-forecast archive |
| Availability | unavailable historically | Feature is NULL in backtests |

Every choice errs **late**. A too-late estimate loses a feature and understates
the model; a too-early one leaks and overstates it. Systematic pessimism is the
correct bias.

The 72-hour statistics lag is not conservatism theatre — it is measured, and it
has a real effect: for a Saturday fixture, the previous Wednesday's results are
visible but Wednesday's *shot and corner counts* may not be. Rolling stat
features legitimately lag rolling goal features, and the backtest must
reproduce that or it will overstate the corners and cards models.

Backtest reports state which fields used estimated knowledge times, and a
sensitivity run at a 24-hour lag bounds the effect.

## 4. Walk-forward protocol

```
Fold 1  train 2017/18–2021/22  calibrate 2022/23  test 2023/24
Fold 2  train 2017/18–2022/23  calibrate 2023/24  test 2024/25
Fold 3  train 2017/18–2023/24  calibrate 2024/25  test 2025/26
```

Expanding window by default; a rolling fixed window is run as a sensitivity
check to test whether older seasons still help. Given the measured home-advantage
shift in 2020/21, the answer is not obvious in advance — which is why it is run
rather than assumed.

Within a test season, the model is refit fortnightly on completed matches only,
so an October prediction comes from a model that could have existed in October.

Three folds is what nine seasons of these leagues permit at this granularity.
That is a real limitation and is reported as one: three folds cannot
distinguish a genuinely better model from a luckier one at small effect sizes.

## 5. What the backtest simulates

Not just predictions — **the whole product**:

1. features as of `T`;
2. model prediction and calibration;
3. recommendation eligibility and ranking;
4. diversity and correlation caps;
5. the notification decision, including minimum-7, cooldown and dedup;
6. the batch that would have been sent;
7. settlement against the real result;
8. evaluation.

This surfaces failures a prediction-only backtest cannot: how many emails would
have gone out in a season, how many would have fallen below seven and stayed
silent, whether the market-share cap ever starved a batch below the minimum,
and what the realised hit rate of *actually recommended* selections was as
distinct from all predictions.

## 6. Outputs

```
backtests/<run_id>/
├── config.yaml              resolved configuration + config_hash
├── predictions.parquet      every simulated prediction with as_of
├── batches.parquet          every batch that would have been sent
├── decisions.parquet        EVERY decision, including skips and reasons
├── metrics/                 by market, competition, season, bucket
├── calibration/             reliability diagrams per market
└── report.html              the human-readable summary
```

`decisions.parquet` matters as much as `predictions.parquet`: the distribution
of skip reasons over a season is the evidence for whether the thresholds are
set sensibly. "Silent for 40% of the season because of
`INSUFFICIENT_QUALIFYING`" is a tuning signal that a prediction-only backtest
would never surface.

## 7. Reproducibility

A backtest is identified by `(config_hash, feature_version, code_commit,
data_snapshot_max_knowledge_time)`. Re-running with identical inputs must
produce byte-identical outputs — random seeds fixed, no wall-clock reads
outside the injected clock, no set iteration order dependence. A regression
test runs a small fixed backtest and compares hashes.

## 8. Backtest red flags

Treated as bugs until disproven:

| Signal | Likely cause |
|---|---|
| Held-out log loss < 0.95 | Leakage — better than de-vigged closing odds is implausible |
| Calibration near-perfect at the extremes on small samples | Calibrator fitted on the test fold |
| A market's hit rate far above its predicted probability | Settlement or window bug |
| Performance improving in later folds without a model change | Information bleeding backwards |
| Zero skipped batches across a season | The planner is not being exercised |
| Corners model matching the goals model's accuracy | Statistics lag not applied |

Each has an automated check in the backtest report, and each prints the
diagnostic rather than a pass/fail alone.
