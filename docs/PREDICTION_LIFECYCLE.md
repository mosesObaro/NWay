# Prediction Lifecycle

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

From fixture discovery to model evaluation. Gameweek and matchday are attached
as metadata at discovery and never consulted again.

```
[1] Fixture discovery      ← football-data.org, every tick
[2] Fixture validation     ← entity resolution, duplicate/dedup checks
[3] Initial prediction     ← T-48h
[4] Prediction refresh     ← T-24h, T-8h, ADHOC
[5] Recommendation eval    ← every tick, on the 72h window
[6] Notification planning  ← should_send_notification()
[7] Email                  ← at earliest_kickoff − 0.5h
[8] Kickoff                ← fixture becomes unpredictable
[9] Result collection      ← polled after kickoff + duration buffer
[10] Market settlement     ← deterministic resolution per market
[11] Prediction validation ← per-prediction metrics
[12] Model evaluation      ← aggregates, drift, reports
```

---

## 1. Fixture discovery

Each tick queries every enabled competition for fixtures in a configurable
discovery horizon (default 30 days — wider than the 72-hour prediction window,
so that schedule changes are seen early).

New fixtures are inserted with `knowledge_time` = fetch time. Existing fixtures
are compared field by field; differences append to `fixture_revision`.

`matchday`, `stage` and `leg` are stored here. Nothing downstream of discovery
reads `matchday`.

**UEFA specifics.** The Champions League league phase has eight fixtures per
team against eight different opponents, and no two-legged structure until the
knockout rounds. Knockout ties get a shared `tie_id` and `leg`. Kickoff times
for later rounds are provisional until the draw, so `kickoff_is_confirmed=0`
until the provider confirms them, and unconfirmed fixtures fail the
recommendation gate.

## 2. Fixture validation

- Both teams resolve to canonical ids; unresolved names go to the queue and the
  fixture is **not** ingested.
- Duplicate detection on `(competition, season, home, away, kickoff ± 48h)`,
  which catches the same match appearing under a corrected kickoff time.
- Sanity: home ≠ away, kickoff within the season bounds, competition enabled.
- Cross-source agreement where two sources cover the fixture; disagreement on
  kickoff time raises a quality check and marks the fixture unconfirmed.

## 3–4. Prediction and refresh

Per [SCHEDULING_DESIGN.md §3](SCHEDULING_DESIGN.md). Each stage appends an
immutable `prediction_run`. Nothing is ever overwritten.

## 5–7. Recommendation, planning, delivery

Per [NOTIFICATION_ARCHITECTURE.md](NOTIFICATION_ARCHITECTURE.md).

## 8. Kickoff

At `kickoff_utc` the fixture leaves the candidate pool. Predictions remain, with
their timestamps, for evaluation.

## 9. Result collection

Polled from `kickoff + 105 minutes`, backing off until the fixture reports
`FINISHED`. Postponement or abandonment after kickoff transitions to
`POSTPONED`/`SUSPENDED`, and all predictions for it settle as `VOID`.

Results are cross-checked against a second source where available
(openfootball, with its known multi-day lag). Disagreement raises a `BLOCKING`
quality check and **settlement is held** until resolved — an incorrect result
corrupts every downstream metric permanently.

## 10. Market settlement

Deterministic resolution from `match_result` and `team_match_stats`:

| Market | Settles on |
|---|---|
| 1X2, double chance, DNB | 90-minute score |
| Over/Under N.5 | total goals at 90 minutes |
| BTTS | both sides ≥ 1 at 90 minutes |
| Team goals, clean sheet, margin, ranges | 90-minute score |
| Half-time markets | half-time score |
| Corners, cards | `team_match_stats`, **when present** — otherwise `UNSETTLEABLE` |

**Extra time and penalties are excluded**, which matters for UEFA knockouts:
`went_to_et` exists so the exclusion is explicit and testable.

`VOID` (abandoned, postponed) is excluded from hit-rate and calibration
statistics but recorded. `UNSETTLEABLE` (the statistic never arrived) is
tracked separately — a rising `UNSETTLEABLE` rate is a data-source problem
wearing a modelling disguise.

## 11. Prediction validation

For every settled prediction: outcome, hit flag, per-prediction log loss and
Brier, and whether it was recommended and notified. A settled prediction is
never re-settled; a corrected result creates a settlement revision with an
audit trail.

Crucially, **every** prediction is validated, not only the recommended ones.
Evaluating only what was recommended measures the selection policy, not the
model, and makes it impossible to tell whether the filter is helping.

## 12. Model evaluation

Aggregates recomputed on a schedule and on demand, at these grains: market;
market × competition; market × season; market × confidence bucket; model
version; recommended versus all.

Each carries `n_samples` and a Wilson interval. The reporting layer refuses to
draw conclusions below `min_samples_for_conclusion` (default 200) and renders
those rows greyed with "insufficient sample" rather than omitting them — an
omitted row looks like an absent problem.

Example of the intended output:

```
Market: Over 1.5 Goals          Window: 2026-08-01 .. 2026-09-20
Predictions:            500      (recommended: 118)
Mean predicted:        82.1%
Actual hit rate:       80.4%     95% CI [76.7%, 83.7%]
Brier:                 0.1402
ECE (10 bins):         0.0218
Verdict: calibrated within interval; continue.
```

A gap of 1.7 points on n=500 sits inside the interval and is not evidence of
miscalibration. The report says so explicitly, because the temptation to
over-read small samples is the most common way a project like this reaches a
wrong conclusion about its own quality.
