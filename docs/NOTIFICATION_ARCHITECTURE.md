# Notification Architecture

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

This is the component that distinguishes a rolling prediction service from a
weekly tips script. It answers one question on every tick: **should an email be
sent right now, and if so, containing exactly what?**

The default answer is **no**. Measured against real fixture calendars, this
system will be silent most of the time, and that is correct behaviour rather
than a fault to be tuned away.

---

## 1. What the fixture calendar actually looks like

Measured over the 2024/25 season, seven domestic leagues, 2,364 matches, using
rolling 72-hour windows stepped every 6 hours (1,160 windows):

| Window contains | Share of windows |
|---|---|
| ≥ 1 match | 84.3% |
| ≥ 5 matches | 73.1% |
| **≥ 7 matches** | **68.0%** |
| ≥ 10 matches | 61.2% |
| ≥ 20 matches | 51.1% |
| **0 matches** | **15.7%** |

Median 21 matches, mean 24.4, maximum 65.

Longest empty stretches, all of them FIFA international windows:
2024-09-02 → 09-10, 2024-10-07 → 10-15, 2024-11-11 → 11-19,
2025-03-17 → 03-24 — seven to eight days each.

Matches cluster hard: Sat 974, Sun 912, Fri 204, Mon 115, Wed 73, Tue 44,
Thu 42. Roughly 80% of fixtures fall at the weekend.

**Three design conclusions follow directly.**

1. The minimum-7 rule is achievable: two thirds of windows hold enough
   *matches*. Whether they hold enough *qualifying recommendations* is §3.
2. Empty windows are not an edge case at 15.7% of the time, in runs of a week
   or more. Silent operation is a first-class, tested, non-error path.
3. Because fixtures cluster at weekends, a naive "send when ≥ 7" rule would
   fire several times each Saturday. Cooldown and duplicate suppression are
   load-bearing, not decoration.

---

## 2. Recommendation engine

Separate from the prediction models by design: it reads stored, calibrated
predictions plus historical reliability, and **never imports from
`models/`** — an import-linting test enforces it. Prediction quality and
selection policy change for different reasons and at different rates.

```
predictions (calibrated)
        │
        ▼
 [1] Eligibility gates        ── hard pass/fail, reasons recorded
        ▼
 [2] Reliability scoring      ── Beta posterior lower bound
        ▼
 [3] Ranking
        ▼
 [4] Diversity & correlation  ── caps and group constraints
        ▼
 [5] Selection (7..20)
```

### 2.1 Eligibility gates — hard filters

A candidate fails if any of these hold. Failures are written to
`recommendation.rejection_reasons`, including for candidates never selected,
so that "why was there no email?" is always answerable.

| Gate | Default | Rationale |
|---|---|---|
| Market enabled for this competition | per `config/markets.yaml` | Eredivisie/Primeira Liga corners have only ~2,750 training matches |
| Calibrated probability ≥ the market's floor | per market, §3 | A flat floor is not comparable across markets |
| Settled sample for (market, competition) ≥ | `500` | No claims from small samples |
| Held-out ECE for the market ≤ | `0.05` | A miscalibrated probability is not a probability |
| Data completeness ≥ | `0.85` | Share of expected features actually present |
| Feature staleness ≤ | `96h` | football-data.co.uk lags up to ~3 days |
| Both teams have ≥ N prior matches in the competition | `5` | Promoted sides, August |
| Fixture status is `SCHEDULED`/`TIMED` | — | Never recommend a postponed match |
| Kickoff is confirmed | — | Unconfirmed UEFA kickoff times drift |
| Lead time remaining ≥ minimum | `0.25h` | A recommendation arriving after kickoff is useless |
| No blocking data-quality issue on the fixture | — | Decline rather than predict from corrupt data |

### 2.2 Reliability scoring

Ranking by raw probability is wrong, and the brief says so: an 85% prediction
from a poorly calibrated market can be worth less than an 82% one from a
well-calibrated market. The fix is to rank by **what this system has actually
achieved historically when it said this**, penalised for uncertainty.

For a candidate in (market *m*, competition *c*, probability bucket *b*), take
the settled history — *k* hits out of *n* — and form a Beta posterior:

```
posterior = Beta(α₀ + k, β₀ + n − k)
```

with the prior `(α₀, β₀)` centred on the market-level pooled hit rate and
weighted as ~50 pseudo-observations, so a thin cell is shrunk toward the
market's overall behaviour rather than trusted on its own.

```
reliability_score = Beta.ppf(0.10 | α₀ + k, β₀ + n − k)
```

— the 10th percentile of the posterior hit rate: a deliberately pessimistic
estimate of how often this kind of prediction has actually come in.

This single quantity handles three of the brief's requirements at once. Small
samples widen the posterior and drag the lower bound down. Miscalibration shows
up as a realised hit rate below the predicted probability, which lowers the
bound directly. Well-evidenced markets with a long track record are rewarded
without any hand-tuned weighting.

```
ranking_score = reliability_score × freshness_factor × completeness_factor
```

where the two factors decay linearly from 1.0 to 0.8 across their allowed
ranges. They break ties; they cannot rescue a candidate that failed a gate.

**Confidence bands** are reported from the posterior, not from the raw
probability: `HIGH` when the 10th percentile ≥ 0.75 and n ≥ 1,000; `MEDIUM`
when ≥ 0.65 and n ≥ 500; `LOW` otherwise (and `LOW` is not eligible by
default).

### 2.3 Match quality versus market quality

Kept separate, as the brief requires.

**Match quality** asks whether the fixture is *predictable at all*: data
completeness, feature freshness, both teams' history depth, no blocking quality
issues, confirmed kickoff. A fixture that fails match quality contributes
nothing, whatever its probabilities look like.

**Market quality** asks whether *this particular prediction* is strong:
calibrated probability, reliability score, market-level ECE and sample.

So the brief's example resolves as intended — Arsenal vs Liverpool can
contribute `Over 1.5 Goals 87%` while `BTTS 59%` and `Home Win 51%` are
rejected for falling below the probability floor.

### 2.4 Correlation management

Within one fixture, `Over 1.5`, `Over 2.5`, `Over 3.5` and `BTTS` all come from
the same goal matrix. They are not four opportunities; they are four views of
one estimate, and presenting them as independent inflates the apparent breadth
of the batch.

Two mechanisms:

1. **Exact within-fixture correlation.** Because every goal market is derived
   from the stored goal matrix, the joint probability of any two selections is
   computable exactly by summing the matrix cells satisfying both — no
   assumption needed. The engine computes φ (the correlation coefficient for
   two binary events) from the matrix and refuses to co-select a pair above
   `max_within_fixture_correlation` (default `0.30`).
2. **Hard caps.** `max_selections_per_fixture: 1` by default — one fixture, one
   recommendation, the highest-ranked. Corners or cards from the same fixture
   are structurally near-independent of the goal markets, so the configuration
   permits raising this to 2 with a same-family exclusion, but 1 is the
   shipping default.

**Cross-fixture common-mode risk** is the subtler problem, and the measured
data makes it vivid: unconstrained selection produces batches that are **81%
double chance** (§3.3). Twenty double-chance selections across twenty different
matches are statistically distinct events, but they share one model, one
calibrator and one set of assumptions — if the goal model drifts, all twenty
fail together.

Hence **per-family concentration caps** (`config/recommendations.yaml`): double
chance 30%, totals 40%, result 40%, corners and cards 25%, plus
`max_share_per_competition: 0.50`. Caps are computed against the **actual**
batch size rather than the 20 maximum, and are relaxed by one slot before a
viable batch is abandoned. These are diversification constraints against model
risk, not aesthetic preferences, and §3.3 prices them.

### 2.5 Selection

Rank by `ranking_score`, then greedily admit candidates subject to every cap,
until `max_recommendations` (20) or the pool is exhausted. If the admitted set
is smaller than `min_recommendations` (7), the batch is **not** created.

There is no code path that relaxes a threshold when the count falls short. The
function signature makes it structurally impossible: thresholds are an
immutable input to selection, never a return value, and a property test asserts
that selection output is monotone in the candidate pool — adding a weak
candidate can never change which strong ones were chosen.

---

## 3. Choosing the probability floors — empirically

### 3.1 Why there is no single global floor

Market base rates differ enormously: Over 1.5 comes in 77.1% of the time,
a home win 43.8%, a draw 24.8%. A single flat floor therefore means completely
different things in different markets. At a flat 72%, an "Over 1.5" selection
would be **weaker than simply assuming the league average**, while a 72% home
win would be a genuinely strong call. A flat floor is not a threshold; it is an
accident.

The rule used instead:

```
floor(market) = max(absolute_floor, base_rate + lift_k · (1 − base_rate))
absolute_floor = 0.65      lift_k = 0.25
```

The **relative** term makes the floor mean the same thing everywhere: the model
must be a quarter of the way from the market's own base rate to certainty. The
**absolute** term stops a low-base-rate market — an away win at 49% — from
qualifying as "high confidence" merely by beating its base rate.

Resulting floors, and the measured share of fixtures clearing each:

| Market | Base rate | Floor | Lift required | Fixtures clearing |
|---|---|---|---|---|
| Home win | 0.438 | 0.650 | +0.212 | 15.5% |
| Away win | 0.313 | 0.650 | +0.337 | 5.1% |
| Double chance 1X | 0.687 | 0.765 | +0.078 | 34.5% |
| Double chance X2 | 0.562 | 0.671 | +0.109 | 27.6% |
| Over 1.5 | 0.771 | 0.828 | +0.057 | 19.4% |
| Over 2.5 | 0.533 | 0.650 | +0.117 | 13.3% |
| Under 2.5 | 0.467 | 0.650 | +0.183 | 2.9% |
| BTTS | 0.535 | 0.651 | +0.116 | 2.2% |
| Draw | 0.248 | 0.650 | +0.402 | ~0% — never recommendable |

### 3.2 The supply sweep

Bookmaker closing odds were inverted into a goal matrix per match and every
market derived from it, exactly as the real system derives markets. Across
9,530 matches (2022/23–2025/26), varying the two parameters:

| absolute | lift_k | Qualifying fixtures | Median per 72h window | Windows with ≥ 7 |
|---|---|---|---|---|
| 0.60 | 0.15 | 83.6% | 17.0 | 63.5% |
| 0.65 | 0.15 | 78.2% | 15.9 | 62.5% |
| 0.65 | 0.20 | 73.7% | 15.0 | 61.7% |
| **0.65** | **0.25** | **66.3%** | **13.5** | **61.0%** |
| 0.65 | 0.30 | 57.9% | 11.8 | 60.4% |
| 0.70 | 0.25 | 60.5% | 12.2 | 60.1% |
| 0.70 | 0.30 | 55.4% | 10.9 | 59.8% |

The striking result: **the "≥ 7" column barely moves**, from 63.5% down to
59.8% while the floor tightens substantially. The binding constraint is the
fixture calendar, not the threshold — only 68.0% of windows contain seven
matches *at all* (§1). Tightening the floor costs batch *size*, not send
*frequency*, until it becomes severe.

That makes the choice comfortable rather than delicate: **0.65 / 0.25** is
selected because it sits near the calendar ceiling while requiring a real lift
in every market. Loosening to 0.15 would buy roughly two percentage points of
coverage in exchange for recommending materially weaker predictions.

### 3.3 The monoculture problem, and what diversity costs

Selecting each fixture's highest-probability qualifying market, with no
diversity constraint, produces batches that are **81% double chance** with a
median size of 20. Those genuinely are the highest probabilities available —
and as a product they are close to worthless, because one goal-model error
takes down the entire batch at once.

Applying the family caps from `config/recommendations.yaml` (double chance 30%,
totals 40%, result 40%), computed against the **actual** batch size and relaxed
by one slot before a viable batch is abandoned:

| | No caps | With family caps |
|---|---|---|
| Windows yielding ≥ 7 | ~61% | ~53% |
| Median batch size | 20 | 13 |
| Family mix | 81% double chance, 19% totals | 46% / 41% totals / 13% result |
| Otherwise-viable batches starved below 7 | — | **8.45% of windows** |

**Diversity costs about eight percentage points of send frequency, and that
cost is accepted.** A batch spread across three market families survives a
single model's drift in a way that a monoculture cannot. The number is recorded
here so the trade-off is a decision with a price attached rather than an
unexamined preference.

### 3.4 Honest caveat

Every figure in §3 comes from **bookmaker closing odds**, which price team news
and money this system will not have — they are measurably sharper than any
model here will be (log loss 0.956 against 1.071 for the base rate). Real
supply at these floors **will be lower**, so the system will be quieter than
these tables suggest.

These numbers are a ceiling and a method, not a final answer. The floors are
re-derived from the system's own walk-forward backtest before launch
([BACKTESTING.md](BACKTESTING.md)), and `research/threshold_study.py`
reproduces every figure above.

## 4. The rolling fixture window

```
now ──────────────────────────────────────────────▶ now + 72h
     │                                                   │
     └── all enabled competitions, one query ────────────┘
```

```sql
SELECT * FROM fixture
WHERE kickoff_utc >  :now
  AND kickoff_utc <= :now_plus_horizon
  AND status IN ('SCHEDULED','TIMED')
  AND competition_id IN (:enabled)
ORDER BY kickoff_utc;
```

One query across every enabled competition. There is no per-competition
scheduling, no gameweek grouping, and no notion of a "matchday batch" — a
single batch routinely spans the Premier League, La Liga, Serie A, the
Bundesliga and the Champions League, because that is simply what falls inside
the next 72 hours. `matchday` is carried into reporting as metadata and touches
nothing here.

The horizon is configurable (`prediction_horizon_hours: 72`). Widening it
increases batch size and lead time but reduces freshness, since a prediction
made 72 hours out cannot know Saturday's team news.

---

## 5. The decision function

Pure, deterministic, side-effect free, and exhaustively testable:

```python
def should_send_notification(
    now: datetime,               # UTC, from the injected clock
    upcoming_fixtures: Sequence[Fixture],
    qualifying_predictions: Sequence[ScoredRecommendation],
    previous_batches: Sequence[PredictionBatch],
    notified_selections: Sequence[NotifiedSelection],
    config: NotificationConfig,
) -> NotificationDecision: ...
```

Evaluated in order, short-circuiting on the first failure:

| # | Check | Outcome when it fails |
|---|---|---|
| 1 | Any supported fixture in the window? | `NO_FIXTURES` — silent, **not an error** |
| 2 | Any candidates surviving eligibility? | `NO_QUALIFYING_PREDICTIONS` |
| 3 | After duplicate suppression (§6), ≥ `min_recommendations`? | `INSUFFICIENT_QUALIFYING` |
| 4 | Predictions fresh enough (`max_prediction_age_hours`)? | `STALE_PREDICTIONS` — triggers a refresh, not a send |
| 5 | Cooldown since the last send elapsed? | `COOLDOWN_ACTIVE` |
| 6 | Daily batch cap not reached? | `DAILY_CAP_REACHED` |
| 7 | Enough *new* coverage versus the last batch? | `INSUFFICIENT_NEW_COVERAGE` |
| 8 | Selected matches still have ≥ minimum lead time? | drop those matches, re-check #3 |
| 9 | Is now within the send window around the optimal time? | `WAITING_FOR_SEND_WINDOW` |

Returning:

```json
{
  "should_send": true,
  "reason": "12 qualifying predictions across 5 competitions",
  "decision_code": "SEND",
  "prediction_count": 12,
  "match_count": 12,
  "competitions": ["premier_league", "la_liga", "serie_a", "bundesliga", "champions_league"],
  "window_start": "2026-09-20T12:00:00Z",
  "window_end": "2026-09-23T12:00:00Z",
  "recommended_send_time": "2026-09-20T16:30:00Z",
  "earliest_kickoff": "2026-09-20T17:00:00Z",
  "lead_time_hours": 0.5,
  "prediction_ids": [8812, 8817, 8840],
  "market_mix": {"OVER_1_5": 5, "HOME_WIN": 4, "BTTS": 2, "OVER_2_5": 1},
  "rejected_count": 31,
  "rejection_summary": {"BELOW_PROBABILITY_FLOOR": 19, "ALREADY_NOTIFIED": 7,
                        "MARKET_SHARE_CAP": 3, "STALE_FEATURES": 2}
}
```

The decision is persisted verbatim on `prediction_batch.decision_payload`,
including for skips. Every silence has a stored, inspectable reason — which is
what makes a mostly-quiet system debuggable.

---

## 6. Duplicate prevention

The tick runs every 30 minutes; a busy Saturday would otherwise produce the
same email repeatedly.

`notified_selection` is the ledger, keyed on
`(fixture_id, market_key, selection, batch_id)`. A candidate already present
for the same fixture and market is excluded **unless all three hold**:

1. `|p_new − p_sent| ≥ material_change_threshold` (default `0.10`);
2. `hours_since_sent ≥ min_resend_gap_hours` (default `6`);
3. the fixture has not kicked off.

A genuine re-send — a key striker ruled out, moving Over 2.5 from 71% to 58% —
is marked as an **update** in the subject line and shows the previous
probability beside the new one, so the reader sees a correction rather than
a contradiction.

**Batch-level novelty (check #7).** Even with per-selection dedup, a second
batch could be 90% the same fixtures with one addition. A new batch must
contribute at least `min_new_fixtures` (default `3`) fixtures not in the
previous batch, or `min_new_fixture_share` (default `0.30`).

---

## 7. Timing the send

```
TARGET_LEAD_TIME = 0.5h   (configurable)
```

A batch spans up to 72 hours, so exactly 0.5 hours of lead time for *every*
match is impossible. The anchor is the **earliest kickoff in the batch**:

```
recommended_send_time = min(kickoff for selected) − target_lead_time_hours
```

with:

- **A send window**, not an instant: the tick fires every 30 minutes, so the
  planner sends when `now ∈ [send_time − interval, send_time + grace]`
  (`grace` default 10 minutes). Without the window, a 30-minute tick would
  routinely step over a single target instant.
- **A freshness ceiling**: if the earliest kickoff is more than
  `max_prediction_age_hours` (default 12) after the newest prediction in the
  batch, refresh first, then send.
- **A late-arrival rule**: a match whose lead time has fallen below
  `min_lead_time_hours` (0.25) is dropped from the batch and the minimum-count
  check re-runs. A recommendation arriving after kickoff is worse than no
  recommendation.
- **Deliberate under-shooting for later matches.** Sunday's 19:00 match, in a
  batch anchored on Friday 18:00, gets ~49 hours of notice. That is the stated
  trade-off of batching, and the email presents kickoff times prominently so
  the reader can see it.

### Worked example (the brief's scenario)

Now: Friday 10:00. Eleven qualifying predictions, Friday 18:00 through Sunday
19:00. All inside the 72-hour horizon.

- Earliest kickoff: Friday 18:00 → `recommended_send_time` = **Friday 17:30**.
- Ticks from 10:00 to 17:00 return `WAITING_FOR_SEND_WINDOW`, each recording
  the eleven candidates.
- The tick at 17:30 sends one batch of eleven, spanning five competitions and
  three days.
- Subject: **"Football Predictions — Next 72 Hours"**. Not "Saturday Gameweek":
  the fixtures span competitions with different matchday numbering, and calling
  it a gameweek would be wrong in both directions.
- Saturday's ticks find the same eleven already in `notified_selection` and
  return `ALREADY_NOTIFIED`, unless new fixtures enter the window and clear the
  novelty check.

---

## 8. Sparse and empty periods

| Situation | Behaviour |
|---|---|
| 0 fixtures in 72h | `NO_FIXTURES`. No email, no "no matches" email, no error, no alert. Logged at INFO. |
| 3–6 qualifying | `INSUFFICIENT_QUALIFYING`. No email. Thresholds untouched. Re-evaluated next tick as fixtures enter the window. |
| 5 strong + 2 weak available | Send only if the 2 weak pass every gate on their own merits. They do not, so: **wait**. |
| International break | The 7–8 day empty stretches measured in §1. Silent throughout. |
| Summer off-season | Silent for weeks. A `last_activity` heartbeat in the log distinguishes "quiet" from "crashed". |
| Postponed after notification | `fixture_revision` detects it; a correction email is sent only if a notified match is affected. |
| Kickoff moved | Treated as material new information; the batch's lead-time check re-runs. |

The 15.7%-empty measurement is why these are tested paths rather than
afterthoughts.

---

## 9. Timezones

Every timestamp in the database is UTC. Server-local time is never consulted:
`clock.py` returns timezone-aware UTC, and a lint test bans naive
`datetime.now()` across the codebase.

Conversion happens **only in the render layer**, using the configured
`user_timezone` (default `Africa/Lagos`) via `zoneinfo`:

```
stored     2026-09-20T17:30:00Z
user tz    Africa/Lagos
displayed  Sunday 20 Sep, 18:30 WAT
```

Tests cover a DST transition in Europe against a non-DST user timezone — the
case where a European kickoff moves by an hour relative to the reader while UTC
stays put, which is the bug this arrangement exists to prevent.

---

## 10. Email

### Content

```
Subject: Football Predictions — Next 72 Hours (12 selections)

Prediction window: Sun 20 Sep 18:00 – Wed 23 Sep 22:00 WAT
12 selections across 5 competitions
Generated 20 Sep 17:30 WAT · model goals_dc_v3 · features fs_2026_09_20_a

 1. Liverpool vs Newcastle                        Premier League
    Kickoff      Sat 18:00 WAT (in 30 minutes)
    Market       Over 1.5 Goals
    Probability  87%          Confidence  High
    Expected goals  2.9 (Liverpool 2.0, Newcastle 0.9)
    Supporting   + Combined attacking output above league average (last 5)
                 + Opponent conceding 1.8/match away (last 10)
                 + League baseline for this market: 78%
    Risk         − Newcastle's last 3 matches averaged 1.7 total goals
    Prediction made 20 Sep 17:30 WAT · model goals_dc_v3
```

Every recommendation carries match, competition, kickoff in the user's
timezone, market, probability, confidence band, supporting and risk factors,
model version and prediction timestamp.

Each factor line is generated from a `prediction_explanation` row — a
`feature_key`, its value, the reference it is compared against, and a fixed
template. There is no free-text path, so the system cannot produce a
plausible-sounding reason that no feature supports.

Every email ends with the uncertainty notice: these are probability estimates
from a statistical model, not predictions of certain outcomes, and a stated 87%
estimate is expected to be wrong roughly one time in eight.

### Delivery

`EmailProvider` protocol with `send(message) -> DeliveryResult`.
Implementations: `SMTPProvider` **[MVP]**, `ConsoleProvider` (development),
`FileProvider` (tests). An API provider can be added without touching the
planner.

Multipart HTML plus a plain-text alternative generated from the same data
structure — never by stripping tags from the HTML. Inline CSS, table-based
layout, no external images, so it renders in Gmail and Outlook.

Retries: 3 attempts, exponential backoff with jitter, every attempt written to
`notification_log`. The batch reaches `SENT` only on provider confirmation;
otherwise `FAILED`, and the ledger is **not** written — so the next tick may
legitimately retry the same content rather than losing the batch.

Send is the last step, after the batch is persisted, so a crash between
persisting and sending is recoverable and cannot produce a silent double-send.

---

## 11. Configuration

```yaml
notifications:
  enabled: true
  min_recommendations: 7
  max_recommendations: 20
  prediction_horizon_hours: 72
  target_lead_time_hours: 0.5
  min_lead_time_hours: 0.25
  send_window_grace_minutes: 10
  scheduler_interval_minutes: 30
  cooldown_hours: 12
  max_batches_per_day: 2
  max_prediction_age_hours: 12
  material_change_threshold: 0.10
  min_resend_gap_hours: 6
  min_new_fixtures: 3
  min_new_fixture_share: 0.30
  user_timezone: Africa/Lagos
```

## 12. Test matrix

Counts, per the brief:

| Qualifying | Expected |
|---|---|
| 0 | no email, `NO_FIXTURES` or `NO_QUALIFYING_PREDICTIONS` |
| 3 | no email, `INSUFFICIENT_QUALIFYING` |
| 6 | no email, `INSUFFICIENT_QUALIFYING` |
| 7 | eligible, sends 7 |
| 12 | sends 12 |
| 20 | sends 20 |
| 25 | sends the best 20 |
| 100 | sends 20, caps respected |

Behavioural:

- same fixtures already notified → suppressed;
- notified fixture's probability moves 0.10 after 6h → update permitted;
- notified fixture's probability moves 0.03 → suppressed;
- international break (real 2024-11-11 → 11-19 calendar) → silence throughout;
- fixture postponed after notification → correction path;
- kickoff moved earlier, below minimum lead time → dropped, count re-checked;
- 6 qualifying at T, a 7th enters the window at T+30min → sends at T+30min;
- clock advanced past every kickoff → no send, no crash;
- 25 candidates of which 22 are Over 1.5 → market-share cap holds, and the
  batch is still ≥ 7 or is skipped.

Property tests: the selected count is never < 7 unless zero, never > 20; no
selection lacks a stored explanation; no fixture appears twice in a batch; the
decision function is deterministic given identical inputs.
