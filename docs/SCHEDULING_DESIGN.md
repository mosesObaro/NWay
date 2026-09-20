# Scheduling Design

> **Status:** implemented. This document is the design; the code that
> realises it lives under `src/nway/`. Where the two differ, the code is
> the source of truth and this document is a bug.

---

## 1. One idempotent tick, not one job per fixture

Per-fixture cron entries would mean thousands of jobs a season, each needing
rescheduling when a kickoff moves, and no way to reason about the system's
state. Instead a single command evaluates the world:

```bash
nway tick
```

invoked every `scheduler_interval_minutes` (default 30) by `launchd` on macOS,
or `cron`. In development, APScheduler runs the same function in-process.

**Idempotency is the core property.** Two ticks in the same minute must produce
identical state and at most one email. It is achieved by:

- a file lock (`data/nway.lock`) so overlapping ticks cannot interleave;
- natural keys and `INSERT ... ON CONFLICT DO NOTHING` on every ingestion write;
- `prediction_run`'s uniqueness on `(fixture_id, prediction_timestamp,
  model_version)`, with `prediction_timestamp` quantised to the tick boundary
  so a retry within the same tick reuses the run rather than creating a twin;
- `notified_selection` as the send ledger, written inside the send transaction.

A crash at any point leaves a state the next tick can resume from.

## 2. What a tick does

```
 1  acquire lock (exit quietly if held)
 2  refresh fixtures for enabled competitions        [ingestion]
 3  detect fixture changes -> fixture_revision       [postponements, kickoff moves]
 4  collect results for finished fixtures            [ingestion]
 5  settle markets for newly finished fixtures       [validation]
 6  refresh statistics if the source has updated     [ingestion]
 7  run data-quality checks                          [monitoring]
 8  select fixtures due for prediction/refresh       [refresh ladder, §3]
 9  compute features + predict + calibrate + store   [prediction]
10  score candidates                                 [recommendation]
11  should_send_notification(...)                    [planner]
12  if SEND: build batch, persist, render, deliver, log
13  update evaluation aggregates (throttled)         [evaluation]
14  release lock
```

Steps 2–7 run even when no prediction is due — result collection and settlement
must continue through international breaks. Steps 8–12 are frequently no-ops,
which is normal.

Each step is independently retryable and logs a structured record. A failure in
step 6 does not prevent step 5 from having committed.

## 3. The refresh ladder

Predictions are refreshed as legitimate pre-match information changes:

| Stage | Trigger | Purpose |
|---|---|---|
| `T48` | first tick with kickoff ≤ 48h away | initial prediction; enters the candidate pool |
| `T24` | kickoff ≤ 24h | absorbs midweek results and new statistics |
| `T8` | kickoff ≤ 8h | absorbs late team news |
| `FINAL` | kickoff ≤ `target_lead_time + interval` | the prediction that goes in the email |
| `ADHOC` | material change detected | postponement, kickoff move, significant availability change |

Each stage writes a **new** `prediction_run`; the previous one is marked
`superseded_by` and never modified. The full history of what the system
believed, and when, is preserved — which is what makes the refresh ladder
auditable rather than merely a cache-invalidation strategy.

A refresh is skipped when no input has changed: if `snapshot_id` watermarks and
the feature hash are identical to the previous run, the run is reused. On a
quiet Tuesday most ticks therefore do almost no work.

## 4. Cost per tick

With 10 competitions and the measured median of 21 fixtures in a 72-hour
window:

| Work | Per tick |
|---|---|
| football-data.org calls | ~10 (one window query per competition) + result polls |
| football-data.co.uk | only when the remote file's `ETag` changes (~2×/week) |
| Predictions computed | 0 on most ticks; ≤ ~30 at stage boundaries |
| Wall clock | seconds |

Against a 10 req/min budget this is comfortable. The HTTP layer's token bucket
enforces the limit regardless.

## 5. Failure handling

| Failure | Behaviour |
|---|---|
| Source unreachable | Retry with backoff; continue with cached data; raise a `WARN` quality check; after `max_source_outage_hours` features go stale and candidates fail the freshness gate — the system stops recommending rather than recommending on old data |
| Partial ingestion | Raw response stored with `parse_status='PARTIAL'`; affected fixtures flagged |
| Model artefact missing | Tick aborts at step 9, logs `ERROR`; ingestion and settlement still committed |
| Email provider down | `notification_log` records the failure; batch `FAILED`; ledger unwritten so the next tick may retry |
| Database locked | Lock file prevents concurrent ticks; SQLite WAL handles readers |
| Clock skew | All decisions use the injected UTC clock; a tick that observes time moving backwards logs `ERROR` and exits |

## 6. Operating modes

```bash
nway tick                              # the scheduled operation
nway tick --dry-run                    # decide and report, send nothing
nway tick --as-of 2026-03-14T13:30:00Z # replay a moment in history
nway ingest --source football_data_org --since 2026-08-01
nway train  --model goals_dixon_coles --through 2026-06-30
nway backtest --from 2023-08-01 --to 2025-05-31
nway report --window 90d
nway explain --prediction-id 8812
```

`--as-of` is the same code path as production with a different clock. The
backtester is not a parallel implementation, which is the only way to be sure
the thing being tested is the thing that runs.

## 7. launchd entry

```xml
<!-- ~/Library/LaunchAgents/com.nway.tick.plist -->
<key>ProgramArguments</key>
<array>
  <string>/Users/obaromoses/Documents/NWay/.venv/bin/python</string>
  <string>-m</string><string>nway.cli</string><string>tick</string>
</array>
<key>StartInterval</key><integer>1800</integer>
<key>WorkingDirectory</key><string>/Users/obaromoses/Documents/NWay</string>
```

The laptop will be asleep sometimes, and a missed tick is not a crisis: the
next tick re-evaluates the whole world. Only a tick missed inside a send window
loses a batch, and the ledger makes that visible rather than silent.
