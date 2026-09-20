# Automation

> **Status:** implemented. `.github/workflows/predict.yml` runs the system
> unattended and schedules its own next run.

---

## 1. The problem

GitHub Actions has **no delayed dispatch**. A job cannot ask to be re-run in
four hours, and `schedule:` is a static cron committed to the default branch.

But the right cadence is anything but static. Measured on the 2024/25
calendar, a rolling 72-hour window across the seven leagues was **completely
empty 15.7% of the time**, in runs of seven to eight days during FIFA
international windows. As this is written the next kickoff is **19 days away**.
A fixed `*/30 * * * *` cron would run 912 times to do nothing.

## 2. The loop

Each run computes when the next one should happen and writes that down. The
next run reads it.

```
       ┌──────────────────────────────────────────────┐
       │                                              │
       ▼                                              │
  cron fires ──▶ gate: is it due? ──no──▶ exit (~30s) │
                      │ yes                           │
                      ▼                               │
        restore cache + committed ledger              │
                      ▼                               │
              ingest ▶ tick ▶ (email)                 │
                      ▼                               │
           nway schedule ──▶ next_run_at ─────────────┤
                      ▼                               │
     rewrite this workflow's cron, commit state ──────┘
```

Two distinct mechanisms, because either alone is fragile:

- **The cron is rewritten** to a dense window around the computed time, so the
  run actually fires when it should.
- **The gate** re-checks the recorded time at the start of every run, so a
  stale or over-eager cron costs thirty seconds rather than a wrong batch.

## 3. What decides the next run

`nway schedule` reads real fixture timestamps, never a fixed interval:

| Situation | Next run | Code |
|---|---|---|
| ≥ 7 fixtures in the 72h window, send window ahead | 20 min before the window opens | `SEND_WINDOW_APPROACHING` |
| Inside the send window | one tick interval (30 min) | `IN_SEND_WINDOW` |
| Fixtures in the window but below the minimum | two tick intervals | `BELOW_MINIMUM` |
| Fixtures exist but beyond the horizon | when the earliest enters the 72h window | `WAITING_FOR_FIXTURES` |
| No upcoming fixtures at all | 24 hours | `NO_FIXTURES_KNOWN` |

**The result is always clamped to between 10 minutes and 24 hours.** Never
less, because GitHub's schedules drift and a tight loop would thrash. Never
more, because a longer sleep would let the Actions cache expire (7 days idle),
let the scheduled workflow be disabled (60 days of repository inactivity), and
turn any bug in this function into something only a human can recover.

Today, during the break, it settles to one cheap run a day. On a matchday it
tightens to every fifteen minutes around the send window.

## 4. State, and why it lives in two places

| State | Size | Where | Why |
|---|---|---|---|
| Database, model, calibrators | ~15 MB | Actions cache | Too big to commit; rebuildable in ~6 minutes |
| Next run time + notified ledger | a few KB | `state/nway-state.json`, committed | **Must not be lost** |

The split matters. The Actions cache is evictable. If the ledger lived only
there, an eviction would produce a rebuilt database with no memory of what had
already been emailed — and the next run would **re-send recommendations the
user already received**. Keeping it in the repository means an evicted cache
costs a six-minute rebuild, not a duplicate email, and the git history doubles
as an audit trail of what went out and when.

This is also why `notified_ledger` exists alongside `notified_selection`. The
live table has foreign keys to fixtures and predictions, which is correct for
normal operation and is exactly what makes it unrestorable: after a rebuild
those rows have different ids. The ledger table carries no foreign keys, and
duplicate suppression reads the union of both.

## 5. Setup

Add four repository secrets under **Settings → Secrets and variables →
Actions**:

| Secret | Value |
|---|---|
| `NWAY_FOOTBALL_DATA_ORG_TOKEN` | free token from football-data.org |
| `NWAY_RESEND_API_KEY` | key from resend.com/api-keys |
| `NWAY_EMAIL_FROM` | `NWay Predictions <onboarding@resend.dev>` until a domain is verified |
| `NWAY_EMAIL_TO` | the address your Resend account is registered with |

`NWAY_CONTACT_EMAIL` is optional and goes in the outgoing User-Agent header.

**Until all four are set the workflow skips**, writes a note to the run
summary, and stays green. It does not thrash and does not turn the repository
red.

Then either wait for the daily safety-net cron, or start the loop by hand:
**Actions → Predict → Run workflow**, with `force_tick` ticked to bypass the
gate on the first run.

## 6. Guard rails

- **The safety-net cron `17 6 * * *` is never removed.** The rewriter
  reinstates it unconditionally, so a bad computed window cannot leave the
  workflow unscheduled.
- **Cron values are validated** before being written. Anything that is not five
  well-formed fields is rejected, so a malformed computed output cannot inject
  into the workflow file. Tested.
- **`concurrency: cancel-in-progress: false`.** Runs queue rather than cancel;
  a cancelled run would leave the schedule un-advanced and strand the loop.
- **The schedule is recomputed with `if: always()`**, so a failed tick still
  advances the loop.
- **A corrupt or missing state file fails open** — the gate runs rather than
  deadlocking.
- **The commit rebases before pushing**, never force-pushes.
- Rewriting the workflow does not re-trigger it: `push` is not a trigger.

## 7. Known limitations

- **GitHub's scheduled runs are routinely late**, sometimes 15 minutes or more
  under load. The 0.5-hour target lead time absorbs this, and the computed
  window starts 20 minutes early, but a very late run can still miss a send
  window. The batch is not lost — the next run picks it up — but the lead time
  will be shorter than intended.
- **A changed cron takes time to propagate.** GitHub reads schedules from the
  default branch, and a newly committed schedule is not always honoured
  immediately. The daily safety net covers the gap.
- **Scheduled workflows are disabled after 60 days of repository inactivity.**
  The daily run commits state often enough to prevent this, but only while the
  loop is healthy.
- **Actions minutes are free on public repositories.** On a private repository,
  budget for roughly one 30-second gate run a day during breaks and a handful
  of 3-minute runs on a matchday.

## 8. Running it locally instead

The workflow is a convenience, not a requirement. `deploy/com.nway.tick.plist`
runs the same `nway tick` under launchd on a Mac, and
`deploy/crontab.example` under cron. Those use a fixed interval, since a
machine that is already awake has no reason to self-schedule.
