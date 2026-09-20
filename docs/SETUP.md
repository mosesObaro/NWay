# Setup

Everything below was verified on macOS with Python 3.12.4 and SQLite 3.53.

## 1. Install

```bash
git clone https://github.com/mosesObaro/NWay.git
cd NWay
make install          # creates .venv and installs dependencies
.venv/bin/pip install -e .
```

Dependencies are deliberately few: numpy and scipy for the goal model, PyYAML
for configuration, pytest for the tests. Everything else — the database, the
HTTP client, email composition, timezone handling — is standard library.

## 2. Credentials

```bash
cp .env.example .env
```

Then, at any point, ask the system whether it is configured correctly:

```bash
.venv/bin/nway check
```

It verifies each credential against the real API — it does not merely check
that a variable is set — and prints where to get anything missing. It never
prints a key, only whether it works.

Two keys are needed for live operation. Neither is needed to run the tests or
the demo.

### football-data.org (fixtures, kickoff times, results)

Register at <https://www.football-data.org/client/register> for a free token.
The free tier covers all seven domestic leagues plus the Champions League, at
10 requests per minute — comfortably more than the ~12 requests each tick uses.

```
NWAY_FOOTBALL_DATA_ORG_TOKEN=your_token
```

### Resend (email delivery)

Create an API key at <https://resend.com/api-keys>.

```
NWAY_RESEND_API_KEY=re_xxxxxxxxxxxx
NWAY_EMAIL_FROM=NWay Predictions <predictions@yourdomain.com>
NWAY_EMAIL_TO=you@example.com
```

**The sender domain must be verified in Resend.** Until it is, Resend accepts
only its onboarding sender (`onboarding@resend.dev`) and delivers only to the
address on your own Resend account. For a school project that is usually
enough:

```
NWAY_EMAIL_FROM=NWay Predictions <onboarding@resend.dev>
NWAY_EMAIL_TO=your-resend-account-email@example.com
```

To switch providers, change `notifications.email.provider` in
`config/notifications.yaml` to `smtp`, `console` or `file`. The planner never
imports a provider, so nothing else changes.

## 3. Load history and train

```bash
.venv/bin/nway init-db
.venv/bin/nway ingest --history --seasons 2017/18..2025/26   # ~5 minutes
.venv/bin/nway train --through 2026-06-30
```

The ingest is rate-limited to one file every five seconds out of politeness to
a free source; 63 files take roughly five minutes. Expect ~21,500 matches.

Training prints the held-out log loss beside the measured baseline. It must
beat **1.0711** (the pooled base rate). A score below **0.95** would beat
de-vigged bookmaker closing odds, which is implausible for this feature set and
is reported as a leakage alarm rather than a result.

## 4. See it work, without credentials

```bash
.venv/bin/nway demo
```

This copies the database, rewinds to a real historical weekend, presents that
weekend's fixtures as upcoming, hides their results from the as-of repository,
and runs the production tick against them. It prints the email to the terminal
and never touches your real database or sends anything.

## 5. Run it live

```bash
.venv/bin/nway ingest              # live fixtures and results
.venv/bin/nway tick --dry-run      # decide and report, send nothing
.venv/bin/nway tick                # decide, and send if the rules are satisfied
```

Then schedule it:

```bash
cp deploy/com.nway.tick.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nway.tick.plist
```

Edit the paths in the plist first. On Linux, use `deploy/crontab.example`.

## 6. What to expect

**Most ticks send nothing, and that is correct.** Measured on the 2024/25
calendar, a rolling 72-hour window across the seven leagues was completely
empty 15.7% of the time — runs of seven to eight days during FIFA
international windows — and only 68% of windows held seven matches at all.

Every decision, including every silence, is stored with its reason:

```bash
.venv/bin/nway status               # recent decisions, model, counts
.venv/bin/nway report --window 90   # calibration and hit rate once settled
.venv/bin/nway explain --prediction-id 1234
```

## 7. Tests

```bash
make test           # the full suite
make test-leakage   # the temporal-leakage suite alone
```

The leakage suite is the one to run after any change to features or storage.
It includes a canary — a deliberately leaky feature that the harness *must*
catch — so that the suite cannot quietly decay into one that passes because it
has stopped checking.
