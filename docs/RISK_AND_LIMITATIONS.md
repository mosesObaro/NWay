# Risks and Limitations

What this system cannot do, where it will be wrong, and what would have to
change for that to improve. Written before implementation so the claims stay
honest once results start arriving.

---

## 1. Statistical limitations

**Football is mostly noise.** The measured ceiling makes this concrete: over
21,544 matches, de-vigged bookmaker closing odds — aggregating team news,
professional modelling and real money — score a log loss of **0.956** against
**1.071** for simply predicting the league base rate. The entire predictable
signal available to a very well-informed participant is about **0.115 nats**.
A good student model might capture half of that. Any claim of dramatically
better performance is a bug, not a discovery.

**Sample sizes are small.** Nine seasons of seven leagues is ~21,500 matches.
Split by market, competition and probability bucket, individual cells fall to
the low hundreds. The Wilson interval on a 500-prediction market at an 80% hit
rate spans roughly ±3.5 percentage points — wide enough that most
season-to-season differences will be indistinguishable from chance. The
reporting layer refuses conclusions below 200 samples for this reason.

**Three backtest folds cannot separate skill from luck** at small effect sizes.
Reported improvements carry bootstrap intervals, and a model is promoted only
when it beats the incumbent by more than the standard error.

**Calibration drifts.** The 2020/21 season is the proof: home wins fell to
40.2% from 46.1% two seasons earlier, and mean home goal difference fell to
+0.155 from +0.356. A model trained through 2019/20 was systematically wrong
about the single largest effect in football for an entire season. It will
happen again, differently.

**Draws are barely predictable.** Draws run at 24.8% overall with little
variance across leagues (23.4%–26.5%). Models rarely assign a draw more than
~33%, so the draw market almost never clears a 72% floor. This is a property of
football, not a gap to be engineered away.

## 2. Data limitations

**No true xG.** Verified: FBref's terms prohibit building tools on scraped
data, and Understat's `robots.txt` disallows all crawlers. The `sxg_proxy` uses
shot counts and accuracy, and cannot distinguish a tap-in from a thirty-yard
shot. It will be meaningfully worse than commercial xG, and the documentation
never calls it xG.

**No player data.** Therefore no player markets at all. This removes a whole
family the brief describes, for a reason that is external to the project.

**No Europa or Conference League** on any free tier. The MVP covers seven
leagues plus the Champions League.

**Statistics lag up to ~3 days.** football-data.co.uk updates roughly twice
weekly. Corner and card features for a Saturday fixture may exclude the
preceding midweek round. The backtest models this with a 72-hour knowledge lag;
if that estimate is wrong in either direction, the corners and cards models are
respectively understated or leaking.

**Eredivisie and Primeira Liga have statistics only from 2017/18** — about
2,750 matches each against ~9,900 for the Premier League. Corner and card
models for those leagues rest on a third of the evidence, which is why market
eligibility is per league.

**Referee identity is available only for the Premier League.** The other six
card models lack the single most predictive feature for cards.

**No possession, no attendance, no injury data** in any free source.

**Single-source dependence.** football-data.org is the only free feed covering
all ten competitions with UTC timestamps. If it changes terms, is rate-limited
harder, or disappears, the live system stops. openfootball is the documented
fallback and lags by days.

## 3. Modelling limitations

- **Dixon–Coles assumes conditional independence** between the two sides' goal
  counts beyond the low-score correction. Real matches have game states — a
  team two down attacks more — that the model cannot represent.
- **Team strength is a scalar pair.** Style mismatches, tactical matchups and
  manager changes are invisible. A new manager's first match is modelled with
  the previous manager's ratings.
- **No in-play information.** A red card in the 20th minute makes the
  pre-match distribution obsolete, and this is a pre-match system.
- **Promoted teams are a structural weakness.** No competition history, and
  they arrive in August when the fixture calendar is dense.
- **Knockout football differs from league football.** Two-legged ties with
  aggregate scores create incentives (a 2-0 first-leg lead changes second-leg
  behaviour) that a model trained mostly on league matches will not capture.
  With eight UEFA knockout ties a season, there will never be enough data to
  learn it — an acknowledged, unfixable limitation at this scale.
- **The `sxg_proxy` may be actively misleading** for teams whose shot profile is
  unusual — a side taking many low-quality long-range shots will be flattered.

## 4. Recommendation and selection risks

**Selection bias in reported performance.** Recommended predictions are chosen
for high probability and good historical reliability, so their hit rate is not
an unbiased estimate of model quality. Both are reported: all predictions, and
recommended ones separately.

**The reliability score is circular early on.** The Beta posterior needs
settled history, and at launch there is none — so the prior does all the work
and the initial ranking is close to ranking by calibrated probability. It takes
a season of settled predictions before the score earns its keep. The backtest
seeds it, with the caveat that backtest reliability is not live reliability.

**Common-mode market risk.** At any workable threshold, Over 1.5 dominates the
candidate pool — 72.3% of candidates at a 72% floor. The market-share cap
limits concentration, but a systematic goal-model drift still degrades most
selections simultaneously.

**The thresholds are derived from a proxy.** The per-market floors come from
inverting bookmaker odds, which are measurably sharper than this system's model
will be (log loss 0.956 against 1.071 for the base rate). Real qualifying
supply will be lower, so the system will be quieter than the tables predict.
The floors are re-derived from the system's own backtest before launch.

**Diversity caps cost coverage.** Measured: family caps drop send-eligible
windows from ~61% to ~53% and starve 8.45% of otherwise-viable batches below
seven. That is a deliberate trade against common-mode model risk, but it is a
real cost, and if the live system proves too quiet the caps are the first
parameter to revisit — not the probability floors.

**The minimum-7 rule can conflict with quality.** By design, quality wins and
the system stays silent. If it is silent for most of a season, that is
information about the model, not a reason to lower the floor — and the
temptation to lower it will be strong.

## 5. Operational risks

| Risk | Mitigation |
|---|---|
| Laptop asleep at a send window | Ticks are stateless re-evaluations; only a missed send window loses a batch, and the skip is logged |
| SQLite single-writer contention | Lock file plus WAL; the workload is tiny |
| Clock skew or timezone bug | Single injected UTC clock; naive-datetime lint; DST tests |
| Email in spam | SMTP with a properly configured sender; delivery logged |
| Silent failure mistaken for quiet | Heartbeat log line every tick distinguishes quiet from dead |
| API token leak | `.env` only, never in YAML, DB or logs; redaction tested |
| Duplicate emails | Ledger written inside the send transaction; idempotency tested |
| Unnoticed data corruption | Quality checks with `BLOCKING` severity halt prediction |

## 6. Scope risks

The brief describes roughly forty markets across six families. The MVP ships
six: 1X2, Over 1.5, Over 2.5, BTTS, expected goals and goal margin. That is not
under-delivery; it is the set for which data quality, sample size and
validation can actually be demonstrated. A system with forty markets and no
calibration evidence for any of them would be a worse project.

The largest schedule risk is Phase 3. The as-of feature store with a working
leakage harness is the hardest part and cannot be shortened, because every
later number depends on it being right.

## 7. Ethical and legal

- **Not gambling advice.** Outputs are probability estimates with quantified
  uncertainty. Every email carries the uncertainty notice, and no output is
  framed as a tip or a guarantee.
- **Source terms are respected.** FBref and Understat are excluded on their
  terms and robots directives rather than worked around. That costs the project
  xG, and the cost is accepted.
- **No redistribution** of raw provider data.
- **Politeness.** Rate limiting, conditional requests, caching and an
  identifying `User-Agent` on every client.
- **Personal data.** Only the recipient's email address and timezone. Player
  statistics, when eventually added, are public professional performance data.

## 8. What would most improve the system

In order of measured impact per unit of effort:

1. **A licensed xG feed.** Shot quality is the largest single modelling gap,
   and `sxg_proxy` is a weak substitute.
2. **Lineup and injury data before kickoff.** Closing odds beat pre-match
   models largely because they price team news.
3. **More seasons for the Eredivisie and Primeira Liga**, which would move
   corners and cards from gated to enabled.
4. **A live season of settled predictions**, which is the only thing that makes
   the reliability score meaningful and lets the threshold be set from the
   system's own behaviour rather than a proxy.
5. **Referee data beyond England**, which would roughly double the card model's
   usable signal in six leagues.
