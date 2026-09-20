# Data Sources

Every claim below was verified on **2026-09-20** by fetching the source, not
from memory. Reproduce with `python3 research/feasibility_study.py --download`.

Labels: **[MVP]** required to ship · **[Recommended]** ship soon after ·
**[Future]** deferred, usually behind a paywall.

---

## 1. Summary of the decision

| Need | Source chosen | Why |
|---|---|---|
| Live fixtures, kickoff times (UTC), results, standings | **football-data.org** free tier | Only free source covering 7/7 leagues + UCL with UTC timestamps and fast updates |
| Historical match statistics (shots, SoT, corners, fouls, cards, HT scores) | **football-data.co.uk** CSV | Only free source with complete per-match stats and deep history |
| Cross-check / fallback fixtures | **openfootball/football.json** | Public domain, but lags several days |
| Weather at kickoff | **Open-Meteo** | Free non-commercial, no key, and archives forecasts (leak-safe) |
| Europa League / Conference League | *Gap — see §7* | Not available free |
| True xG | *Gap — see §8* | No free, terms-compliant source exists |
| Player-level data, lineups, injuries | *Gap — see §9* | Free tiers cannot sustain the request volume |

The MVP therefore covers **seven domestic leagues plus the Champions League**,
and the goals/result/BTTS/corners/cards market families. Player markets and the
two secondary UEFA competitions are explicitly out of MVP scope, for documented
data reasons rather than effort reasons.

---

## 2. football-data.org **[MVP — primary fixture & result feed]**

- **Access**: REST, `https://api.football-data.org/v4/`, header `X-Auth-Token`.
- **Auth**: free API token by email registration.
- **Verified rate limits** (from the published policy page and live response
  headers): free plan **10 requests/minute**; Standard 30/min; Premium+ 60/min.
  Unauthenticated clients get **100 requests / 24h** and can read only the
  areas and competitions lists. Live responses carry `X-Requests-Available`
  and `X-RequestCounter-Reset`.
- **Timestamps**: `utcDate`, genuinely UTC. This is the system's scheduling
  source of truth.
- **Verified coverage** — the free tier is `TIER_ONE`, and a live call to
  `/v4/competitions` on 2026-09-20 returned 190 competitions, of which exactly
  12 are `TIER_ONE`:

  | Target competition | API code | Plan | Seasons available |
  |---|---|---|---|
  | Premier League | `PL` | TIER_ONE (free) | 128 |
  | Primera División | `PD` | TIER_ONE (free) | 95 |
  | Bundesliga | `BL1` | TIER_ONE (free) | 64 |
  | Serie A | `SA` | TIER_ONE (free) | 95 |
  | Ligue 1 | `FL1` | TIER_ONE (free) | 83 |
  | Eredivisie | `DED` | TIER_ONE (free) | 71 |
  | Primeira Liga | `PPL` | TIER_ONE (free) | 78 |
  | UEFA Champions League | `CL` | TIER_ONE (free) | 47 |
  | UEFA Europa League | `EL` | **TIER_TWO (paid)** | 10 |
  | UEFA Conference League | `UCL` | **TIER_FOUR (paid)** | 6 |

- **Naming trap, must be encoded in the competition registry**: this provider
  uses `CL` for the **Champions** League and `UCL` for the **Conference**
  League. A reader who assumes `UCL` means "UEFA Champions League" will silently
  ingest the wrong competition. See `config/competitions.yaml`.
- **Fields**: competitions, seasons, matchdays, teams, squads, fixtures with
  UTC kickoff, status (`SCHEDULED`/`POSTPONED`/`FINISHED`/...), full-time and
  half-time scores, standings, scorers. Detailed match statistics are **not**
  in the free tier (sold as a €15/mo Statistic add-on).
- **Reliability**: good. Status transitions and postponements are reflected,
  which the fixture-change detector depends on.
- **Limitations**: no shots/corners/cards free; no xG; no lineups on free tier.
- **Rate-limit budget**: 10 req/min is ample. The planned tick uses ~12
  requests per 30-minute cycle (one fixture-window call per competition plus
  result polling), i.e. under 600/day.

## 3. football-data.co.uk **[MVP — primary historical statistics]**

- **Access**: static CSV, `https://football-data.co.uk/mmz4281/<SSSS>/<DIV>.csv`
  (e.g. `2526/E0.csv`). **Requests to `www.` return HTTP 302** to the apex
  domain — follow redirects or every download silently yields an empty file.
- **robots.txt**: `User-agent: * / Disallow:` — crawling is permitted.
- **Terms**: the site offers the data as free downloads and carries no
  machine-readable licence. Treat it as *free to use for a personal/academic
  project, not to redistribute*. Do not republish the raw CSVs.
- **Update frequency**: the homepage carried "Updated: 17/09/26" when checked on
  2026-09-20, i.e. roughly twice weekly, **lagging by up to ~3 days**. This is
  why it is a training/feature source and never the live fixture feed, and why
  the system tracks `feature_staleness_hours` as a recommendation gate.
- **Fields**: `Div, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG,
  HTAG, HTR, Referee, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR`, plus
  ~100 columns of bookmaker odds. 132 columns total for `E0` in 2025/26.
- **Verified completeness, 2025/26 season**: shots, shots on target, corners,
  fouls, cards and half-time scores are **100% populated in all seven
  leagues**. `Referee` is present **only for the Premier League**.
  `Attendance` is not present in the current format.
- **Verified historical depth** (Y = shots *and* corners present):

  | League | Stats available from | Matches/season |
  |---|---|---|
  | Premier League (`E0`) | 2000/01 | 380 |
  | Bundesliga (`D1`) | 2000/01 | 306 |
  | La Liga (`SP1`) | 2005/06 | 380 |
  | Serie A (`I1`) | 2005/06 | 380 |
  | Ligue 1 (`F1`) | shots 2005/06, **corners only from 2010/11** | 380 → 306 from 2023/24 |
  | Eredivisie (`N1`) | **2017/18** | 306 |
  | Primeira Liga (`P1`) | **2017/18** | 306 |

  Half-time scores go back much further (1995/96 for `E0`/`D1`) — half markets
  are not constrained by the statistics gap.

- **Consequence for the design**: a corners or cards model for the Eredivisie
  or Primeira Liga has ~9 seasons (~2,750 matches) of history, against ~26
  seasons for England. Market eligibility is therefore **per league**, gated on
  a minimum training-sample rule, not enabled globally. See
  `config/markets.yaml`.
- **Known data traps** (each needs an ingestion test):
  - Files are **latin-1**, not UTF-8.
  - The first header cell carries a UTF-8 BOM, parsing as `﻿Div`.
  - `Date` is `dd/mm/yy` in older files and `dd/mm/yyyy` in newer ones.
  - `Time` is **local kickoff time, not UTC** — never use it for scheduling.
  - Truncated seasons exist: Ligue 1 2019/20 has 279 rows and the Eredivisie
    2019/20 has 232 (COVID abandonment). Ligue 1 dropped to 306 matches from
    2023/24 (18 teams). Row-count assertions must be per season, not constant.
  - Team names are provider-specific (`Man United`, `Nott'm Forest`) and must
    go through entity resolution against the football-data.org names.

## 4. openfootball / football.json **[Recommended — cross-check]**

- **Access**: raw GitHub JSON, e.g.
  `https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/en.1.json`.
  No key.
- **Licence**: public domain.
- **Verified coverage**: the `2026-27` directory holds exactly the seven target
  leagues (`de.1, en.1, es.1, fr.1, it.1, nl.1, pt.1`) with `round`, `date`,
  `time`, teams and half-time/full-time scores.
- **Verified freshness problem**: on 2026-09-20 the Premier League file's most
  recent played match was **2026-09-14** — a ~6-day lag. Fine as an independent
  cross-check for result disagreements; unusable as a live feed.
- **`openfootball/champions-league` is stale**: last pushed 2026-07-02 with no
  `2026-27` directory. Do not rely on it for the current UCL season.

## 5. Open-Meteo **[Recommended — match context]**

- **Access**: REST, no API key for non-commercial use.
- **Licence**: free for non-commercial use; a key is only required for
  commercial resources. A school project qualifies.
- **Three distinct endpoints, and the choice matters for leakage**:
  - *Forecast API* — what will happen. **This is what the live pipeline uses**,
    because it is what was knowable at prediction time.
  - *Historical Forecast API* — archived model runs stitched into an hourly
    series, from ~2022 (ECMWF IFS from 2017). **This is what backtests use.**
  - *Historical Weather API (ERA5 reanalysis)* — what actually happened, back
    to 1940. **Never use this for features.** Actual kickoff weather was not
    knowable beforehand; training on it is textbook leakage.
- **Variables**: temperature, precipitation, wind speed/gusts, cloud cover,
  humidity, hourly.
- **Limitation**: needs stadium coordinates, which no free feed supplies
  cleanly. A hand-curated `venues.yaml` (about 140 clubs) is required, which is
  why weather is Recommended rather than MVP.

## 6. API-Football (api-sports.io) **[Future — gated on budget]**

- **Free tier**: **100 requests/day, 10/minute**, all endpoints, but
  **historical seasons are restricted** on the free plan (the vendor states the
  restriction without publishing the exact range — treat depth as unverified).
- **Would provide**: Europa League, Conference League, lineups, injuries,
  player statistics, some statistics endpoints.
- **Why it is not in the MVP**: the seven leagues plus UCL produce roughly
  2,500 fixtures a season. Lineups alone are one request per fixture, before
  refresh cycles. 100 requests/day cannot sustain it, and the season
  restriction makes it unsuitable for training data. It becomes viable at the
  paid tier.

## 7. Gap: UEFA Europa League and Conference League

Both are outside every free tier that was checked. Options, in the order they
should be considered:

1. **Ship the MVP with 7 leagues + Champions League.** Verified free, and the
   fixture-density study shows this already clears the minimum-7 rule
   comfortably. *Recommended.*
2. football-data.org Standard (€49/mo) adds `EL`; Conference League is
   `TIER_FOUR` and needs a higher plan.
3. API-Football paid tier covers both plus lineups and injuries.

The competition registry is configuration-driven precisely so that adding these
later is a YAML edit plus a provider mapping row — no code change.

## 7b. Gap: UEFA Nations League and Women's Champions League

Both are registered in `config/competitions.yaml` and both are disabled.
Verified 2026-09-20.

**UEFA Nations League (`UNL`)** exists on football-data.org but is `TIER_FOUR`;
a direct request with a free-tier token returns **HTTP 403**. Buying access
would not be enough on its own. It is national-team football: the goal model is
fitted on club sides, so no national team has a rating and every one would fall
back to the competition average. Teams play roughly ten matches a year in
windows months apart, which breaks the rolling-form features the system is
built on, and football-data.co.uk publishes no international fixtures, so there
is no statistics history to train a replacement on. It needs its own model
trained on international results.

**UEFA Women's Champions League** has no source at all in this stack:

| Source | Result |
|---|---|
| football-data.org | Not listed at any tier. Its only women's entry is `ECF`, the Women's Euro — a national-team tournament |
| football-data.co.uk | No women's divisions (see the trap below) |
| openfootball | No women's repository among its 36 |
| StatsBomb open data | Carries FA WSL, Frauen Bundesliga, Liga F, NWSL, Serie A Women and the Women's Euro — **not** this competition, and only showcase seasons |

Even with fixtures it could not use the existing model, for the same reason:
women's club sides share no ratings with the men's club sides it was fitted on.

**A trap worth naming.** football-data.co.uk answers some unknown division
codes with *another division's file*, byte for byte: a request for `I1W`
returns men's Serie A with HTTP 200 and an identical SHA-256 to `I1`. Clearly
invalid codes (`ZZ9`, `WCL`) do 404, so this only affects near-misses — which
is exactly what a speculative women's or second-tier code looks like. The `Div`
column is empty in current files, so ingestion cannot validate against it and
instead compares content across divisions within a season, raising a
`BLOCKING` quality check when two codes return the same fixtures.

Adding either competition later is a configuration change plus a separately
trained model, not a code change. The `model_scope` field on each competition
records which model may serve it, and a model refuses to predict a scope it was
not fitted on.

## 8. Gap: expected goals (xG)

There is **no free, terms-compliant, continuously-updated xG source** for these
leagues. Verified:

- **FBref / Sports-Reference** — the terms of use prohibit automated access that
  affects the site, and separately prohibit creating tools based on scraped data
  or using it to train models without written permission. **Excluded.** The
  published limit (10 requests/minute, session blocked for up to a day on
  violation) is moot given the terms.
- **Understat** — `robots.txt` is `User-agent: * / Disallow: /`, a blanket
  crawl prohibition. **Excluded.**
- **StatsBomb open data** — permissively licensed with an attribution
  requirement, but verified to be 80 showcase competition-seasons: Premier
  League 2003/04 and 2015/16 only, Serie A 1986/87 and 2015/16, La Liga
  weighted to historic Barcelona matches, nothing current. **Unusable as a
  feed**; useful for learning event data.
- Paid options (Sportmonks, API-Football Pro, TheStatsAPI) carry xG from ~€19–50/mo.

**Design response — do not fake it.** The MVP derives a **shot-based expected
goals proxy** from the shots and shots-on-target that football-data.co.uk
actually provides, and names it `sxg_proxy` everywhere — never `xg`. Its
definition, fitted weights and the size of its gap to true xG are documented in
[MODEL_DESIGN.md](MODEL_DESIGN.md). Any column named `xg` in this system means a
licensed provider's xG, and is null until one is purchased.

## 9. Gap: player data, lineups, injuries, suspensions

No free source supplies these at the volume required. Consequently **all player
markets are excluded from the MVP** — anytime scorer, first scorer, player
shots, shots on target, assists. This is a data decision, not a modelling one:
a player scoring model without expected minutes is not a model, it is a guess,
and the brief is explicit that player predictions must not be generated when the
underlying data is insufficient.

The schema still carries `player`, `player_match_stats` and
`player_availability` tables so that the upgrade is additive.

## 10. Bookmaker odds

football-data.co.uk ships historical closing odds. They are used for **exactly
two purposes**, both non-training:

1. As a **benchmark ceiling** in evaluation. Measured over 21,544 matches
   (2017/18–2025/26): de-vigged closing odds score **log loss 0.956 / multiclass
   Brier 0.567**, against **1.071 / 0.648** for the pooled base rate and
   **1.099 / 0.667** for uniform. The model must beat the base rate; the market
   is the practical ceiling.
2. As a **market-comparison column** in reporting, kept strictly separate from
   model probability, per the brief.

They are **never** a training target and never an input feature. The system must
function with the odds columns dropped entirely, and a test asserts that.

## 11. Rate-limit and politeness budget

| Source | Limit | Planned usage | Headroom |
|---|---|---|---|
| football-data.org | 10 req/min, free | ~12 req per 30-min tick | large |
| football-data.co.uk | none stated; robots permits | 7 files, twice weekly | large |
| openfootball | GitHub raw | 7 files, daily | large |
| Open-Meteo | fair use | ~1 req per fixture per refresh, cached | moderate |

All clients share one HTTP layer with a token-bucket limiter, exponential
backoff with jitter, a descriptive `User-Agent` including a contact address,
conditional requests (`ETag`/`If-Modified-Since`), and an on-disk response
cache keyed by URL and fetch time. Every raw response is persisted before
parsing so that ingestion is replayable — see [BACKTESTING.md](BACKTESTING.md).
