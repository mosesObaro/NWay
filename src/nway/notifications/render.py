"""Email rendering.

HTML and plain text are generated from the same data structure. The text part
is never produced by stripping tags from the HTML -- that yields something that
reads like broken HTML rather than a readable email, and it is the difference
between a fallback that works and one that merely exists.

Timezone conversion happens here and nowhere else. Everything upstream is UTC.
"""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import dataclass
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from nway import clock
from nway.storage.db import Database

MARKET_LABELS = {
    "HOME_WIN": "Home Win", "DRAW": "Draw", "AWAY_WIN": "Away Win",
    "DOUBLE_CHANCE_1X": "Home Win or Draw", "DOUBLE_CHANCE_X2": "Draw or Away Win",
    "OVER_0_5": "Over 0.5 Goals", "OVER_1_5": "Over 1.5 Goals",
    "OVER_2_5": "Over 2.5 Goals", "OVER_3_5": "Over 3.5 Goals",
    "UNDER_2_5": "Under 2.5 Goals", "BTTS": "Both Teams To Score",
    "HOME_CLEAN_SHEET": "Home Clean Sheet", "AWAY_CLEAN_SHEET": "Away Clean Sheet",
    "HOME_TO_SCORE": "Home Team To Score", "AWAY_TO_SCORE": "Away Team To Score",
}

UNCERTAINTY_NOTICE = (
    "These are probability estimates from a statistical model, not predictions "
    "of certain outcomes. A stated 85% estimate is expected to be wrong roughly "
    "one time in seven. Nothing here is advice."
)


@dataclass
class RenderedEmail:
    subject: str
    html: str
    text: str


def _local(moment: dt.datetime, timezone: str) -> dt.datetime:
    return clock.ensure_utc(moment).astimezone(ZoneInfo(timezone))


def _format_kickoff(moment: dt.datetime, timezone: str, now: dt.datetime) -> str:
    local = _local(moment, timezone)
    hours = clock.hours_between(now, moment)
    if hours < 1:
        relative = f"in {int(round(hours * 60))} minutes"
    elif hours < 24:
        relative = f"in {hours:.0f} hours"
    else:
        relative = f"in {hours / 24:.0f} days"
    return f"{local.strftime('%a %d %b, %H:%M')} {local.tzname()} ({relative})"


def load_explanations(db: Database, prediction_id: int) -> tuple[list[str], list[str]]:
    """Read back the stored explanation rows.

    Rendering reads from the database rather than from an in-memory object so
    that what the reader sees is provably what was persisted -- there is no
    path for text to reach the email without a row behind it.
    """
    rows = db.query(
        "SELECT direction, feature_key, feature_value, reference_value, "
        "contribution, template_key FROM prediction_explanation "
        "WHERE prediction_id = ? ORDER BY rank", (prediction_id,))
    from nway.prediction.explain import render_template

    support: list[str] = []
    risk: list[str] = []
    for row in rows:
        text = render_template(row["template_key"], row["feature_value"],
                               row["reference_value"])
        if not text:
            continue
        (support if row["direction"] == "SUPPORT" else risk).append(text)
    return support, risk


def render(db: Database, *, selections: Sequence[Any], decision: Any,
           now: dt.datetime, timezone: str = "Africa/Lagos",
           model_version: str = "", feature_version: str = "",
           subject_template: str | None = None) -> RenderedEmail:
    now = clock.ensure_utc(now)
    ordered = sorted(selections, key=lambda item: item.candidate.kickoff_utc)
    count = len(ordered)
    horizon = 72
    if decision.window_start and decision.window_end:
        horizon = int(round(clock.hours_between(
            clock.from_iso(decision.window_start), clock.from_iso(decision.window_end))))

    subject = (subject_template or
               "Football Predictions — Next {horizon_hours} Hours ({count} selections)"
               ).format(horizon_hours=horizon, count=count)

    first_kick = _local(ordered[0].candidate.kickoff_utc, timezone)
    last_kick = _local(ordered[-1].candidate.kickoff_utc, timezone)
    window_label = (f"{first_kick.strftime('%a %d %b %H:%M')} – "
                    f"{last_kick.strftime('%a %d %b %H:%M')} {last_kick.tzname()}")
    competitions = sorted({item.candidate.competition_name for item in ordered})
    generated = _local(now, timezone).strftime('%d %b %H:%M %Z')

    text = _render_text(db, ordered, window_label, competitions, count,
                        generated, model_version, feature_version, timezone, now)
    body = _render_html(db, ordered, window_label, competitions, count,
                        generated, model_version, feature_version, timezone, now)
    return RenderedEmail(subject=subject, html=body, text=text)


def _render_text(db, ordered, window_label, competitions, count, generated,
                 model_version, feature_version, timezone, now) -> str:
    lines = [
        "FOOTBALL PREDICTIONS — NEXT 72 HOURS",
        "=" * 52,
        "",
        f"Prediction window: {window_label}",
        f"{count} selections across {len(competitions)} competitions",
        f"Generated {generated} · model {model_version} · features {feature_version}",
        "",
    ]
    for index, item in enumerate(ordered, start=1):
        candidate = item.candidate
        support, risk = load_explanations(db, candidate.prediction_id)
        lines.append(f"{index}. {candidate.match_label}")
        lines.append(f"   Competition  {candidate.competition_name}")
        lines.append(f"   Kickoff      {_format_kickoff(candidate.kickoff_utc, timezone, now)}")
        lines.append(f"   Market       {MARKET_LABELS.get(candidate.market_key, candidate.market_key)}")
        lines.append(f"   Probability  {candidate.probability:.0%}")
        lines.append(f"   Confidence   {item.confidence_band.title()}")
        if candidate.lambda_home is not None:
            lines.append(f"   Expected goals  {candidate.lambda_home:.2f} – "
                         f"{candidate.lambda_away:.2f} "
                         f"(total {candidate.lambda_home + candidate.lambda_away:.2f})")
        for line in support:
            lines.append(f"   +  {line}")
        for line in risk:
            lines.append(f"   -  {line}")
        lines.append(f"   Predicted at {_local(candidate.prediction_timestamp, timezone)
                                        .strftime('%d %b %H:%M %Z')}")
        lines.append("")
    lines += ["-" * 52, UNCERTAINTY_NOTICE, ""]
    return "\n".join(lines)


def _render_html(db, ordered, window_label, competitions, count, generated,
                 model_version, feature_version, timezone, now) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    cards = []
    for index, item in enumerate(ordered, start=1):
        candidate = item.candidate
        support, risk = load_explanations(db, candidate.prediction_id)
        band_colour = {"HIGH": "#0f7b3f", "MEDIUM": "#8a6d00",
                       "LOW": "#8a2b2b"}.get(item.confidence_band, "#444")
        factors = "".join(
            f'<li style="margin:2px 0;color:#0f7b3f;">+ {esc(line)}</li>' for line in support)
        factors += "".join(
            f'<li style="margin:2px 0;color:#8a2b2b;">− {esc(line)}</li>' for line in risk)
        expected = ""
        if candidate.lambda_home is not None:
            expected = (f'<tr><td style="padding:2px 0;color:#555;">Expected goals</td>'
                        f'<td style="padding:2px 0;">{candidate.lambda_home:.2f} – '
                        f'{candidate.lambda_away:.2f} '
                        f'(total {candidate.lambda_home + candidate.lambda_away:.2f})</td></tr>')
        cards.append(f"""
        <tr><td style="padding:14px 0;border-bottom:1px solid #e6e6e6;">
          <div style="font-size:16px;font-weight:600;color:#111;">
            {index}. {esc(candidate.match_label)}
          </div>
          <div style="font-size:12px;color:#666;margin:2px 0 8px;">
            {esc(candidate.competition_name)}
          </div>
          <table style="font-size:13px;border-collapse:collapse;width:100%;">
            <tr><td style="padding:2px 0;color:#555;width:130px;">Kickoff</td>
                <td style="padding:2px 0;">{esc(_format_kickoff(candidate.kickoff_utc, timezone, now))}</td></tr>
            <tr><td style="padding:2px 0;color:#555;">Market</td>
                <td style="padding:2px 0;font-weight:600;">
                  {esc(MARKET_LABELS.get(candidate.market_key, candidate.market_key))}</td></tr>
            <tr><td style="padding:2px 0;color:#555;">Probability</td>
                <td style="padding:2px 0;font-weight:700;font-size:15px;">
                  {candidate.probability:.0%}</td></tr>
            <tr><td style="padding:2px 0;color:#555;">Confidence</td>
                <td style="padding:2px 0;color:{band_colour};font-weight:600;">
                  {esc(item.confidence_band.title())}</td></tr>
            {expected}
          </table>
          <ul style="font-size:12px;margin:8px 0 0;padding-left:18px;">{factors}</ul>
          <div style="font-size:11px;color:#999;margin-top:6px;">
            Predicted {esc(_local(candidate.prediction_timestamp, timezone)
                            .strftime('%d %b %H:%M %Z'))} · {esc(candidate.model_version)}
          </div>
        </td></tr>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f5f5f4;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
       style="background:#f5f5f4;padding:24px 12px;">
<tr><td align="center">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
       style="max-width:620px;background:#ffffff;border-radius:10px;padding:26px;">
  <tr><td>
    <div style="font-size:20px;font-weight:700;color:#111;">
      Football Predictions</div>
    <div style="font-size:13px;color:#666;margin-top:4px;">Next 72 hours</div>
    <div style="height:1px;background:#e6e6e6;margin:16px 0;"></div>
    <table style="font-size:13px;width:100%;">
      <tr><td style="color:#555;width:130px;padding:2px 0;">Window</td>
          <td style="padding:2px 0;">{esc(window_label)}</td></tr>
      <tr><td style="color:#555;padding:2px 0;">Selections</td>
          <td style="padding:2px 0;">{count} across {len(competitions)} competitions</td></tr>
      <tr><td style="color:#555;padding:2px 0;">Competitions</td>
          <td style="padding:2px 0;">{esc(', '.join(competitions))}</td></tr>
      <tr><td style="color:#555;padding:2px 0;">Generated</td>
          <td style="padding:2px 0;">{esc(generated)}</td></tr>
    </table>
  </td></tr>
  {''.join(cards)}
  <tr><td style="padding-top:18px;">
    <div style="font-size:11px;color:#777;line-height:1.55;background:#faf9f7;
                border-left:3px solid #ddd;padding:10px 12px;border-radius:4px;">
      {esc(UNCERTAINTY_NOTICE)}
    </div>
    <div style="font-size:10px;color:#aaa;margin-top:10px;">
      model {esc(model_version)} · features {esc(feature_version)}
    </div>
  </td></tr>
</table>
</td></tr></table></body></html>"""
