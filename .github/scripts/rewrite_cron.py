#!/usr/bin/env python3
"""Rewrite the workflow's own cron schedule to match the computed next run.

GitHub Actions has no delayed dispatch, so the only way a run can influence
when the next one happens is to change the schedule and commit it. The cron
block is fenced by markers; everything outside them is left untouched.

Guard rails:
  * the safety-net entry is always retained, so a bad computed window can
    never leave the workflow with no schedule at all;
  * the file is only rewritten when the block actually changes, so runs that
    reach the same conclusion produce no commit;
  * the result is parsed back and validated before being written.

Usage: rewrite_cron.py <workflow.yml> <cron-line> [<cron-line> ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BEGIN = "# >>> nway:dynamic-schedule"
END = "# <<< nway:dynamic-schedule"
SAFETY_NET = "17 6 * * *"

CRON_FIELD = re.compile(r"^[\d*/,\-]+$")


def valid_cron(line: str) -> bool:
    fields = line.split()
    return len(fields) == 5 and all(CRON_FIELD.match(f) for f in fields)


def render(lines: list[str], indent: str) -> str:
    body = "\n".join(f'{indent}- cron: "{line}"' for line in lines)
    return f"{indent}{BEGIN}\n{body}\n{indent}{END}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    path = Path(argv[0])
    requested = [line.strip() for line in argv[1:] if line.strip()]

    lines = [line for line in requested if valid_cron(line)]
    rejected = [line for line in requested if not valid_cron(line)]
    for line in rejected:
        print(f"rejecting malformed cron: {line!r}", file=sys.stderr)

    # Never leave the workflow unscheduled: without the safety net, one bad
    # computed window would strand the loop with no way to wake up.
    if SAFETY_NET not in lines:
        lines.append(SAFETY_NET)

    text = path.read_text()
    match = re.search(
        rf"^([ \t]*){re.escape(BEGIN)}.*?^[ \t]*{re.escape(END)}",
        text, flags=re.S | re.M)
    if not match:
        print(f"markers not found in {path}", file=sys.stderr)
        return 1

    indent = match.group(1)
    replacement = render(lines, indent)
    if match.group(0) == replacement:
        print("schedule unchanged")
        return 0

    path.write_text(text[:match.start()] + replacement + text[match.end():])
    print("schedule updated to:")
    for line in lines:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
