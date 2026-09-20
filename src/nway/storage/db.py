"""SQLite access layer.

Deliberately plain: sqlite3 from the standard library, explicit SQL, no ORM.
At ~2,500 fixtures a season with a single writer, an ORM would add a dependency
and a layer of indirection without buying anything. The SQL avoids
SQLite-specific constructs so moving to PostgreSQL later is a connection
change rather than a rewrite.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from nway import clock
from nway.config import PROJECT_ROOT
from nway.logging_setup import get_logger
from nway.storage.schema import DDL, SCHEMA_VERSION

log = get_logger(__name__)

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nway.db"


def _database_path(url: str | None = None) -> Path:
    raw = url or os.environ.get("NWAY_DATABASE_URL") or str(DEFAULT_DB_PATH)
    if raw.startswith("sqlite:///"):
        raw = raw[len("sqlite:///"):]
    if raw == ":memory:":
        return Path(":memory:")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class Database:
    def __init__(self, url: str | None = None) -> None:
        self.path = _database_path(url)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=30.0,
            detect_types=0, check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")

    # -- lifecycle -------------------------------------------------------
    def migrate(self) -> None:
        self.conn.executescript(DDL)
        current = self.conn.execute(
            "SELECT MAX(version) AS v FROM schema_version").fetchone()["v"]
        if current != SCHEMA_VERSION:
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?,?)",
                (SCHEMA_VERSION, clock.to_iso(clock.now())),
            )
        log.info("schema ready", context={"path": str(self.path), "version": SCHEMA_VERSION})

    def close(self) -> None:
        self.conn.close()

    # -- queries ---------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    def insert(self, table: str, values: dict[str, Any], or_ignore: bool = False) -> int:
        """Insert a row and return its id, or 0 when an OR IGNORE was ignored.

        sqlite3 does NOT reset ``lastrowid`` when an INSERT OR IGNORE inserts
        nothing -- it keeps the id from the previous successful insert, on any
        table. Returning that stale value makes callers link child rows to an
        unrelated parent, so the row count is the only trustworthy signal.
        """
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        verb = "INSERT OR IGNORE INTO" if or_ignore else "INSERT INTO"
        cursor = self.conn.execute(
            f"{verb} {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))
        if or_ignore and cursor.rowcount == 0:
            return 0
        return cursor.lastrowid

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction. Rolls back on any exception."""
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")


def connect(url: str | None = None, migrate: bool = True) -> Database:
    database = Database(url)
    if migrate:
        database.migrate()
    return database
