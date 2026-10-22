"""SQLite storage layer (Phase 2 — auth + DB security).

Design notes
------------
The studio used to keep *everything* in JSON files next to the media, with a
single shared ``RECAP_ACCESS_PASSWORD``. That is fine for one person on a
laptop, but on a public AWS box it means: no accounts, no per-user isolation,
no audit trail and secrets sitting in world-readable files.

This module is the storage foundation for all of that:

* **One SQLite file** (``DATA_DIR/recap.db``) created with ``0600``
  permissions — only the service user can read the password hashes.
* **WAL journal + busy timeout** so the API, the janitor and the job workers
  can write concurrently without "database is locked" errors.
* **Thread-local connections** (FastAPI runs handlers on a thread pool, and
  SQLite connections are not safe to share across threads).
* **Parameterised queries only** — every helper takes ``params``; no string
  formatting of user input anywhere in the codebase.
* **Versioned migrations** so an existing deployment upgrades itself on
  startup (``python -m recapstudio.useradmin migrate`` does it manually).

Nothing here imports FastAPI, so the CLI (``recapstudio.useradmin``) and the
tests can use the same code without spinning up the web app.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from . import config
from .util import get_logger

log = get_logger("recap.db")

_local = threading.local()
_init_lock = threading.RLock()
_initialised: set[str] = set()

#: bumped whenever MIGRATIONS grows
SCHEMA_VERSION = 4


MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        CREATE TABLE IF NOT EXISTS users (
            id                   TEXT PRIMARY KEY,
            username             TEXT NOT NULL,
            username_lower       TEXT NOT NULL UNIQUE,
            display_name         TEXT NOT NULL DEFAULT '',
            password_hash        TEXT NOT NULL,
            role                 TEXT NOT NULL DEFAULT 'user',
            status               TEXT NOT NULL DEFAULT 'active',
            created_at           REAL NOT NULL,
            updated_at           REAL NOT NULL,
            last_login_at        REAL NOT NULL DEFAULT 0,
            password_changed_at  REAL NOT NULL DEFAULT 0,
            must_change_password INTEGER NOT NULL DEFAULT 0,
            quota_bytes          INTEGER NOT NULL DEFAULT 0,
            max_concurrent_jobs  INTEGER NOT NULL DEFAULT 0,
            daily_job_limit      INTEGER NOT NULL DEFAULT 0,
            notes                TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token_hash   TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            csrf_token   TEXT NOT NULL,
            created_at   REAL NOT NULL,
            expires_at   REAL NOT NULL,
            last_seen_at REAL NOT NULL,
            ip           TEXT NOT NULL DEFAULT '',
            user_agent   TEXT NOT NULL DEFAULT '',
            revoked_at   REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_exp  ON sessions(expires_at);

        CREATE TABLE IF NOT EXISTS login_attempts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL DEFAULT '',
            ip         TEXT NOT NULL DEFAULT '',
            ok         INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_user ON login_attempts(username, created_at);
        CREATE INDEX IF NOT EXISTS idx_attempts_ip   ON login_attempts(ip, created_at);

        CREATE TABLE IF NOT EXISTS audit_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            user_id    TEXT NOT NULL DEFAULT '',
            username   TEXT NOT NULL DEFAULT '',
            ip         TEXT NOT NULL DEFAULT '',
            action     TEXT NOT NULL,
            target     TEXT NOT NULL DEFAULT '',
            detail     TEXT NOT NULL DEFAULT '',
            ok         INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id, created_at);
    """),
    (2, """
        CREATE TABLE IF NOT EXISTS user_secrets (
            user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            slot       INTEGER NOT NULL,
            value      TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (user_id, slot)
        );

        CREATE TABLE IF NOT EXISTS job_index (
            job_id     TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL DEFAULT '',
            kind       TEXT NOT NULL DEFAULT 'recap',
            status     TEXT NOT NULL DEFAULT 'queued',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_user ON job_index(user_id, created_at);
    """),
    (3, """
        CREATE TABLE IF NOT EXISTS settings_kv (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
    """),
    (4, """
        CREATE TABLE IF NOT EXISTS user_preferences (
            scope              TEXT PRIMARY KEY,
            preferences_json   TEXT NOT NULL DEFAULT '{}',
            updated_at         REAL NOT NULL
        );
    """),
]


def db_path() -> Path:
    return Path(config.DB_PATH)


def _harden_file(path: Path) -> None:
    """Password hashes must never be world readable."""
    try:
        for candidate in (path, path.with_name(path.name + "-wal"),
                          path.with_name(path.name + "-shm")):
            if candidate.exists():
                os.chmod(candidate, 0o600)
    except Exception as exc:  # pragma: no cover - platform dependent
        log.debug("could not chmod %s: %s", path, exc)


def connect() -> sqlite3.Connection:
    """Thread-local, hardened connection (auto-commit mode)."""
    path = db_path()
    key = f"{path}"
    conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", "") == key:
        return conn
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=15000")
    cur.execute("PRAGMA secure_delete=ON")
    try:
        cur.execute("PRAGMA trusted_schema=OFF")
    except sqlite3.DatabaseError:
        pass  # older SQLite
    cur.close()
    if fresh:
        _harden_file(path)
    _local.conn = conn
    _local.path = key
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def execute(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
    return connect().execute(sql, params)


def query(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
    return list(connect().execute(sql, params).fetchall())


def query_one(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Optional[sqlite3.Row]:
    row = connect().execute(sql, params).fetchone()
    return row


def executemany(sql: str, rows: Iterable[Sequence[Any]]) -> None:
    connect().executemany(sql, rows)


# ── migrations ───────────────────────────────────────────────────────────
def _current_version(conn: sqlite3.Connection) -> int:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL
        )
    """)
    row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
    return int(row["v"] or 0) if row else 0


def init(force: bool = False) -> int:
    """Create/upgrade the schema. Safe to call from every process start."""
    path = str(db_path())
    with _init_lock:
        if path in _initialised and not force:
            return SCHEMA_VERSION
        conn = connect()
        version = _current_version(conn)
        for target, script in MIGRATIONS:
            if target <= version:
                continue
            log.info("applying DB migration %d", target)
            conn.executescript(script)
            conn.execute("INSERT OR REPLACE INTO schema_migrations(version, applied_at) "
                         "VALUES (?, ?)", (target, time.time()))
            version = target
        _harden_file(db_path())
        _initialised.add(path)
        return version


def reset_for_tests() -> None:
    """Drop the cached connection (used by the test-suite / CLI)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    _local.conn = None
    _local.path = ""
    _initialised.clear()


def stats() -> dict[str, Any]:
    init()
    out: dict[str, Any] = {"path": str(db_path()), "schema_version": SCHEMA_VERSION}
    try:
        out["size_bytes"] = db_path().stat().st_size
        out["mode"] = oct(db_path().stat().st_mode & 0o777)
    except OSError:
        out["size_bytes"] = 0
        out["mode"] = "?"
    for table in ("users", "sessions", "audit_log", "job_index", "user_preferences"):
        try:
            row = query_one(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608 - fixed names
            out[f"{table}_count"] = int(row["n"]) if row else 0
        except sqlite3.DatabaseError:
            out[f"{table}_count"] = 0
    return out
