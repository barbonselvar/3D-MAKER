"""SQLite (WAL) persistence for uploads, jobs, models, artifacts, workers and spend.

The jobs table doubles as the queue (blueprint section C); Postgres with
SKIP LOCKED replaces it when more than one API instance runs (P4-03).
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    mime TEXT NOT NULL,
    original_mime TEXT NOT NULL,
    has_alpha INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    target TEXT NOT NULL,
    engine TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    worker_id TEXT,
    requeue_count INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    error_message TEXT,
    error_retryable INTEGER,
    model_id TEXT,
    gpu_seconds REAL,
    cost_usd REAL,
    retry_of TEXT,
    created_at REAL NOT NULL,
    queued_at REAL NOT NULL,
    assigned_at REAL,
    started_at REAL,
    heartbeat_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner_id, created_at);
CREATE TABLE IF NOT EXISTS job_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts REAL NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    progress REAL,
    message TEXT
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, seq);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    manifest TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS models_owner ON models(owner_id, created_at);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    model_id TEXT,
    format TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    validated INTEGER NOT NULL DEFAULT 0,
    validation TEXT,
    created_at REAL NOT NULL,
    UNIQUE(job_id, format)
);
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'internal',
    probe TEXT,
    engines TEXT,
    heartbeat_at REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spend_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    reserved_usd REAL NOT NULL,
    actual_usd REAL,
    created_at REAL NOT NULL,
    reconciled_at REAL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.execute('PRAGMA journal_mode=WAL')  # must run outside a transaction
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per unit of work; commits on success, rolls back on error."""
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=10000')
        try:
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.execute('COMMIT')
        except BaseException:
            conn.execute('ROLLBACK')
            raise
        finally:
            conn.close()


def loads(value: str | None):
    return None if value is None else json.loads(value)


def dumps(value) -> str:
    return json.dumps(value, separators=(',', ':'), sort_keys=True)
