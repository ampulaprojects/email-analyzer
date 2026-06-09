"""Database module — SQLite persistence layer for email metadata."""

import sqlite3
from pathlib import Path

# ── DDL: core tables (Fáza 1) ─────────────────────────────────────────────────

_DDL_EMAILS = """
CREATE TABLE IF NOT EXISTS emails (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id       TEXT    UNIQUE NOT NULL,
    thread_id        TEXT,
    in_reply_to      TEXT,
    "references"     TEXT,
    date             TEXT,
    from_address     TEXT,
    from_name        TEXT,
    to_addresses     TEXT,
    cc_addresses     TEXT,
    subject          TEXT,
    folder           TEXT,
    has_attachments  INTEGER,
    attachment_names TEXT,
    attachment_types TEXT,
    size_bytes       INTEGER,
    imap_uid         INTEGER,
    synced_at        TEXT
);
"""

_DDL_SYNC_STATE = """
CREATE TABLE IF NOT EXISTS sync_state (
    id        INTEGER PRIMARY KEY,
    folder    TEXT    UNIQUE,
    last_uid  INTEGER DEFAULT 0,
    last_sync TEXT
);
"""

_DDL_INDEXES_PHASE1 = [
    "CREATE INDEX IF NOT EXISTS idx_emails_date         ON emails (date)",
    "CREATE INDEX IF NOT EXISTS idx_emails_from_address ON emails (from_address)",
    "CREATE INDEX IF NOT EXISTS idx_emails_thread_id    ON emails (thread_id)",
    "CREATE INDEX IF NOT EXISTS idx_emails_folder       ON emails (folder)",
]

# ── DDL: Phase 2 tables ───────────────────────────────────────────────────────

# Stĺpce pridávané cez ALTER TABLE — emails tabuľka už existuje s dátami
_PHASE2_COLUMNS: list[tuple[str, str]] = [
    ("body_text",    "TEXT"),   # prvých 1000 znakov plain textu
    ("body_snippet", "TEXT"),   # prvých 150 znakov pre UI
    ("embedding",    "BLOB"),   # float32 vektor 768-dim (nomic-embed-text)
    ("language",     "TEXT"),   # sk / en / de / other
]

_DDL_CLUSTERS = """
CREATE TABLE IF NOT EXISTS clusters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    description TEXT,
    size        INTEGER DEFAULT 0,
    created_at  TEXT,
    updated_at  TEXT
);
"""

# email_clusters — definícia so všetkými požadovanými stĺpcami
_DDL_EMAIL_CLUSTERS = """
CREATE TABLE IF NOT EXISTS email_clusters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id   INTEGER REFERENCES emails(id),
    cluster_id INTEGER REFERENCES clusters(id),
    confidence REAL,
    source     TEXT,
    created_at TEXT
);
"""

_DDL_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id           INTEGER REFERENCES emails(id),
    old_cluster_id     INTEGER,
    correct_cluster_id INTEGER,
    note               TEXT,
    created_at         TEXT
);
"""

_DDL_INDEXES_PHASE2 = [
    "CREATE INDEX IF NOT EXISTS idx_emails_language        ON emails (language)",
    "CREATE INDEX IF NOT EXISTS idx_email_clusters_email   ON email_clusters (email_id)",
    "CREATE INDEX IF NOT EXISTS idx_email_clusters_cluster ON email_clusters (cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_email         ON feedback (email_id)",
]


# ── public API ────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> None:
    """Create all tables and indexes for a fresh database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _configure(conn)
        for ddl in [_DDL_EMAILS, _DDL_SYNC_STATE, _DDL_CLUSTERS,
                    _DDL_EMAIL_CLUSTERS, _DDL_FEEDBACK]:
            conn.execute(ddl)
        for idx in _DDL_INDEXES_PHASE1 + _DDL_INDEXES_PHASE2:
            conn.execute(idx)
        conn.commit()
    finally:
        conn.close()


def migrate_phase2(db_path: str) -> dict:
    """Safely apply Phase 2 schema changes to an existing database.

    - Adds body_text / body_snippet / embedding / language to emails if missing.
    - Recreates email_clusters with the correct schema if it is empty and outdated.
    - Creates clusters / feedback tables if missing.
    - Verifies row count in emails is unchanged.
    Returns a dict with migration details.
    """
    conn = sqlite3.connect(db_path)
    report: dict = {}
    try:
        _configure(conn)

        count_before = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]

        # 1. new columns on emails
        added_cols = _add_missing_columns(conn)
        report["columns_added"] = added_cols

        # 2. Phase 2 tables
        conn.execute(_DDL_CLUSTERS)
        conn.execute(_DDL_FEEDBACK)

        # 3. email_clusters — recreate only if schema is outdated AND table is empty
        ec_cols = {r[1] for r in conn.execute("PRAGMA table_info(email_clusters)").fetchall()}
        ec_exists = bool(ec_cols)
        if ec_exists and "id" not in ec_cols:
            ec_count = conn.execute("SELECT COUNT(*) FROM email_clusters").fetchone()[0]
            if ec_count > 0:
                raise RuntimeError(
                    f"email_clusters has outdated schema but {ec_count} rows — "
                    "manual migration required"
                )
            conn.execute("DROP TABLE email_clusters")
            report["email_clusters_recreated"] = True
        if not ec_exists or "id" not in ec_cols:
            conn.execute(_DDL_EMAIL_CLUSTERS)
        report["email_clusters_recreated"] = report.get("email_clusters_recreated", False)

        # 4. indexes (Phase 1 + Phase 2, all idempotent)
        for idx in _DDL_INDEXES_PHASE1 + _DDL_INDEXES_PHASE2:
            conn.execute(idx)

        conn.commit()

        count_after = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        assert count_after == count_before, (
            f"Row count changed: {count_before} → {count_after}"
        )
        report["emails_count"] = count_after
        report["ok"] = True
    finally:
        conn.close()
    return report


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a configured SQLite connection with row_factory set."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _configure(conn)
    return conn


def get_emails_without_body(db_path: str, limit: int = 500) -> list[sqlite3.Row]:
    """Return emails that have no body_text yet, oldest-first."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            """
            SELECT id, imap_uid, folder, subject, message_id
            FROM emails
            WHERE body_text IS NULL
            ORDER BY date ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()


# ── internal helpers ──────────────────────────────────────────────────────────

def _configure(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")


def _add_missing_columns(conn: sqlite3.Connection) -> list[str]:
    """Add Phase 2 columns to emails if they don't exist. Returns names of added cols."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(emails)").fetchall()}
    added = []
    for col_name, col_type in _PHASE2_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE emails ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
    return added
