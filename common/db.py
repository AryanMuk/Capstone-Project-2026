from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'celeba',
    key_version  INTEGER NOT NULL DEFAULT 1,
    active       INTEGER NOT NULL DEFAULT 1,
    enrolled_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates(
    user_id     TEXT NOT NULL,
    scheme      TEXT NOT NULL,
    key_version INTEGER NOT NULL,
    blob        BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, scheme, key_version),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS auth_log(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    scheme       TEXT,
    mode         TEXT,
    claimed_user TEXT,
    decision     TEXT NOT NULL,
    score        REAL,
    reason       TEXT,
    latency_ms   REAL
);
CREATE TABLE IF NOT EXISTS nonces(
    nonce TEXT PRIMARY KEY,
    ts    TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_user(conn: sqlite3.Connection, user_id: str, display_name: str,
                source: str = "celeba", key_version: int = 1) -> None:
    conn.execute(
        """INSERT INTO users(user_id, display_name, source, key_version, active, enrolled_at)
           VALUES (?, ?, ?, ?, 1, ?)
           ON CONFLICT(user_id) DO UPDATE SET display_name = excluded.display_name,
               source = excluded.source, key_version = excluded.key_version, active = 1""",
        (user_id, display_name, source, key_version, utc_now()))


def put_template(conn: sqlite3.Connection, user_id: str, scheme: str,
                 key_version: int, blob: bytes) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO templates(user_id, scheme, key_version, blob, created_at) "
        "VALUES (?, ?, ?, ?, ?)", (user_id, scheme, key_version, blob, utc_now()))


def get_active_template(conn: sqlite3.Connection, user_id: str,
                        scheme: str) -> Optional[Tuple[int, bytes]]:
    row = conn.execute(
        """SELECT t.key_version, t.blob FROM templates t JOIN users u
           ON u.user_id = t.user_id AND u.key_version = t.key_version
           WHERE t.user_id = ? AND t.scheme = ? AND u.active = 1""",
        (user_id, scheme)).fetchone()
    return (row["key_version"], bytes(row["blob"])) if row else None


def iter_active_templates(conn: sqlite3.Connection, scheme: str
                          ) -> Iterator[Tuple[str, int, bytes]]:
    cur = conn.execute(
        """SELECT t.user_id, t.key_version, t.blob FROM templates t JOIN users u
           ON u.user_id = t.user_id AND u.key_version = t.key_version
           WHERE t.scheme = ? AND u.active = 1 ORDER BY t.user_id""", (scheme,))
    for row in cur:
        yield row["user_id"], row["key_version"], bytes(row["blob"])


def list_users(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT user_id, display_name, source, key_version, active, enrolled_at "
                        "FROM users ORDER BY user_id").fetchall()
    return [dict(r) for r in rows]


def get_user(conn: sqlite3.Connection, user_id: str) -> Optional[Dict]:
    row = conn.execute("SELECT user_id, display_name, source, key_version, active, enrolled_at "
                       "FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def bump_key_version(conn: sqlite3.Connection, user_id: str) -> int:
    """Start revocation: the user's active key_version becomes old+1. Returns the new version."""
    cur = conn.execute("UPDATE users SET key_version = key_version + 1 WHERE user_id = ?",
                       (user_id,))
    if cur.rowcount != 1:
        raise KeyError(f"unknown user {user_id}")
    return conn.execute("SELECT key_version FROM users WHERE user_id = ?",
                        (user_id,)).fetchone()["key_version"]


def purge_old_versions(conn: sqlite3.Connection, user_id: str) -> int:
    cur = conn.execute(
        "DELETE FROM templates WHERE user_id = ? AND key_version < "
        "(SELECT key_version FROM users WHERE user_id = ?)", (user_id, user_id))
    return cur.rowcount


def delete_user(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute("DELETE FROM templates WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))


def log_auth(conn: sqlite3.Connection, scheme: Optional[str], mode: Optional[str],
             claimed_user: Optional[str], decision: str, score: Optional[float],
             reason: Optional[str], latency_ms: Optional[float]) -> None:
    conn.execute(
        "INSERT INTO auth_log(ts, scheme, mode, claimed_user, decision, score, reason, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (utc_now(), scheme, mode, claimed_user, decision, score, reason, latency_ms))
    conn.commit()


def recent_auth_log(conn: sqlite3.Connection, limit: int = 50) -> List[Dict]:
    rows = conn.execute("SELECT * FROM auth_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def register_nonce(conn: sqlite3.Connection, nonce: str) -> bool:
    """Record a request nonce. Returns False if it was already used (a replay)."""
    try:
        conn.execute("INSERT INTO nonces(nonce, ts) VALUES (?, ?)", (nonce, utc_now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def template_stats(conn: sqlite3.Connection) -> Dict[str, Dict[str, float]]:
    """Per scheme: number of active templates, total and mean blob size in bytes."""
    out: Dict[str, Dict[str, float]] = {}
    rows = conn.execute(
        """SELECT t.scheme AS scheme, COUNT(*) AS n, SUM(LENGTH(t.blob)) AS total
           FROM templates t JOIN users u
           ON u.user_id = t.user_id AND u.key_version = t.key_version
           WHERE u.active = 1 GROUP BY t.scheme""").fetchall()
    for r in rows:
        out[r["scheme"]] = {"count": r["n"], "total_bytes": r["total"],
                            "mean_bytes": r["total"] / r["n"]}
    return out
