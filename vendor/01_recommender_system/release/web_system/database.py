"""Minimal SQLite persistence for the packaged web demo."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users(
    user_id TEXT PRIMARY KEY,
    country TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS auth_accounts(
    username TEXT PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS products(
    item_id TEXT PRIMARY KEY,
    description TEXT,
    chinese_name TEXT,
    category TEXT
);
CREATE TABLE IF NOT EXISTS recommendation_sessions(
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    request_k INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS recommendation_items(
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    rank_position INTEGER NOT NULL,
    final_score REAL NOT NULL,
    source_type TEXT,
    PRIMARY KEY(session_id,item_id),
    UNIQUE(session_id,rank_position),
    FOREIGN KEY(session_id) REFERENCES recommendation_sessions(session_id),
    FOREIGN KEY(item_id) REFERENCES products(item_id)
);
CREATE TABLE IF NOT EXISTS feedback_events(
    feedback_id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK(feedback_type IN('click','like','dislike')),
    idempotency_key TEXT UNIQUE NOT NULL,
    client_time DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES recommendation_sessions(session_id),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(item_id) REFERENCES products(item_id)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    return connection


def record_feedback(connection: sqlite3.Connection, event: dict) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO feedback_events(
               feedback_id,session_id,user_id,item_id,feedback_type,idempotency_key,client_time
           ) VALUES(?,?,?,?,?,?,?)""",
        (
            event.get("feedback_id", str(uuid.uuid4())),
            event.get("session_id"),
            event["user_id"],
            event["item_id"],
            event["feedback_type"],
            event["idempotency_key"],
            event.get("client_time"),
        ),
    )
    connection.commit()
