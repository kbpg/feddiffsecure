from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "demo_portal" / "demo_users.sqlite3"

DEFAULT_ACCOUNTS = [
    {
        "username": os.environ.get("PAPER_DEMO_ADMIN_USER", "admin"),
        "password": os.environ.get("PAPER_DEMO_ADMIN_PASSWORD", "fed-demo-2026"),
        "role": "admin",
        "display_name": "管理员",
    },
    {
        "username": os.environ.get("PAPER_DEMO_VIEWER_USER", "viewer"),
        "password": os.environ.get("PAPER_DEMO_VIEWER_PASSWORD", "viewer-demo-2026"),
        "role": "user",
        "display_name": "普通用户",
    },
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                display_name TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        for account in DEFAULT_ACCOUNTS:
            conn.execute(
                """
                INSERT INTO users(username, password_hash, role, display_name, is_active)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(username) DO UPDATE SET
                    role=excluded.role,
                    display_name=excluded.display_name,
                    is_active=1
                """,
                (
                    account["username"],
                    generate_password_hash(account["password"]),
                    account["role"],
                    account["display_name"],
                ),
            )
        conn.commit()


def authenticate_user(username: str, password: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, password_hash, role, display_name, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None or int(row["is_active"]) != 1:
        return None
    if not check_password_hash(str(row["password_hash"]), password):
        return None
    return {
        "username": str(row["username"]),
        "role": str(row["role"]),
        "display_name": str(row["display_name"]),
    }


def list_seed_accounts() -> list[dict[str, str]]:
    return [
        {
            "username": str(account["username"]),
            "password": str(account["password"]),
            "role": str(account["role"]),
            "display_name": str(account["display_name"]),
        }
        for account in DEFAULT_ACCOUNTS
    ]
