import sqlite3
from pathlib import Path
from datetime import datetime

from ..utils.config import Config

DB_PATH = Config.LOGS_DIR / "saas.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------
# INIT DATABASE
# ---------------------------------------------------
def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT UNIQUE,
        user TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT,
        timestamp TEXT,
        success INTEGER
    )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------
# API KEY MANAGEMENT
# ---------------------------------------------------
def add_api_key(api_key: str, user: str = "default"):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO api_keys (api_key, user, created_at)
    VALUES (?, ?, ?)
    """, (api_key, user, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def validate_api_key(api_key: str) -> bool:

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM api_keys WHERE api_key = ?",
        (api_key,)
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


# ---------------------------------------------------
# USAGE TRACKING
# ---------------------------------------------------
def log_usage(api_key: str, success: bool):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO usage (api_key, timestamp, success)
    VALUES (?, ?, ?)
    """, (api_key, datetime.now().isoformat(), int(success)))

    conn.commit()
    conn.close()


def get_usage(api_key: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*) FROM usage WHERE api_key = ?
    """, (api_key,))
    total = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*) FROM usage WHERE api_key = ? AND success = 1
    """, (api_key,))
    success = cursor.fetchone()[0]

    conn.close()

    return {
        "total_requests": total,
        "successful_requests": success,
        "failed_requests": total - success
    }