import sqlite3
from datetime import datetime
from pathlib import Path

from ..utils.config import Config

DB_PATH = Config.LOGS_DIR / "saas.db"


# ---------------------------------------------------
# CONNECTION (THREAD-SAFE)
# ---------------------------------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# ---------------------------------------------------
# INIT DATABASE
# ---------------------------------------------------
def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # ---------------- API KEYS ----------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT UNIQUE,
        user TEXT,
        created_at TEXT
    )
    """)

    # ---------------- USAGE ----------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT,
        timestamp TEXT,
        success INTEGER
    )
    """)

    # ---------------- PATIENTS ----------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE,
        name TEXT,
        age INTEGER,
        gender TEXT,
        created_at TEXT
    )
    """)

    # ---------------- PREDICTIONS ----------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    """)

    # ---------------- INDEXES (PERFORMANCE) ----------------
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_patient_id
    ON predictions (patient_id)
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


# ---------------------------------------------------
# PATIENT MANAGEMENT
# ---------------------------------------------------
def add_patient(patient_id: str, name: str, age: int, gender: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (patient_id, name, age, gender, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_patient(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT patient_id, name, age, gender
    FROM patients
    WHERE patient_id = ?
    """, (patient_id,))

    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "patient_id": result[0],
            "name": result[1],
            "age": result[2],
            "gender": result[3]
        }

    return None


# ---------------------------------------------------
# PREDICTION LOGGING
# ---------------------------------------------------
def log_prediction(patient_id: str, prediction: str, confidence: float):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (patient_id, prediction, confidence, timestamp)
    VALUES (?, ?, ?, ?)
    """, (patient_id, prediction, confidence, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_patient_history(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT prediction, confidence, timestamp
    FROM predictions
    WHERE patient_id = ?
    ORDER BY timestamp DESC
    """, (patient_id,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "prediction": r[0],
            "confidence": r[1],
            "timestamp": r[2]
        }
        for r in rows
    ]