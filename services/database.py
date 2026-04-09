import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "shyam_steel.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            emp_id      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            department  TEXT DEFAULT '',
            enrolled    INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            visitor_id       TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            host             TEXT DEFAULT '',
            permitted_floors TEXT DEFAULT '',
            check_in         TEXT,
            enrolled         INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   TEXT NOT NULL,
            person_type TEXT NOT NULL,
            embedding   TEXT NOT NULL        -- stored as JSON string
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS detection_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   TEXT,
            label       TEXT,
            camera_id   TEXT,
            confidence  REAL,
            timestamp   TEXT
        )
    """)

    conn.commit()
    conn.close()