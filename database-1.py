import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "gesture_predictions.db"

def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = connect()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gesture TEXT NOT NULL,
        confidence REAL NOT NULL,
        filename TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.commit()
    c.close()

def add(gesture, confidence, filename):
    c = connect()
    cur = c.execute(
        "INSERT INTO predictions(gesture,confidence,filename) VALUES(?,?,?)",
        (gesture, confidence, filename)
    )
    c.commit()
    row_id = cur.lastrowid
    c.close()
    return row_id

def all_rows():
    c = connect()
    rows = [dict(x) for x in c.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()]
    c.close()
    return rows

def clear():
    c = connect()
    c.execute("DELETE FROM predictions")
    c.commit()
    c.close()
