import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "matching_history.db"

def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = connect()
    c.execute("""CREATE TABLE IF NOT EXISTS matches(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        score REAL,
        matched_keywords TEXT,
        missing_keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.commit()
    c.close()

def add(filename, score, matched, missing):
    c = connect()
    cur = c.execute(
        """INSERT INTO matches
        (filename,score,matched_keywords,missing_keywords)
        VALUES(?,?,?,?)""",
        (filename, score, matched, missing)
    )
    c.commit()
    row_id = cur.lastrowid
    c.close()
    return row_id

def all_rows():
    c = connect()
    rows = [dict(x) for x in c.execute(
        "SELECT * FROM matches ORDER BY id DESC"
    ).fetchall()]
    c.close()
    return rows

def clear():
    c = connect()
    c.execute("DELETE FROM matches")
    c.commit()
    c.close()
