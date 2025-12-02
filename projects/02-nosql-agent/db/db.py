import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "company.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db_connection = get_connection()
    cursor = db_connection.cursor()

    cursor.execute(
        """
        create table if not exists employees (
            id integer primary key autoincrement,
            name text not null,
            role text not null,
            salary integer not null
        )
        """
    )

    db_connection.commit()
    db_connection.close()
    return "Database initialized"
