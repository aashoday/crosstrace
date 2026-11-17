import sqlite3
from pathlib import Path

from crosstrace.config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS issues (
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    title TEXT,
    body TEXT,
    state TEXT,
    author TEXT,
    created_at TEXT,
    closed_at TEXT,
    PRIMARY KEY (repo, number)
);

CREATE TABLE IF NOT EXISTS cross_refs (
    source_repo TEXT NOT NULL,
    source_number INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    target_repo TEXT NOT NULL,
    target_number INTEGER NOT NULL,
    created_at TEXT,
    will_close_target INTEGER,
    PRIMARY KEY (source_repo, source_number, target_repo, target_number)
);
"""


def connect(path: Path = DATA_DIR / "crosstrace.sqlite") -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn