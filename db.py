import sqlite3
from datetime import datetime, date
import numpy as np
from config import DB_PATH


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            vector BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            day TEXT NOT NULL,
            time TEXT NOT NULL,
            score REAL,
            source TEXT NOT NULL DEFAULT 'camera',
            UNIQUE(student_id, day)
        );
        """)


def get_or_create_student(code, name):
    with connect() as c:
        row = c.execute("SELECT id FROM students WHERE code=?", (code,)).fetchone()
        if row:
            return row[0], False
        cur = c.execute("INSERT INTO students(code, name) VALUES(?,?)", (code, name))
        return cur.lastrowid, True


def add_embeddings(student_id, vectors):
    with connect() as c:
        c.executemany(
            "INSERT INTO embeddings(student_id, vector) VALUES(?,?)",
            [(student_id, np.asarray(v, dtype=np.float32).tobytes()) for v in vectors],
        )


def load_gallery():
    """برمی‌گرداند: ماتریس بردارها، شمارهٔ دانش‌آموز هر ردیف، و دیکشنری اطلاعات دانش‌آموزان"""
    with connect() as c:
        students = {r[0]: (r[1], r[2]) for r in c.execute("SELECT id, code, name FROM students")}
        rows = c.execute("SELECT student_id, vector FROM embeddings").fetchall()
    if not rows:
        return np.empty((0, 512), np.float32), np.empty((0,), np.int64), students
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return mat, ids, students


def today():
    return date.today().isoformat()


def mark_present(student_id, score=None, source="camera"):
    """اگر امروز قبلاً ثبت نشده باشد ثبت می‌کند و True برمی‌گرداند"""
    with connect() as c:
        try:
            c.execute(
                "INSERT INTO attendance(student_id, day, time, score, source) VALUES(?,?,?,?,?)",
                (student_id, today(), datetime.now().strftime("%H:%M:%S"), score, source),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def present_ids_today():
    with connect() as c:
        return {r[0] for r in c.execute("SELECT student_id FROM attendance WHERE day=?", (today(),))}


def report(day=None):
    day = day or today()
    with connect() as c:
        present = c.execute("""
            SELECT s.code, s.name, a.time, a.score, a.source FROM attendance a
            JOIN students s ON s.id=a.student_id WHERE a.day=? ORDER BY a.time""", (day,)).fetchall()
        absent = c.execute("""
            SELECT code, name FROM students WHERE id NOT IN
            (SELECT student_id FROM attendance WHERE day=?) ORDER BY code""", (day,)).fetchall()
    return present, absent
