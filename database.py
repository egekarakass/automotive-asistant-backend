import sqlite3
from datetime import datetime
import os

DB_NAME = 'chat_history.db'


def initialize_database():
    """
    Creates the necessary tables if they do not exist.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id INTEGER,
            username TEXT,
            language TEXT,
            has_image BOOLEAN,
            file_path TEXT,
            question TEXT,
            answer TEXT,
            tool_used TEXT,
            model_used TEXT,
            token_usage INTEGER,
            timestamp TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            api_name TEXT,
            request_endpoint TEXT,
            response_data TEXT,
            timestamp TEXT
        )
    ''')

    # Task 3: rolling per-session summary. Updated periodically (every
    # SUMMARY_INTERVAL messages, see ai_handler.py) instead of on every turn,
    # so we don't burn a model call per message just to keep it fresh.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id TEXT PRIMARY KEY,
            summary TEXT,
            last_summarized_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    ''')

    # Tracks which session_id is "currently active" for each Telegram user.
    # A new row/update is written every time /start creates a fresh session.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_sessions (
           user_id INTEGER PRIMARY KEY,
           session_id TEXT
       )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            expected_tool TEXT,
            actual_tool TEXT,
            correct_tool_called BOOLEAN,
            answer_correct BOOLEAN,
            reasoning TEXT,
            evaluated_by TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_attempts (
            username TEXT PRIMARY KEY,
            failed_count INTEGER DEFAULT 0,
            locked_until TEXT
        )
    ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS users(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT UNIQUE NOT NULL,
                       password_hash TEXT NOT NULL,
                       created_at TEXT
                       )
                    ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS session_owners
                   (
                       session_id TEXT PRIMARY KEY,
                       user_id INTEGER NOT NULL,
                       title TEXT,
                       created_at TEXT
                    )
                   ''')

    conn.commit()
    conn.close()

    if not os.path.exists('downloads'):
        os.makedirs('downloads')


def claim_session(session_id, user_id, title=None):
    """İlk mesajda session'ı kullanıcıya bağlar. Zaten sahiplenilmişse dokunmaz."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM session_owners WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if row is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO session_owners (session_id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, title, timestamp)
        )
        conn.commit()
        conn.close()
        return user_id
    conn.close()
    return row[0]

def get_session_owner(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM session_owners WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_user_sessions(user_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT so.session_id, so.title, so.created_at,
               MAX(m.timestamp) as last_activity,
               COUNT(m.id) as message_count
        FROM session_owners so
        LEFT JOIN messages m ON m.session_id = so.session_id
        WHERE so.user_id = ?
        GROUP BY so.session_id
        ORDER BY last_activity DESC
    ''', (user_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_last_tool_used(session_id):
    """
    Returns the tool(s) used in the most recent message that actually
    used a tool, or None if none found. Used to detect follow-up
    questions ("peki X?", "aynı soruyu Y için de") that don't repeat the
    original trigger keyword but clearly continue the same intent.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT tool_used FROM messages
        WHERE session_id = ? AND tool_used IS NOT NULL AND tool_used != 'None'
        ORDER BY id DESC LIMIT 1
    ''', (session_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def log_message(session_id, user_id, username, language, has_image, file_path,
                 question, answer, tool_used, model_used, token_usage):
    """
    Inserts a single conversation record into the messages table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO messages
        (session_id, user_id, username, language, has_image, file_path,
         question, answer, tool_used, model_used, token_usage, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, user_id, username, language, has_image, file_path,
          question, answer, tool_used, model_used, token_usage, timestamp))

    conn.commit()
    conn.close()

def log_manual_evaluation(message_id, correct_tool_called, answer_correct, reasoning, evaluated_by):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, tool_used FROM messages WHERE id = ?", (message_id,))
    msg = cursor.fetchone()
    if not msg:
        conn.close()
        return False
    cursor.execute("""
        SELECT expected_tool FROM evaluations
        WHERE message_id = ? ORDER BY id DESC LIMIT 1
    """, (message_id,))
    prev = cursor.fetchone()
    cursor.execute("""
        INSERT INTO evaluations
        (message_id, session_id, expected_tool, actual_tool,
         correct_tool_called, answer_correct, reasoning, evaluated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (message_id, msg["session_id"], prev["expected_tool"] if prev else None,
          msg["tool_used"], correct_tool_called, answer_correct, reasoning, evaluated_by))
    conn.commit()
    conn.close()
    return True

def log_api_call(session_id, api_name, request_endpoint, response_data):
    """
    Inserts a record into the api_logs table (for Task 2 - tool calling / external services).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO api_logs
        (session_id, api_name, request_endpoint, response_data, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, api_name, request_endpoint, response_data, timestamp))

    conn.commit()
    conn.close()



def get_recent_history(session_id, limit=5):
    """
    Fetches the last N interactions for a given session to build conversational memory.
    Returns a list of dicts in chronological order.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
                   SELECT question, answer
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?
                   ''', (session_id, limit))

    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in reversed(rows):
        history.append({
            "question": row[0],
            "answer": row[1]
        })

    return history


def get_message_count(session_id):
    """
    Returns the total number of logged messages for a session. Used to decide
    when the rolling summary is due for an update.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM messages WHERE session_id = ?', (session_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_messages_since(session_id, offset):
    """
    Returns all messages for a session after the first `offset` messages,
    in chronological order. Used to fetch only the "new" exchanges that
    haven't been folded into the rolling summary yet.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT question, answer FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
    ''', (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"question": r[0], "answer": r[1]} for r in rows[offset:]]


def get_summary(session_id):
    """
    Returns the current rolling summary for a session, or None if one
    hasn't been generated yet.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT summary, last_summarized_count FROM session_summaries
        WHERE session_id = ?
    ''', (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return {"summary": row[0], "last_summarized_count": row[1]}


def upsert_summary(session_id, summary, message_count):
    """
    Creates or updates the rolling summary for a session.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO session_summaries (session_id, summary, last_summarized_count, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            summary = excluded.summary,
            last_summarized_count = excluded.last_summarized_count,
            updated_at = excluded.updated_at
    ''', (session_id, summary, message_count, timestamp))

    conn.commit()
    conn.close()


def get_active_session(user_id):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT session_id FROM active_sessions WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def set_active_session(user_id, session_id):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO active_sessions (user_id, session_id)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET session_id = excluded.session_id
    ''', (user_id, session_id))
    conn.commit()
    conn.close()


def init_evaluations_table():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS evaluations
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       message_id
                       INTEGER
                       NOT
                       NULL,
                       session_id
                       TEXT
                       NOT
                       NULL,
                       expected_tool
                       TEXT,
                       actual_tool
                       TEXT,
                       correct_tool_called
                       BOOLEAN,
                       answer_correct
                       BOOLEAN,
                       reasoning
                       TEXT,
                       evaluated_by
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       FOREIGN
                       KEY
                   (
                       message_id
                   ) REFERENCES messages
                   (
                       id
                   )
                       )
                   ''')
    conn.commit()
    conn.close()


def log_evaluation(message_id, session_id, expected_tool, actual_tool,
                   correct_tool_called, answer_correct, reasoning, evaluated_by):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
                   INSERT INTO evaluations
                   (message_id, session_id, expected_tool, actual_tool,
                    correct_tool_called, answer_correct, reasoning, evaluated_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   """, (message_id, session_id, expected_tool, actual_tool,
                         correct_tool_called, answer_correct, reasoning, evaluated_by))
    conn.commit()
    conn.close()


def get_unevaluated_messages(limit=50):

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT m.id, m.session_id, m.question, m.answer, m.tool_used
                   FROM messages m
                            LEFT JOIN evaluations e ON m.id = e.message_id
                   WHERE e.id IS NULL
                   ORDER BY m.timestamp ASC LIMIT ?
                   """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows] if rows else []

if __name__ == "__main__":
    initialize_database()
    print("Database infrastructure and downloads folder successfully created!")