import os
import bcrypt
import jwt
import sqlite3
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from database import DB_NAME
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 12
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

def _get_admin_user(username):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin_users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def _get_lockout_status(username):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM login_attempts WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def _register_failed_attempt(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    existing = _get_lockout_status(username)
    new_count = (existing["failed_count"] if existing else 0) + 1
    locked_until = None
    if new_count >= MAX_FAILED_ATTEMPTS:
        locked_until = (datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO login_attempts (username, failed_count, locked_until)
        VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            failed_count = excluded.failed_count,
            locked_until = excluded.locked_until
    ''', (username, new_count, locked_until))
    conn.commit()
    conn.close()

def _clear_failed_attempts(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM login_attempts WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def authenticate(username: str, password: str) -> str:
    lockout = _get_lockout_status(username)
    if lockout and lockout["locked_until"]:
        locked_until_dt = datetime.strptime(lockout["locked_until"], "%Y-%m-%d %H:%M:%S")
        if datetime.now() < locked_until_dt:
            raise HTTPException(
                status_code=429,
                detail=f"Çok fazla başarısız deneme. {LOCKOUT_MINUTES} dakika sonra tekrar deneyin."
            )

    user = _get_admin_user(username)
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        _register_failed_attempt(username)
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

    _clear_failed_attempts(username)
    payload = {
        "sub": username,
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Yetkilendirme başlığı eksik")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Bu işlem için yönetici yetkisi gerekir")
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Oturum süresi doldu, tekrar giriş yapın")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Geçersiz token")