import os
import bcrypt
import jwt
import sqlite3
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from dotenv import load_dotenv
from database import DB_NAME

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24 * 7  # normal kullanıcılar için daha uzun ömürlü (1 hafta)

def register_user(username: str, password: str) -> str:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, hashed, timestamp)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return _issue_token(user_id, username)

def authenticate_user(username: str, password: str) -> str:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row or not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

    return _issue_token(row["id"], username)

def _issue_token(user_id: int, username: str) -> str:
    payload = {
        "sub": username,
        "user_id": user_id,
        "role": "user",
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)

def verify_user_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Yetkilendirme başlığı eksik")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        if payload.get("role") != "user":
            raise HTTPException(status_code=403, detail="Geçersiz yetki")
        return {"user_id": payload["user_id"], "username": payload["sub"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Oturum süresi doldu, tekrar giriş yapın")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Geçersiz token")