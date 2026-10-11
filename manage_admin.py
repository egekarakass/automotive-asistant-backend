import sys
import bcrypt
import sqlite3
from datetime import datetime
from database import DB_NAME, initialize_database

def set_admin_password(username, password):
    initialize_database()
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO admin_users (username, password_hash, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET password_hash = excluded.password_hash
    ''', (username, hashed, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # Reset any lockout on manual password reset
    cursor.execute("DELETE FROM login_attempts WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    print(f"'{username}' için şifre başarıyla ayarlandı/sıfırlandı.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanım: python manage_admin.py <kullanıcı_adı> <yeni_şifre>")
        sys.exit(1)
    set_admin_password(sys.argv[1], sys.argv[2])