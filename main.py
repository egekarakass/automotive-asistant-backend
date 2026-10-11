import telebot
import os
import time
from dotenv import load_dotenv
import database
import threading
import ai_handler
import requests

# Load environment variables
load_dotenv()

# Securely fetch the token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Ensure database and tables are ready
database.initialize_database()

TELEGRAM_MESSAGE_LIMIT = 4096

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def _call_api(endpoint, timeout=120, **kwargs):
    """API'ye istek atar, hata durumunda None döner — çağıran taraf
    kullanıcıya nazik bir hata mesajı gösterebilsin diye, bot'u
    çökertmez."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[API CALL FAILED] {endpoint}: {e}")
        return None

def send_long_message(chat_id, text, reply_to_message_id=None):
    """
    Telegram rejects any single message over 4096 characters. Splits long
    answers into safe chunks and sends them as consecutive messages
    instead of crashing.
    """
    if not text or not text.strip():
        text = "Üzgünüm, bu soru için bir cevap üretemedim. Lütfen tekrar deneyin."
        print("[SEND WARNING] Empty bot_response — sending fallback text instead of crashing.")

    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id)
        return

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= TELEGRAM_MESSAGE_LIMIT:
            chunks.append(remaining)
            break
        cut_point = remaining.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if cut_point < TELEGRAM_MESSAGE_LIMIT // 2:
            cut_point = remaining.rfind(" ", 0, TELEGRAM_MESSAGE_LIMIT)
        if cut_point <= 0:
            cut_point = TELEGRAM_MESSAGE_LIMIT
        chunks.append(remaining[:cut_point])
        remaining = remaining[cut_point:].lstrip()

    for i, chunk in enumerate(chunks):
        bot.send_message(
            chat_id, chunk,
            reply_to_message_id=reply_to_message_id if i == 0 else None
        )


def _get_or_create_session(user_id):
    """
    Returns the user's currently active session_id, creating one (as a
    fallback) if they somehow send a message without ever running /start.
    """
    session_id = database.get_active_session(user_id)
    if session_id is None:
        session_id = f"session_{user_id}_{int(time.time())}"
        database.set_active_session(user_id, session_id)
    return session_id

def _get_display_name(user):
    """
    Telegram's `username` field (the public @handle) is often unset, since
    it's optional — falling back to a generic "Unknown" loses real
    identifying info we already have. Prefer the person's actual display
    name (first_name + last_name, always present) and only fall back to
    the @username or "Unknown" as a last resort.
    """
    parts = [user.first_name, user.last_name]
    display_name = " ".join(p for p in parts if p)
    if display_name:
        return display_name
    if user.username:
        return f"@{user.username}"
    return "Unknown"

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    print(f"[DEBUG] Handler triggered for message_id={message.message_id}, text={message.text}")
    user_id = message.from_user.id
    session_id = _get_or_create_session(user_id)
    user_question = message.text

    bot.send_chat_action(message.chat.id, 'typing')

    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"message": user_question, "session_id": session_id},
        timeout=120
    )
    response.raise_for_status()
    ai_result = response.json()
    bot_response = ai_result["response"]

    send_long_message(message.chat.id, bot_response, reply_to_message_id=message.message_id)
    # NOT: database.log_message artık burada ÇAĞRILMIYOR —
    # /ask endpoint'i bu logu zaten kendisi yapıyor. Burada tekrar
    # çağırırsak her mesaj DB'ye iki kez yazılır.


@bot.message_handler(content_types=['photo'])
def handle_photo_message(message):
    user_id = message.from_user.id
    session_id = _get_or_create_session(user_id)
    caption = message.caption or ""

    bot.send_chat_action(message.chat.id, 'typing')

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    files = {"image": (f"{message.message_id}.jpg", downloaded_file)}
    data = {"caption": caption, "session_id": session_id}
    response = requests.post(f"{API_BASE_URL}/ask_vision", files=files, data=data, timeout=120)
    response.raise_for_status()
    ai_result = response.json()
    bot_response = ai_result["response"]

    send_long_message(message.chat.id, bot_response, reply_to_message_id=message.message_id)


@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    user_id = message.from_user.id
    session_id = _get_or_create_session(user_id)

    bot.send_chat_action(message.chat.id, 'typing')

    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    files = {"audio": (f"{message.message_id}.ogg", downloaded_file)}
    data = {"session_id": session_id}
    response = requests.post(f"{API_BASE_URL}/ask_voice", files=files, data=data, timeout=120)
    response.raise_for_status()
    ai_result = _call_api("/ask_voice", files=files, data=data, timeout=300)
    bot_response = ai_result[
        "response"] if ai_result else "Üzgünüm, sesli mesajı işlerken sunucuya ulaşamadım. Lütfen tekrar deneyin."
    send_long_message(message.chat.id, bot_response, reply_to_message_id=message.message_id)

print("Automotive Bot is successfully running and listening for messages...")
bot.infinity_polling()