import os
import telebot
from dotenv import load_dotenv

load_dotenv()

bot = telebot.TeleBot(os.getenv("TELEGRAM_TOKEN"))
bot.remove_webhook()
bot.set_webhook(
    url=f"{os.getenv('PUBLIC_BASE_URL')}/telegram_webhook",
    secret_token=os.getenv("TELEGRAM_WEBHOOK_SECRET"),
)
info = bot.get_webhook_info()
print("url:", info.url)
print("pending:", info.pending_update_count)
print("last_error:", info.last_error_message)