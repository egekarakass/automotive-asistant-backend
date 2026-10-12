import os
import hmac
import threading
from dotenv import load_dotenv

load_dotenv()

import telebot
import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException

import main  # handler'ları kaydeder, main.bot'u getirir

WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
app = FastAPI()


@app.post("/telegram_webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(None),
):
    if not WEBHOOK_SECRET or not hmac.compare_digest(
        x_telegram_bot_api_secret_token or "", WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=403, detail="Forbidden")

    update = telebot.types.Update.de_json(await request.json())
    threading.Thread(
        target=main.bot.process_new_updates, args=([update],), daemon=True
    ).start()
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)