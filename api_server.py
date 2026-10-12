import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import ai_handler
import database
import sqlite3
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from stt import transcribe_audio
from fastapi import Depends
from pydantic import BaseModel
from auth import authenticate, verify_token
from user_auth import register_user, authenticate_user, verify_user_token, verify_user_or_bot
from evaluator import run_evaluation_batch
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Automotive Assistant API",
    description="Otomotiv ve Taşıma Sistemleri Test Arayüzü",
    version="1.0.0"
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def preload_models():
    database.initialize_database()
    from stt import _get_whisper_model
    print("Ses tanıma modeli yükleniyor (ilk kurulumda internetten indirilebilir, biraz sürebilir)...")
    _get_whisper_model()
    print("Ses tanıma modeli hazır.")

# Metin istekleri için JSON şeması
class TextRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.post("/ask")
def ask(request: TextRequest, current_user: dict = Depends(verify_user_or_bot)):
    session_id = request.session_id or "http_api_unknown"
    owner_id = database.claim_session(session_id, current_user["user_id"], title=request.message[:50])
    if owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Bu sohbet başka bir kullanıcıya ait")

    result = ai_handler.get_ai_response(request.message, session_id=session_id)
    database.log_message(
        session_id=session_id,
        user_id=current_user["user_id"],
        username=current_user["username"],
        language=ai_handler.detect_language_for_logging(request.message),
        has_image=False,
        file_path="",
        question=request.message,
        answer=result["response"],
        tool_used=result.get("tool_used", "None"),
        model_used=result.get("model_used", "None"),
        token_usage=result.get("token_usage", 0)
    )
    return ai_handler.sanitize_for_external_response(result)


@app.post("/ask_vision")
async def ask_vision(
    image: UploadFile = File(...),
    caption: str = Form(""),
    session_id: str = Form(None),
    current_user: dict = Depends(verify_user_or_bot)
):
    session_id = session_id or "http_api_unknown"
    owner_id = database.claim_session(session_id, current_user["user_id"], title=caption[:50] or "Görsel sohbet")
    if owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Bu sohbet başka bir kullanıcıya ait")

    temp_file_path = f"temp_{image.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        result = ai_handler.get_vision_response(temp_file_path, caption, session_id)

        database.log_message(
            session_id=session_id,
            user_id=current_user["user_id"],
            username=current_user["username"],
            language=ai_handler.detect_language_for_logging(caption),
            has_image=True,
            file_path=temp_file_path,
            question=caption,
            answer=result["response"],
            tool_used=result.get("tool_used", "None"),
            model_used=result.get("model_used", "None"),
            token_usage=result.get("token_usage", 0)
        )

        return ai_handler.sanitize_for_external_response(result)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask_voice")
async def ask_voice(
    audio: UploadFile = File(...),
    session_id: str = Form(None),
    current_user: dict = Depends(verify_user_or_bot)
):
    session_id = session_id or "http_api_unknown"
    owner_id = database.claim_session(session_id, current_user["user_id"], title="Sesli sohbet")
    if owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Bu sohbet başka bir kullanıcıya ait")

    temp_file_path = f"temp_voice_{audio.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        transcribed_text, _ = transcribe_audio(temp_file_path)
        print(f"[VOICE DEBUG] Transcribed: {transcribed_text!r}")
        if not transcribed_text:
            return {
                "response": "Üzgünüm, ses kaydını anlayamadım. Lütfen tekrar deneyin.",
                "transcribed_text": ""
            }

        result = ai_handler.get_ai_response(transcribed_text, session_id=session_id)

        database.log_message(
            session_id=session_id,
            user_id=current_user["user_id"],
            username=current_user["username"],
            language=ai_handler.detect_language_for_logging(transcribed_text),
            has_image=False,
            file_path=temp_file_path,
            question=f"[voice] {transcribed_text}",
            answer=result["response"],
            tool_used=result.get("tool_used", "None"),
            model_used=result.get("model_used", "None"),
            token_usage=result.get("token_usage", 0)
        )

        sanitized = ai_handler.sanitize_for_external_response(result)
        sanitized["transcribed_text"] = transcribed_text
        return sanitized
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/my_sessions")
def my_sessions(current_user: dict = Depends(verify_user_token)):
    return database.get_user_sessions(current_user["user_id"])

@app.get("/my_sessions/{session_id}")
def my_session_detail(session_id: str, current_user: dict = Depends(verify_user_token)):
    owner_id = database.get_session_owner(session_id)
    if owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Bu sohbete erişim izniniz yok")
    history = database.get_messages_since(session_id, offset=0)
    return {"session_id": session_id, "messages": history}
@app.post("/admin/login")
def login(request: LoginRequest):
    token = authenticate(request.username, request.password)
    return {"access_token": token, "token_type": "bearer"}

class ManualEvaluation(BaseModel):
    message_id: int
    correct_tool_called: bool
    answer_correct: bool
    notes: str | None = None

@app.post("/admin/evaluations/override")
def override_evaluation(body: ManualEvaluation, admin_user: str = Depends(verify_token)):
    ok = database.log_manual_evaluation(
        body.message_id, body.correct_tool_called, body.answer_correct,
        body.notes, f"manual:{admin_user}"
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Mesaj bulunamadı")
    return {"status": "saved"}

@app.post("/admin/run_evaluation")
async def trigger_evaluation(limit: int = 50, admin_user: str = Depends(verify_token)):
    success_count, total = run_evaluation_batch(limit=limit)
    return {
        "status": "completed",
        "evaluated": success_count,
        "total_attempted": total
    }

@app.get("/admin/sessions")
def list_sessions(limit: int = 50, admin_user: str = Depends(verify_token)):
    conn = sqlite3.connect(database.DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id,
               COUNT(*) as message_count,
               MAX(timestamp) as last_activity,
               MIN(timestamp) as started_at,
               MAX(username) as username,
               MAX(language) as language
        FROM messages
        GROUP BY session_id
        ORDER BY last_activity DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

@app.get("/admin/sessions/{session_id}")
def get_session_detail(session_id: str, admin_user: str = Depends(verify_token)):
    conn = sqlite3.connect(database.DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, question, answer, tool_used, model_used, has_image, timestamp
        FROM messages WHERE session_id = ? ORDER BY timestamp ASC
    """, (session_id,))
    messages = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT api_name, request_endpoint, response_data, timestamp
        FROM api_logs WHERE session_id = ? ORDER BY timestamp ASC
    """, (session_id,))
    tool_calls = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT * FROM evaluations WHERE session_id = ? ORDER BY timestamp ASC
    """, (session_id,))
    evaluations = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,))
    summary_row = cursor.fetchone()
    summary = summary_row["summary"] if summary_row else None

    conn.close()
    return {
        "session_id": session_id,
        "summary": summary,
        "messages": messages,
        "tool_calls": tool_calls,
        "evaluations": evaluations
    }


@app.get("/admin/stats")
def get_stats(admin_user: str = Depends(verify_token)):
    conn = sqlite3.connect(database.DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT model_used, COUNT(*) as c FROM messages GROUP BY model_used")
    model_usage = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT language, COUNT(*) as c FROM messages GROUP BY language")
    language_dist = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT has_image, COUNT(*) as c FROM messages GROUP BY has_image")
    image_ratio = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
                   SELECT SUM(CASE WHEN tool_rel = 1 AND correct_tool_called = 1 THEN 1 ELSE 0 END) as correct_tool,
                          SUM(CASE WHEN tool_rel = 1 THEN 1 ELSE 0 END)                             as tool_total,
                          SUM(CASE WHEN answer_correct = 1 THEN 1 ELSE 0 END)                       as correct_answer,
                          COUNT(*)                                                                  as total
                   FROM (SELECT *,
                                (COALESCE(actual_tool, 'None') NOT IN ('None', 'none', '')
                                    OR COALESCE(expected_tool, 'none') NOT IN ('none', 'None', '')) AS tool_rel
                         FROM evaluations
                         WHERE id IN (SELECT MAX(id) FROM evaluations GROUP BY message_id))
                   """)
    eval_summary = dict(cursor.fetchone())

    conn.close()
    return {
        "model_usage": model_usage,
        "language_distribution": language_dist,
        "image_ratio": image_ratio,
        "evaluation_summary": eval_summary
    }

class AuthRequest(BaseModel):
    username: str
    password: str

@app.post("/register")
def register(request: AuthRequest):
    token = register_user(request.username, request.password)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login")
def user_login(request: AuthRequest):
    token = authenticate_user(request.username, request.password)
    return {"access_token": token, "token_type": "bearer"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)