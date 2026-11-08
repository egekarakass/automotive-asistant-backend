import json
from database import DB_NAME, log_evaluation, get_unevaluated_messages
from ai_handler import client

JUDGE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
JUDGE_FALLBACK_MODEL = "cohere/north-mini-code:free"

JUDGE_SYSTEM_PROMPT = """You are a QA evaluator for an automotive assistant bot.
Given a user question, the tool the bot actually called, and the bot's answer,
determine:
1. Which tool SHOULD have been called (expected_tool): one of
   get_recalls, get_complaints, decode_vin, get_fuel_prices, or "none" if no
   tool was needed.
2. Whether the ACTUAL tool called matches the expected one (correct_tool_called).
3. Whether the answer itself is factually reasonable and addresses the question
   (answer_correct) — judge based on internal consistency and whether it directly
   answers what was asked, not external fact-checking.

Respond with ONLY a JSON object, no markdown fences, no extra text:
{"expected_tool": "...", "correct_tool_called": true/false,
 "answer_correct": true/false, "reasoning": "one short sentence"}
"""


def _parse_judge_response(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return json.loads(cleaned)


def _call_judge(user_content, model):
    completion = client.chat.completions.create(
        model=model,
        timeout=30,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    )
    return completion.choices[0].message.content


def evaluate_message(message_id, session_id, question, answer, tool_used):
    user_content = (
        f"Question: {question}\n"
        f"Tool called by bot: {tool_used or 'none'}\n"
        f"Bot's answer: {answer}"
    )

    used_model = JUDGE_MODEL
    try:
        raw = _call_judge(user_content, JUDGE_MODEL)
    except Exception as e:
        print(f"[EVALUATOR] Primary judge ({JUDGE_MODEL}) failed: {e} — trying fallback")
        try:
            raw = _call_judge(user_content, JUDGE_FALLBACK_MODEL)
            used_model = JUDGE_FALLBACK_MODEL
        except Exception as e2:
            print(f"[EVALUATOR] Fallback judge ({JUDGE_FALLBACK_MODEL}) also failed: {e2}")
            return False

    try:
        result = _parse_judge_response(raw)
    except Exception as e:
        print(f"[EVALUATOR] Failed to parse judge response from {used_model}: {e} — raw: {raw}")
        return False

    log_evaluation(
        message_id=message_id,
        session_id=session_id,
        expected_tool=result.get("expected_tool"),
        actual_tool=tool_used,
        correct_tool_called=result.get("correct_tool_called"),
        answer_correct=result.get("answer_correct"),
        reasoning=result.get("reasoning"),
        evaluated_by=f"auto_llm:{used_model}"
    )
    return True


def run_evaluation_batch(limit=50):
    rows = get_unevaluated_messages(limit=limit)
    success_count = 0
    for row in rows:
        # Veritabanından gelen sözlük (dictionary) yapısına uygun çekim:
        if evaluate_message(row["id"], row["session_id"], row["question"], row["answer"], row["tool_used"]):
            success_count += 1

    print(f"[EVALUATOR] Evaluated {success_count}/{len(rows)} messages")
    return success_count, len(rows)