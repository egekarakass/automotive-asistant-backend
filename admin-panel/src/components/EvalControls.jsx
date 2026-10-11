import { useState } from "react";
import { useLang } from "../i18n";

function EvalControls({ apiBase, token, messageId, current, toolInvolved, onSaved, onAuthError }) {
  const { t } = useLang();
  const [editing, setEditing] = useState(false);
  const [toolOk, setToolOk] = useState(true);
  const [answerOk, setAnswerOk] = useState(true);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

  async function save(tool, answer, note) {
    setSaving(true);
    try {
      const res = await fetch(`${apiBase}/admin/evaluations/override`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          message_id: messageId,
          correct_tool_called: tool,
          answer_correct: answer,
          notes: note || null,
        }),
      });
      if (res.status === 401) return onAuthError();
      if (!res.ok) throw new Error();
      setEditing(false);
      onSaved();
    } catch {
      alert(t("saveFailed"));
    } finally {
      setSaving(false);
    }
  }

  function openEditor() {
    setToolOk(current ? !!current.correct_tool_called : true);
    setAnswerOk(current ? !!current.answer_correct : true);
    setNotes("");
    setEditing(true);
  }

  if (!editing) {
    const isAuto = current && !current.evaluated_by.startsWith("manual");
    return (
      <div className="eval-controls">
        {isAuto && (
          <button
            disabled={saving}
            onClick={() => save(!!current.correct_tool_called, !!current.answer_correct, "Otomatik değerlendirme onaylandı")}
          >
            {t("approve")}
          </button>
        )}
        <button onClick={openEditor}>{current ? t("fix") : t("evaluate")}</button>
      </div>
    );
  }

  return (
    <div className="eval-form">
      {toolInvolved && (
        <label>
          <input type="checkbox" checked={toolOk} onChange={(e) => setToolOk(e.target.checked)} />
          {t("toolCalledCorrectly")}
        </label>
      )}
      <label>
        <input type="checkbox" checked={answerOk} onChange={(e) => setAnswerOk(e.target.checked)} />
        {t("answerIsCorrect")}
      </label>
      <input
        type="text"
        placeholder={t("notePlaceholder")}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
      />
      <div className="eval-controls">
        <button disabled={saving} onClick={() => save(toolOk, answerOk, notes)}>{t("save")}</button>
        <button onClick={() => setEditing(false)}>{t("cancel")}</button>
      </div>
    </div>
  );
}

export default EvalControls;