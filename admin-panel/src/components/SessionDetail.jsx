import { useState, useEffect } from "react";
import EvalControls from "./EvalControls";
import { useLang } from "../i18n";

function SessionDetail({ apiBase, sessionId, refreshKey, token, onAuthError, onEvaluationChanged }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const { t } = useLang();

  useEffect(() => {
    if (!sessionId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    fetch(`${apiBase}/admin/sessions/${sessionId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (res.status === 401) {
          onAuthError();
          throw new Error("Oturum süresi doldu");
        }
        return res.json();
      })
      .then((data) => {
        setDetail(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [apiBase, token, sessionId, refreshKey]);

  if (!sessionId) {
    return <div className="session-detail empty">{t("selectConversation")}</div>;
  }
  if (loading) return <div className="session-detail">{t("loading")}</div>;
  if (!detail) return <div className="session-detail">{t("noData")}</div>;

  return (
    <div className="session-detail">
      <h2>{t("conversationDetail")}</h2>
      <p className="session-id-label">Session: {sessionId}</p>

      {detail.summary && (
        <div className="detail-block">
          <h3>{t("summary")}</h3>
          <p>{detail.summary}</p>
        </div>
      )}

      <div className="detail-block">
        <h3>{t("messages")} ({detail.messages.length})</h3>
        <div className="transcript">
          {detail.messages.map((m) => {
            const msgEvals = detail.evaluations.filter((e) => e.message_id === m.id);
            const evalForMsg = msgEvals[msgEvals.length - 1];
            const toolInvolved =
              (m.tool_used && m.tool_used !== "None") ||
              (evalForMsg?.expected_tool && !["none", "None"].includes(evalForMsg.expected_tool));
            return (
              <div key={m.id} className="message-pair">
                <div className="bubble user-bubble">
                  <span className="bubble-meta">
                    {m.timestamp} {m.has_image ? "📷" : ""}
                  </span>
                  {m.question}
                </div>
                <div className="bubble bot-bubble">
                  <span className="bubble-meta">
                    {m.model_used} {m.tool_used !== "None" ? `· ${m.tool_used}` : ""}
                  </span>
                  {m.answer}
                  {evalForMsg && (
                    <div className="eval-tag">
                      {toolInvolved && (
                        <>
                          {evalForMsg.correct_tool_called ? `✅ ${t("correctTool")}` : `❌ ${t("wrongTool")}`}
                          {" · "}
                        </>
                      )}
                      {evalForMsg.answer_correct ? `✅ ${t("correctAnswer")}` : `❌ ${t("wrongAnswer")}`}
                      {" · "}
                      {evalForMsg.evaluated_by.startsWith("manual") ? `👤 ${t("manual")}` : `🤖 ${t("automatic")}`}
                      <div className="eval-reasoning">{evalForMsg.reasoning}</div>
                    </div>
                  )}
                  <EvalControls
                    apiBase={apiBase}
                    token={token}
                    messageId={m.id}
                    current={evalForMsg}
                    toolInvolved={toolInvolved}
                    onSaved={onEvaluationChanged}
                    onAuthError={onAuthError}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="detail-block">
        <h3>{t("toolCalls")} ({detail.tool_calls.length})</h3>
        <table className="tool-calls-table">
          <thead>
            <tr>
              <th>{t("colTime")}</th>
              <th>{t("colApi")}</th>
              <th>{t("colParams")}</th>
            </tr>
          </thead>
          <tbody>
            {detail.tool_calls.map((tc, i) => (
              <tr key={i}>
                <td>{tc.timestamp}</td>
                <td>{tc.api_name}</td>
                <td className="mono-cell">{tc.request_endpoint}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default SessionDetail;