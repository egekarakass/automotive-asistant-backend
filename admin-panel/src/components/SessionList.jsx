import { useState, useEffect } from "react";
import { useLang } from "../i18n";

function SessionList({ apiBase, token, selectedSessionId, onSelectSession, refreshKey, onAuthError }) {
  const { t } = useLang();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${apiBase}/admin/sessions?limit=50`, {
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
        setSessions(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [apiBase, token, refreshKey]);

  if (loading) return <div className="session-list">{t("loading")}</div>;
  if (error) return <div className="session-list">{t("error")}: {error}</div>;

  return (
    <div className="session-list">
      <h2>{t("conversations")} ({sessions.length})</h2>
      <ul>
        {sessions.map((s) => (
          <li
            key={s.session_id}
            className={s.session_id === selectedSessionId ? "selected" : ""}
            onClick={() => onSelectSession(s.session_id)}
          >
            <div className="session-row-top">
              <strong>{s.username || t("unknown")}</strong>
              <span className="lang-badge">{s.language || "?"}</span>
            </div>
            <div className="session-id-mini">{s.session_id}</div>
            <div className="session-row-bottom">
              <span>{s.message_count} {t("messagesShort")}</span>
              <span>{s.last_activity}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default SessionList;