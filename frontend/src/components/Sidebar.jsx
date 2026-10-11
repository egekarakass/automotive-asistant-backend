import { useState, useEffect } from "react";
import { t } from "../i18n";

function Sidebar({ apiBase, token, lang, currentSessionId, onSelectSession, onNewChat, username, onLogout, refreshKey }) {
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    fetch(`${apiBase}/my_sessions`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then(setSessions)
      .catch(() => setSessions([]));
  }, [apiBase, token, refreshKey]);

  return (
    <div className="sidebar">
      <button className="new-chat-btn" onClick={onNewChat}>
        {t("newChat", lang)}
      </button>
      <div className="sidebar-sessions">
        {sessions.length === 0 && <div className="sidebar-empty">{t("noSessions", lang)}</div>}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`sidebar-item ${s.session_id === currentSessionId ? "active" : ""}`}
            onClick={() => onSelectSession(s.session_id)}
          >
            {s.title || s.session_id}
          </div>
        ))}
      </div>
      <div className="sidebar-footer">
        <span className="sidebar-username">{username}</span>
        <button className="logout-btn" onClick={onLogout}>{t("logout", lang)}</button>
      </div>
    </div>
  );
}

export default Sidebar;