import { useState, useEffect } from "react";
import Login from "./components/Login";
import SessionList from "./components/SessionList";
import SessionDetail from "./components/SessionDetail";
import StatsPanel from "./components/StatsPanel";
import "./App.css";
import { LangContext, makeT } from "./i18n";

const API_BASE = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [token, setToken] = useState(() => localStorage.getItem("admin_token"));
  const [selectedSessionId, setSelectedSessionId] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [theme, setTheme] = useState(() => localStorage.getItem("admin_theme") || "dark");
  const [lang, setLang] = useState(() => localStorage.getItem("admin_lang") || "tr");

  useEffect(() => {
    localStorage.setItem("admin_lang", lang);
  }, [lang]);

  const t = makeT(lang);
  const toggleLang = () => setLang((l) => (l === "tr" ? "en" : "tr"));
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("admin_theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  };

  const handleLogout = () => {
    localStorage.removeItem("admin_token");
    setToken(null);
  };

  const handleAuthError = () => {
    handleLogout();
  };

  if (!token) {
    return (
        <LangContext.Provider value={{lang, t, toggleLang}}>
          <Login apiBase={API_BASE} onLoginSuccess={setToken}/>
        </LangContext.Provider>
    );
  }

  return (

      <LangContext.Provider value={{lang, t, toggleLang}}>
        <div className="app-layout">
          <header className="app-header">
            <h1>{t("panelTitle")}</h1>
            <div className="header-actions">
              <button onClick={toggleTheme}>
                {theme === "dark" ? t("lightMode") : t("darkMode")}
              </button>
              <button onClick={toggleLang}>{lang === "tr" ? "EN" : "TR"}</button>
              <button onClick={() => setRefreshKey((k) => k + 1)}>{t("refresh")}</button>
              <button onClick={handleLogout}>{t("logout")}</button>
            </div>
          </header>

          <StatsPanel apiBase={API_BASE} token={token} refreshKey={refreshKey} onAuthError={handleAuthError}/>

          <div className="main-content">
            <SessionList
                apiBase={API_BASE}
                token={token}
                selectedSessionId={selectedSessionId}
                onSelectSession={setSelectedSessionId}
                refreshKey={refreshKey}
                onAuthError={handleAuthError}
            />
            <SessionDetail
                apiBase={API_BASE}
                token={token}
                sessionId={selectedSessionId}
                refreshKey={refreshKey}
                onAuthError={handleAuthError}
                onEvaluationChanged={() => setRefreshKey((k) => k + 1)}
            />
          </div>
        </div>
      </LangContext.Provider>
  );
}
export default App;