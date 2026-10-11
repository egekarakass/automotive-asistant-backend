import { useState } from "react";
import { t } from "../i18n";

function AuthScreen({ apiBase, lang, onAuthSuccess }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    const endpoint = mode === "login" ? "/login" : "/register";
    try {
      const res = await fetch(`${apiBase}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error");
      localStorage.setItem("user_token", data.access_token);
      localStorage.setItem("username", username);
      onAuthSuccess(data.access_token, username);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-screen">
      <form className="auth-box" onSubmit={handleSubmit}>
        <h2>{mode === "login" ? t("loginTitle", lang) : t("registerTitle", lang)}</h2>
        <input
          type="text"
          placeholder={t("username", lang)}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
        />
        <input
          type="password"
          placeholder={t("password", lang)}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="auth-error">{error}</div>}
        <button type="submit" disabled={loading}>
          {loading ? t("loading", lang) : mode === "login" ? t("loginButton", lang) : t("registerButton", lang)}
        </button>
        <span className="auth-switch" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login" ? t("switchToRegister", lang) : t("switchToLogin", lang)}
        </span>
      </form>
    </div>
  );
}

export default AuthScreen;