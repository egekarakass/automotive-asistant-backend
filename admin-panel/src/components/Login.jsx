import { useState } from "react";
import { useLang } from "../i18n";

function Login({ apiBase, onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const { t, lang, toggleLang } = useLang();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/admin/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || t("loginFailed"));
      }
      const data = await res.json();
      localStorage.setItem("admin_token", data.access_token);
      onLoginSuccess(data.access_token);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-screen">
      <form className="login-box" onSubmit={handleSubmit}>
        <h2>{t("adminLogin")}</h2>
        <input
          type="text"
          placeholder={t("username")}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
        />
        <input
          type="password"
          placeholder={t("password")}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="login-error">{error}</div>}
        <button type="submit" disabled={loading}>
          {loading ? t("loggingIn") : t("loginButton")}
        </button>

        {/* Dil Değiştirme Butonu */}
        <button
          type="button"
          onClick={toggleLang}
          style={{
            background: "rgba(255, 255, 255, 0.05)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            boxShadow: "none",
            color: "#9ca3af",
            marginTop: "0px"
          }}
        >
          {lang === "tr" ? "English" : "Türkçe"}
        </button>
      </form>
    </div>
  );
}

export default Login;