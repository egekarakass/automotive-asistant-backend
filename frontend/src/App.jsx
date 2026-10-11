import { useState, useRef, useEffect } from "react";
import AuthScreen from "./components/AuthScreen";
import Sidebar from "./components/Sidebar";
import { t, getLang, setLang } from "./i18n";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("user_token"));
  const [username, setUsername] = useState(() => localStorage.getItem("username"));
  const [lang, setLangState] = useState(getLang());
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [sidebarRefresh, setSidebarRefresh] = useState(0);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [pendingImage, setPendingImage] = useState(null);
  const [speakingId, setSpeakingId] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function toggleLang() {
    const newLang = lang === "tr" ? "en" : "tr";
    setLang(newLang);
    setLangState(newLang);
  }

  const [theme, setTheme] = useState(() => localStorage.getItem("chat_theme") || "light");

useEffect(() => {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("chat_theme", theme);
}, [theme]);

function toggleTheme() {
  setTheme((t) => (t === "light" ? "dark" : "light"));
}

  function handleAuthSuccess(newToken, newUsername) {
    setToken(newToken);
    setUsername(newUsername);
  }

  function handleLogout() {
    localStorage.removeItem("user_token");
    localStorage.removeItem("username");
    setToken(null);
    setUsername(null);
  }

  function handleNewChat() {
    setSessionId(crypto.randomUUID());
    setMessages([]);
  }

  async function handleSelectSession(id) {
    setSessionId(id);
    try {
      const res = await fetch(`${API_BASE}/my_sessions/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.status === 401) return handleLogout();
      const data = await res.json();
      const loaded = data.messages.flatMap((m) => [
        { content: m.question, sender: "user", type: "text", id: crypto.randomUUID() },
        { content: m.answer, sender: "bot", type: "text", id: crypto.randomUUID() },
      ]);
      setMessages(loaded);
    } catch (err) {
      console.error(err);
    }
  }

  function speak(text, id) {
    if (!("speechSynthesis" in window)) return;
    if (speakingId === id) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = /[çğıöşüÇĞİÖŞÜ]/.test(text) ? "tr-TR" : "en-US";
    utterance.onend = () => setSpeakingId(null);
    window.speechSynthesis.speak(utterance);
    setSpeakingId(id);
  }

  function addMessage(content, sender, type) {
    setMessages((prev) => [...prev, { content, sender, type, id: crypto.randomUUID() }]);
  }

  function authHeader() {
    return { Authorization: `Bearer ${token}` };
  }

  async function sendText(message) {
    addMessage(message, "user", "text");
    setInput("");

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeader() },
        body: JSON.stringify({ message, session_id: sessionId }),
      });
      if (res.status === 401) return handleLogout();
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      addMessage(data.response, "bot", "text");
      setSidebarRefresh((k) => k + 1);
    } catch (err) {
      addMessage(t("connectionError", lang), "bot", "text");
      console.error(err);
    }
  }

  async function sendImage(file, caption) {
    const imageUrl = URL.createObjectURL(file);
    addMessage(imageUrl, "user", "image");
    if (caption) addMessage(caption, "user", "text");

    const formData = new FormData();
    formData.append("image", file);
    formData.append("caption", caption || "");
    formData.append("session_id", sessionId);

    try {
      const res = await fetch(`${API_BASE}/ask_vision`, {
        method: "POST",
        headers: authHeader(),
        body: formData,
      });
      if (res.status === 401) return handleLogout();
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      addMessage(data.response, "bot", "text");
      setSidebarRefresh((k) => k + 1);
    } catch (err) {
      addMessage(t("imageError", lang), "bot", "text");
      console.error(err);
    }
  }

  function handleImageChange(e) {
    const file = e.target.files[0];
    if (file) setPendingImage({ file, url: URL.createObjectURL(file) });
    e.target.value = "";
  }

  async function toggleRecording() {
    if (isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });

        const audioUrl = URL.createObjectURL(audioBlob);
        addMessage(audioUrl, "user", "audio");

        const formData = new FormData();
        formData.append("audio", audioBlob, "voice.webm");
        formData.append("session_id", sessionId);

        try {
          const res = await fetch(`${API_BASE}/ask_voice`, {
            method: "POST",
            headers: authHeader(),
            body: formData,
          });
          if (res.status === 401) return handleLogout();
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          if (data.transcribed_text) addMessage(data.transcribed_text, "user", "text");
          addMessage(data.response, "bot", "text");
          setSidebarRefresh((k) => k + 1);
        } catch (err) {
          addMessage(t("voiceError", lang), "bot", "text");
          console.error(err);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      alert("Mikrofona erişilemedi: " + err.message);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (pendingImage) {
      sendImage(pendingImage.file, text);
      setPendingImage(null);
      setInput("");
      return;
    }
    if (text) sendText(text);
  }

  if (!token) {
    return <AuthScreen apiBase={API_BASE} lang={lang} onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <div className="app-shell">
      <Sidebar
        apiBase={API_BASE}
        token={token}
        lang={lang}
        currentSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        username={username}
        onLogout={handleLogout}
        refreshKey={sidebarRefresh}
      />

      <div className="chat-main">
        <div className="chat-topbar">
          <span className="chat-title">{t("appTitle", lang)}</span>
      <div className="topbar-actions">
        <button className="theme-toggle" onClick={toggleTheme}>
      {theme === "light" ? "🌙" : "☀️"}
          </button>
          <button className="lang-toggle" onClick={toggleLang}>
          {lang === "tr" ? "EN" : "TR"}
          </button>
        </div>
      </div>

        <div className="messages">
          {messages.map((m) => (
            <div key={m.id} className={`message ${m.sender}`}>
              {m.type === "audio" ? (
                <audio controls src={m.content} />
              ) : m.type === "image" ? (
                <img src={m.content} alt="" className="message-image" />
              ) : (
                <>
                  {m.content}
                  {m.sender === "bot" && (
                    <button className="speak-button" onClick={() => speak(m.content, m.id)} title={t("micTitle", lang)}>
                      🔊
                    </button>
                  )}
                </>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {pendingImage && (
          <div className="pending-image">
            <img src={pendingImage.url} alt="" />
            <button type="button" onClick={() => setPendingImage(null)}>✕</button>
          </div>
        )}

        <form className="input-row" onSubmit={handleSubmit}>
          <button type="button" className={`mic-button ${isRecording ? "recording" : ""}`} onClick={toggleRecording}>
            🎤
          </button>
          <button type="button" className="image-button" onClick={() => fileInputRef.current.click()}>
            📷
          </button>
          <input type="file" accept="image/*" ref={fileInputRef} style={{ display: "none" }} onChange={handleImageChange} />
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t("placeholder", lang)}
            autoComplete="off"
          />
          <button type="submit">{t("send", lang)}</button>
        </form>
      </div>
    </div>
  );
}