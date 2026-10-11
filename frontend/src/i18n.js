export const translations = {
  tr: {
    appTitle: "Otomotiv Asistanı",
    newChat: "+ Yeni Sohbet",
    logout: "Çıkış Yap",
    loginTitle: "Giriş Yap",
    registerTitle: "Kayıt Ol",
    username: "Kullanıcı adı",
    password: "Şifre",
    loginButton: "Giriş Yap",
    registerButton: "Kayıt Ol",
    switchToRegister: "Hesabın yok mu? Kayıt ol",
    switchToLogin: "Zaten hesabın var mı? Giriş yap",
    placeholder: "Bir soru yazın...",
    send: "Gönder",
    micTitle: "Seslendir",
    noSessions: "Henüz sohbet yok",
    loading: "Yükleniyor...",
    connectionError: "Üzgünüm, sunucuya ulaşamadım. Lütfen tekrar deneyin.",
    imageError: "Üzgünüm, fotoğrafı işlerken sunucuya ulaşamadım.",
    voiceError: "Üzgünüm, sesli mesajı işlerken sunucuya ulaşamadım.",
  },
  en: {
    appTitle: "Automotive Assistant",
    newChat: "+ New Chat",
    logout: "Log Out",
    loginTitle: "Log In",
    registerTitle: "Register",
    username: "Username",
    password: "Password",
    loginButton: "Log In",
    registerButton: "Register",
    switchToRegister: "No account? Register",
    switchToLogin: "Already have an account? Log in",
    placeholder: "Type a question...",
    send: "Send",
    micTitle: "Speak",
    noSessions: "No conversations yet",
    loading: "Loading...",
    connectionError: "Sorry, I couldn't reach the server. Please try again.",
    imageError: "Sorry, I couldn't process the photo.",
    voiceError: "Sorry, I couldn't process the voice message.",
  },
};

export function getLang() {
  return localStorage.getItem("ui_lang") || "tr";
}

export function setLang(lang) {
  localStorage.setItem("ui_lang", lang);
}

export function t(key, lang) {
  return translations[lang]?.[key] || translations.tr[key] || key;
}