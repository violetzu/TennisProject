// components/AuthModal.tsx
"use client";

import { useState } from "react";
import { useAuth } from "@/components/AuthProvider";

type Mode = "login" | "register";

export default function AuthModal({
  onClose,
}: {
  onClose: () => void;
}) {
  const { login } = useAuth();

  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok?: boolean } | null>(null);

  async function submit() {
    setMsg(null);
    setBusy(true);
    try {
      if (mode === "login") {
        const body = new URLSearchParams({ username, password });
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: body.toString(),
        });
        if (!res.ok) throw new Error(await res.text().catch(() => "登入失敗"));
        const data = await res.json();
        if (!data?.access_token) throw new Error("缺少 access_token");
        await login(data.access_token);
        onClose();
      } else {
        if (!username.trim()) throw new Error("請輸入 username");
        if (!password) throw new Error("請輸入 password");
        if (password !== password2) throw new Error("兩次密碼不一致");
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        if (!res.ok) throw new Error(await res.text().catch(() => "註冊失敗"));
        // 註冊成功後自動切換到登入
        setMode("login");
        setPassword("");
        setPassword2("");
        setMsg({ text: "註冊成功，請登入", ok: true });
      }
    } catch (e: any) {
      setMsg({ text: e?.message || String(e) });
    } finally {
      setBusy(false);
    }
  }

  const title = mode === "login" ? "登入" : "註冊";

  return (
    <div className="auth-backdrop" role="dialog" aria-modal="true">
      <div className="glass-base auth-card">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontSize: 18, fontWeight: 800 }}>{title}</div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button className="btn" type="button" onClick={() => { setMode("login"); setMsg(null); }} disabled={busy}>
              登入
            </button>
            <button className="btn" type="button" onClick={() => { setMode("register"); setMsg(null); }} disabled={busy}>
              註冊
            </button>
            <button className="btn" type="button" onClick={onClose} disabled={busy}>
              關閉
            </button>
          </div>
        </div>

        <div style={{ height: 12 }} />

        <div style={{ display: "grid", gap: 10 }}>
          <input
            value={username} onChange={(e) => setUsername(e.target.value)}
            placeholder="username" className="input"
            autoComplete="username" disabled={busy}
          />
          <input
            value={password} onChange={(e) => setPassword(e.target.value)}
            placeholder="password" className="input" type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            disabled={busy}
            onKeyDown={(e) => { if (e.key === "Enter" && mode === "login") void submit(); }}
          />

          {mode === "register" && (
            <input
              value={password2} onChange={(e) => setPassword2(e.target.value)}
              placeholder="confirm password" className="input" type="password"
              autoComplete="new-password" disabled={busy}
              onKeyDown={(e) => { if (e.key === "Enter") void submit(); }}
            />
          )}

          {msg && (
            <div style={{ color: msg.ok ? "lightgreen" : "salmon", fontSize: 12 }}>
              {msg.text}
            </div>
          )}

          <button className="btn btn-green" type="button" onClick={submit} disabled={busy}>
            {busy ? "處理中..." : title}
          </button>
        </div>
      </div>
    </div>
  );
}
