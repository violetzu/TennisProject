// components/AuthModal.tsx
"use client";

import { useMemo, useState } from "react";
import { useAuth } from "@/components/AuthProvider";

type Mode = "login" | "register";

export default function AuthModal({
  mode,
  onClose,
  onSwitch,
}: {
  mode: Mode;
  onClose: () => void;
  onSwitch: (m: Mode) => void;
}) {
  const { login } = useAuth();

  const title = useMemo(() => (mode === "login" ? "登入" : "註冊"), [mode]);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);
    setBusy(true);
    try {
      if (mode === "login") {
        const body = new URLSearchParams();
        body.set("username", username);
        body.set("password", password);

        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: body.toString(),
        });

        if (!res.ok) throw new Error(await res.text().catch(() => "login failed"));
        const data = await res.json();
        if (!data?.access_token) throw new Error("缺少 access_token");

        login(data.access_token);
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

        if (!res.ok) throw new Error(await res.text().catch(() => "register failed"));

        // 註冊成功後：自動切到登入（也可做自動登入，但這裡先走安全保守流程）
        onSwitch("login");
        setErr("註冊成功，請登入");
      }
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-backdrop" role="dialog" aria-modal="true">
      <div className="glass-base auth-card">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontSize: 18, fontWeight: 800 }}>{title}</div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button className="btn" type="button" onClick={() => onSwitch("login")} disabled={busy}>
              登入
            </button>
            <button
              className="btn"
              type="button"
              onClick={() => onSwitch("register")}
              disabled={busy}
            >
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
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="username"
            className="input"
            autoComplete="username"
            disabled={busy}
          />
          <input
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="password"
            className="input"
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            disabled={busy}
          />

          {mode === "register" && (
            <input
              value={password2}
              onChange={(e) => setPassword2(e.target.value)}
              placeholder="confirm password"
              className="input"
              type="password"
              autoComplete="new-password"
              disabled={busy}
            />
          )}

          {err && (
            <div style={{ color: err === "註冊成功，請登入" ? "lightgreen" : "salmon", fontSize: 12 }}>
              {err}
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
