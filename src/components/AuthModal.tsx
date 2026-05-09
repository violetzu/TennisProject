// components/AuthModal.tsx
"use client";

import { useState } from "react";
import { useAuth } from "@/components/AuthProvider";

type Mode = "login" | "register";

function EyeIcon({ open }: { open: boolean }) {
  return open ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
      <line x1="1" y1="1" x2="23" y2="23"/>
    </svg>
  );
}

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
  const [showPwd, setShowPwd] = useState(false);
  const [showPwd2, setShowPwd2] = useState(false);
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
    <div className="fixed inset-0 z-[9999] grid place-items-center bg-black/35 backdrop-blur-[10px]" role="dialog" aria-modal="true">
      <div className="glass w-[360px] px-[22px] py-6">
        <div className="flex items-center gap-2.5">
          <div className="text-lg font-extrabold">{title}</div>
          <div className="ml-auto flex gap-2">
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

        <div className="h-3" />

        <div className="grid gap-3">
          <div className="grid gap-1">
            <label className="text-base text-gray-500 dark:text-gray-400">帳號</label>
            <input
              value={username} onChange={(e) => setUsername(e.target.value)}
              placeholder="請輸入帳號" className="input"
              type="text" autoComplete="off" disabled={busy}
              onKeyDown={(e) => { if (e.key === "Enter") void submit(); }}
            />
          </div>

          <div className="grid gap-1">
            <label className="text-base text-gray-500 dark:text-gray-400">密碼</label>
            <div className="input-wrap">
              <input
                value={password} onChange={(e) => setPassword(e.target.value)}
                placeholder="請輸入密碼" className="input"
                type={showPwd ? "text" : "password"}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                disabled={busy}
                onKeyDown={(e) => { if (e.key === "Enter" && mode === "login") void submit(); }}
              />
              <button type="button" className="input-eye" onClick={() => setShowPwd(v => !v)} tabIndex={-1}>
                <EyeIcon open={showPwd} />
              </button>
            </div>
          </div>

          {mode === "register" && (
            <div className="grid gap-1">
              <label className="text-base text-gray-500 dark:text-gray-400">確認密碼</label>
              <div className="input-wrap">
                <input
                  value={password2} onChange={(e) => setPassword2(e.target.value)}
                  placeholder="再次輸入密碼" className="input"
                  type={showPwd2 ? "text" : "password"}
                  autoComplete="new-password" disabled={busy}
                  onKeyDown={(e) => { if (e.key === "Enter") void submit(); }}
                />
                <button type="button" className="input-eye" onClick={() => setShowPwd2(v => !v)} tabIndex={-1}>
                  <EyeIcon open={showPwd2} />
                </button>
              </div>
            </div>
          )}

          {msg && (
            <div className={`text-base ${msg.ok ? "text-green-400" : "text-red-400"}`}>
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
