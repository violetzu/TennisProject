// components/AuthProvider.tsx
"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { clearToken, getToken, setToken as persistToken } from "@/lib/auth";

type User = { id: number; username: string };

type AuthContextValue = {
  token: string | null;
  user: User | null;
  isAuthed: boolean;
  login: (token: string) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);

  const refresh = useCallback(async () => {
    const t = getToken();
    setTokenState(t);

    if (!t) {
      setUser(null);
      return;
    }

    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${t}` },
      });
      if (!res.ok) throw new Error("me failed");
      const data = (await res.json()) as User;
      setUser(data);
    } catch {
      // token 失效 → 清除，回到登出狀態
      clearToken();
      setTokenState(null);
      setUser(null);
    }
  }, []);

  useEffect(() => {
    refresh();

    // 多分頁同步用（同分頁 setItem 不會觸發 storage，但我們本來就有 login() 直接 setState）
    const onStorage = (e: StorageEvent) => {
      if (e.key === "access_token") refresh();
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [refresh]);

  const login = useCallback(
    async (newToken: string) => {
      persistToken(newToken);
      setTokenState(newToken);
      await refresh(); // 驗證 token 並取得 user → isAuthed 才會變 true
    },
    [refresh]
  );

  const logout = useCallback(() => {
    clearToken();
    setTokenState(null);
    setUser(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      user,
      isAuthed: !!token && !!user,
      login,
      logout,
      refresh,
    }),
    [token, user, login, logout, refresh]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider />");
  return ctx;
}
