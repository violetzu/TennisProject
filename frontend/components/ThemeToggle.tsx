// components/ThemeToggle.tsx
"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [mode, setMode] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", mode === "dark");
    document.documentElement.classList.toggle("light", mode === "light");
    document.body.classList.toggle("dark", mode === "dark");
    document.body.classList.toggle("light", mode === "light");
  }, [mode]);

  return (
    <button
      className="glass-base theme-toggle"
      onClick={() => setMode((m) => (m === "light" ? "dark" : "light"))}
      type="button"
    >
      {mode === "light" ? "🌞" : "🌙"}
    </button>
  );
}
