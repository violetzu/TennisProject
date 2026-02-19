"use client";

import { useEffect, useState } from "react";
import "./ThemeToggle.css";

type Theme = "light" | "dark";

function applyTheme(theme: Theme) {
  const body = document.body;
  body.classList.remove("light", "dark");
  body.classList.add(theme);

  try {
    localStorage.setItem("theme", theme);
  } catch {}
}

export default function ThemeToggle() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const onToggle = () => {
    const isLight = document.body.classList.contains("light");
    applyTheme(isLight ? "dark" : "light");
  };

  return (
    <button
      id="themeToggle"
      className="glass-base theme-toggle"
      type="button"
      aria-label="Toggle theme"
      onClick={mounted ? onToggle : undefined}
    >
      <div className="ts-track">
        <div className="ts-sky ts-sky-day" />
        <div className="ts-sky ts-sky-night" />
        <div className="ts-stars" />
        <div className="ts-clouds ts-clouds-front" />
        <div className="ts-clouds ts-clouds-back" />
      </div>

      <div className="ts-orb">
        <div className="ts-orb-inner" />
      </div>
    </button>
  );
}
