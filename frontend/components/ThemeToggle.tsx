"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

const STYLE_ID = "theme-toggle-style";

const CSS = `
/* ===== 夜間 / 日間 動態切換開關 ===== */
.theme-toggle {
  position: relative;
  width: 130px;
  height: 44px;
  padding: 0;
  border-radius: 999px;
  border: none;
  cursor: pointer;
  overflow: hidden;
  display: inline-flex;
  align-items: center;
  justify-content: flex-start;
  background: transparent;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.25);
}

/* 裡面的軌道 */
.ts-track {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  overflow: hidden;
}

/* 兩層天空：白天 & 夜晚 疊在一起，靠 transform 切換 */
.ts-sky {
  position: absolute;
  inset: 0;
  transition: opacity .5s ease, transform .5s ease;
}

/* 白天：藍天 */
.ts-sky-day {
  background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 45%, #38bdf8 100%);
}

/* 夜晚：深藍星空底 */
.ts-sky-night {
  background: radial-gradient(circle at 20% 0%, #1d4ed8 0%, #020617 60%, #000 100%);
}

/* light 狀態只顯示白天 */
body.light .ts-sky-day {
  opacity: 1;
  transform: translateX(0);
}

body.light .ts-sky-night {
  opacity: 0;
  transform: translateX(20px);
}

/* dark 狀態只顯示夜晚 */
body.dark .ts-sky-day {
  opacity: 0;
  transform: translateX(-20px);
}

body.dark .ts-sky-night {
  opacity: 1;
  transform: translateX(0);
}

/* 星星群 */
.ts-stars::before {
  content: "";
  position: absolute;
  top: 6px;
  left: 18px;
  width: 2px;
  height: 2px;
  border-radius: 999px;
  background: #e5e7eb;
  box-shadow:
    10px 2px 0 #e5e7eb,
    24px 5px 0 #e5e7eb,
    38px 1px 0 #e5e7eb,
    52px 4px 0 #e5e7eb,
    70px 3px 0 #e5e7eb,
    88px 6px 0 #e5e7eb;
  opacity: 0;
  transform: translateY(-3px);
  transition: opacity .4s ease, transform .4s ease;
}

/* 夜晚才亮起來 */
body.dark .ts-stars::before {
  opacity: 0.95;
  transform: translateY(0);
}

/* 雲層：前後兩層做一點 parallax */
.ts-clouds {
  position: absolute;
  bottom: -4px;
  left: 10px;
  width: 52px;
  height: 18px;
  border-radius: 999px;
  background: #f9fafb;
  box-shadow:
    20px -4px 0 0 #f9fafb,
    40px -1px 0 0 #f9fafb,
    60px -5px 0 0 #f9fafb;
  opacity: 0.96;
  transition: transform .6s ease;
}

.ts-clouds-back {
  opacity: 0.8;
  transform: scale(1.05) translateX(2px);
}

/* 夜晚雲偏灰一點 */
body.dark .ts-clouds,
body.dark .ts-clouds-back {
  background: #e5e7eb;
  box-shadow:
    20px -4px 0 0 #e5e7eb,
    40px -1px 0 0 #e5e7eb,
    60px -5px 0 0 #e5e7eb;
}

/* 切換時雲順便滑一下，製造流動感 */
body.light .ts-clouds {
  transform: translateX(0);
}

body.dark .ts-clouds {
  transform: translateX(4px);
}

body.light .ts-clouds-back {
  transform: scale(1.05) translateX(0);
}

body.dark .ts-clouds-back {
  transform: scale(1.05) translateX(6px);
}

/* 太陽 / 月亮本體 */
.ts-orb {
  position: relative;
  z-index: 1;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  margin-left: 8px;
  transition: transform .45s cubic-bezier(.22, .61, .36, 1);
}

/* 內部做一點光暈 */
.ts-orb-inner {
  width: 100%;
  height: 100%;
  border-radius: inherit;
  box-shadow: 0 8px 14px rgba(15, 23, 42, 0.45);
  transition: background .4s ease, box-shadow .4s ease;
}

/* 白天：左邊太陽，橘色＋柔和陰影 */
body.light .ts-orb {
  transform: translateX(0);
}

body.light .ts-orb-inner {
  background: radial-gradient(circle at 30% 30%, #fed7aa 0%, #fb923c 50%, #ea580c 100%);
  box-shadow: 0 8px 14px rgba(248, 113, 113, 0.55);
}

/* 夜晚：右邊月亮＋冷色陰影 */
body.dark .ts-orb {
  transform: translateX(72px);
}

body.dark .ts-orb-inner {
  background: radial-gradient(circle at 30% 30%, #f9fafb 0%, #cbd5f5 55%, #94a3b8 100%);
  box-shadow: 0 8px 16px rgba(15, 23, 42, 0.75);
}

/* 讓整個開關 hover 只有很輕微的浮動 */
.theme-toggle.glass-base:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.28);
}

#themeToggle {
  border: none;
  cursor: pointer;
  padding: 0 12px;
  gap: 6px;
  color: inherit;
}
`;

function getInitialTheme(): Theme {
  // 1) localStorage 優先
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {}

  // 2) 沒存就看系統偏好
  try {
    const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
    return prefersDark ? "dark" : "light";
  } catch {}

  return "light";
}

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
    // 只注入一次 style
    if (!document.getElementById(STYLE_ID)) {
      const style = document.createElement("style");
      style.id = STYLE_ID;
      style.textContent = CSS;
      document.head.appendChild(style);
    }

    // 初始化 theme
    const initial = getInitialTheme();
    applyTheme(initial);

    setMounted(true);
  }, []);

  const onToggle = () => {
    const isLight = document.body.classList.contains("light");
    applyTheme(isLight ? "dark" : "light");
  };

  // mounted 前先不渲染，避免 hydration 閃一下
  if (!mounted) return null;

  return (
    <button
      id="themeToggle"
      className="glass-base theme-toggle"
      type="button"
      aria-label="Toggle theme"
      onClick={onToggle}
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
