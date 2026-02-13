// app/layout.tsx
import "./globals.css";
import Script from "next/script";

export const metadata = {
  title: "網球比賽分析助手",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant" suppressHydrationWarning>
      <head>
        <Script id="init-theme" strategy="beforeInteractive">{`
(function () {
  try {
    var saved = localStorage.getItem("theme");
    var theme = (saved === "dark" || saved === "light")
      ? saved
      : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    var body = document.body;

    // 先清掉，避免重複/髒狀態
    body.classList.remove("light", "dark");

    // 設定正確 theme
    body.classList.add(theme);
  } catch (e) {
    // 失敗就保底 light
    document.body.classList.add("light");
  }
})();
        `}</Script>
      </head>

      {/* SSR 先給 light，避免 hydration mismatch（script 會在非常早期覆蓋成正確值） */}
      <body className="light" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
