// app/layout.tsx
import "./globals.css";
import Script from "next/script";

import { AuthProvider } from "@/components/AuthProvider";

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
            var t = localStorage.getItem("theme");
            if (!t) {
              t = window.matchMedia("(prefers-color-scheme: dark)").matches
                ? "dark"
                : "light";
            }
            document.body.classList.add(t);
          } catch (e) {
            document.body.classList.add("light");
          }
        })();
        `}</Script>
      </head>

      <body  suppressHydrationWarning>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
