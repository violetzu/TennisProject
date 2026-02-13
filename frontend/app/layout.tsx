// app/layout.tsx
import "./globals.css";
import Script from "next/script";

export const metadata = {
  title: "網球比賽分析助手",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant">
      <body className="light">
        {children}

        {/* 載入 Web Component：<theme-toggle /> */}
        <Script src="/theme-toggle.js" strategy="afterInteractive" />
      </body>
    </html>
  );
}
