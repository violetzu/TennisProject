// app/layout.tsx
import "./globals.css";

export const metadata = {
  title: "網球比賽分析助手",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant" className="light">
      <body className="light">{children}</body>
    </html>
  );
}
