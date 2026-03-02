// app/api/chat/route.ts
// 專門為 /api/chat 做串流 proxy，繞過 Next.js rewrite 的緩衝問題。
// 其他 /api/* 路由仍走 next.config.js 的 rewrites。

import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_DOMAIN || "http://backend:8000";

export async function POST(req: NextRequest) {
  const body = await req.arrayBuffer();

  // 取出授權 header（如果有的話）
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const auth = req.headers.get("authorization");
  if (auth) headers["Authorization"] = auth;

  const upstream = await fetch(`${BACKEND}/api/chat`, {
    method: "POST",
    headers,
    body,
    // Node 18+ fetch 預設支援串流 body，不會緩衝
    // @ts-expect-error — Next.js 環境需要關閉自動解壓縮以保持串流
    duplex: "half",
  });

  // 直接把 upstream 的 ReadableStream 透傳給瀏覽器
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "X-Accel-Buffering": "no",
      "Cache-Control": "no-cache",
    },
  });
}
