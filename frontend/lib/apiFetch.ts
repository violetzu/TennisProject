// lib/apiFetch.ts
// 統一的 API fetch 封裝：
// - 自動帶 Authorization header
// - 非 2xx 自動 throw，錯誤訊息從 response body 讀取
// - 呼叫端只需要 try/catch，不需要每次寫 await res.text().catch(...)

import { getToken } from "@/lib/auth";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readErrorText(res: Response): Promise<string> {
  try {
    const text = await res.text();
    // FastAPI 的錯誤通常是 JSON { detail: "..." }
    try {
      const json = JSON.parse(text);
      if (json?.detail) return String(json.detail);
    } catch {}
    return text || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(input, { ...init, headers });

  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(msg, res.status);
  }

  return res;
}

/** 直接拿到 JSON，非 2xx 自動 throw ApiError */
export async function apiFetchJson<T = unknown>(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<T> {
  const res = await apiFetch(input, init);
  return res.json() as Promise<T>;
}
