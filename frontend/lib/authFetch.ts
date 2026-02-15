// lib/authFetch.ts
import { getToken } from "./auth";

export async function authFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  const token = getToken();

  const headers = new Headers(init.headers || {});
  // 如果 caller 沒塞 Content-Type，這裡不強制（讓 FormData/XHR 自己處理）
  if (token) headers.set("Authorization", `Bearer ${token}`);

  return fetch(input, { ...init, headers });
}
