// lib/guestToken.ts
// guest_token 存在 sessionStorage（關閉分頁就失效，符合 guest 設計）

const GUEST_TOKEN_KEY = "guest_token";
const GUEST_RECORD_KEY = "guest_record_id";

export function getGuestToken(): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(GUEST_TOKEN_KEY);
}

export function setGuestToken(token: string) {
  sessionStorage.setItem(GUEST_TOKEN_KEY, token);
}

export function clearGuestToken() {
  sessionStorage.removeItem(GUEST_TOKEN_KEY);
  sessionStorage.removeItem(GUEST_RECORD_KEY);
}

export function getGuestRecordId(): number | null {
  if (typeof window === "undefined") return null;
  const v = sessionStorage.getItem(GUEST_RECORD_KEY);
  return v ? Number(v) : null;
}

export function setGuestRecordId(id: number) {
  sessionStorage.setItem(GUEST_RECORD_KEY, String(id));
}
