"""
VLM 回合勝負判斷模組

Phase 1 decode 迴圈中每 THUMB_STRIDE 幀儲存縮圖到 thumb_dir；
Phase 3 對每個回合的最後一拍後取縮圖，由 VLM 判斷得分方。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests

from config import VIDEO_DIR, VIDEO_URL_DOMAIN

# ── 常數 ──────────────────────────────────────────────────────────────────────
THUMB_STRIDE             = 5    # Phase 1：每 N 幀儲存一張縮圖（5 → 每 ~0.08s@60fps）
THUMB_W                  = 320  # 縮圖寬度（px）
THUMB_H                  = 180  # 縮圖高度（px）
WINNER_LOOK_AHEAD_FRAMES = 240  # 最後一拍後最多看幾幀（~4s@60fps）
MAX_FRAMES_PER_WINNER    = 8    # 每次 VLM 最多送幾張縮圖（winner 判斷）
VLM_TIMEOUT              = 120  # VLLM 呼叫 timeout（秒）

_SYSTEM_PROMPT = (
    "You are a tennis analysis assistant. "
    "Reply ONLY with valid XML in the exact format requested, no other text. /no_think"
)


# ── 縮圖工具 ──────────────────────────────────────────────────────────────────

def save_thumbnail(frame: np.ndarray, thumb_dir: Path, idx: int) -> None:
    """在 Phase 1 decode 迴圈中每 THUMB_STRIDE 幀呼叫一次，儲存原始幀縮圖。"""
    small = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(
        str(thumb_dir / f"frame_{idx:06d}.jpg"),
        small,
        [cv2.IMWRITE_JPEG_QUALITY, 70],
    )


def _thumb_to_url(thumb_path: Path) -> str:
    """縮圖絕對路徑 → vLLM 可抓取的 HTTP URL。"""
    rel = thumb_path.resolve().relative_to(VIDEO_DIR.resolve()).as_posix()
    return f"{VIDEO_URL_DOMAIN}/videos/{rel}"


def select_rally_thumbs(
    thumb_dir: Path,
    start_f: int,
    end_f: int,
    max_n: int = MAX_FRAMES_PER_WINNER,
) -> List[Path]:
    """從 thumb_dir 選出落在 [start_f, end_f] 範圍內、最多 max_n 張等距縮圖。"""
    candidates = sorted(
        (
            p for p in thumb_dir.glob("frame_*.jpg")
            if start_f <= int(p.stem.split("_")[1]) <= end_f
        ),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not candidates:
        return []
    if len(candidates) <= max_n:
        return candidates
    step = (len(candidates) - 1) / (max_n - 1)
    return [candidates[round(i * step)] for i in range(max_n)]


# ── VLM 呼叫 ─────────────────────────────────────────────────────────────────

def _call_vlm(
    vllm_cfg,
    image_urls: List[str],
    user_text: str,
    max_tokens: int = 128,
) -> Optional[str]:
    """
    送圖 + 文字給 VLM，回傳 content 字串。
    任何失敗（網路 / timeout / HTTP error）回傳 None。
    """
    img_blocks = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    payload = {
        "model": vllm_cfg.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": img_blocks + [
                {"type": "text", "text": user_text},
            ]},
        ],
        "stream":               False,
        "max_tokens":           max_tokens,
        "temperature":          0.1,
        "top_p":                0.95,
        "top_k":                20,
        "min_p":                0.0,
        "repetition_penalty":   1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    headers = {"Content-Type": "application/json"}
    if vllm_cfg.api_key:
        headers["Authorization"] = f"Bearer {vllm_cfg.api_key}"

    resp = requests.post(
        f"{vllm_cfg.url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=VLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── 回合勝負判斷 ─────────────────────────────────────────────────────────────

def verify_rally_winner(
    last_contact_f: int,
    next_rally_start_f: Optional[int],
    thumb_dir: Path,
    fps: float,
    vllm_cfg,
) -> str:
    """
    VLM 看最後一拍之後的幾幀，判斷是否有人得分及得分方。
    Returns: 'top' | 'bottom' | 'unknown'
    """
    end_f = (
        min(last_contact_f + WINNER_LOOK_AHEAD_FRAMES, next_rally_start_f - 1)
        if next_rally_start_f is not None
        else last_contact_f + WINNER_LOOK_AHEAD_FRAMES
    )
    thumbs = select_rally_thumbs(thumb_dir, last_contact_f, end_f, MAX_FRAMES_PER_WINNER)
    if not thumbs:
        print(f"[VLM-winner] f={last_contact_f}: no thumbnails in window, skip")
        return "unknown"

    def _ft(p: Path) -> float:
        return int(p.stem.split("_")[1]) / max(fps, 1.0)

    last_t       = last_contact_f / max(fps, 1.0)
    frame_labels = ", ".join(
        f"Frame {i + 1} (t={_ft(p):.2f}s)" for i, p in enumerate(thumbs)
    )

    user_text = (
        f"These {len(thumbs)} frames show the end of a tennis rally "
        f"(last detected shot at t={last_t:.2f}s).\n"
        f"{frame_labels}\n\n"
        "The TOP player occupies the UPPER half of the image; "
        "the BOTTOM player occupies the LOWER half.\n\n"
        "Determine who scored the point at the end of this rally:\n"
        "  - top: TOP player scored (ball landed unreturned in BOTTOM player's court, "
        "or BOTTOM player hit the ball out/into the net)\n"
        "  - bottom: BOTTOM player scored (ball landed unreturned in TOP player's court, "
        "or TOP player hit the ball out/into the net)\n"
        "  - unknown: cannot determine from these frames\n\n"
        "Reply ONLY with this XML:\n"
        "<outcome>\n"
        "  <winner>top|bottom|unknown</winner>\n"
        "</outcome>"
    )

    try:
        content = _call_vlm(
            vllm_cfg,
            [_thumb_to_url(p) for p in thumbs],
            user_text,
            max_tokens=64,
        )
        if content is None:
            return "unknown"

        m = re.search(r"<winner>\s*(.+?)\s*</winner>", content, re.IGNORECASE)
        if m:
            v = m.group(1).strip().lower()
            if v in ("top", "bottom", "unknown"):
                print(f"[VLM-winner] f={last_contact_f} t={last_t:.2f}s → {v.upper()}")
                return v
        print(f"[VLM-winner] f={last_contact_f}: parse failed, content={content!r}")
        return "unknown"

    except Exception as exc:
        print(f"[VLM-winner] f={last_contact_f} t={last_t:.2f}s failed: {exc}")
        return "unknown"
