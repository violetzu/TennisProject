"""
VLM 驗證模組（Phase 4）

Phase 1 decode 迴圈中每 THUMB_STRIDE 幀儲存縮圖到 thumb_dir；
Phase 4 在 Phase 3 彙整前，對每個 contact 幀逐一驗證：
  是「真實擊球（racket hit）」還是「落地反彈（bounce）」，
  同時分類擊球類型（overhead / swing / unknown）。

VLM 判定為 bounce 的 contact 幀會被移除；hit / unknown 保留（保守策略）。
過濾後重新 segment_rallies → Phase 3 使用驗證後的 contacts 與 shot_types。

Qwen3 系列：加 /no_think 並透過 chat_template_kwargs 關閉推理模式，
可讓回應速度從 ~60s 降至 ~1s，且不輸出 <think>...</think> 雜訊。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from config import VIDEO_DIR, VIDEO_URL_DOMAIN

# ── 常數 ──────────────────────────────────────────────────────────────────────
THUMB_STRIDE             = 5    # Phase 1：每 N 幀儲存一張縮圖（5 → 每 ~0.08s@60fps）
THUMB_W                  = 320  # 縮圖寬度（px）
THUMB_H                  = 180  # 縮圖高度（px）
CONTACT_CONTEXT_FRAMES   = 30   # 每個 contact 前後各取幾幀（±0.5s@60fps）
MAX_FRAMES_PER_CONTACT   = 8    # 每次 VLM 呼叫最多送幾張圖（contact 驗證）
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
    max_n: int = MAX_FRAMES_PER_CONTACT,
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


# ── 逐 contact 驗證 ────────────────────────────────────────────────────────────

def verify_one_contact(
    frame_idx: int,
    thumb_dir: Path,
    fps: float,
    vllm_cfg,
) -> Tuple[str, str]:
    """
    對單一 contact 幀呼叫 VLM。
    Returns: (event_type, shot_type)
      event_type: 'hit' | 'bounce' | 'unknown'
      shot_type:  'overhead' | 'swing' | 'unknown'
    任何失敗（無縮圖、網路、timeout、解析錯誤）均回傳 ('unknown', 'unknown')。
    """
    thumbs = select_rally_thumbs(
        thumb_dir,
        frame_idx - CONTACT_CONTEXT_FRAMES,
        frame_idx + CONTACT_CONTEXT_FRAMES,
    )
    if not thumbs:
        print(f"[VLM] f={frame_idx}: no thumbnails in ±{CONTACT_CONTEXT_FRAMES} frames, skip")
        return "unknown", "unknown"

    def _ft(p: Path) -> float:
        return int(p.stem.split("_")[1]) / max(fps, 1.0)

    contact_t    = frame_idx / max(fps, 1.0)
    frame_labels = ", ".join(
        f"Frame {i + 1} (t={_ft(p):.2f}s)" for i, p in enumerate(thumbs)
    )
    urls = [_thumb_to_url(p) for p in thumbs]

    user_text = (
        f"{len(thumbs)} frames around a detected ball event at t={contact_t:.2f}s.\n"
        f"{frame_labels}\n\n"
        "Examine the ball trajectory and racket/player positions carefully.\n"
        "1. Determine whether the event is a RACKET HIT or a COURT BOUNCE.\n"
        "2. If it is a HIT, classify the shot type:\n"
        "   - overhead: arm raised above head (smash / overhead)\n"
        "   - swing: normal groundstroke or volley (forehand/backhand)\n"
        "   - unknown: cannot determine\n\n"
        "Reply ONLY with this XML:\n"
        "<verification>\n"
        "  <event_type>hit|bounce|unknown</event_type>\n"
        "  <shot_type>overhead|swing|unknown</shot_type>\n"
        "</verification>"
    )

    img_blocks = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    payload = {
        "model": vllm_cfg.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": img_blocks + [
                {"type": "text", "text": user_text},
            ]},
        ],
        "stream":               False,
        "max_tokens":           128,
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

    try:
        resp = requests.post(
            f"{vllm_cfg.url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=VLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        m_event = re.search(r"<event_type>\s*(.+?)\s*</event_type>", content, re.IGNORECASE)
        event_type = "unknown"
        if m_event:
            v = m_event.group(1).strip().lower()
            if v in ("hit", "bounce", "unknown"):
                event_type = v

        m_shot = re.search(r"<shot_type>\s*(.+?)\s*</shot_type>", content, re.IGNORECASE)
        shot_type = "unknown"
        if m_shot:
            v = m_shot.group(1).strip().lower()
            if v in ("overhead", "swing", "unknown"):
                shot_type = v

        return event_type, shot_type

    except Exception as exc:
        print(f"[VLM] contact f={frame_idx} (t={contact_t:.2f}s) failed: {exc}")
        return "unknown", "unknown"


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
    任何失敗均回傳 'unknown'。
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

    last_t      = last_contact_f / max(fps, 1.0)
    frame_labels = ", ".join(
        f"Frame {i + 1} (t={_ft(p):.2f}s)" for i, p in enumerate(thumbs)
    )
    urls = [_thumb_to_url(p) for p in thumbs]

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

    img_blocks = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    payload = {
        "model": vllm_cfg.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": img_blocks + [
                {"type": "text", "text": user_text},
            ]},
        ],
        "stream":               False,
        "max_tokens":           64,
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

    try:
        resp = requests.post(
            f"{vllm_cfg.url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=VLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
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


def verify_contacts_vlm(
    contacts_f: List[int],
    thumb_dir: Path,
    fps: float,
    vllm_cfg,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    progress_start: int = 72,
    progress_end: int = 95,
) -> Tuple[List[int], List[int], Dict[int, str]]:
    """
    Phase 3：逐一驗證 contacts_f 中每個幀是否為真實擊球，同時取得擊球類型。

    detect_events 以場地轉向作為候選事件（包含落地彈跳），
    此函式由 VLM 判斷真正的揮拍 / 發球 vs 落地反彈。

    Returns:
        (verified_contacts, vlm_bounces, shot_types)
        verified_contacts: VLM 判 bounce 的幀移除；hit 或 unknown 保留（保守策略）
        vlm_bounces:       VLM 判定為 bounce 的幀（供 bounces_f 使用）
        shot_types:        frame_idx → 'overhead' | 'swing' | 'unknown'
    """
    if not contacts_f:
        return contacts_f, [], {}

    verified: List[int] = []
    vlm_bounces: List[int] = []
    shot_types: Dict[int, str] = {}

    n = len(contacts_f)
    for i, fi in enumerate(contacts_f):
        event_type, shot_type = verify_one_contact(fi, thumb_dir, fps, vllm_cfg)
        t = fi / max(fps, 1.0)
        if event_type == "bounce":
            print(f"[VLM] f={fi} t={t:.2f}s → BOUNCE")
            vlm_bounces.append(fi)
        else:
            verified.append(fi)
            shot_types[fi] = shot_type
            print(f"[VLM] f={fi} t={t:.2f}s → {event_type.upper()}  shot={shot_type}")
        if progress_cb:
            pct = progress_start + int((i + 1) / n * (progress_end - progress_start))
            progress_cb(pct, 100)

    print(f"[VLM] contacts: {len(contacts_f)} total → "
          f"{len(verified)} hits, {len(vlm_bounces)} bounces")
    return verified, vlm_bounces, shot_types
