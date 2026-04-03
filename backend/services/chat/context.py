# services/chat/context.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import requests

from config import EmbeddingConfig

_EMBED_TIMEOUT = 30


# ── 分析資料載入 ───────────────────────────────────────────────────────────────

def load_analysis_data(json_path: Path) -> Optional[dict]:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── 永遠包含的摘要統計 ─────────────────────────────────────────────────────────

def build_summary_context(data: dict) -> str:
    s   = data.get("summary", {})
    spd = s.get("speed", {})
    dep = s.get("depth", {}).get("total", {})

    lines = [
        "=== 影片分析數據（請優先參考此數據回答問題）===",
        f"回合數：{s.get('total_rallies', '?')}，"
        f"總擊球：{s.get('total_shots', '?')}，"
        f"得分球：{s.get('total_winners', '?')}，"
        f"平均每回合：{s.get('avg_rally_length', '?')} 拍",
    ]

    for side, label in (("top", "上方球員"), ("bottom", "下方球員")):
        p  = s.get("players", {}).get(side, {})
        st = p.get("shot_types", {})
        lines.append(
            f"{label}：擊球 {p.get('shots', 0)}（不含發球），"
            f"發球 {p.get('serves', 0)}，得分 {p.get('winners', 0)}；"
            f"揮拍 {st.get('swing', 0)}，高壓 {st.get('overhead', 0)}，未知 {st.get('unknown', 0)}"
        )

    a, sv, rv = spd.get("all", {}), spd.get("serves", {}), spd.get("rally", {})
    lines.append(
        f"球速：整體均 {a.get('avg_kmh', '?')} km/h / 最高 {a.get('max_kmh', '?')} km/h；"
        f"發球均 {sv.get('avg_kmh', '?')} km/h / 最高 {sv.get('max_kmh', '?')} km/h；"
        f"回合球均 {rv.get('avg_kmh', '?')} km/h"
    )
    lines.append(
        f"站位：底線 {dep.get('baseline', 0)} 次，發球區 {dep.get('service', 0)} 次，網前 {dep.get('net', 0)} 次"
    )
    return "\n".join(lines)


# ── 單回合描述文字（供 embedding 使用）────────────────────────────────────────

def build_rally_text(rally: dict) -> str:
    shots_s = "→".join(
        f"{'上' if sh['player'] == 'top' else '下'}"
        f"{sh.get('shot_type', '?')[:2]}"
        f"({sh.get('speed_kmh') or '—'}km/h)"
        for sh in rally.get("shots", [])
    )
    oc = rally.get("outcome", {})
    wp = oc.get("winner_player")
    outcome = f"{wp} 得分" if wp else oc.get("type", "unknown")
    return (
        f"回合{rally['id']} "
        f"{rally.get('start_time_sec', 0):.1f}s–{rally.get('end_time_sec', 0):.1f}s "
        f"{rally.get('shot_count', 0)}拍 "
        f"發球方:{rally.get('server', '?')} "
        f"擊球:{shots_s} "
        f"結果:{outcome}"
    )


# ── Embedding 客戶端 ───────────────────────────────────────────────────────────

def _embed_texts(texts: List[str], cfg: EmbeddingConfig) -> List[List[float]]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    resp = requests.post(
        f"{cfg.url}/v1/embeddings",
        headers=headers,
        json={"model": cfg.model, "input": texts},
        timeout=_EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # 依 index 排序後回傳
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── 懶生成 + 快取 rally embeddings ────────────────────────────────────────────

def ensure_rally_embeddings(
    json_path: Path,
    data: dict,
    cfg: EmbeddingConfig,
) -> Dict:
    """讀取或生成 rally_embeddings.json。回傳快取 dict。"""
    cache_path = json_path.parent / "rally_embeddings.json"

    rallies = data.get("rallies", [])
    if not rallies:
        return {}

    texts = [build_rally_text(r) for r in rallies]

    # 若快取存在且內容相符，直接回傳
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_rallies = cache.get("rallies", [])
            cached_texts = [r.get("text", "") for r in cached_rallies]
            if cache.get("model") == cfg.model and cached_texts == texts:
                return cache
        except Exception:
            pass

    # 生成
    print(f"[chat/context] 生成 rally embeddings，共 {len(rallies)} 個回合…")
    try:
        embeddings = _embed_texts(texts, cfg)
    except Exception as e:
        print(f"[chat/context] embedding 生成失敗：{e}")
        return {}

    cache = {
        "model": cfg.model,
        "rallies": [
            {"id": r["id"], "text": t, "embedding": emb}
            for r, t, emb in zip(rallies, texts, embeddings)
        ],
    }
    try:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as e:
        print(f"[chat/context] 快取寫入失敗：{e}")

    return cache


# ── 檢索最相關回合 ─────────────────────────────────────────────────────────────

def _fmt_rally_detail(rally: dict) -> str:
    """格式化回合詳情（供 context 注入）。"""
    oc      = rally.get("outcome", {})
    wp      = oc.get("winner_player")
    outcome = f"→ {wp} 得分" if wp else f"→ {oc.get('type', 'unknown')}"
    shots_s = "→".join(
        f"{'上' if sh['player'] == 'top' else '下'}"
        f"{sh.get('shot_type', '?')[:2]}"
        f"({sh.get('speed_kmh') or '—'}km/h)"
        for sh in rally.get("shots", [])
    )
    return (
        f"[回合{rally['id']} {rally.get('start_time_sec', 0):.1f}s–"
        f"{rally.get('end_time_sec', 0):.1f}s "
        f"{rally.get('shot_count', 0)}拍] {shots_s} {outcome}"
    )


def retrieve_relevant_rallies(
    question: str,
    embeddings_cache: Dict,
    data: dict,
    cfg: EmbeddingConfig,
    top_k: int = 5,
) -> str:
    """嵌入問題，回傳 top-K 最相關回合的格式化詳情文字。"""
    cached_rallies = embeddings_cache.get("rallies", [])
    all_rallies    = {r["id"]: r for r in data.get("rallies", [])}

    if not cached_rallies or not all_rallies:
        # fallback：全部回合直接輸出（原有行為）
        lines = [_fmt_rally_detail(r) for r in data.get("rallies", [])]
        return "\n".join(lines)

    # 嵌入問題
    try:
        q_emb = _embed_texts([question], cfg)[0]
    except Exception as e:
        print(f"[chat/context] 問題 embedding 失敗：{e}，改用全部回合")
        lines = [_fmt_rally_detail(r) for r in data.get("rallies", [])]
        return "\n".join(lines)

    # 計算相似度並排序
    scored = [
        (cr["id"], _cosine_similarity(q_emb, cr["embedding"]))
        for cr in cached_rallies
        if cr.get("embedding")
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [rid for rid, _ in scored[:top_k]]

    total = len(all_rallies)
    header = f"=== 以下為與問題最相關的回合詳情（共 {total} 個回合中的 {len(top_ids)} 個）==="
    lines  = [header]
    for rid in top_ids:
        r = all_rallies.get(rid)
        if r:
            lines.append(_fmt_rally_detail(r))

    return "\n".join(lines)


# ── 組裝完整分析上下文 ─────────────────────────────────────────────────────────

def build_analysis_context(
    json_path: Path,
    data: dict,
    question: str,
    cfg: EmbeddingConfig,
) -> str:
    """摘要統計 + embedding 檢索的相關回合詳情。"""
    summary = build_summary_context(data)
    cache   = ensure_rally_embeddings(json_path, data, cfg)
    rallies = retrieve_relevant_rallies(question, cache, data, cfg)
    return summary + "\n\n" + rallies
