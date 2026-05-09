# services/chat/prompt.py
from __future__ import annotations

from typing import Optional

from config import BASE_DIR

_BASE_PROMPT_CACHE: Optional[str] = None

_ANALYZED_PREAMBLE = """\
你現在是根據結構化分析數據回答問題的網球分析師。你無法直接觀看影片，但擁有完整的比賽分析數據。
分析數據已附在本系統提示的末尾，請優先根據這些數據回答問題。
若使用者詢問特定回合的視覺細節（如球員動作、姿態、站位調整等），你可以使用 view_rally_clip 工具查看該回合的標註影片截圖，獲取視覺資訊後再作答。

核心限制調整：
一 根據分析數據和工具回傳的視覺描述作答，不再要求「只能根據畫面中可視覺觀察到的內容」。
二 分析數據中，球員以「上方球員」(top) 和「下方球員」(bottom) 標識，回答時沿用此稱呼。
三 數據不足時坦承說明，不捏造內容。
四 除非使用者明確要求翻譯或保留原文引用，否則一律使用繁體中文回答。
"""

_UNANALYZED_PREAMBLE = """\
你是一位專業網球知識助手。目前影片尚未完成分析，你可以回答一般網球知識問題，但無法分析影片內容。
若使用者詢問影片相關問題（如球速、回合、站位等），請禮貌地提醒他們先點擊「開始分析」按鈕完成分析，分析完成後即可使用完整的影片分析功能。
請使用繁體中文回答。
"""


def load_base_prompt() -> str:
    global _BASE_PROMPT_CACHE
    if _BASE_PROMPT_CACHE is None:
        path = BASE_DIR / "tennis_prompt.txt"
        _BASE_PROMPT_CACHE = path.read_text(encoding="utf-8") if path.exists() else ""
    return _BASE_PROMPT_CACHE


def build_system_prompt(analysis_context: Optional[str]) -> str:
    """組裝完整 system prompt。

    analysis_context 為 None → 未分析模式（一般知識問答）
    analysis_context 有值  → 已分析模式（RAG + 工具）
    """
    base = load_base_prompt()

    if analysis_context is None:
        return _UNANALYZED_PREAMBLE.rstrip()

    return (
        _ANALYZED_PREAMBLE.rstrip()
        + "\n\n"
        + base.rstrip()
        + "\n\n"
        + analysis_context
    )
