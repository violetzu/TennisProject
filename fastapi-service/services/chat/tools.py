# services/chat/tools.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config import DATA_DIR, VIDEO_URL_DOMAIN
from services.analyze.vlm_verify import select_rally_thumbs

if TYPE_CHECKING:
    from services.chat.llm import VLLMClient

# ── 工具定義（OpenAI function calling 格式）────────────────────────────────────

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "view_rally_clip",
            "description": (
                "查看指定回合的標註影片截圖，獲取該回合的視覺描述。"
                "當使用者詢問特定回合的視覺細節時使用此工具，"
                "例如：球員動作、擊球姿態、站位調整、球的軌跡等視覺資訊。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rally_id": {
                        "type": "integer",
                        "description": "回合編號（從 1 開始，對應分析數據中的回合 id）",
                    }
                },
                "required": ["rally_id"],
            },
        },
    }
]


# ── 縮圖路徑 → HTTP URL ────────────────────────────────────────────────────────

def _thumb_to_url(thumb_path: Path) -> str:
    rel = thumb_path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    return f"{VIDEO_URL_DOMAIN}/videos/{rel}"


# ── 工具執行器 ────────────────────────────────────────────────────────────────

def execute_view_rally_clip(
    rally_id: int,
    analysis_data: dict,
    thumb_dir: Path,
    fps: float,
    vlm_client: "VLLMClient",
) -> str:
    """查看指定回合的縮圖，呼叫 VLM 取得視覺描述。回傳描述文字。"""
    rallies = {r["id"]: r for r in analysis_data.get("rallies", [])}
    rally   = rallies.get(rally_id)
    if rally is None:
        return f"找不到回合 {rally_id} 的資料，請確認回合編號是否正確。"

    start_f = rally.get("start_frame", 0)
    end_f   = rally.get("end_frame", start_f)

    thumbs = select_rally_thumbs(thumb_dir, start_f, end_f, max_n=12)
    if not thumbs:
        return f"回合 {rally_id} 沒有可用的影片截圖（可能是縮圖尚未生成）。"

    image_urls = [_thumb_to_url(p) for p in thumbs]

    start_s = rally.get("start_time_sec", start_f / max(fps, 1.0))
    end_s   = rally.get("end_time_sec",   end_f   / max(fps, 1.0))
    server  = "上方球員" if rally.get("server") == "top" else "下方球員"

    user_text = (
        f"以下 {len(thumbs)} 張截圖來自網球比賽第 {rally_id} 回合"
        f"（時間：{start_s:.1f}s–{end_s:.1f}s，發球方：{server}）。\n"
        "上方球員在畫面上半部，下方球員在畫面下半部。\n\n"
        "請詳細描述以下內容（以繁體中文回答）：\n"
        "1. 雙方球員的移動和步伐\n"
        "2. 擊球動作與技術（正手、反手、截擊等）\n"
        "3. 球員的站位和場地深度\n"
        "4. 球的飛行軌跡和彈跳\n"
        "5. 回合的戰術特點或值得注意的細節\n"
    )

    try:
        description = vlm_client.call_vision(image_urls, user_text, max_tokens=1024)
        return description or f"無法取得回合 {rally_id} 的視覺描述。"
    except Exception as e:
        print(f"[chat/tools] VLM 呼叫失敗 rally_id={rally_id}: {e}")
        return f"無法取得回合 {rally_id} 的視覺分析結果。"


def dispatch_tool(
    tool_name: str,
    arguments: dict,
    analysis_data: dict,
    thumb_dir: Path,
    fps: float,
    vlm_client: "VLLMClient",
) -> str:
    """工具分派器。"""
    if tool_name == "view_rally_clip":
        rally_id = arguments.get("rally_id")
        if not isinstance(rally_id, int):
            return "工具呼叫參數錯誤：rally_id 必須為整數。"
        return execute_view_rally_clip(rally_id, analysis_data, thumb_dir, fps, vlm_client)
    return f"未知工具：{tool_name}"
