# app.py
import os
import json
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ollama import Client, ChatResponse

from src_llm.tools_ball import ball_position_on_frame  # WJㄉ球偵測
from src_llm.tools_score import score_video            # PHㄉ逐分記分
from src_llm.tools_clip import explain_clip_segment    # WYㄉ分析

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "llm_server/uploads")
OUT_DIR = os.path.join(APP_DIR, "llm_server/outputs")
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 改成你現在要連的那台 Ollama
#現在是WJ電腦
OLLAMA_CLIENT = Client(host="http://220.132.170.233:11434")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 單人簡易版狀態（多人要自己改成 session / token）
CURRENT_VIDEO_ID: str | None = None
CURRENT_VIDEO_INFO: Dict[str, Any] = {}

# ====== 將 YOLO 工具包成 LLM 可呼叫的 tool ======
def tool_get_video_info() -> Dict[str, Any]:
    if CURRENT_VIDEO_ID is None:
        return {
            "ok": False,
            "error": "no video uploaded",
            "message": "目前沒有上傳中的影片",
        }
    return {
        "ok": True,
        "video_id": CURRENT_VIDEO_ID,
        "frames": CURRENT_VIDEO_INFO.get("frames", 0),
        "fps": CURRENT_VIDEO_INFO.get("fps", 0.0),
    }

def tool_get_ball_position(frame_id: int) -> Dict[str, Any]:
    if CURRENT_VIDEO_ID is None:
        return {
            "ok": False,
            "error": "no video uploaded",
            "message": "目前沒有上傳中的影片",
        }
    video_path = os.path.join(UPLOAD_DIR, CURRENT_VIDEO_ID)
    return ball_position_on_frame(video_path, frame_id, OUT_DIR, base_url="/image")

def tool_score_video(debug: bool = False) -> Dict[str, Any]:
    if CURRENT_VIDEO_ID is None:
        return {
            "ok": False,
            "error": "no video uploaded",
            "message": "目前沒有上傳中的影片",
        }

    video_path = os.path.join(UPLOAD_DIR, CURRENT_VIDEO_ID)

    # debug 輸出檔名（存在 OUT_DIR）
    debug_out = None
    if debug:
        base, _ = os.path.splitext(CURRENT_VIDEO_ID)
        debug_name = f"{base}_score_debug.mp4"
        debug_out = os.path.join(OUT_DIR, debug_name)

    result = score_video(video_path, debug=debug, debug_out=debug_out)

    # 包一層 ok + 可能的 debug 影片 URL
    out: Dict[str, Any] = {
        "ok": True,
        "fps": result.get("fps", 0.0),
        "points": result.get("points", []),
    }
    if debug and debug_out:
        # 給前端一個可以抓 debug.mp4 的 URL
        out["debug_url"] = f"/output/{os.path.basename(debug_out)}"

    return out

def tool_explain_clip_segment(start_ts: str, end_ts: str) -> Dict[str, Any]:
    """
    解析目前上傳影片在 [start_ts, end_ts] 這段的網球情境：
    行為類型、球落地時間、觸球序列與勝方。
    """
    if CURRENT_VIDEO_ID is None:
        return {
            "ok": False,
            "error": "no video uploaded",
            "message": "目前沒有上傳中的影片",
        }

    video_path = os.path.join(UPLOAD_DIR, CURRENT_VIDEO_ID)
    # 這邊 out_dir 用現有的 OUT_DIR，就跟 ball_position_on_frame 一樣放同個根目錄
    return explain_clip_segment(
        video_path=video_path,
        start_ts=start_ts,
        end_ts=end_ts,
        out_dir=OUT_DIR,
    )


# ====== LLM理解用的 ======
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_video_info",
            "description": "取得影片的總幀數與 fps",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ball_position",
            "description": "取得第 n 幀球的位置",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "integer"},
                },
                "required": ["frame_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_video",
            "description": "分析目前這部影片的每一分（發球方、關鍵接觸、得分方與原因等）。影片路徑由後端管理，不需要傳。",
            "parameters": {
                "type": "object",
                "properties": {
                    "debug": {
                        "type": "boolean",
                        "description": "是否輸出帶疊字標註的 debug 影片（mp4）",
                        "default": False,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_clip_segment",
            "description": "解析影片在指定時間區間內的網球回合情境，推論行為、落地時間與勝方",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_ts": {
                        "type": "string",
                        "description": "片段開始時間，例如 '00:12' 或 '00:12:03'",
                    },
                    "end_ts": {
                        "type": "string",
                        "description": "片段結束時間，例如 '00:19' 或 '00:19:05'",
                    },
                },
                "required": ["start_ts", "end_ts"],
            },
        },
    },
]


# ====== 路由 ======
@app.get("/")
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return HTMLResponse(open(index_path, "r", encoding="utf-8").read())


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global CURRENT_VIDEO_ID, CURRENT_VIDEO_INFO

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        return JSONResponse(
            {"ok": False, "error": "僅支援 mp4/mov/avi/mkv"}, status_code=400
        )

    vid = f"video_{os.urandom(4).hex()}{ext}"
    path = os.path.join(UPLOAD_DIR, vid)
    with open(path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    CURRENT_VIDEO_ID = vid
    CURRENT_VIDEO_INFO = {"frames": frames, "fps": fps}

    return {
        "ok": True,
        "video_id": vid,
        "frames": frames,
        "fps": fps,
    }


@app.get("/image/{name}")
def get_image(name: str):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "圖檔不存在"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/ask")
async def ask(payload: Dict[str, Any]):
    """
    接收：
    {
      "query": "第17幀球在哪？",
      "model_name": "qwen2.5"
    }
    影片資訊從 CURRENT_VIDEO_ID / CURRENT_VIDEO_INFO 取
    """
    if CURRENT_VIDEO_ID is None:
        return JSONResponse({"ok": False, "error": "請先上傳影片"}, status_code=400)

    query = payload.get("query", "")
    model_name = payload.get("model_name", "qwen2.5")

    messages: list[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是影片分析助手。\n"
                "使用者的影片已經由後端記住，因此你在呼叫工具時只需要提供 frame_id 等參數，"
                "不需要管影片路徑。\n"
                "工具說明：\n"
                "1) get_ball_position(frame_id:int): 回傳該幀球的位置資訊（grid9 / center / bbox / image_url 等）。\n"
                "2) get_video_info(): 回傳目前影片的總幀數與 fps。\n"
                "3) score_video(debug:bool=False): 對整部影片做逐分分析，輸出每一分的事件時間點與勝負原因，"
                "debug=True 時會產生一支帶疊字標註的 debug 影片（後端會回傳網址）。\n"
                "4) explain_clip_segment(start_ts:str, end_ts:str): 解析指定時間區間的回合狀況，包含動作類型、球落地時間、觸球順序與勝方推定。\n"

                "回答時請用簡短自然的繁體中文，必要時可說明幀數是否超出範圍。"
            ),
        },
        {"role": "user", "content": query},
    ]

    tool_results: Dict[str, Any] = {}

    while True:
        resp: ChatResponse = OLLAMA_CLIENT.chat(
            model=model_name,
            messages=messages,
            tools=OLLAMA_TOOLS,
            options={"temperature": 0.0},
        )

        messages.append(resp.message)

        # 沒有再呼叫 tool，這就是最終回答
        if not resp.message.tool_calls:
            answer_text = resp.message.content
            result: Dict[str, Any] = {"ok": True, "answer": answer_text}

            if "get_ball_position" in tool_results:
                result.update(tool_results["get_ball_position"])
            if "get_video_info" in tool_results:
                result["video_info"] = tool_results["get_video_info"]
            if "score_video" in tool_results:
                # 不直接展開，包在一個欄位裡
                result["score"] = tool_results["score_video"]
            if "explain_clip_segment" in tool_results:
                result["clip_explain"] = tool_results["explain_clip_segment"]


            return result



        # 執行所有工具呼叫
        for call in resp.message.tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if name == "get_ball_position":
                result = tool_get_ball_position(**args)
            elif name == "get_video_info":
                result = tool_get_video_info()
            elif name == "score_video":
                # args 目前只會有 debug:boolean
                result = tool_score_video(**args)
            elif name == "explain_clip_segment":
                result = tool_explain_clip_segment(**args)
            else:
                result = {"ok": False, "error": f"unknown tool: {name}"}

            tool_results[name] = result

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

# PH寫的  
# 增加一個下載 output 檔案的路由（給 debug.mp4 用）
@app.get("/output/{name}")
def get_output(name: str):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "檔案不存在"}, status_code=404)

    ext = os.path.splitext(name)[1].lower()
    if ext == ".mp4":
        media_type = "video/mp4"
    elif ext in [".png", ".jpg", ".jpeg"]:
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"

    return FileResponse(path, media_type=media_type)
