# src_llm/tools_clip.py
import glob
import math
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from ultralytics import YOLO

# ---------- 時間工具 ----------
def ts_to_seconds(ts: str) -> float:
    parts = [float(x) for x in ts.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def seconds_to_hhmmss_ms(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------- 抽幀 ----------
def extract_frames(
    video_path: str,
    start_ts: str,
    end_ts: str,
    n: int,
    tmpdir: Path,
):
    tmpdir.mkdir(parents=True, exist_ok=True)
    clip = tmpdir / "clip.mp4"
    t0 = ts_to_seconds(start_ts)
    dur = max(0.01, ts_to_seconds(end_ts) - t0)

    # 先切出時間窗
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            start_ts,
            "-i",
            video_path,
            "-t",
            f"{dur:.3f}",
            "-c",
            "copy",
            str(clip),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    fps = max(1.0, n / dur)
    outpat = tmpdir / "f_%03d.jpg"

    # 再從 clip 抽幀
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clip),
            "-vf",
            f"fps={fps}",
            str(outpat),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frames = sorted(glob.glob(str(tmpdir / "f_*.jpg")))
    return frames, dur, fps, t0


# ---------- YOLO 偵測 ----------
def detect(
    image_path: str,
    model_path: str = "../model/yolov8x.pt",
    conf: float = 0.03,
    imgsz: Optional[int] = None,
):

    model = YOLO(model_path)

    kwargs: Dict[str, Any] = {
        "conf": conf,
        "verbose": False,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz

    r = model.predict(
        image_path,
        **kwargs,
    )[0]

    names = r.names
    dets = []
    for b in r.boxes:
        dets.append(
            {
                "cls": names[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "xyxy": [float(x) for x in b.xyxy[0]],
            }
        )
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets[:25]


def summarize_frame(
    dets: List[Dict[str, Any]],
    W: int,
    H: int,
    baseline_y: Optional[int] = None,
) -> Dict[str, Any]:
    persons = []
    for d in dets:
        if d["cls"] not in ("person", "player"):
            continue
        x1, y1, x2, y2 = d["xyxy"]
        cx = 0.5 * (x1 + x2)
        h = y2 - y1
        if 0.15 * W <= cx <= 0.85 * W and h >= 0.08 * H:
            d["_cx"] = cx
            d["_h"] = h
            d["_head"] = y1
            d["_foot"] = y2
            persons.append(d)

    persons.sort(key=lambda d: d["_foot"], reverse=True)
    near = persons[0] if persons else None
    far = max(persons[1:], key=lambda d: d["_h"], default=None) if len(persons) > 1 else None

    balls = [d for d in dets if d["cls"] in ("sports ball", "ball")]
    ball = balls[0] if balls else None

    if baseline_y is None:
        baseline_y = int(0.92 * H)
    opp = H - baseline_y

    out = {
        "W": W,
        "H": H,
        "baseline_y": baseline_y,
        "opp_baseline_y": opp,
        "near": {"cx": None, "head": None, "foot": None, "h": None},
        "far": {"cx": None, "head": None, "foot": None, "h": None},
        "ball": {"cx": None, "cy": None, "conf": None},
    }

    if near:
        out["near"].update(
            {
                "cx": near["_cx"],
                "head": near["_head"],
                "foot": near["_foot"],
                "h": near["_h"],
            }
        )
    if far:
        out["far"].update(
            {
                "cx": far["_cx"],
                "head": far["_head"],
                "foot": far["_foot"],
                "h": far["_h"],
            }
        )
    if ball:
        x1, y1, x2, y2 = ball["xyxy"]
        out["ball"].update(
            {
                "cx": 0.5 * (x1 + x2),
                "cy": 0.5 * (y1 + y2),
                "conf": ball["conf"],
            }
        )
    return out


# ---------- fallback 規則 ----------
def fallback_explain(snaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not snaps:
        return {"action": "unknown", "explanation": "無快照", "evidence": [], "confidence": 0.3}
    H = snaps[0]["H"]
    W = snaps[0]["W"]
    bx = [s["ball"]["cx"] for s in snaps if s["ball"]["cx"] is not None]
    by = [s["ball"]["cy"] for s in snaps if s["ball"]["cy"] is not None]

    if len(bx) >= max(2, len(snaps) // 3):
        dx = (max(bx) - min(bx)) if bx else 0
        dy = (max(by) - min(by)) if by else 0
        if dx > 0.04 * W or dy > 0.035 * H:
            return {
                "action": "rally",
                "explanation": "多幀觀察到球在兩人之間移動",
                "evidence": ["ball_motion_between_players"],
                "confidence": 0.62,
            }

    for s in snaps:
        b = s["ball"]
        near = s["near"]
        base = s["baseline_y"]
        if b["cy"] and near["head"]:
            if b["cy"] < near["head"] - 0.06 * H and (near["foot"] and near["foot"] > base - 0.02 * H):
                return {
                    "action": "serve_pre_toss",
                    "explanation": "近端選手在底線附近拋球至頭頂上方",
                    "evidence": ["near_at_baseline", "ball_above_head"],
                    "confidence": 0.58,
                }

    if len(by) >= max(2, len(snaps) // 3):
        return {
            "action": "idle",
            "explanation": "多幀可見網球但位移有限，可能為準備或暫停片段",
            "evidence": ["ball_visible_low_motion"],
            "confidence": 0.5,
        }

    return {"action": "unknown", "explanation": "證據仍不足", "evidence": [], "confidence": 0.35}


# ---------- 落地偵測 ----------
def find_bounces(snaps: List[Dict[str, Any]], fps: float) -> List[int]:
    """回傳落地發生的幀索引列表（以球y速度從下降變上升，且靠近地面為準）"""
    ys = [s["ball"]["cy"] for s in snaps]
    H = snaps[0]["H"]
    out_idx = []
    for i in range(1, len(ys) - 1):
        if ys[i - 1] is None or ys[i] is None or ys[i + 1] is None:
            continue
        vy1 = ys[i] - ys[i - 1]  # 正：向下
        vy2 = ys[i + 1] - ys[i]  # 正：向下
        near_ground = ys[i] > 0.9 * H
        if vy1 > 0 and vy2 < 0 and near_ground:
            out_idx.append(i)

    merged = []
    for idx in out_idx:
        if not merged or idx - merged[-1] > max(2, int(0.12 * fps)):
            merged.append(idx)
    return merged


# ---------- 觸球推定 ----------
def _pdist(ax: float, ay: float, bx: Optional[float], by: Optional[float]) -> float:
    if bx is None or by is None:
        return 1e9
    return math.hypot(ax - bx, ay - by)


def infer_contacts(snaps: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    holder = []
    for i, s in enumerate(snaps):
        bx, by = s["ball"]["cx"], s["ball"]["cy"]
        if bx is None or by is None:
            holder.append(None)
            continue
        nd = _pdist(bx, by, s["near"]["cx"], s["near"]["foot"])
        fd = _pdist(bx, by, s["far"]["cx"], s["far"]["foot"])
        who = "near" if nd < fd else "far"
        holder.append(who)

    contacts: List[Tuple[int, str]] = []
    last = None
    last_i = -10
    for i, w in enumerate(holder):
        if w is None:
            continue
        if last is None:
            last = w
            last_i = i
            continue
        if w != last and (i - last_i) >= 3:
            contacts.append((i, w))
            last = w
            last_i = i
    return contacts


def decide_winner(contacts: List[Tuple[int, str]], snaps: List[Dict[str, Any]]) -> str:
    if not contacts:
        return "unknown"

    H = snaps[0]["H"]
    last_idx, last_holder = contacts[-1]

    last_ball = None
    for s in reversed(snaps):
        if s["ball"]["cy"] is not None:
            last_ball = s["ball"]
            break
    if not last_ball:
        return "unknown"

    cy = last_ball["cy"]
    near_ground = cy > 0.9 * H
    far_ground = cy < 0.1 * H

    if last_holder == "near" and far_ground:
        return "near"
    if last_holder == "far" and near_ground:
        return "far"

    mid = 0.5 * H
    if last_holder == "near" and cy < mid:
        return "near"
    if last_holder == "far" and cy > mid:
        return "far"

    return "unknown"


# ---------- 對外主函式：explain_clip_segment ----------
def explain_clip_segment(
    video_path: str,
    start_ts: str,
    end_ts: str,
    out_dir: str,
    *,
    model_path: str = "../model/yolov8x.pt",
    imgsz: Optional[int] = None,
    conf: float = 0.03,
    frames: int = 15,
    baseline_y: Optional[int] = None,
) -> Dict[str, Any]:
    """
    解析一段影片時間窗的行為、落地(bounce)、觸球與勝方。

    現在不再呼叫內建 Ollama，只使用 fallback 規則。
    imgsz 可為 None（自由解析度）或指定整數。

    回傳格式：
    {
      "ok": True/False,
      "error": "...",          # 若 ok=False
      "action": "...",
      "explanation": "...",
      "evidence": [...],
      "confidence": 0.xx,
      "bounces": {...},
      "contacts": [...],
      "winner": "near|far|unknown",
      "image_size": {"W":..., "H":...}
    }
    """

    out_base_dir = Path(out_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            frames_paths, dur, fps, clip_t0 = extract_frames(
                video_path, start_ts, end_ts, frames, td_path
            )
            if not frames_paths:
                return {
                    "ok": False,
                    "error": "無法抽影格",
                }

            snaps = []
            W = H = None
            for p in frames_paths:
                dets = detect(p, model_path, conf, imgsz)
                with Image.open(p) as im:
                    W, H = im.size
                snaps.append(summarize_frame(dets, W, H, baseline_y))

            # 1) 行為敘述：只用 fallback 規則
            out_llm = fallback_explain(snaps)

            # 2) 落地偵測
            bidx = find_bounces(snaps, fps=fps)
            bounce_rel = [i / fps for i in bidx]
            bounce_abs = [clip_t0 + t for t in bounce_rel]

            # 3) 觸球序列 & 勝方推定
            contacts = infer_contacts(snaps)
            winner = decide_winner(contacts, snaps)

            # 4) 整理輸出
            result: Dict[str, Any] = {
                "ok": True,
                "action": out_llm.get("action", "unknown"),
                "explanation": out_llm.get("explanation", ""),
                "evidence": out_llm.get("evidence", []),
                "confidence": out_llm.get("confidence", 0.3),
                "bounces": {
                    "indices": bidx,
                    "relative_sec": [round(t, 3) for t in bounce_rel],
                    "absolute_sec": [round(t, 3) for t in bounce_abs],
                    "absolute_hhmmss": [
                        seconds_to_hhmmss_ms(t).replace(",", ".") for t in bounce_abs
                    ],
                },
                "contacts": [{"frame": i, "side": s} for (i, s) in contacts],
                "winner": winner,
                "image_size": {"W": W, "H": H},
            }

            return result

    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "error": f"ffmpeg 執行失敗：{e}",
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"clip 分析失敗：{e}",
        }
