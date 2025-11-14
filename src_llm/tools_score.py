# src_llm/tools_score.py
import os
from typing import Dict, Any

import cv2
from ultralytics import YOLO


def score_video(
    video_path: str,
    step: int = 1,
    conf: float = 0.15,
    imgsz: int = 1280,
    use_class_filter: bool = True,
    near_top_th: float = 0.3,
    near_bottom_th: float = 0.7,
    start_hold_frames: int = 1,
    min_post_serve_visible: float = 0.15,
    ema_alpha: float = 0.45,
    vy_delta_min: float = 30.0,
    contact_cooldown: float = 0.06,
    net_hys: int = 6,
    miss_seconds_end: float = 0.80,
    min_point_duration: float = 0.60,
    require_cross_or_contact: bool = False,
    debug: bool = False,
    debug_out: str | None = None,
) -> Dict[str, Any]:
    """
    回傳格式：
    {
        "fps": <float>,
        "points": [
            {
                "id": 1,
                "events": [
                    {"t": 0.xxx, "type": "serve", "by": "bottom"},
                    {"t": 1.xxx, "type": "contact", "by": "top"},
                    ...,
                    {"t": 12.xxx, "type": "end"}
                ],
                "winner": "top" / "bottom",
                "reason": "ace" / "serve_won_no_contact" / "return_miss" / "rally_winner"
            },
            ...
        ]
    }
    """
    model = YOLO("model/yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_dur = (max(0, total_frames - 1)) / fps if total_frames > 0 else 1e9

    def clamp_time(t: float) -> float:
        return max(0.0, min(t, video_dur)) if video_dur < 1e8 else max(0.0, t)

    writer = None
    if debug:
        if not debug_out:
            # 沒給路徑就寫在同資料夾
            base, _ = os.path.splitext(video_path)
            debug_out = base + "_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_out, fourcc, fps / max(1, step), (W, H))

    points: list[Dict[str, Any]] = []
    cur: Dict[str, Any] | None = None
    last_ball = None
    last_side = None
    serve_side = None

    crossed_once = False
    miss_frames = 0
    fno = -1
    prev_vy = None
    start_hold = 0
    post_serve_visible_t = 0.0
    net_y = H / 2

    def side_of(y):
        return "bottom" if y > net_y else "top"

    def crossed(a, b):
        if a <= net_y - net_hys and b >= net_y + net_hys:
            return True
        if a >= net_y + net_hys and b <= net_y - net_hys:
            return True
        return False

    def append_event(ev_type, t_now, by=None):
        ev = {"t": round(clamp_time(t_now), 3), "type": ev_type}
        if by is not None:
            ev["by"] = by
        cur["events"].append(ev)

    def dump_end(t_now):
        nonlocal cur, last_ball, last_side, serve_side, crossed_once
        nonlocal prev_vy, start_hold, post_serve_visible_t

        if not cur:
            return

        last_seen_t = last_ball[0] if last_ball is not None else t_now
        t_end = clamp_time(min(t_now, last_seen_t + miss_seconds_end, video_dur))
        start_t = cur["events"][0]["t"] if cur["events"] else t_end
        dur = t_end - start_t
        valid = True
        if dur < min_point_duration:
            valid = False
        if post_serve_visible_t < min_post_serve_visible:
            valid = False
        if require_cross_or_contact and (not crossed_once) and (last_side is None):
            valid = False

        if valid:
            if last_side:
                cur["winner"] = "top" if last_side == "bottom" else "bottom"
                cur["reason"] = "rally_winner" if crossed_once else "return_miss"
            else:
                if crossed_once:
                    cur["winner"] = serve_side
                    cur["reason"] = "serve_won_no_contact"
                else:
                    cur["winner"] = serve_side
                    cur["reason"] = "ace"
            append_event("end", t_end)
            points.append(cur)

        cur = None
        last_ball = None
        last_side = None
        serve_side = None
        crossed_once = False
        prev_vy = None
        start_hold = 0
        post_serve_visible_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fno += 1

        if fno % step:
            if debug and writer:
                writer.write(frame)
            continue

        t = fno / fps

        res = YOLO.predict(
            model,
            frame,
            imgsz=imgsz,
            conf=conf,
            classes=[32] if use_class_filter else None,
            verbose=False,
        )[0]

        boxes = res.boxes
        ball_idx = None
        xyxy = None

        if len(boxes):
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            cand_idx = list(range(len(confs)))
            if cand_idx:
                if last_ball is None:
                    ball_idx = max(cand_idx, key=lambda i: confs[i])
                else:
                    lx, ly = last_ball[1], last_ball[2]

                    def dist2(i):
                        x1, y1, x2, y2 = xyxy[i]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        return (cx - lx) ** 2 + (cy - ly) ** 2

                    near_idx = min(cand_idx, key=dist2)
                    far = dist2(near_idx) > (max(W, H) * 0.15) ** 2
                    ball_idx = (
                        max(cand_idx, key=lambda i: confs[i]) if far else near_idx
                    )

        if ball_idx is not None:
            miss_frames = 0
            x1, y1, x2, y2 = xyxy[ball_idx]
            cx, cy_raw = (x1 + x2) / 2, (y1 + y2) / 2
            cy_ema = (
                cy_raw
                if last_ball is None
                else ema_alpha * cy_raw + (1 - ema_alpha) * last_ball[3]
            )

            near_top = cy_ema < H * near_top_th
            near_bottom = cy_ema > H * near_bottom_th
            in_edge = near_top or near_bottom

            if cur is None:
                start_hold = start_hold + 1 if in_edge else 0
                vy_ok = True
                if last_ball is not None:
                    dt = max(1e-6, t - last_ball[0])
                    vy = (cy_ema - last_ball[3]) / dt
                    if near_top and vy <= 0:
                        vy_ok = False
                    if near_bottom and vy >= 0:
                        vy_ok = False
                if start_hold >= start_hold_frames and vy_ok:
                    cur = {
                        "id": len(points) + 1,
                        "events": [],
                        "winner": None,
                        "reason": None,
                    }
                    serve_side = side_of(cy_ema)
                    crossed_once = False
                    last_side = None
                    prev_vy = None
                    post_serve_visible_t = 0.0
                    append_event("serve", t, serve_side)
            else:
                post_serve_visible_t += step / fps
                if last_ball is not None:
                    dt = max(1e-6, t - last_ball[0])
                    vy = (cy_ema - last_ball[3]) / dt

                    if crossed(last_ball[3], cy_ema):
                        from_side = side_of(last_ball[3])

                        if not crossed_once:
                            crossed_once = True
                        else:
                            last_side = from_side
                            append_event("contact", t, last_side)

                    prev_vy = vy

            last_ball = (t, cx, cy_raw, cy_ema)

            if debug and writer:
                cv2.circle(frame, (int(cx), int(cy_raw)), 6, (0, 255, 255), -1)
                cv2.line(frame, (0, int(H / 2)), (W, int(H / 2)), (200, 200, 200), 1)
                cv2.rectangle(
                    frame,
                    (0, int(H * near_top_th)),
                    (W, int(H * near_top_th) + 2),
                    (255, 255, 0),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (0, int(H * near_bottom_th) - 2),
                    (W, int(H * near_bottom_th)),
                    (255, 255, 0),
                    2,
                )
                txt = (
                    f"t={clamp_time(t):.2f}s state={'RUN' if cur else 'IDLE'} "
                    f"serve={serve_side} last={last_side} crossed={crossed_once}"
                )
                cv2.putText(
                    frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if cur:
                    y0 = 60
                    for i, e in enumerate(cur["events"][-8:]):
                        cv2.putText(
                            frame,
                            f"{e['type']}:{e.get('by','')}",
                            (10, y0 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )
                writer.write(frame)
        else:
            miss_frames += 1
            if cur is not None and (miss_frames / fps) * step > miss_seconds_end:
                dump_end(t)
            if debug and writer:
                cv2.putText(
                    frame,
                    "BALL MISSING",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                writer.write(frame)

    if cur is not None:
        dump_end(video_dur)

    cap.release()
    if writer:
        writer.release()

    result: Dict[str, Any] = {"fps": fps, "points": points}
    if debug and debug_out:
        result["debug_out"] = debug_out
    return result
