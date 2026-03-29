"""
統計彙整 (Aggregate)

將分析結果轉換為最終 JSON 結構。

- build_single_rally()  處理單回合 → rally JSON 片段 + per-rally 統計增量
- build_summary()       彙總所有回合 → summary + heatmap
- build_rallies()       （向下相容）全局一次組裝（呼叫上面兩者）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .court import project_to_world
from .analysis import (
    MIN_BALL_SPEED_KMH,
    assign_court_side, bounce_zone, find_serve_index,
    find_winner_landing, player_court_zone,
)


# ─────────────────────────────────────────────────────────────────────────────
# 內部工具
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_pos(pos: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    return round(pos[0] / width, 4), round(pos[1] / height, 4)


def _speed_stats(speeds: List[float]) -> Dict:
    if not speeds:
        return {"avg_kmh": None, "max_kmh": None, "min_kmh": None, "count": 0}
    return {
        "avg_kmh": round(sum(speeds) / len(speeds), 1),
        "max_kmh": round(max(speeds), 1),
        "min_kmh": round(min(speeds), 1),
        "count": len(speeds),
    }


SPEED_LOOK_AHEAD_SEC = 0.4  # 擊球後找球速峰值的時間窗口

def _get_speed_after(
    frame_idx: int, smooth_speeds: List[Optional[float]], look_ahead: int = 12,
) -> Optional[float]:
    peaks = [
        smooth_speeds[j]
        for j in range(frame_idx, min(frame_idx + look_ahead, len(smooth_speeds)))
        if smooth_speeds[j] is not None and smooth_speeds[j] >= MIN_BALL_SPEED_KMH
    ]
    return round(max(peaks), 1) if peaks else None


# ─────────────────────────────────────────────────────────────────────────────
# 單回合組裝
# ─────────────────────────────────────────────────────────────────────────────

def build_single_rally(
    *,
    rally_idx: int,
    rally_contacts: List[int],
    bounces_f: List[int],
    pos_interp: List[Optional[Tuple[float, float]]],
    smooth_speeds: List[Optional[float]],
    all_player_top: List[Optional[Tuple[float, float]]],
    all_player_bottom: List[Optional[Tuple[float, float]]],
    scene_cut_frames: List[int],
    vlm_shot_types: Dict[int, str],
    last_valid_H: Optional,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    next_rally_start: Optional[int],
) -> Tuple[Dict, Dict]:
    """處理單回合。

    Returns:
        (rally_json, stats)
        stats 包含: player_stats 增量、speed lists、heatmap entries、depth counts
    """

    def _pn(pos: Tuple[float, float]) -> Tuple[float, float]:
        return _normalize_pos(pos, width, height)

    def _side(fi: int) -> str:
        return assign_court_side(
            fi, pos_interp, all_player_top, all_player_bottom,
            last_valid_H, height)

    # ── 發球偵測 ──────────────────────────────────────────────────────────
    serve_idx = find_serve_index(
        rally_contacts, pos_interp, all_player_top, all_player_bottom,
        last_valid_H, height, fps)

    contacts_work = list(rally_contacts[serve_idx:])
    if serve_idx > 0:
        print(f"[SERVE] rally#{rally_idx+1} skipped {serve_idx} pre-serve event(s), "
              f"actual serve f={contacts_work[0]}({contacts_work[0]/fps:.2f}s)")

    server = _side(contacts_work[0])

    r_end_f = (next_rally_start - 1) if next_rally_start else total_frames
    rally_bounces_f = [b for b in bounces_f if contacts_work[0] <= b <= r_end_f]

    # ── per-rally stats 累計器 ────────────────────────────────────────────
    stats: Dict = {
        "player_stats": {
            "top": {"shots": 0, "serves": 0, "winners": 0,
                    "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
            "bottom": {"shots": 0, "serves": 0, "winners": 0,
                       "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
        },
        "all_speeds": [],
        "serve_speeds": [],
        "rally_speeds": [],
        "depth_counts": {"net": 0, "service": 0, "baseline": 0},
        "player_depth": {
            "top": {"net": 0, "service": 0, "baseline": 0},
            "bottom": {"net": 0, "service": 0, "baseline": 0},
        },
        "heatmap_contacts": [],
        "heatmap_bounces": [],
        "heatmap_top": [],
        "heatmap_bot": [],
    }

    stats["player_stats"][server]["serves"] += 1

    # ── 逐拍 shots ──────────────────────────────────────────────────────
    shots_out: List[Dict] = []
    for seq, fi in enumerate(contacts_work):
        ball_pos = pos_interp[fi] if fi < len(pos_interp) else None
        if ball_pos is None:
            continue

        player = _side(fi)
        is_serve = (seq == 0)
        shot_type = "serve" if is_serve else vlm_shot_types.get(fi, "unknown")
        speed = _get_speed_after(fi, smooth_speeds,
                                 look_ahead=max(1, int(SPEED_LOOK_AHEAD_SEC * fps)))
        x_norm, y_norm = _pn(ball_pos)

        plist = all_player_top if player == "top" else all_player_bottom
        ppos = plist[fi] if fi < len(plist) else None
        player_pos: Optional[Dict] = None
        _ppx, _ppy = 0.0, 0.0
        if ppos:
            _ppx, _ppy = _pn(ppos)
            player_pos = {"x": _ppx, "y": _ppy}

        ball_world = (project_to_world(ball_pos, last_valid_H)
                      if last_valid_H is not None else None)
        player_world = (project_to_world(ppos, last_valid_H)
                        if ppos is not None and last_valid_H is not None else None)

        if player_world:
            p_zone = player_court_zone(player_world[1])
        else:
            dist_net = abs(y_norm - 0.5) / 0.5
            p_zone = ("net" if dist_net < 0.17
                      else "service" if dist_net < 0.54
                      else "baseline")

        if not is_serve:
            stats["player_stats"][player]["shots"] += 1
        stats["player_stats"][player]["shot_types"][shot_type] += 1
        stats["depth_counts"][p_zone] += 1
        stats["player_depth"][player][p_zone] += 1
        if speed is not None:
            stats["all_speeds"].append(speed)
            (stats["serve_speeds"] if is_serve else stats["rally_speeds"]).append(speed)

        hm_entry: Dict = {"player": player}
        if ball_world:
            hm_entry.update(x=round(ball_world[0], 3),
                            y=round(ball_world[1], 3), coord="world")
        else:
            hm_entry.update(x=x_norm, y=y_norm, coord="pixel")
        stats["heatmap_contacts"].append(hm_entry)

        if ppos:
            bucket = stats["heatmap_top"] if player == "top" else stats["heatmap_bot"]
            if player_world:
                bucket.append({"x": round(player_world[0], 3),
                               "y": round(player_world[1], 3), "coord": "world"})
            else:
                bucket.append({"x": _ppx, "y": _ppy, "coord": "pixel"})

        shot: Dict = {
            "seq": seq + 1,
            "frame": fi,
            "time_sec": round(fi / fps, 3),
            "player": player,
            "is_serve": is_serve,
            "shot_type": shot_type,
            "speed_kmh": speed,
            "ball_pos": {"x": x_norm, "y": y_norm},
            "player_pos": player_pos,
            "player_zone": p_zone,
        }
        if ball_world:
            shot["ball_world"] = {"x": round(ball_world[0], 2),
                                  "y": round(ball_world[1], 2)}
        if player_world:
            shot["player_world"] = {"x": round(player_world[0], 2),
                                    "y": round(player_world[1], 2)}
        shots_out.append(shot)

    # ── bounces ──────────────────────────────────────────────────────────
    bounces_out: List[Dict] = []
    for bf in rally_bounces_f:
        bpos = pos_interp[bf] if bf < len(pos_interp) else None
        if bpos is None:
            continue
        x_n, y_n = _pn(bpos)
        b_world = (project_to_world(bpos, last_valid_H)
                   if last_valid_H is not None else None)
        b_entry: Dict = {
            "frame": bf,
            "time_sec": round(bf / fps, 3),
            "pos": {"x": x_n, "y": y_n},
        }
        if b_world:
            b_entry["world"] = {"x": round(b_world[0], 2),
                                "y": round(b_world[1], 2)}
            b_entry["zone"] = bounce_zone(b_world[0], b_world[1])
        bounces_out.append(b_entry)
        if b_world:
            stats["heatmap_bounces"].append(
                {"x": round(b_world[0], 3), "y": round(b_world[1], 3), "coord": "world"})
        else:
            stats["heatmap_bounces"].append({"x": x_n, "y": y_n, "coord": "pixel"})

    # ── rally JSON（outcome 暫留 placeholder，VLM 結果稍後填入）────────
    rally_json: Dict = {
        "id": rally_idx + 1,
        "start_frame": rally_contacts[0],
        "end_frame": rally_contacts[-1],
        "start_time_sec": round(rally_contacts[0] / fps, 2),
        "end_time_sec": round(rally_contacts[-1] / fps, 2),
        "duration_sec": round((rally_contacts[-1] - rally_contacts[0]) / fps, 2),
        "shot_count": len(shots_out),
        "server": server,
        "shots": shots_out,
        "bounces": bounces_out,
        "outcome": {
            "type": "unknown",
            "winner_player": None,
            "winner_land": None,
        },
    }

    return rally_json, stats


# ─────────────────────────────────────────────────────────────────────────────
# 全局彙總
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(
    rally_results: List[Dict],
    per_rally_stats: List[Dict],
) -> Dict:
    """彙總所有回合的 summary + heatmap。

    Args:
        rally_results: 每回合的 rally JSON（已填入 outcome）。
        per_rally_stats: 每回合的 stats dict（由 build_single_rally 產出）。

    Returns:
        {"summary": {...}, "heatmap": {...}}
    """
    player_stats: Dict = {
        "top": {"shots": 0, "serves": 0, "winners": 0,
                "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
        "bottom": {"shots": 0, "serves": 0, "winners": 0,
                   "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
    }
    all_speeds: List[float] = []
    serve_speeds: List[float] = []
    rally_speeds: List[float] = []
    depth_counts: Dict = {"net": 0, "service": 0, "baseline": 0}
    player_depth: Dict = {
        "top": {"net": 0, "service": 0, "baseline": 0},
        "bottom": {"net": 0, "service": 0, "baseline": 0},
    }
    heatmap_contacts: List[Dict] = []
    heatmap_bounces: List[Dict] = []
    heatmap_top: List[Dict] = []
    heatmap_bot: List[Dict] = []

    for s in per_rally_stats:
        for side in ("top", "bottom"):
            ps = s["player_stats"][side]
            for key in ("shots", "serves", "winners"):
                player_stats[side][key] += ps[key]
            for st_key, st_val in ps["shot_types"].items():
                player_stats[side]["shot_types"][st_key] += st_val
        all_speeds.extend(s["all_speeds"])
        serve_speeds.extend(s["serve_speeds"])
        rally_speeds.extend(s["rally_speeds"])
        for zone in ("net", "service", "baseline"):
            depth_counts[zone] += s["depth_counts"][zone]
            player_depth["top"][zone] += s["player_depth"]["top"][zone]
            player_depth["bottom"][zone] += s["player_depth"]["bottom"][zone]
        heatmap_contacts.extend(s["heatmap_contacts"])
        heatmap_bounces.extend(s["heatmap_bounces"])
        heatmap_top.extend(s["heatmap_top"])
        heatmap_bot.extend(s["heatmap_bot"])

    total_rallies = len(rally_results)
    total_shots = sum(r["shot_count"] for r in rally_results)
    avg_rally_length = round(total_shots / total_rallies, 1) if total_rallies > 0 else 0.0

    return {
        "summary": {
            "total_rallies": total_rallies,
            "total_shots": total_shots,
            "total_winners": sum(
                1 for r in rally_results if r["outcome"]["type"] == "winner"),
            "avg_rally_length": avg_rally_length,
            "players": {
                "top": player_stats["top"],
                "bottom": player_stats["bottom"],
            },
            "speed": {
                "all": _speed_stats(all_speeds),
                "serves": _speed_stats(serve_speeds),
                "rally": _speed_stats(rally_speeds),
            },
            "depth": {
                "total": depth_counts,
                "top": player_depth["top"],
                "bottom": player_depth["bottom"],
            },
        },
        "heatmap": {
            "contacts": heatmap_contacts,
            "bounces": heatmap_bounces,
            "top_player": heatmap_top,
            "bottom_player": heatmap_bot,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 向下相容（原有 build_rallies 介面）
# ─────────────────────────────────────────────────────────────────────────────

def build_rallies(
    *,
    rally_groups: List[List[int]],
    bounces_f: List[int],
    pos_interp: List[Optional[Tuple[float, float]]],
    smooth_speeds: List[Optional[float]],
    all_player_top: List[Optional[Tuple[float, float]]],
    all_player_bottom: List[Optional[Tuple[float, float]]],
    scene_cut_frames: List[int],
    vlm_shot_types: Dict[int, str],
    vlm_winner_results: Dict[int, str],
    last_valid_H: Optional,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
) -> Dict:
    """向下相容包裝：依序呼叫 build_single_rally + build_summary。"""

    rally_results: List[Dict] = []
    per_rally_stats: List[Dict] = []

    for rally_idx, rally_contacts in enumerate(rally_groups):
        if not rally_contacts:
            continue
        next_start = (rally_groups[rally_idx + 1][0]
                      if rally_idx + 1 < len(rally_groups) else None)

        rally_json, stats = build_single_rally(
            rally_idx=rally_idx,
            rally_contacts=rally_contacts,
            bounces_f=bounces_f,
            pos_interp=pos_interp,
            smooth_speeds=smooth_speeds,
            all_player_top=all_player_top,
            all_player_bottom=all_player_bottom,
            scene_cut_frames=scene_cut_frames,
            vlm_shot_types=vlm_shot_types,
            last_valid_H=last_valid_H,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            next_rally_start=next_start,
        )

        # 填入 VLM 勝負結果
        winner_player = vlm_winner_results.get(rally_idx)
        if winner_player in ("top", "bottom"):
            stats["player_stats"][winner_player]["winners"] += 1
            rally_json["outcome"]["type"] = "winner"
            rally_json["outcome"]["winner_player"] = winner_player
            if last_valid_H is not None:
                rally_json["outcome"]["winner_land"] = find_winner_landing(
                    winner_player, rally_contacts, bounces_f, pos_interp,
                    next_start, total_frames, last_valid_H, fps)
        else:
            cut_near = any(
                rally_contacts[0] <= sc <= rally_contacts[-1] + int(fps * 3)
                for sc in scene_cut_frames)
            rally_json["outcome"]["type"] = "scene_cut" if cut_near else "unknown"
            rally_json["outcome"]["winner_player"] = winner_player

        rally_results.append(rally_json)
        per_rally_stats.append(stats)

    return {
        **build_summary(rally_results, per_rally_stats),
        "rallies": rally_results,
    }
