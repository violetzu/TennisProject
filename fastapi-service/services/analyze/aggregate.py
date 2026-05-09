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
    assign_court_side, bounce_zone, find_serve_index,
    find_winner_landing, player_court_zone, get_speed_after,
)
from .player import PlayerDetection


# ─────────────────────────────────────────────────────────────────────────────
# 內部工具
# ─────────────────────────────────────────────────────────────────────────────


def _empty_player_stats() -> Dict:
    return {
        "top": {"shots": 0, "serves": 0, "winners": 0,
                "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
        "bottom": {"shots": 0, "serves": 0, "winners": 0,
                   "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
    }


def _speed_stats(speeds: List[float]) -> Dict:
    if not speeds:
        return {"avg_kmh": None, "max_kmh": None, "min_kmh": None, "count": 0}
    return {
        "avg_kmh": round(sum(speeds) / len(speeds), 1),
        "max_kmh": round(max(speeds), 1),
        "min_kmh": round(min(speeds), 1),
        "count": len(speeds),
    }




# ─────────────────────────────────────────────────────────────────────────────
# 單拍屬性計算
# ─────────────────────────────────────────────────────────────────────────────

def _build_shot(
    seq: int,
    fi: int,
    pos_interp: List[Optional[Tuple[float, float]]],
    smooth_speeds: List[Optional[float]],
    speed_offset: int,
    fps: float,
    all_top: List,
    all_bot: List,
    last_valid_H,
    width: int,
    height: int,
    vlm_shot_types: Dict[int, str],
) -> Optional[Tuple[Dict, Dict, Optional[Dict]]]:
    """單拍計算：屬性 + shot JSON + heatmap entry。

    Returns (shot_json, hm_ball_entry, hm_player_entry) 或 None（球位置不可用）。
    hm_player_entry 為 None 表示該拍球員位置無資料。
    """
    ball_pos = pos_interp[fi] if fi < len(pos_interp) else None
    if ball_pos is None:
        return None

    is_serve = (seq == 0)
    player = assign_court_side(fi, pos_interp, all_top, all_bot, last_valid_H, height)
    shot_type = "serve" if is_serve else vlm_shot_types.get(fi, "unknown")
    speed = get_speed_after(fi, smooth_speeds, fps, speed_offset=speed_offset)

    x_norm = round(ball_pos[0] / width, 4)
    y_norm = round(ball_pos[1] / height, 4)

    p_det = (all_top[fi] if fi < len(all_top) else None) if player == "top" else \
            (all_bot[fi] if fi < len(all_bot) else None)
    ppos = p_det.pos if p_det else None
    player_pos: Optional[Dict] = None
    ppx, ppy = 0.0, 0.0
    if ppos:
        ppx = round(ppos[0] / width, 4)
        ppy = round(ppos[1] / height, 4)
        player_pos = {"x": ppx, "y": ppy}

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
        shot["ball_world"] = {"x": round(ball_world[0], 2), "y": round(ball_world[1], 2)}
    if player_world:
        shot["player_world"] = {"x": round(player_world[0], 2), "y": round(player_world[1], 2)}

    _spd = f"{speed:.0f}km/h" if speed is not None else "—"
    _bp  = f"({ball_pos[0]:.0f},{ball_pos[1]:.0f})"
    _wp  = f"({ball_world[0]:.1f},{ball_world[1]:.1f})m" if ball_world else "—"
    print(f"  [shot#{seq+1}] f={fi} t={fi/fps:.2f}s player={player} "
          f"type={shot_type} ball={_bp} world={_wp} speed={_spd} zone={p_zone}")

    hm_ball: Dict = {"player": player}
    if ball_world:
        hm_ball.update(x=round(ball_world[0], 3), y=round(ball_world[1], 3), coord="world")
    else:
        hm_ball.update(x=x_norm, y=y_norm, coord="pixel")

    hm_player: Optional[Dict] = None
    if ppos:
        if player_world:
            hm_player = {"x": round(player_world[0], 3),
                         "y": round(player_world[1], 3), "coord": "world"}
        else:
            hm_player = {"x": ppx, "y": ppy, "coord": "pixel"}

    return shot, hm_ball, hm_player


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
    speed_offset: int = 0,
    all_top: List[Optional[PlayerDetection]],
    all_bot: List[Optional[PlayerDetection]],
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

    # ── 發球偵測 ──────────────────────────────────────────────────────────
    serve_idx = find_serve_index(
        rally_contacts, pos_interp, all_top, all_bot,
        last_valid_H, height, fps)

    contacts_work = list(rally_contacts[serve_idx:])
    if serve_idx > 0:
        print(f"[SERVE] rally#{rally_idx+1} skipped {serve_idx} pre-serve event(s), "
              f"actual serve f={contacts_work[0]}({contacts_work[0]/fps:.2f}s)")

    server = assign_court_side(
        contacts_work[0], pos_interp, all_top, all_bot, last_valid_H, height)

    r_end_f = (next_rally_start - 1) if next_rally_start else total_frames
    rally_bounces_f = [b for b in bounces_f if contacts_work[0] <= b <= r_end_f]

    # ── per-rally stats 累計器 ────────────────────────────────────────────
    stats: Dict = {
        "player_stats": _empty_player_stats(),
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
        result = _build_shot(seq, fi, pos_interp, smooth_speeds, speed_offset, fps,
                             all_top, all_bot, last_valid_H, width, height, vlm_shot_types)
        if result is None:
            continue
        shot, hm_ball, hm_player = result
        shots_out.append(shot)

        player, is_serve = shot["player"], shot["is_serve"]
        shot_type, speed, p_zone = shot["shot_type"], shot["speed_kmh"], shot["player_zone"]

        if not is_serve:
            stats["player_stats"][player]["shots"] += 1
        stats["player_stats"][player]["shot_types"][shot_type] += 1
        stats["depth_counts"][p_zone] += 1
        stats["player_depth"][player][p_zone] += 1
        if speed is not None:
            stats["all_speeds"].append(speed)
            (stats["serve_speeds"] if is_serve else stats["rally_speeds"]).append(speed)
        stats["heatmap_contacts"].append(hm_ball)
        if hm_player:
            (stats["heatmap_top"] if player == "top" else stats["heatmap_bot"]).append(hm_player)

    # ── bounces ──────────────────────────────────────────────────────────
    bounces_out: List[Dict] = []
    for bf in rally_bounces_f:
        bpos = pos_interp[bf] if bf < len(pos_interp) else None
        if bpos is None:
            continue
        x_n = round(bpos[0] / width, 4)
        y_n = round(bpos[1] / height, 4)
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
    player_stats: Dict = _empty_player_stats()
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

