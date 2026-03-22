"""
Phase 4：統計彙整 (Aggregate)

將 Phase 2/3 的偵測結果轉換為最終 JSON 結構。
純數據轉換，不依賴模型或 FFmpeg。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .analysis import (
    MIN_BALL_SPEED_KMH, _NET_Y_M,
    bounce_zone, player_court_zone, project_to_world,
)


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
    """
    統計彙整主函式，回傳完整 result dict（不含 metadata）。

    Returns:
        {"summary": {...}, "rallies": [...], "heatmap": {...}}
    """

    def _pn(pos: Tuple[float, float]) -> Tuple[float, float]:
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

    def _get_speed_after(frame_idx: int, look_ahead: int = 12) -> Optional[float]:
        peaks = [smooth_speeds[j] for j in range(
            frame_idx, min(frame_idx + look_ahead, len(smooth_speeds)))
            if smooth_speeds[j] is not None and smooth_speeds[j] >= MIN_BALL_SPEED_KMH]
        return round(max(peaks), 1) if peaks else None

    def _court_side_player(fi: int) -> str:
        bp_ = pos_interp[fi] if fi < len(pos_interp) else None
        if bp_ is not None and last_valid_H is not None:
            w = project_to_world(bp_, last_valid_H)
            if w is not None:
                return "bottom" if w[1] < _NET_Y_M else "top"
        if bp_ is not None:
            return "bottom" if bp_[1] > height * 0.5 else "top"
        tp_ = all_player_top[fi] if fi < len(all_player_top) else None
        bp2 = all_player_bottom[fi] if fi < len(all_player_bottom) else None
        if tp_ and bp2:
            return "top" if tp_[1] < bp2[1] else "bottom"
        return "top" if tp_ else "bottom"

    # ── 累計器 ──────────────────────────────────────────────────────────────
    player_stats: Dict = {
        "top":    {"shots": 0, "serves": 0, "winners": 0,
                   "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
        "bottom": {"shots": 0, "serves": 0, "winners": 0,
                   "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
    }
    all_speeds_kmh: List[float] = []
    serve_speeds_kmh: List[float] = []
    rally_speeds_kmh: List[float] = []
    depth_counts: Dict = {"net": 0, "service": 0, "baseline": 0}
    player_depth: Dict = {
        "top":    {"net": 0, "service": 0, "baseline": 0},
        "bottom": {"net": 0, "service": 0, "baseline": 0},
    }
    heatmap_contacts: List[Dict] = []
    heatmap_bounces: List[Dict] = []
    heatmap_top: List[Dict] = []
    heatmap_bot: List[Dict] = []
    rallies_out: List[Dict] = []

    # ── 逐回合 ──────────────────────────────────────────────────────────────
    for rally_idx, rally_contacts in enumerate(rally_groups):
        if not rally_contacts:
            continue

        next_start = (rally_groups[rally_idx + 1][0]
                      if rally_idx + 1 < len(rally_groups) else None)

        # 回合內 bounce
        r_end_f = (next_start - 1) if next_start else total_frames
        rally_bounces_f = [b for b in bounces_f
                           if rally_contacts[0] <= b <= r_end_f]

        # 發球者
        server = _court_side_player(rally_contacts[0])
        player_stats[server]["serves"] += 1

        # ── 發球前置過濾 ────────────────────────────────────────────────────
        _MAX_SERVE_BUILDUP_S = 5.0
        _contacts_work: List[int] = list(rally_contacts)
        _pre_serve_n = 0
        for _fi in rally_contacts:
            if ((_fi - rally_contacts[0]) / fps <= _MAX_SERVE_BUILDUP_S
                    and _court_side_player(_fi) == server):
                _pre_serve_n += 1
            else:
                break
        if _pre_serve_n >= 2:
            _contacts_work = list(rally_contacts[_pre_serve_n - 1:])
            print(f"[SERVE] rally#{rally_idx+1} merged {_pre_serve_n-1} pre-serve event(s), "
                  f"actual serve f={_contacts_work[0]}({_contacts_work[0]/fps:.2f}s)")

        # ── 逐拍 shots ──────────────────────────────────────────────────────
        shots_out: List[Dict] = []
        for seq, fi in enumerate(_contacts_work):
            ball_pos = pos_interp[fi] if fi < len(pos_interp) else None
            if ball_pos is None:
                continue

            player = _court_side_player(fi)
            is_serve = (seq == 0)
            shot_type = "serve" if is_serve else vlm_shot_types.get(fi, "unknown")
            speed = _get_speed_after(fi)
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
                player_stats[player]["shots"] += 1
            player_stats[player]["shot_types"][shot_type] += 1
            depth_counts[p_zone] += 1
            player_depth[player][p_zone] += 1
            if speed is not None:
                all_speeds_kmh.append(speed)
                (serve_speeds_kmh if is_serve else rally_speeds_kmh).append(speed)

            _hm_entry: Dict = {"player": player}
            if ball_world:
                _hm_entry.update(x=round(ball_world[0], 3),
                                 y=round(ball_world[1], 3), coord="world")
            else:
                _hm_entry.update(x=x_norm, y=y_norm, coord="pixel")
            heatmap_contacts.append(_hm_entry)

            if ppos:
                bucket = heatmap_top if player == "top" else heatmap_bot
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
                heatmap_bounces.append({"x": round(b_world[0], 3),
                                        "y": round(b_world[1], 3), "coord": "world"})
            else:
                heatmap_bounces.append({"x": x_n, "y": y_n, "coord": "pixel"})

        # ── 勝負 + 勝利球落點 ───────────────────────────────────────────────
        winner_player: Optional[str] = vlm_winner_results.get(rally_idx)
        if winner_player in ("top", "bottom"):
            player_stats[winner_player]["winners"] += 1

        cut_near = any(rally_contacts[0] <= sc <= rally_contacts[-1] + int(fps * 3)
                       for sc in scene_cut_frames)
        outcome_type = ("winner" if winner_player in ("top", "bottom") else
                        "scene_cut" if cut_near else "unknown")

        # 勝利球落點：最後一拍後，在對方半場的第一個彈跳（bounce）
        # 落點 = 球碰到場地的彈跳位置；因為對手未能回擊才成為勝利落點
        winner_land: Optional[Dict] = None
        if winner_player in ("top", "bottom") and last_valid_H is not None:
            last_fi = rally_contacts[-1]
            # 1) 優先從 bounces_f 找最後一拍之後的彈跳
            for bf in bounces_f:
                if bf <= last_fi:
                    continue
                if next_start and bf >= next_start:
                    break
                bpos = pos_interp[bf] if bf < len(pos_interp) else None
                if bpos is None:
                    continue
                bw = project_to_world(bpos, last_valid_H)
                if bw is None:
                    continue
                in_opp = (bw[1] > _NET_Y_M) if winner_player == "bottom" else (bw[1] < _NET_Y_M)
                if in_opp:
                    winner_land = {"x": round(bw[0], 2), "y": round(bw[1], 2)}
                    break

            # 2) 無彈跳資料時 fallback：最後一拍後在對方半場的第一個球位置
            if winner_land is None:
                end_fi = min(
                    (next_start - 1) if next_start else total_frames,
                    last_fi + int(fps * 3),
                )
                for _fi in range(last_fi + 3, end_fi):
                    _bp = pos_interp[_fi] if _fi < len(pos_interp) else None
                    if _bp is None:
                        continue
                    _bw = project_to_world(_bp, last_valid_H)
                    if _bw is None:
                        continue
                    in_opp = (_bw[1] > _NET_Y_M) if winner_player == "bottom" else (_bw[1] < _NET_Y_M)
                    if in_opp:
                        winner_land = {"x": round(_bw[0], 2), "y": round(_bw[1], 2)}
                        break

        rallies_out.append({
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
                "type": outcome_type,
                "winner_player": winner_player,
                "winner_land": winner_land,
            },
        })

    # ── summary ──────────────────────────────────────────────────────────────
    total_rallies = len(rallies_out)
    total_shots = sum(r["shot_count"] for r in rallies_out)
    avg_rally_length = round(total_shots / total_rallies, 1) if total_rallies > 0 else 0.0

    return {
        "summary": {
            "total_rallies": total_rallies,
            "total_shots": total_shots,
            "total_winners": sum(1 for r in rallies_out
                                 if r["outcome"]["type"] == "winner"),
            "avg_rally_length": avg_rally_length,
            "players": {
                "top":    player_stats["top"],
                "bottom": player_stats["bottom"],
            },
            "speed": {
                "all":    _speed_stats(all_speeds_kmh),
                "serves": _speed_stats(serve_speeds_kmh),
                "rally":  _speed_stats(rally_speeds_kmh),
            },
            "depth": {
                "total": depth_counts,
                "top":   player_depth["top"],
                "bottom": player_depth["bottom"],
            },
        },
        "rallies": rallies_out,
        "heatmap": {
            "contacts":       heatmap_contacts,
            "bounces":        heatmap_bounces,
            "top_player":     heatmap_top,
            "bottom_player":  heatmap_bot,
        },
    }
