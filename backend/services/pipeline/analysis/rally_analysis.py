"""
回合分析模組 / Rally Analysis Module

此模組分析網球比賽中的回合結構，提供以下功能：
1. 發球偵測 (Serve Detection) - 每個回合的第一擊
2. 勝利球偵測 (Winner Detection) - 回合結束後的落點
3. 回合統計 (Rally Statistics)

使用方法 / Usage:
    from analysis.rally_analysis import analyze_rallies
    
    rallies = analyze_rallies(world_json_path)

依賴 / Dependencies:
    - world_coordinate_info.json (需要有 racket_contact 事件)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 確保可匯入 backend 模組
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from event.event_utils import load_json


# =============================================================================
# 資料結構 / Data Structures
# =============================================================================

@dataclass
class ContactEvent:
    """
    擊球事件資料結構
    """
    frame_index: int        # 影格索引
    time: float            # 時間 (秒)
    x: float               # 球的 X 座標 (公尺)
    y: float               # 球的 Y 座標 (公尺)
    speed: Optional[float] = None  # 擊球時球速 (m/s)
    post_contact_speed: Optional[float] = None  # 擊球後最大球速


@dataclass
class BouncePoint:
    """
    彈地點資料結構
    """
    frame_index: int
    time: float
    x: float
    y: float
    speed: Optional[float] = None


@dataclass
class Rally:
    """
    單一回合資料結構
    """
    rally_id: int                          # 回合編號 (從 1 開始)
    serve: Optional[ContactEvent] = None   # 發球 (第一擊)
    shots: List[ContactEvent] = field(default_factory=list)  # 回合中的擊球
    winner_bounce: Optional[BouncePoint] = None  # 勝利球落點
    start_time: float = 0.0                # 回合開始時間
    end_time: float = 0.0                  # 回合結束時間
    shot_count: int = 0                    # 總擊球數（含發球）


@dataclass
class RallyAnalysisResult:
    """
    回合分析總結果
    """
    total_rallies: int = 0
    total_shots: int = 0
    total_serves: int = 0
    total_winners: int = 0
    avg_rally_length: float = 0.0
    rallies: List[Rally] = field(default_factory=list)


# =============================================================================
# 核心函數 / Core Functions
# =============================================================================

def extract_contacts(frames: List[Dict], fps: float = 30.0) -> List[ContactEvent]:
    """
    從 frames 提取所有 racket_contact 事件
    
    Args:
        frames: JSON frames 列表
        fps: 影片 FPS
        
    Returns:
        ContactEvent 列表
    """
    contacts: List[ContactEvent] = []
    
    for i, frame in enumerate(frames):
        events = frame.get("events", [])
        
        for evt in events:
            if not isinstance(evt, dict):
                continue
            if evt.get("type") != "racket_contact":
                continue
            
            # 取得球的世界座標
            ball = frame.get("ball", {})
            world_coords = ball.get("world")
            speed = ball.get("speed")
            
            if world_coords and len(world_coords) >= 2:
                x, y = world_coords[0], world_coords[1]
                time = frame.get("time", i / fps)
                
                # 計算擊球後 5 幀內的最大球速
                post_speed = None
                for j in range(i, min(i + 5, len(frames))):
                    s = frames[j].get("ball", {}).get("speed")
                    if s is not None and s > 0:
                        if post_speed is None or s > post_speed:
                            post_speed = s
                
                contacts.append(ContactEvent(
                    frame_index=i,
                    time=time,
                    x=x,
                    y=y,
                    speed=speed,
                    post_contact_speed=post_speed,
                ))
    
    return contacts


def find_bounce_after_contact(
    frames: List[Dict],
    start_frame: int,
    search_window: int = 90,
    fps: float = 30.0,
    is_last_rally: bool = False,
) -> Optional[BouncePoint]:
    """
    在指定幀後尋找彈地點
    
    策略：
    1. 優先使用 bounce 事件
    2. 使用球速局部最小值偵測（彈地時速度會減少然後回升）
    3. 使用 Y 座標方向變化偵測
    
    Args:
        frames: JSON frames 列表
        start_frame: 開始搜索的幀
        search_window: 搜索範圍（幀數）
        fps: 影片 FPS
        is_last_rally: 是否為最後一個回合（擴大搜索範圍）
        
    Returns:
        BouncePoint 或 None
    """
    # 如果是最後一個回合，搜索到影片結尾
    end_frame = len(frames) if is_last_rally else min(start_frame + search_window, len(frames))
    
    # 方法 1: 找 bounce 事件
    for i in range(start_frame, end_frame):
        events = frames[i].get("events", [])
        for evt in events:
            if isinstance(evt, dict) and evt.get("type") == "bounce":
                ball = frames[i].get("ball", {})
                world = ball.get("world")
                if world and len(world) >= 2:
                    return BouncePoint(
                        frame_index=i,
                        time=frames[i].get("time", i / fps),
                        x=world[0],
                        y=world[1],
                        speed=ball.get("speed"),
                    )
    
    # 方法 2: 使用速度局部最小值偵測
    # 彈地時球速會降到局部最小值，然後回升
    speeds = []
    for i in range(start_frame, end_frame):
        ball = frames[i].get("ball", {})
        speed = ball.get("speed") or 0
        speeds.append((i, speed))
    
    if len(speeds) >= 5:
        # 找局部最小值（速度下降後上升）
        for j in range(2, len(speeds) - 2):
            prev_speed = speeds[j-1][1]
            curr_speed = speeds[j][1]
            next_speed = speeds[j+1][1]
            next2_speed = speeds[j+2][1]
            
            # 條件：當前速度是局部最小值，且之後速度有上升趨勢
            if curr_speed < prev_speed and curr_speed < next_speed:
                # 確認不是噪音：後續速度持續上升
                if next_speed < next2_speed or next_speed > curr_speed * 1.1:
                    frame_idx = speeds[j][0]
                    ball = frames[frame_idx].get("ball", {})
                    world = ball.get("world")
                    if world and len(world) >= 2:
                        return BouncePoint(
                            frame_index=frame_idx,
                            time=frames[frame_idx].get("time", frame_idx / fps),
                            x=world[0],
                            y=world[1],
                            speed=curr_speed,
                        )
    
    # 方法 3: 用 Y 座標變化偵測 (備用)
    for i in range(start_frame + 1, end_frame):
        ball_prev = frames[i - 1].get("ball", {})
        ball_curr = frames[i].get("ball", {})
        
        world_prev = ball_prev.get("world")
        world_curr = ball_curr.get("world")
        
        if world_prev and world_curr and len(world_prev) >= 2 and len(world_curr) >= 2:
            y_prev = world_prev[1]
            y_curr = world_curr[1]
            
            if i > start_frame + 1:
                y_prev2 = frames[i - 2].get("ball", {}).get("world", [0, 0])[1]
                dy_prev = y_prev - y_prev2
                dy_curr = y_curr - y_prev
                
                # 方向改變：向一個方向移動後反向
                if abs(dy_prev) > 0.2 and abs(dy_curr) > 0.2:
                    if (dy_prev > 0 and dy_curr < 0) or (dy_prev < 0 and dy_curr > 0):
                        return BouncePoint(
                            frame_index=i,
                            time=frames[i].get("time", i / fps),
                            x=world_curr[0],
                            y=world_curr[1],
                            speed=ball_curr.get("speed"),
                        )
    
    return None


def analyze_rallies(
    world_json_path: str,
    gap_threshold_sec: float = 2.5,
    serve_speed_threshold: float = 0.0,  # 0 表示不過濾
) -> RallyAnalysisResult:
    """
    分析回合結構
    
    邏輯：
    1. 影片第一個擊球 = 發球
    2. 如果兩次擊球間隔超過 gap_threshold_sec 秒 → 回合結束
    3. 回合結束後的最後一個彈地 = 勝利球落點
    4. 下一個擊球 = 新發球
    
    Args:
        world_json_path: JSON 檔案路徑
        gap_threshold_sec: 回合結束的間隔閾值（秒）
        serve_speed_threshold: 發球速度閾值（m/s），低於此則不算發球
        
    Returns:
        RallyAnalysisResult
    """
    payload = load_json(world_json_path)
    frames = payload.get("frames", [])
    metadata = payload.get("metadata", {})
    fps = float(metadata.get("fps", 30.0))
    
    # 計算幀數閾值
    gap_frames = int(gap_threshold_sec * fps)
    
    # 提取所有擊球事件
    contacts = extract_contacts(frames, fps)
    
    if not contacts:
        return RallyAnalysisResult()
    
    # 分群回合
    rallies: List[Rally] = []
    current_rally_contacts: List[ContactEvent] = []
    rally_id = 1
    
    for i, contact in enumerate(contacts):
        if current_rally_contacts:
            # 檢查是否需要結束當前回合
            last_contact = current_rally_contacts[-1]
            frame_gap = contact.frame_index - last_contact.frame_index
            
            if frame_gap > gap_frames:
                # 結束當前回合（不是最後一個回合）
                rally = finalize_rally(
                    rally_id=rally_id,
                    contacts=current_rally_contacts,
                    frames=frames,
                    fps=fps,
                    is_last_rally=False,
                )
                rallies.append(rally)
                rally_id += 1
                current_rally_contacts = []
        
        current_rally_contacts.append(contact)
    
    # 處理最後一個回合（搜索到影片結尾）
    if current_rally_contacts:
        rally = finalize_rally(
            rally_id=rally_id,
            contacts=current_rally_contacts,
            frames=frames,
            fps=fps,
            is_last_rally=True,
        )
        rallies.append(rally)
    
    # 計算統計
    result = RallyAnalysisResult(
        total_rallies=len(rallies),
        rallies=rallies,
    )
    
    for rally in rallies:
        result.total_shots += rally.shot_count
        if rally.serve:
            result.total_serves += 1
        if rally.winner_bounce:
            result.total_winners += 1
    
    if rallies:
        result.avg_rally_length = result.total_shots / len(rallies)
    
    return result


def finalize_rally(
    rally_id: int,
    contacts: List[ContactEvent],
    frames: List[Dict],
    fps: float,
    is_last_rally: bool = False,
) -> Rally:
    """
    完成回合資料結構
    
    Args:
        rally_id: 回合編號
        contacts: 該回合的擊球事件列表
        frames: 所有 frames
        fps: FPS
        is_last_rally: 是否為最後一個回合
        
    Returns:
        Rally
    """
    if not contacts:
        return Rally(rally_id=rally_id)
    
    serve = contacts[0]  # 第一擊 = 發球
    shots = contacts[1:] if len(contacts) > 1 else []
    
    # 找最後一擊後的彈地作為勝利球
    last_contact = contacts[-1]
    winner_bounce = find_bounce_after_contact(
        frames=frames,
        start_frame=last_contact.frame_index,
        search_window=90,  # 3 秒 @30fps
        fps=fps,
        is_last_rally=is_last_rally,
    )
    
    return Rally(
        rally_id=rally_id,
        serve=serve,
        shots=shots,
        winner_bounce=winner_bounce,
        start_time=serve.time,
        end_time=last_contact.time,
        shot_count=len(contacts),
    )


# =============================================================================
# 輸出函數 / Output Functions
# =============================================================================

def generate_rally_report(world_json_path: str) -> Dict[str, Any]:
    """
    產生回合分析報告
    
    Args:
        world_json_path: JSON 檔案路徑
        
    Returns:
        分析報告字典
    """
    result = analyze_rallies(world_json_path)
    
    rallies_data = []
    for rally in result.rallies:
        rally_info = {
            "id": rally.rally_id,
            "shot_count": rally.shot_count,
            "start_time": round(rally.start_time, 2),
            "end_time": round(rally.end_time, 2),
            "duration": round(rally.end_time - rally.start_time, 2),
        }
        
        if rally.serve:
            rally_info["serve"] = {
                "frame": rally.serve.frame_index,
                "time": round(rally.serve.time, 2),
                "position": [round(rally.serve.x, 2), round(rally.serve.y, 2)],
                "speed": round(rally.serve.post_contact_speed, 1) if rally.serve.post_contact_speed else None,
            }
        
        if rally.winner_bounce:
            rally_info["winner"] = {
                "frame": rally.winner_bounce.frame_index,
                "time": round(rally.winner_bounce.time, 2),
                "position": [round(rally.winner_bounce.x, 2), round(rally.winner_bounce.y, 2)],
            }
        
        rallies_data.append(rally_info)
    
    return {
        "summary": {
            "total_rallies": result.total_rallies,
            "total_shots": result.total_shots,
            "total_serves": result.total_serves,
            "total_winners": result.total_winners,
            "avg_rally_length": round(result.avg_rally_length, 1),
        },
        "rallies": rallies_data,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """命令列測試入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tennis rally analysis")
    parser.add_argument("--json", type=str, default=None, help="Path to world_coordinate_info.json")
    parser.add_argument("--gap", type=float, default=2.5, help="Gap threshold in seconds")
    args = parser.parse_args()
    
    # json_path = args.json or str(_BACKEND_ROOT / "data" / "world_coordinate_info.json")
    json_path = args.json or str(_BACKEND_ROOT / "data" / "world_info_input_video3.json")
    
    print("=" * 60)
    print("Tennis Rally Analysis Report")
    print("=" * 60)
    
    report = generate_rally_report(json_path)
    
    print(f"\n總回合數: {report['summary']['total_rallies']}")
    print(f"總擊球數: {report['summary']['total_shots']}")
    print(f"發球數: {report['summary']['total_serves']}")
    print(f"勝利球數: {report['summary']['total_winners']}")
    print(f"平均回合長度: {report['summary']['avg_rally_length']} 擊")
    
    print("\n--- 回合詳情 ---")
    for rally in report["rallies"]:
        print(f"\n回合 {rally['id']}:")
        print(f"  擊球數: {rally['shot_count']}")
        print(f"  時間: {rally['start_time']}s - {rally['end_time']}s (持續 {rally['duration']}s)")
        
        if "serve" in rally:
            s = rally["serve"]
            print(f"  發球: Frame {s['frame']}, 位置 {s['position']}, 速度 {s['speed']} m/s")
        
        if "winner" in rally:
            w = rally["winner"]
            print(f"  勝利球: Frame {w['frame']}, 落點 {w['position']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
