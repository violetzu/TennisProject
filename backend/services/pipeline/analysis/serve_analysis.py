"""
發球與擊球落點分析模組 / Serve and Shot Landing Analysis Module

此模組分析網球比賽中的發球和擊球落點，提供以下功能：
1. 發球落點分析 (Serve Landing Analysis)
2. 擊球深度分類 (Shot Depth Classification)
3. 發球優勢統計 (Serve Advantage Statistics)

使用方法 / Usage:
    from analysis.serve_analysis import analyze_serve_landings, classify_shot_depths
    
    # 分析發球落點
    serve_stats = analyze_serve_landings(world_json_path)
    
    # 分類擊球深度
    depth_stats = classify_shot_depths(world_json_path)

依賴 / Dependencies:
    - world_coordinate_info.json (需要有 bounce 事件)
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

from event.event_utils import load_json, save_json


# =============================================================================
# 球場區域常數 / Court Zone Constants
# =============================================================================

@dataclass
class CourtDimensions:
    """
    標準網球場尺寸 (單位: 公尺)
    Standard tennis court dimensions (in meters)
    
    球場座標系統:
    Y=0 ─────────────────── 頂部底線 (Top Baseline)
    Y=6.4m ───────────────── 發球線 (Service Line)
    Y=11.885m ─── 網子 ───── 網 (Net)
    Y=17.485m ─────────────── 發球線 (Service Line)
    Y=23.77m ─────────────── 底部底線 (Bottom Baseline)
    
    X座標:
    X=0 ──── 左邊 (Left)
    X=5.485m ── 中線 (Center)
    X=10.97m ── 右邊 (Right)
    """
    # 球場總長寬
    length: float = 23.77  # 底線到底線
    width: float = 10.97   # 單打邊線到邊線
    
    # 網子位置
    net_y: float = 11.885  # 球場中心
    
    # 發球區
    service_line_distance: float = 6.4  # 從底線到發球線
    service_box_width: float = 4.115    # 發球區寬度 (半邊)
    
    # 深度分類區域
    front_zone_depth: float = 4.0   # 前場區域 (網前)
    mid_zone_depth: float = 10.0    # 中場區域
    # back_zone 是剩餘部分
    
    def get_zone(self, y: float, from_net: bool = True) -> str:
        """
        根據 Y 座標判斷球場區域
        
        Args:
            y: Y 座標 (公尺)
            from_net: 是否從網子計算距離
            
        Returns:
            區域名稱: "front", "mid", "back"
        """
        # 計算與網的距離
        dist_from_net = abs(y - self.net_y)
        
        if dist_from_net < self.front_zone_depth:
            return "front"  # 前場 (網前截擊區)
        elif dist_from_net < self.mid_zone_depth:
            return "mid"    # 中場
        else:
            return "back"   # 後場 (底線區)
    
    def is_in_service_box(self, x: float, y: float, serving_from_top: bool) -> bool:
        """
        判斷落點是否在發球區內
        
        Args:
            x: X 座標
            y: Y 座標
            serving_from_top: 是否從頂部發球
            
        Returns:
            是否在發球區內
        """
        if serving_from_top:
            # 從頂部發球，落點應在底部發球區 (Y > net_y)
            in_y = self.net_y < y < self.net_y + self.service_line_distance
        else:
            # 從底部發球，落點應在頂部發球區 (Y < net_y)
            in_y = self.net_y - self.service_line_distance < y < self.net_y
        
        # X 座標在發球區內
        in_x = 0 < x < self.width
        
        return in_y and in_x
    
    def get_service_box_side(self, x: float) -> str:
        """
        判斷發球落點在左邊還是右邊發球區
        
        Returns:
            "deuce" (右邊) 或 "ad" (左邊)
        """
        center = self.width / 2
        if x < center:
            return "ad"    # 左邊 (Advantage)
        else:
            return "deuce"  # 右邊 (Deuce)


# 預設球場尺寸
COURT = CourtDimensions()


# =============================================================================
# 資料結構 / Data Structures
# =============================================================================

@dataclass
class BounceEvent:
    """
    彈跳事件資料結構
    Bounce event data structure
    """
    frame_index: int        # 影格索引
    time: float            # 時間 (秒)
    x: float               # X 座標 (公尺)
    y: float               # Y 座標 (公尺)
    speed: Optional[float] = None  # 球速 (m/s)
    zone: str = ""         # 區域: front/mid/back
    is_serve: bool = False # 是否為發球


@dataclass
class ServeAnalysisResult:
    """
    發球分析結果
    Serve analysis result
    """
    total_serves: int = 0           # 發球總數
    serves_in_box: int = 0          # 落在發球區內的發球
    deuce_court_count: int = 0      # Deuce 側發球數
    ad_court_count: int = 0         # Ad 側發球數
    avg_serve_speed: float = 0.0    # 平均發球球速
    serve_landing_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DepthAnalysisResult:
    """
    深度分析結果
    Depth analysis result
    """
    total_shots: int = 0      # 擊球總數
    front_count: int = 0      # 網前落點數
    mid_count: int = 0        # 中場落點數
    back_count: int = 0       # 後場落點數
    avg_depth: float = 0.0    # 平均深度 (距網距離)
    depth_distribution: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 核心函數 / Core Functions
# =============================================================================

def extract_bounce_events(world_json_path: str) -> List[BounceEvent]:
    """
    從 world_coordinate_info.json 提取所有彈跳事件
    Extract all bounce events from world_coordinate_info.json
    
    Args:
        world_json_path: JSON 檔案路徑
        
    Returns:
        彈跳事件列表
    """
    payload = load_json(world_json_path)
    frames = payload.get("frames", [])
    metadata = payload.get("metadata", {})
    fps = float(metadata.get("fps", 30.0))
    
    bounces: List[BounceEvent] = []
    
    for frame in frames:
        frame_idx = frame.get("frame_index", 0)
        time = frame.get("time", frame_idx / fps)
        events = frame.get("events", [])
        
        for evt in events:
            if not isinstance(evt, dict):
                continue
            if evt.get("type") != "bounce":
                continue
            
            # 取得球的世界座標
            ball = frame.get("ball", {})
            world_coords = ball.get("world")
            speed = ball.get("speed")
            
            if world_coords and len(world_coords) >= 2:
                x, y = world_coords[0], world_coords[1]
                zone = COURT.get_zone(y)
                
                bounces.append(BounceEvent(
                    frame_index=frame_idx,
                    time=time,
                    x=x,
                    y=y,
                    speed=speed,
                    zone=zone,
                ))
    
    return bounces


def analyze_serve_landings(
    world_json_path: str,
    first_n_bounces_as_serve: int = 2,
) -> ServeAnalysisResult:
    """
    分析發球落點
    Analyze serve landing points
    
    策略: 將比賽開始的前幾個彈跳視為發球
    Strategy: Treat first N bounces as serves
    
    Args:
        world_json_path: JSON 檔案路徑
        first_n_bounces_as_serve: 前幾個彈跳視為發球 (預設 2)
        
    Returns:
        發球分析結果
    """
    bounces = extract_bounce_events(world_json_path)
    
    if not bounces:
        return ServeAnalysisResult()
    
    # 取前 N 個 bounce 作為發球
    serves = bounces[:first_n_bounces_as_serve]
    
    result = ServeAnalysisResult()
    result.total_serves = len(serves)
    
    speeds = []
    
    for serve in serves:
        serve.is_serve = True
        result.serve_landing_points.append((serve.x, serve.y))
        
        # 判斷是否在發球區
        # 假設從頂部發球 (可根據實際情況調整)
        in_box = COURT.is_in_service_box(serve.x, serve.y, serving_from_top=True)
        if not in_box:
            # 試試從底部發球
            in_box = COURT.is_in_service_box(serve.x, serve.y, serving_from_top=False)
        
        if in_box:
            result.serves_in_box += 1
        
        # 發球區側邊
        side = COURT.get_service_box_side(serve.x)
        if side == "deuce":
            result.deuce_court_count += 1
        else:
            result.ad_court_count += 1
        
        if serve.speed is not None:
            speeds.append(serve.speed)
    
    if speeds:
        result.avg_serve_speed = sum(speeds) / len(speeds)
    
    return result


def classify_shot_depths(world_json_path: str) -> DepthAnalysisResult:
    """
    分類所有擊球的深度
    Classify depth of all shots
    
    Args:
        world_json_path: JSON 檔案路徑
        
    Returns:
        深度分析結果
    """
    bounces = extract_bounce_events(world_json_path)
    
    if not bounces:
        return DepthAnalysisResult()
    
    result = DepthAnalysisResult()
    result.total_shots = len(bounces)
    
    depths = []  # 距離網的距離
    
    for bounce in bounces:
        depth = abs(bounce.y - COURT.net_y)
        depths.append(depth)
        
        if bounce.zone == "front":
            result.front_count += 1
        elif bounce.zone == "mid":
            result.mid_count += 1
        else:
            result.back_count += 1
    
    if depths:
        result.avg_depth = sum(depths) / len(depths)
    
    # 計算百分比分布
    if result.total_shots > 0:
        result.depth_distribution = {
            "front": round(result.front_count / result.total_shots * 100, 1),
            "mid": round(result.mid_count / result.total_shots * 100, 1),
            "back": round(result.back_count / result.total_shots * 100, 1),
        }
    
    return result


def get_shot_speeds_summary(world_json_path: str) -> Dict[str, Any]:
    """
    取得球速統計摘要
    Get ball speed statistics summary
    
    Args:
        world_json_path: JSON 檔案路徑
        
    Returns:
        球速統計字典
    """
    payload = load_json(world_json_path)
    frames = payload.get("frames", [])
    
    contact_speeds = []
    bounce_speeds = []
    all_speeds = []
    
    for frame in frames:
        ball = frame.get("ball", {})
        speed = ball.get("speed")
        events = frame.get("events", [])
        
        if speed is not None and speed > 0:
            all_speeds.append(speed)
            
            for evt in events:
                if isinstance(evt, dict):
                    evt_type = evt.get("type")
                    if evt_type == "racket_contact":
                        contact_speeds.append(speed)
                    elif evt_type == "bounce":
                        bounce_speeds.append(speed)
    
    def calc_stats(speeds: List[float]) -> Dict[str, float]:
        if not speeds:
            return {"count": 0, "avg": 0, "max": 0, "min": 0}
        return {
            "count": len(speeds),
            "avg": round(sum(speeds) / len(speeds), 2),
            "max": round(max(speeds), 2),
            "min": round(min(speeds), 2),
        }
    
    return {
        "contact_speeds": calc_stats(contact_speeds),
        "bounce_speeds": calc_stats(bounce_speeds),
        "all_speeds": calc_stats(all_speeds),
    }


def generate_analysis_report(world_json_path: str) -> Dict[str, Any]:
    """
    產生完整分析報告
    Generate complete analysis report
    
    Args:
        world_json_path: JSON 檔案路徑
        
    Returns:
        分析報告字典，可餵給 VLM 作為上下文
    """
    serve_result = analyze_serve_landings(world_json_path)
    depth_result = classify_shot_depths(world_json_path)
    speed_stats = get_shot_speeds_summary(world_json_path)
    
    report = {
        "serve_analysis": {
            "total_serves": serve_result.total_serves,
            "serves_in_box": serve_result.serves_in_box,
            "serve_in_rate": (
                round(serve_result.serves_in_box / serve_result.total_serves * 100, 1)
                if serve_result.total_serves > 0 else 0
            ),
            "deuce_count": serve_result.deuce_court_count,
            "ad_count": serve_result.ad_court_count,
            "avg_serve_speed_ms": round(serve_result.avg_serve_speed, 1),
            "avg_serve_speed_kmh": round(serve_result.avg_serve_speed * 3.6, 1),
        },
        "depth_analysis": {
            "total_bounces": depth_result.total_shots,
            "front_count": depth_result.front_count,
            "mid_count": depth_result.mid_count,
            "back_count": depth_result.back_count,
            "depth_distribution_percent": depth_result.depth_distribution,
            "avg_depth_from_net_m": round(depth_result.avg_depth, 2),
        },
        "speed_analysis": speed_stats,
    }
    
    return report


# =============================================================================
# 命令列介面 / CLI
# =============================================================================

def main():
    """命令列測試入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tennis serve and shot analysis")
    parser.add_argument("--json", type=str, default=None, help="Path to world_coordinate_info.json")
    args = parser.parse_args()
    
    json_path = args.json or str(_BACKEND_ROOT / "data" / "world_coordinate_info.json")
    # json_path = args.json or str(_BACKEND_ROOT / "data" / "video_info_video2.json")
    
    print("=" * 60)
    print("Tennis Match Analysis Report")
    print("=" * 60)
    
    report = generate_analysis_report(json_path)
    
    print("\n[發球分析 / Serve Analysis]")
    serve = report["serve_analysis"]
    print(f"  發球總數: {serve['total_serves']}")
    print(f"  落在發球區: {serve['serves_in_box']} ({serve['serve_in_rate']}%)")
    print(f"  Deuce側: {serve['deuce_count']}, Ad側: {serve['ad_count']}")
    print(f"  平均發球速度: {serve['avg_serve_speed_kmh']} km/h")
    
    print("\n[深度分析 / Depth Analysis]")
    depth = report["depth_analysis"]
    print(f"  總落點數: {depth['total_bounces']}")
    print(f"  前場: {depth['front_count']} ({depth['depth_distribution_percent'].get('front', 0)}%)")
    print(f"  中場: {depth['mid_count']} ({depth['depth_distribution_percent'].get('mid', 0)}%)")
    print(f"  後場: {depth['back_count']} ({depth['depth_distribution_percent'].get('back', 0)}%)")
    print(f"  平均深度 (距網): {depth['avg_depth_from_net_m']} m")
    
    print("\n[球速分析 / Speed Analysis]")
    speeds = report["speed_analysis"]
    print(f"  擊球時球速: avg={speeds['contact_speeds']['avg']} m/s, max={speeds['contact_speeds']['max']} m/s")
    print(f"  彈跳時球速: avg={speeds['bounce_speeds']['avg']} m/s, max={speeds['bounce_speeds']['max']} m/s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
