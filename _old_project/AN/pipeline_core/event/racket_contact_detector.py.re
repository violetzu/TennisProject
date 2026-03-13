"""Improved heuristic racket-contact detector.

Key improvements:
1. Uses X-axis velocity change to distinguish contact from bounce
2. Contact: Y-axis reversal + significant X-axis change (ball direction changes)
3. Bounce: Y-axis reversal + minimal X-axis change (ball just bounces)
4. Uses player proximity as additional signal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .event_utils import (
    DetectedEvent,
    extract_axis_velocity,
    extract_speeds,
    extract_times,
    extract_world_coords,
    compute_velocities,
    inject_events,
    load_json,
    save_json,
)


@dataclass
class ContactDetectionConfig:
    # Y-axis (along court) velocity thresholds
    min_direction_flip: float = 4.0  # m/s change along court axis (relaxed from 5.0)
    min_velocity_magnitude: float = 4.0  # |vy| must exceed before/after (relaxed from 6.0)
    
    # X-axis (across court) velocity thresholds
    min_x_velocity_change: float = 0.8  # m/s - contact should change X direction (relaxed from 1.5)
    
    # Speed thresholds
    min_speed_jump: float = 3.0  # m/s magnitude delta (relaxed from 4.0)
    min_accel_peak: float = 20.0  # m/s^2 change (relaxed from 25.0)
    
    # Timing
    cooldown_frames: int = 10  # (relaxed from 12)
    
    # Player proximity check (world coordinates)
    player_proximity_threshold: float = 4.0  # meters (relaxed from 3.0)


def _get_player_positions(frame: dict) -> List[Tuple[float, float]]:
    """Extract player world positions from frame."""
    positions = []
    players = frame.get("players", [])
    for player in players:
        if isinstance(player, dict):
            world = player.get("world")
            if world and len(world) >= 2:
                positions.append((float(world[0]), float(world[1])))
    return positions


def _ball_near_player(
    ball_pos: Optional[Tuple[float, float]],
    player_positions: List[Tuple[float, float]],
    threshold: float,
) -> bool:
    """Check if ball is near any player."""
    if ball_pos is None or not player_positions:
        return True  # Default to True if we can't check
    
    bx, by = ball_pos
    for px, py in player_positions:
        dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
        if dist < threshold:
            return True
    return False


def _detect_contacts(
    frames: List[dict],
    speeds: List[Optional[float]],
    velocities: List[Optional[Tuple[float, float]]],  # (vx, vy) pairs
    times: List[float],
    cfg: ContactDetectionConfig,
) -> List[DetectedEvent]:
    """Detect racket contacts using improved logic."""
    events: List[DetectedEvent] = []
    last_idx = -10**9
    
    for idx in range(2, len(speeds)):
        # Get velocity data
        vel_prev = velocities[idx - 1] if idx - 1 < len(velocities) else None
        vel_curr = velocities[idx] if idx < len(velocities) else None
        
        if vel_prev is None or vel_curr is None:
            continue
        
        vx_prev, vy_prev = vel_prev
        vx_curr, vy_curr = vel_curr
        
        # Cooldown check
        if idx - last_idx <= cfg.cooldown_frames:
            continue
        
        # Check if velocities are significant enough
        if abs(vy_prev) < cfg.min_velocity_magnitude or abs(vy_curr) < cfg.min_velocity_magnitude:
            continue
        
        dt = times[idx] - times[idx - 1]
        if dt <= 0:
            continue
        
        # ===== Key criteria for CONTACT (not bounce) =====
        
        # 1. Y-axis direction change (ball changes vertical direction)
        direction_delta_y = vy_curr - vy_prev
        switched_y_direction = (vy_prev * vy_curr) < 0
        
        # 2. X-axis velocity change (ball changes horizontal direction)
        direction_delta_x = abs(vx_curr - vx_prev)
        significant_x_change = direction_delta_x > cfg.min_x_velocity_change
        
        # For contact: we need Y direction change AND significant X change
        # (A bounce would have Y change but minimal X change)
        
        if not switched_y_direction and abs(direction_delta_y) < cfg.min_direction_flip:
            continue
        
        # Speed analysis
        speed_prev = speeds[idx - 1]
        speed_curr = speeds[idx]
        speed_jump = 0.0
        if speed_prev is not None and speed_curr is not None:
            speed_jump = abs(speed_curr - speed_prev)
        
        # Acceleration peak
        accel_peak = abs(direction_delta_y) / dt
        
        # ===== Scoring system =====
        # Higher score = more likely to be a contact
        score = 0.0
        
        # Y-direction switch is strong signal
        if switched_y_direction:
            score += 0.3
        
        # X-axis change is crucial for distinguishing from bounce
        if significant_x_change:
            score += 0.4
        else:
            # If no significant X change, this is likely a bounce, not contact
            # Reduce score or skip
            if switched_y_direction and direction_delta_x < 0.5:
                # Very likely a bounce - skip
                continue
        
        # High acceleration peak
        if accel_peak > cfg.min_accel_peak:
            score += 0.2
        
        # Speed jump
        if speed_jump > cfg.min_speed_jump:
            score += 0.1
        
        # Check player proximity
        frame = frames[idx]
        ball = frame.get("ball", {})
        ball_world = ball.get("world")
        ball_pos = None
        if ball_world and len(ball_world) >= 2:
            ball_pos = (float(ball_world[0]), float(ball_world[1]))
        
        player_positions = _get_player_positions(frame)
        near_player = _ball_near_player(ball_pos, player_positions, cfg.player_proximity_threshold)
        
        if near_player:
            score += 0.1
        
        # Minimum threshold to count as contact
        if score < 0.5:
            continue
        
        confidence = min(1.0, score)
        
        events.append(
            DetectedEvent(
                frame_index=idx,
                confidence=confidence,
                timestamp=times[idx],
                details={
                    "vy_prev": round(vy_prev, 3),
                    "vy_curr": round(vy_curr, 3),
                    "vx_change": round(direction_delta_x, 3),
                    "speed_jump": round(speed_jump, 3),
                    "near_player": near_player,
                },
            )
        )
        last_idx = idx
    
    return events


def detect_racket_contacts(
    world_json_path: str,
    video_json_path: Optional[str] = None,
    config: Optional[ContactDetectionConfig] = None,
) -> List[DetectedEvent]:
    """Detect racket contacts in video analysis data."""
    cfg = config or ContactDetectionConfig()
    world_payload = load_json(world_json_path)
    frames = world_payload.get("frames", [])
    metadata = world_payload.get("metadata", {})
    fps = float(metadata.get("fps", 0.0))
    if not isinstance(frames, list) or fps <= 0:
        return []

    times = extract_times(frames, fps)
    coords = extract_world_coords(frames)
    speeds = extract_speeds(frames)
    
    # Compute full 2D velocity (vx, vy)
    velocities = compute_velocities(coords, times)
    
    events = _detect_contacts(frames, speeds, velocities, times, cfg)

    inject_events(frames, "racket_contact", events)
    save_json(world_json_path, world_payload)

    if video_json_path is not None:
        video_payload = load_json(video_json_path)
        video_frames = video_payload.get("frames", [])
        if isinstance(video_frames, list) and len(video_frames) == len(frames):
            inject_events(video_frames, "racket_contact", events)
            save_json(video_json_path, video_payload)

    return events


__all__ = ["detect_racket_contacts", "ContactDetectionConfig"]
