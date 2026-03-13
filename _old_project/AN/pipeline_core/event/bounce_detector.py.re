"""Improved bounce detector.

Key improvements:
1. Fixed Y-axis reversal logic - bounces SHOULD have Y reversal
2. Bounces: Y-axis reversal + minimal X-axis change
3. Uses speed dip pattern (drop then recovery)
4. Better segment-based detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .event_utils import (
    DetectedEvent,
    extract_speeds,
    extract_times,
    extract_world_coords,
    compute_velocities,
    inject_events,
    load_json,
    save_json,
)


@dataclass
class BounceDetectionConfig:
    # Speed dip detection (relaxed)
    min_drop: float = 0.5  # m/s speed drop before bounce (relaxed from 1.0)
    min_recovery: float = 0.4  # m/s speed recovery after bounce (relaxed from 0.8)
    max_speed_at_bounce: float = 50.0  # m/s - bounce shouldn't be too fast (relaxed from 40.0)
    
    # X-axis velocity threshold - bounce has minimal X change
    max_x_velocity_change: float = 3.0  # m/s - bounce should NOT change X much (relaxed from 2.0)
    
    # Timing
    cooldown_frames: int = 4  # (relaxed from 5)
    max_bounces_per_segment: int = 3  # (increased from 2)
    min_distance_from_edges: int = 2  # (relaxed from 3)


def _segment_indices(
    total_frames: int,
    contacts: Sequence[DetectedEvent],
) -> List[Tuple[int, int]]:
    """Split frames into segments between contacts."""
    if not contacts:
        return [(0, total_frames - 1)] if total_frames else []
    
    idxs = sorted(evt.frame_index for evt in contacts)
    segments: List[Tuple[int, int]] = []
    start = 0
    
    for contact_idx in idxs:
        if contact_idx > start:
            segments.append((start, contact_idx))
        start = contact_idx
    
    if start < total_frames - 1:
        segments.append((start, total_frames - 1))
    
    return segments


def _detect_bounces_in_segment(
    speeds: Sequence[Optional[float]],
    velocities: Sequence[Optional[Tuple[float, float]]],
    times: Sequence[float],
    seg_start: int,
    seg_end: int,
    cfg: BounceDetectionConfig,
) -> List[DetectedEvent]:
    """Detect bounces within a segment between contacts."""
    events: List[DetectedEvent] = []
    last_idx = -10**9
    
    for idx in range(max(seg_start + 1, 1), min(seg_end, len(speeds) - 2)):
        # Distance from edges check
        if idx - seg_start < cfg.min_distance_from_edges:
            continue
        if seg_end - idx < cfg.min_distance_from_edges:
            continue
        
        # Cooldown check
        if idx - last_idx <= cfg.cooldown_frames:
            continue
        
        # Speed dip pattern: drop then recovery
        s_prev, s_curr, s_next = speeds[idx - 1], speeds[idx], speeds[idx + 1]
        if s_prev is None or s_curr is None or s_next is None:
            continue
        
        drop = s_prev - s_curr
        recovery = s_next - s_curr
        
        if drop < cfg.min_drop or recovery < cfg.min_recovery:
            continue
        
        if s_curr > cfg.max_speed_at_bounce:
            continue
        
        # ===== Key: Check X-axis velocity change =====
        # Bounce should have MINIMAL X-axis change (ball just bounces, no direction change)
        # Contact should have SIGNIFICANT X-axis change (ball goes to other side)
        
        vel_prev = velocities[idx - 1] if idx - 1 < len(velocities) else None
        vel_curr = velocities[idx] if idx < len(velocities) else None
        vel_next = velocities[idx + 1] if idx + 1 < len(velocities) else None
        
        x_change = 0.0
        if vel_prev is not None and vel_next is not None:
            vx_prev = vel_prev[0]
            vx_next = vel_next[0]
            x_change = abs(vx_next - vx_prev)
        
        # If X change is too large, this is likely a contact, not bounce
        if x_change > cfg.max_x_velocity_change:
            continue
        
        # ===== Y-axis behavior for bounce =====
        # For a bounce: ball comes down (vy negative) then goes up (vy positive)
        # So Y velocity should flip from negative to positive (in court coords)
        # Note: We WANT Y reversal for bounce, opposite of old logic
        
        has_y_reversal = False
        if vel_prev is not None and vel_next is not None:
            vy_prev = vel_prev[1]
            vy_next = vel_next[1]
            # Y reversal means the product is negative
            has_y_reversal = (vy_prev * vy_next) < 0
        
        # Calculate confidence
        # Higher drop/recovery = higher confidence
        base_conf = (drop + recovery) / (cfg.min_drop + cfg.min_recovery)
        
        # Bonus for Y reversal (strong bounce signal)
        if has_y_reversal:
            base_conf *= 1.2
        
        # Bonus for low X change (clearly not a contact)
        if x_change < 0.5:
            base_conf *= 1.1
        
        confidence = min(1.0, base_conf)
        
        events.append(
            DetectedEvent(
                frame_index=idx,
                confidence=confidence,
                timestamp=times[idx],
                details={
                    "speed": round(s_curr, 3),
                    "drop": round(drop, 3),
                    "recovery": round(recovery, 3),
                    "x_change": round(x_change, 3),
                    "y_reversal": has_y_reversal,
                },
            )
        )
        last_idx = idx
        
        if len(events) >= cfg.max_bounces_per_segment:
            break
    
    return events


def detect_bounces(
    world_json_path: str,
    video_json_path: Optional[str] = None,
    contact_events: Optional[Iterable[DetectedEvent]] = None,
    config: Optional[BounceDetectionConfig] = None,
) -> List[DetectedEvent]:
    """Detect ball bounces in video analysis data."""
    cfg = config or BounceDetectionConfig()
    world_payload = load_json(world_json_path)
    frames = world_payload.get("frames", [])
    metadata = world_payload.get("metadata", {})
    fps = float(metadata.get("fps", 0.0))
    if not isinstance(frames, list) or fps <= 0:
        return []

    times = extract_times(frames, fps)
    coords = extract_world_coords(frames)
    speeds = extract_speeds(frames)
    velocities = compute_velocities(coords, times)
    
    contacts = list(contact_events or [])
    segments = _segment_indices(len(frames), contacts)
    
    events: List[DetectedEvent] = []
    for seg_start, seg_end in segments:
        events.extend(
            _detect_bounces_in_segment(
                speeds,
                velocities,
                times,
                seg_start,
                seg_end,
                cfg,
            )
        )

    inject_events(frames, "bounce", events)
    save_json(world_json_path, world_payload)

    if video_json_path is not None:
        video_payload = load_json(video_json_path)
        video_frames = video_payload.get("frames", [])
        if isinstance(video_frames, list) and len(video_frames) == len(frames):
            inject_events(video_frames, "bounce", events)
            save_json(video_json_path, video_payload)

    return events


__all__ = ["detect_bounces", "BounceDetectionConfig"]
