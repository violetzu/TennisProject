"""Alternating event detector based on tennis rally rules.

In a baseline rally:
- Player A hits → ball flies → bounce → ball flies → Player B hits
- So events should alternate: CONTACT - BOUNCE - CONTACT - BOUNCE - CONTACT

This detector:
1. First finds ALL significant velocity changes
2. Then classifies them as CONTACT or BOUNCE based on player position
3. Uses alternating pattern to resolve ambiguous cases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .event_utils import (
    DetectedEvent,
    extract_times,
    extract_world_coords,
    extract_speeds,
    compute_velocities,
    inject_events,
    load_json,
    save_json,
)


@dataclass
class AlternatingDetectorConfig:
    # Velocity change thresholds (increased to reduce false positives)
    min_velocity_change: float = 5.0  # m/s - any significant change (raised from 3.0)
    min_velocity_magnitude: float = 4.0  # m/s - ball must be moving (raised from 3.0)
    
    # Timing
    cooldown_frames: int = 12  # Increased from 8 to prevent consecutive same-type events
    
    # Player proximity for contact classification
    player_proximity_threshold: float = 4.0  # meters (reduced from 5.0 for stricter contact)
    
    # Court boundaries (Y coordinates)
    court_length: float = 23.77  # meters
    
    # Video end margin - don't detect events in last N seconds
    end_margin_seconds: float = 0.5


@dataclass
class RawEvent:
    """A raw velocity change event, not yet classified."""
    frame_index: int
    time: float
    vx_before: float
    vy_before: float
    vx_after: float
    vy_after: float
    speed: float
    ball_y: Optional[float]  # Court Y position
    near_player: bool
    player_y: Optional[float]  # Closest player Y


def _get_player_positions(frame: dict) -> List[Tuple[float, float]]:
    """Extract player world positions."""
    positions = []
    for player in frame.get("players", []):
        if isinstance(player, dict):
            world = player.get("world")
            if world and len(world) >= 2:
                positions.append((float(world[0]), float(world[1])))
    return positions


def _find_velocity_changes(
    frames: List[dict],
    velocities: List[Optional[Tuple[float, float]]],
    times: List[float],
    cfg: AlternatingDetectorConfig,
) -> List[RawEvent]:
    """Find all significant velocity changes."""
    raw_events: List[RawEvent] = []
    last_idx = -100
    
    for idx in range(2, len(velocities) - 1):
        vel_prev = velocities[idx - 1]
        vel_curr = velocities[idx]
        
        if vel_prev is None or vel_curr is None:
            continue
        
        if idx - last_idx < cfg.cooldown_frames:
            continue
        
        vx_prev, vy_prev = vel_prev
        vx_curr, vy_curr = vel_curr
        
        # Check velocity magnitude
        if abs(vy_prev) < cfg.min_velocity_magnitude and abs(vy_curr) < cfg.min_velocity_magnitude:
            continue
        
        # Check for significant change
        dvy = abs(vy_curr - vy_prev)
        dvx = abs(vx_curr - vx_prev)
        y_reversed = (vy_prev * vy_curr) < 0
        
        if dvy < cfg.min_velocity_change and not y_reversed:
            continue
        
        # Get ball position
        frame = frames[idx]
        ball = frame.get("ball", {})
        ball_world = ball.get("world")
        ball_y = ball_world[1] if ball_world and len(ball_world) >= 2 else None
        speed = ball.get("speed", 0) or 0
        
        # Check player proximity
        player_positions = _get_player_positions(frame)
        near_player = False
        closest_player_y = None
        
        if ball_world and len(ball_world) >= 2:
            bx, by = ball_world[0], ball_world[1]
            for px, py in player_positions:
                dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
                if dist < cfg.player_proximity_threshold:
                    near_player = True
                    closest_player_y = py
                    break
        
        raw_events.append(RawEvent(
            frame_index=idx,
            time=times[idx],
            vx_before=vx_prev,
            vy_before=vy_prev,
            vx_after=vx_curr,
            vy_after=vy_curr,
            speed=speed,
            ball_y=ball_y,
            near_player=near_player,
            player_y=closest_player_y,
        ))
        last_idx = idx
    
    return raw_events


def _classify_events(
    raw_events: List[RawEvent],
    cfg: AlternatingDetectorConfig,
) -> Tuple[List[DetectedEvent], List[DetectedEvent]]:
    """Classify raw events into contacts and bounces using alternating pattern."""
    contacts: List[DetectedEvent] = []
    bounces: List[DetectedEvent] = []
    
    if not raw_events:
        return contacts, bounces
    
    # Sort by time
    sorted_events = sorted(raw_events, key=lambda e: e.time)
    
    # First pass: classify based on position
    # - Near player = likely contact
    # - Far from both players = likely bounce (mid-court)
    classifications = []
    for evt in sorted_events:
        if evt.near_player:
            classifications.append("contact")
        else:
            # Check if ball is near center of court (where bounces happen)
            if evt.ball_y is not None:
                mid_court = cfg.court_length / 2
                dist_from_mid = abs(evt.ball_y - mid_court)
                # Bounces typically happen in the receiving half, not exact mid
                classifications.append("bounce")
            else:
                classifications.append("unknown")
    
    # Second pass: use alternating pattern to resolve unknowns
    # Rally pattern: contact - bounce - contact - bounce - ...
    # First event should be contact (serve or return hit)
    
    last_type = None
    for i, (evt, classification) in enumerate(zip(sorted_events, classifications)):
        inferred_type = classification
        
        # If first event, assume contact
        if i == 0:
            inferred_type = "contact"
        elif classification == "unknown":
            # Alternate from last
            inferred_type = "bounce" if last_type == "contact" else "contact"
        
        # Create detected event
        confidence = 0.8 if classification != "unknown" else 0.6
        
        if inferred_type == "contact":
            contacts.append(DetectedEvent(
                frame_index=evt.frame_index,
                confidence=confidence,
                timestamp=evt.time,
                details={
                    "vy_before": round(evt.vy_before, 3),
                    "vy_after": round(evt.vy_after, 3),
                    "near_player": evt.near_player,
                    "ball_y": round(evt.ball_y, 3) if evt.ball_y else None,
                },
            ))
        else:
            bounces.append(DetectedEvent(
                frame_index=evt.frame_index,
                confidence=confidence,
                timestamp=evt.time,
                details={
                    "speed": round(evt.speed, 3),
                    "ball_y": round(evt.ball_y, 3) if evt.ball_y else None,
                },
            ))
        
        last_type = inferred_type
    
    return contacts, bounces


def detect_alternating_events(
    world_json_path: str,
    video_json_path: Optional[str] = None,
    config: Optional[AlternatingDetectorConfig] = None,
) -> Tuple[List[DetectedEvent], List[DetectedEvent]]:
    """Detect contacts and bounces using alternating pattern.
    
    Returns:
        Tuple of (contacts, bounces)
    """
    cfg = config or AlternatingDetectorConfig()
    
    world_payload = load_json(world_json_path)
    frames = world_payload.get("frames", [])
    metadata = world_payload.get("metadata", {})
    fps = float(metadata.get("fps", 30.0))
    
    if not frames:
        return [], []
    
    times = extract_times(frames, fps)
    coords = extract_world_coords(frames)
    velocities = compute_velocities(coords, times)
    
    # Calculate video duration for end margin
    video_duration = times[-1] if times else 0
    
    # Find all velocity changes
    raw_events = _find_velocity_changes(frames, velocities, times, cfg)
    
    # Filter out events near video end
    if cfg.end_margin_seconds > 0:
        raw_events = [e for e in raw_events if e.time < video_duration - cfg.end_margin_seconds]
    
    # Classify into contacts and bounces
    contacts, bounces = _classify_events(raw_events, cfg)
    
    # Inject events
    inject_events(frames, "racket_contact", contacts)
    inject_events(frames, "bounce", bounces)
    save_json(world_json_path, world_payload)
    
    if video_json_path:
        video_payload = load_json(video_json_path)
        video_frames = video_payload.get("frames", [])
        if len(video_frames) == len(frames):
            inject_events(video_frames, "racket_contact", contacts)
            inject_events(video_frames, "bounce", bounces)
            save_json(video_json_path, video_payload)
    
    return contacts, bounces


__all__ = ["detect_alternating_events", "AlternatingDetectorConfig"]
