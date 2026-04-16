"""
Visualization utilities — drawing boxes, trails, zones, and heatmaps.
All functions are pure: they take a frame + data and return an annotated frame.
"""

import cv2
import numpy as np
from collections import deque

# BGR color palette per object class
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 120, 255),     # orange
    "car": (255, 200, 0),        # cyan
    "truck": (0, 255, 100),      # green
    "unknown": (200, 200, 200),  # gray
}


def _color_for(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])


# ------------------------------------------------------------------
# Bounding boxes with labels
# ------------------------------------------------------------------
LOCK_COLOR = (255, 255, 0)  # bright cyan (BGR)


def draw_boxes(
    frame: np.ndarray,
    tracked_objects: list[dict],
    locked_track_id: int | None = None,
) -> np.ndarray:
    """Draw colored bounding boxes with ID / class / confidence / velocity labels.

    If *locked_track_id* matches a track, that target gets a distinctive
    cyan highlight with corner brackets.
    """
    annotated = frame.copy()

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        is_locked = locked_track_id is not None and obj["track_id"] == locked_track_id

        if is_locked:
            color = LOCK_COLOR
            thickness = 3
            label_prefix = "LOCKED "
        else:
            color = _color_for(obj["class_name"])
            thickness = 2
            label_prefix = ""

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Corner brackets for locked target
        if is_locked:
            cl = max(12, min(25, (x2 - x1) // 4, (y2 - y1) // 4))
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(annotated, (cx, cy), (cx + dx * cl, cy), LOCK_COLOR, 3)
                cv2.line(annotated, (cx, cy), (cx, cy + dy * cl), LOCK_COLOR, 3)

        # Label text
        parts = [f"{label_prefix}ID:{obj['track_id']} {obj['class_name']}"]
        if obj.get("confidence") is not None:
            parts.append(f"{obj['confidence']:.0%}")
        if obj.get("velocity", 0) > 0:
            parts.append(f"{obj['velocity']:.0f}px/s")
        label = " | ".join(parts)

        # Background rectangle for readability
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - th - 10),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ------------------------------------------------------------------
# Locked-target info panel
# ------------------------------------------------------------------
def draw_locked_target_info(
    frame: np.ndarray,
    locked_obj: dict,
    lock_duration_s: float,
) -> np.ndarray:
    """Draw an info panel in the top-right corner for the locked target."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 270, 130
    x1 = w - panel_w - 10
    y1 = 10
    x2, y2 = x1 + panel_w, y1 + panel_h

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)

    lines = [
        f"TARGET LOCKED: ID {locked_obj['track_id']}",
        f"Class : {locked_obj['class_name']}",
        f"Speed : {locked_obj['velocity']:.0f} px/s",
        f"Pos   : {locked_obj['centroid']}",
        f"Tracked: {lock_duration_s:.1f}s",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x1 + 10, y1 + 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, LOCK_COLOR, 1, cv2.LINE_AA,
        )
    return frame


# ------------------------------------------------------------------
# Movement trails (breadcrumbs)
# ------------------------------------------------------------------
def draw_trails(
    frame: np.ndarray,
    trails: dict[int, deque],
    class_map: dict[int, str],
) -> np.ndarray:
    """Draw polyline trails and a dot at the latest position for each track."""
    for track_id, trail in trails.items():
        if len(trail) < 2:
            continue
        color = _color_for(class_map.get(track_id, "unknown"))
        points = np.array(list(trail), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, trail[-1], 4, color, -1)
    return frame


# ------------------------------------------------------------------
# Zone overlay
# ------------------------------------------------------------------
def draw_zone(
    frame: np.ndarray, zone_polygon: np.ndarray, is_intruded: bool
) -> np.ndarray:
    """Draw a semi-transparent zone polygon — green if safe, red if breached."""
    overlay = frame.copy()
    color = (0, 0, 200) if is_intruded else (0, 180, 0)

    cv2.fillPoly(overlay, [zone_polygon], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [zone_polygon], isClosed=True, color=color, thickness=2)

    # Zone label
    label = "ZONE BREACH" if is_intruded else "MONITORED ZONE"
    tx, ty = zone_polygon[0][0], zone_polygon[0][1] - 8
    cv2.putText(
        frame, label, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )
    return frame


# ------------------------------------------------------------------
# Heatmap overlay
# ------------------------------------------------------------------
def generate_heatmap(
    frame: np.ndarray,
    trails: dict[int, deque],
    heatmap_accumulator: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cumulative heatmap from all trail positions, overlay on frame.

    Returns (blended_frame, updated_accumulator).
    """
    h, w = frame.shape[:2]
    if heatmap_accumulator is None or heatmap_accumulator.shape[:2] != (h, w):
        heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

    # Accumulate current trail points
    for trail in trails.values():
        for cx, cy in trail:
            if 0 <= cy < h and 0 <= cx < w:
                heatmap_accumulator[cy, cx] += 1.0

    # Blur and normalize
    blurred = cv2.GaussianBlur(heatmap_accumulator, (0, 0), sigmaX=20, sigmaY=20)
    max_val = blurred.max()
    if max_val > 0:
        normalized = np.uint8(255 * blurred / max_val)
    else:
        normalized = np.zeros((h, w), dtype=np.uint8)

    heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    return blended, heatmap_accumulator


# ------------------------------------------------------------------
# Zone polygon builder
# ------------------------------------------------------------------
def build_zone_polygon(
    frame_w: int,
    frame_h: int,
    x_pct: float,
    y_pct: float,
    w_pct: float,
    h_pct: float,
) -> np.ndarray:
    """Convert percentage-based zone sliders to a pixel-coordinate polygon.

    Returns an int32 numpy array of shape (4, 2).
    """
    x1 = int(frame_w * x_pct / 100)
    y1 = int(frame_h * y_pct / 100)
    x2 = int(frame_w * (x_pct + w_pct) / 100)
    y2 = int(frame_h * (y_pct + h_pct) / 100)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
