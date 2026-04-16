"""
Aerial Intelligence Engine — DroneVision
YOLOv8 detection + DeepSORT tracking with movement trails,
zone intrusion detection, and velocity estimation.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import numpy as np
import cv2


class DroneVision:
    """Core vision engine: detects, tracks, and analyzes objects from drone footage."""

    DEFAULT_TARGETS = {0: "person", 2: "car", 7: "truck"}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        iou_thresh: float = 0.5,
        target_classes: dict | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.target_classes = target_classes or self.DEFAULT_TARGETS

        # Initialize DeepSORT — fall back to CPU if no GPU available
        try:
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.2,
                nn_budget=100,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
            )
        except Exception:
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.2,
                nn_budget=100,
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False,
            )

        # Per-track state
        self.trails: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self.class_map: dict[int, str] = {}
        self.frame_count: int = 0
        self.fps: float = 30.0  # updated by caller after reading video metadata

    # ------------------------------------------------------------------
    # Main per-frame pipeline
    # ------------------------------------------------------------------
    def detect_and_track(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO detection → DeepSORT tracking on a single frame.

        Returns a list of dicts, one per confirmed track:
            {track_id, bbox, class_name, confidence, centroid, velocity}
        """
        self.frame_count += 1

        # 1. YOLO inference
        results = self.model(
            frame, conf=self.confidence, iou=self.iou_thresh, verbose=False
        )
        boxes = results[0].boxes

        # 2. Filter detections by target classes and format for DeepSORT
        raw_detections: list[tuple] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cls_id = int(classes[i])
                if cls_id not in self.target_classes:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                w, h = x2 - x1, y2 - y1
                conf = float(confs[i])
                class_name = self.target_classes[cls_id]
                raw_detections.append(([x1, y1, w, h], conf, class_name))

        # 3. Update tracker (must be called every frame, even with no detections)
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        # 4. Build output from confirmed tracks
        tracked_objects: list[dict] = []
        active_ids: set[int] = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_ids.add(track_id)

            ltrb = track.to_ltrb()
            if ltrb is None:
                continue

            det_class = track.get_det_class()
            if det_class:
                self.class_map[track_id] = det_class
            class_name = self.class_map.get(track_id, "unknown")

            cx = int((ltrb[0] + ltrb[2]) / 2)
            cy = int((ltrb[1] + ltrb[3]) / 2)
            self.trails[track_id].append((cx, cy))

            tracked_objects.append(
                {
                    "track_id": track_id,
                    "bbox": [int(v) for v in ltrb],
                    "class_name": class_name,
                    "confidence": track.get_det_conf(),
                    "centroid": (cx, cy),
                    "velocity": self._estimate_velocity(track_id),
                }
            )

        # 5. Prune stale trail data
        stale_ids = [tid for tid in self.trails if tid not in active_ids]
        for tid in stale_ids:
            # Keep trails for a few extra frames so they fade visually
            if self.frame_count % 60 == 0:
                del self.trails[tid]
                self.class_map.pop(tid, None)

        return tracked_objects

    # ------------------------------------------------------------------
    # Zone intrusion
    # ------------------------------------------------------------------
    def check_zone_intrusion(
        self, tracked_objects: list[dict], zone_polygon: np.ndarray
    ) -> list[dict]:
        """Return subset of tracked_objects whose centroids are inside the zone polygon."""
        intruders = []
        for obj in tracked_objects:
            cx, cy = obj["centroid"]
            result = cv2.pointPolygonTest(
                zone_polygon, (float(cx), float(cy)), measureDist=False
            )
            if result >= 0:
                intruders.append(obj)
        return intruders

    # ------------------------------------------------------------------
    # Velocity estimation
    # ------------------------------------------------------------------
    def _estimate_velocity(self, track_id: int) -> float:
        """Pixel-per-second velocity based on last two trail positions."""
        trail = self.trails.get(track_id)
        if trail is None or len(trail) < 2:
            return 0.0
        p1, p2 = trail[-2], trail[-1]
        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        return round(float(dist * self.fps), 1)

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------
    @staticmethod
    def get_counts(tracked_objects: list[dict]) -> dict:
        """Aggregate current-frame counts by class."""
        counts = {"person": 0, "car": 0, "truck": 0, "total": 0}
        for obj in tracked_objects:
            cls = obj["class_name"]
            if cls in counts:
                counts[cls] += 1
            counts["total"] += 1
        return counts

    # ------------------------------------------------------------------
    # Reset (for new video)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all tracking state for a fresh run."""
        self.tracker.delete_all_tracks()
        self.trails.clear()
        self.class_map.clear()
        self.frame_count = 0
