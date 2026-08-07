import cv2
import tempfile
from typing import Dict, List

from detector import MultiModelDetector
from pothole_core import compute_iou

detector = MultiModelDetector()


class SimpleTrackManager:
    def __init__(self, iou_threshold: float = 0.35, max_missed: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        matched_track_ids = set()

        for det in detections:
            best_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                iou = compute_iou(det["bbox"], track["bbox"])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                track = self.tracks[best_id]
                track["bbox"] = det["bbox"]
                track["missed"] = 0
                track["confidence"] = max(track["confidence"], det["confidence"])
                track["ml_confidence"] = max(track.get("ml_confidence", 0), det.get("ml_confidence", 0))
                track["model_votes"] = max(track["model_votes"], det["model_votes"])
                track["area_ratio"] = max(track["area_ratio"], det["area_ratio"])
                track["darkness"] = max(track["darkness"], det["darkness"])
                track["texture"] = max(track["texture"], det["texture"])
                track["bright_ratio"] = max(track["bright_ratio"], det["bright_ratio"])
                track["explanation"] = det["explanation"]

                severity_rank = {"Low": 0, "Medium": 1, "High": 2}
                if severity_rank[det["severity"]] > severity_rank[track["severity"]]:
                    track["severity"] = det["severity"]

                matched_track_ids.add(best_id)
                det["track_id"] = best_id
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "bbox": det["bbox"],
                    "severity": det["severity"],
                    "confidence": det["confidence"],
                    "ml_confidence": det.get("ml_confidence", 0),
                    "model_votes": det["model_votes"],
                    "area_ratio": det["area_ratio"],
                    "darkness": det["darkness"],
                    "texture": det["texture"],
                    "bright_ratio": det["bright_ratio"],
                    "explanation": det["explanation"],
                    "missed": 0,
                }
                matched_track_ids.add(track_id)
                det["track_id"] = track_id

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed:
                    del self.tracks[track_id]

        return detections

    def unique_summary(self) -> List[Dict]:
        summary = []
        for track_id, track in self.tracks.items():
            if track["missed"] <= self.max_missed:
                row = track.copy()
                row["track_id"] = track_id
                summary.append(row)
        return summary


def enhance_live_frame(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 1.18, img, -0.10, 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def process_frame(
    img,
    mode="Standard",
    conf_threshold=0.25,
    iou_threshold=0.45,
    overlap_threshold=0.20,
    min_width=16,
    min_height=12,
    use_validation=True,
    allow_dark_frames=True,
    allow_blurry_frames=True,
    max_width=960,
    require_road_scene=True,
    road_scene_threshold=0.42,
):
    return detector.detect(
        img=img,
        mode=mode,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        overlap_threshold=overlap_threshold,
        min_width=min_width,
        min_height=min_height,
        use_validation=use_validation,
        allow_dark_frames=allow_dark_frames,
        allow_blurry_frames=allow_blurry_frames,
        max_width=max_width,
        require_road_scene=require_road_scene,
        road_scene_threshold=road_scene_threshold,
    )


def process_video_file(
    video_path,
    mode="Standard",
    conf_threshold=0.25,
    iou_threshold=0.45,
    overlap_threshold=0.20,
    min_width=16,
    min_height=12,
    use_validation=True,
    allow_dark_frames=True,
    allow_blurry_frames=True,
    max_width=960,
    frame_skip=3,
    require_road_scene=True,
    road_scene_threshold=0.42,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_index = 0
    last_processed_frame = None
    tracker = SimpleTrackManager(iou_threshold=0.35, max_missed=8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            processed_frame_small, detections, _, _, _, _, _ = detector.detect(
                img=frame,
                mode=mode,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                overlap_threshold=overlap_threshold,
                min_width=min_width,
                min_height=min_height,
                use_validation=use_validation,
                allow_dark_frames=allow_dark_frames,
                allow_blurry_frames=allow_blurry_frames,
                max_width=max_width,
                require_road_scene=require_road_scene,
                road_scene_threshold=road_scene_threshold,
            )

            tracker.update(detections)

            processed_frame = cv2.resize(processed_frame_small, (width, height))
            last_processed_frame = processed_frame
        else:
            processed_frame = last_processed_frame if last_processed_frame is not None else frame

        writer.write(processed_frame)
        frame_index += 1

    cap.release()
    writer.release()

    unique_rows = tracker.unique_summary()

    aggregate_counts = {"Low": 0, "Medium": 0, "High": 0}
    for row in unique_rows:
        aggregate_counts[row["severity"]] += 1

    if aggregate_counts["High"] >= 3:
        final_alert = "🚨 UNSAFE TO DRIVE — SEVERE ROAD DAMAGE"
    elif aggregate_counts["High"] > 0 or aggregate_counts["Medium"] >= 4:
        final_alert = "🚨 DRIVE WITH EXTREME CAUTION"
    elif aggregate_counts["Medium"] > 0 or aggregate_counts["Low"] >= 5:
        final_alert = "⚠️ MODERATE ROAD DAMAGE"
    elif aggregate_counts["Low"] > 0:
        final_alert = "⚠️ MINOR ROAD DAMAGE"
    else:
        final_alert = "✅ NO POTHOLES DETECTED IN VIDEO"

    return out_path, unique_rows, aggregate_counts, final_alert