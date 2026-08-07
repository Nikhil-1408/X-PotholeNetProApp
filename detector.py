from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from pothole_core import (
    COCO_VEHICLE_PERSON_CLASSES,
    compute_iou,
    compute_overall_road_assessment,
    draw_badge,
    explain_severity,
    extract_severity_features,
    get_severity_color,
    has_road_texture,
    is_blurry,
    is_too_dark,
    overlaps_any,
    preprocess_by_mode,
    resize_for_speed,
    road_scene_score,
)
from severity_model import SeverityMLModel


class MultiModelDetector:
    def __init__(
        self,
        general_model_path: str = "models/yolov8n.pt",
        pothole_model_1_path: str = "models/pothole_best.pt",
        pothole_model_2_path: str = "models/best.pt",
    ) -> None:
        self.general_model = YOLO(general_model_path)
        self.pothole_model_1 = YOLO(pothole_model_1_path)
        self.pothole_model_2 = YOLO(pothole_model_2_path)
        self.severity_model = SeverityMLModel()

    def _general_objects(
        self,
        img: np.ndarray,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
    ) -> List[Dict]:
        h, w = img.shape[:2]
        results = self.general_model.predict(img, conf=conf_threshold, iou=iou_threshold, verbose=False)

        objects = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in COCO_VEHICLE_PERSON_CLASSES:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                objects.append(
                    {
                        "label": COCO_VEHICLE_PERSON_CLASSES[cls_id],
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return objects

    def _single_pothole_model(
        self,
        model,
        img: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        min_width: int,
        min_height: int,
    ) -> List[Dict]:
        h, w = img.shape[:2]
        results = model.predict(img, conf=conf_threshold, iou=iou_threshold, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                bw = x2 - x1
                bh = y2 - y1
                if bw < min_width or bh < min_height:
                    continue

                if bh > 2.4 * bw:
                    continue

                boxes.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    }
                )
        return boxes

    def _merge_pothole_boxes(
        self,
        all_boxes: List[List[Dict]],
        merge_iou: float = 0.20,
    ) -> List[Dict]:
        merged = []
        flat = []
        for box_list in all_boxes:
            flat.extend(box_list)

        flat = sorted(flat, key=lambda x: x["confidence"], reverse=True)

        for det in flat:
            current_box = det["bbox"]
            matched = False
            for item in merged:
                if compute_iou(current_box, item["bbox"]) >= merge_iou:
                    item["confidence"] = max(item["confidence"], det["confidence"])
                    item["model_votes"] += 1
                    matched = True
                    break
            if not matched:
                merged.append(
                    {
                        "bbox": current_box,
                        "confidence": det["confidence"],
                        "model_votes": 1,
                    }
                )
        return merged

    def detect(
        self,
        img: np.ndarray,
        mode: str = "Standard",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        overlap_threshold: float = 0.20,
        min_width: int = 16,
        min_height: int = 12,
        use_validation: bool = True,
        allow_dark_frames: bool = True,
        allow_blurry_frames: bool = True,
        max_width: int = 960,
        require_road_scene: bool = True,
        road_scene_threshold: float = 0.42,
        object_overlap_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, List[Dict], Dict[str, int], str, int, List[Dict], str]:
        original = img.copy()
        resized_img, _ = resize_for_speed(original, max_width=max_width)
        processed = preprocess_by_mode(resized_img, mode)
        output_img = processed.copy()
        h, w = output_img.shape[:2]

        if use_validation:
            if not allow_blurry_frames and is_blurry(output_img):
                return output_img, [], {"Low": 0, "Medium": 0, "High": 0}, "❌ Blurry frame", 0, [], "Invalid Frame"

            if not allow_dark_frames and is_too_dark(output_img):
                return output_img, [], {"Low": 0, "Medium": 0, "High": 0}, "❌ Dark frame", 0, [], "Invalid Frame"

            if not has_road_texture(output_img):
                return output_img, [], {"Low": 0, "Medium": 0, "High": 0}, "❌ Unclear road frame", 0, [], "Invalid Frame"

        if require_road_scene:
            rs = road_scene_score(output_img)
            if rs < road_scene_threshold:
                return output_img, [], {"Low": 0, "Medium": 0, "High": 0}, "⚠️ Non-road scene rejected", 0, [], "Rejected"

        general_objects = self._general_objects(
            output_img,
            conf_threshold=max(0.28, conf_threshold),
            iou_threshold=iou_threshold,
        )

        potholes_1 = self._single_pothole_model(
            self.pothole_model_1, output_img, conf_threshold, iou_threshold, min_width, min_height
        )
        potholes_2 = self._single_pothole_model(
            self.pothole_model_2, output_img, conf_threshold, iou_threshold, min_width, min_height
        )

        pothole_boxes = self._merge_pothole_boxes([potholes_1, potholes_2], merge_iou=overlap_threshold)

        detections = []

        clean_boxes = []
        for det in pothole_boxes:
            if not overlaps_any(det["bbox"], general_objects, threshold=object_overlap_threshold):
                clean_boxes.append(det)

        count_context = len(clean_boxes)
        counts = {"Low": 0, "Medium": 0, "High": 0}

        for det in clean_boxes:
            x1, y1, x2, y2 = det["bbox"]
            conf = float(det["confidence"])
            votes = int(det["model_votes"])

            roi = output_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            features = extract_severity_features(
                roi=roi,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                img_w=w,
                img_h=h,
                confidence=conf,
                model_votes=votes,
                count_context=count_context,
            )

            severity, ml_confidence = self.severity_model.predict(
                area_ratio=features["area_ratio"],
                darkness=features["darkness"],
                texture=features["texture"],
                bright_ratio=features["bright_ratio"],
                confidence=features["confidence"],
                model_votes=features["model_votes"],
                count_context=features["count_context"],
            )

            if features["area_ratio"] >= 0.055:
                severity = "High"
            elif features["area_ratio"] >= 0.020 and severity == "Low":
                severity = "Medium"

            if severity == "Low" and conf < 0.28:
                continue
            if severity == "Medium" and conf < 0.24:
                continue
            if severity == "High" and conf < 0.22:
                continue

            reasons = explain_severity(features, severity)
            color = get_severity_color(severity)

            counts[severity] += 1

            label = f"{severity} ({conf:.2f})"
            y_text = max(y1 - 8, 22)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            draw_badge(output_img, label, color, x1, y_text)

            detections.append(
                {
                    "severity": severity,
                    "confidence": round(conf, 2),
                    "ml_confidence": round(float(ml_confidence), 2),
                    "bbox": [x1, y1, x2, y2],
                    "model_votes": votes,
                    "explanation": ", ".join(reasons),
                    "area_ratio": features["area_ratio"],
                    "darkness": features["darkness"],
                    "texture": features["texture"],
                    "bright_ratio": features["bright_ratio"],
                }
            )

        for obj in general_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(
                output_img,
                f'{obj["label"]} {obj["confidence"]:.2f}',
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 180, 0),
                1,
            )

        risk_score, road_status = compute_overall_road_assessment(counts)

        if road_status == "Unsafe to Drive":
            alert = "🚨 UNSAFE TO DRIVE — SEVERE ROAD DAMAGE"
            alert_color = (0, 0, 255)
        elif road_status == "Drive With Extreme Caution":
            alert = "🚨 DRIVE WITH EXTREME CAUTION"
            alert_color = (0, 0, 255)
        elif road_status == "Moderate Road Damage":
            alert = "⚠️ MODERATE ROAD DAMAGE DETECTED"
            alert_color = (0, 165, 255)
        elif road_status == "Minor Road Damage":
            alert = "⚠️ MINOR ROAD DAMAGE DETECTED"
            alert_color = (0, 255, 255)
        else:
            alert = "✅ NO POTHOLES DETECTED"
            alert_color = (0, 255, 0)

        cv2.putText(output_img, alert, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, alert_color, 2)
        cv2.putText(
            output_img,
            f"Risk Score: {risk_score}/100",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )

        return output_img, detections, counts, alert, risk_score, general_objects, road_status