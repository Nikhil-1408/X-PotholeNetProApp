import cv2
import numpy as np
from typing import Dict, List, Tuple

COCO_VEHICLE_PERSON_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

SEVERITY_COLORS = {
    "Low": (0, 200, 0),
    "Medium": (0, 200, 255),
    "High": (0, 0, 255),
}


def enhance_standard(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_rainy(img: np.ndarray) -> np.ndarray:
    base = enhance_standard(img)
    blur = cv2.GaussianBlur(base, (3, 3), 0)
    return cv2.addWeighted(base, 1.18, blur, -0.18, 0)


def enhance_lowlight(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhance_standard(bright)


def preprocess_by_mode(img: np.ndarray, mode: str = "Standard") -> np.ndarray:
    if mode == "Rainy / Wet Road":
        return enhance_rainy(img)
    if mode == "Low Light / Night":
        return enhance_lowlight(img)
    return enhance_standard(img)


def is_blurry(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 18


def is_too_dark(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() < 32


def has_road_texture(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 130)
    density = edges.sum() / (img.shape[0] * img.shape[1] * 255)
    return density > 0.0035


def road_scene_score(img: np.ndarray) -> float:
    h, _ = img.shape[:2]
    lower = img[int(h * 0.45):, :]

    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 40, 140)
    edge_density = edges.sum() / (lower.shape[0] * lower.shape[1] * 255)

    sat_mean = hsv[:, :, 1].mean() / 255.0
    val_mean = hsv[:, :, 2].mean() / 255.0

    sat_score = 1.0 - min(sat_mean / 0.65, 1.0)

    if 0.14 <= val_mean <= 0.78:
        brightness_score = 1.0
    else:
        brightness_score = max(0.0, 1.0 - abs(val_mean - 0.45) / 0.45)

    if edge_density < 0.004:
        edge_score = edge_density / 0.004
    elif edge_density > 0.14:
        edge_score = max(0.0, 1.0 - (edge_density - 0.14) / 0.14)
    else:
        edge_score = 1.0

    score = 0.45 * edge_score + 0.25 * sat_score + 0.30 * brightness_score
    return round(float(score), 4)


def resize_for_speed(img: np.ndarray, max_width: int = 960) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    xi1 = max(x1, x1p)
    yi1 = max(y1, y1p)
    xi2 = min(x2, x2p)
    yi2 = min(y2, y2p)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(1, (x2 - x1) * (y2 - y1))
    area2 = max(1, (x2p - x1p) * (y2p - y1p))
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def overlaps_any(box: List[int], others: List[Dict], threshold: float = 0.30) -> bool:
    for obj in others:
        if compute_iou(box, obj["bbox"]) >= threshold:
            return True
    return False


def extract_severity_features(
    roi: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    confidence: float,
    model_votes: int,
    count_context: int,
) -> Dict[str, float]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

    darkness = 1 - (gray.mean() / 255.0)
    edges = cv2.Canny(gray, 35, 130)
    texture = edges.sum() / (roi.shape[0] * roi.shape[1] * 255)
    bright_ratio = float((gray > 210).sum()) / gray.size

    return {
        "area_ratio": round(float(area_ratio), 4),
        "darkness": round(float(darkness), 4),
        "texture": round(float(texture), 4),
        "bright_ratio": round(float(bright_ratio), 4),
        "confidence": round(float(confidence), 4),
        "model_votes": int(model_votes),
        "count_context": int(count_context),
    }


def explain_severity(features: Dict[str, float], severity: str) -> List[str]:
    reasons = []

    area_ratio = features["area_ratio"]
    darkness = features["darkness"]
    texture = features["texture"]
    bright_ratio = features["bright_ratio"]
    votes = features["model_votes"]
    count_context = features["count_context"]

    if area_ratio >= 0.055:
        reasons.append("very large pothole region")
    elif area_ratio >= 0.020:
        reasons.append("moderate pothole size")
    else:
        reasons.append("small pothole size")

    if darkness > 0.40:
        reasons.append("dark depression visible")

    if texture > 0.10:
        reasons.append("rough damaged texture")

    if bright_ratio > 0.22:
        reasons.append("water/reflection present")

    if votes >= 2:
        reasons.append("multi-model agreement")

    if count_context >= 8:
        reasons.append("many potholes on road")

    if severity == "High" and area_ratio < 0.020:
        reasons.append("elevated by confidence + road context")

    return reasons


def get_severity_color(severity: str) -> Tuple[int, int, int]:
    return SEVERITY_COLORS.get(severity, (255, 255, 255))


def compute_overall_road_assessment(counts: Dict[str, int]) -> Tuple[int, str]:
    total = counts["Low"] + counts["Medium"] + counts["High"]

    risk_score = min(
        100,
        int(
            counts["Low"] * 6
            + counts["Medium"] * 20
            + counts["High"] * 42
            + total * 2
        ),
    )

    if counts["High"] >= 3 or risk_score >= 80:
        road_status = "Unsafe to Drive"
    elif counts["High"] >= 1 or counts["Medium"] >= 4 or risk_score >= 55:
        road_status = "Drive With Extreme Caution"
    elif total >= 5 or counts["Medium"] >= 2 or risk_score >= 30:
        road_status = "Moderate Road Damage"
    elif total > 0:
        road_status = "Minor Road Damage"
    else:
        road_status = "Road Looks Safe"

    return risk_score, road_status


def draw_badge(img: np.ndarray, text: str, color: Tuple[int, int, int], x: int, y: int) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 12, y + 4), color, -1)
    cv2.putText(img, text, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 2)