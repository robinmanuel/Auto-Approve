import cv2
import numpy as np
from image_model import predict_damage
from detector import CarPartDetector


class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        self.detector = CarPartDetector(yolo_weights)

    # -----------------------------
    # DETECT PARTS
    # -----------------------------
    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    # -----------------------------
    # SAFE CROP
    # -----------------------------
    def crop(self, image, bbox):

        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = map(int, bbox)

        h, w = image.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop is None or crop.shape[0] < 30 or crop.shape[1] < 30:
            return None

        return crop

    # -----------------------------
    # IOU (FOR DUPLICATE REMOVAL)
    # -----------------------------
    def iou(self, a, b):

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        union = area_a + area_b - inter

        return inter / union if union > 0 else 0

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------
    def deduplicate(self, parts, threshold=0.75):

        filtered = []

        for p in parts:

            keep = True

            for f in filtered:

                if self.iou(p["bbox"], f["bbox"]) > threshold:
                    keep = False
                    break

            if keep:
                filtered.append(p)

        return filtered

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def analyze(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Invalid image path")

        # STEP 1: DETECT
        raw_parts = self.detect_parts(image_path)

        if not raw_parts:
            return {
                "status": "no_damage_detected",
                "parts": [],
                "total_estimated_cost": 0
            }

        # STEP 2: REMOVE DUPLICATES
        parts = self.deduplicate(raw_parts)

        results = []
        total_cost = 0

        # STEP 3: PROCESS EACH PART
        for part in parts:

            # FILTER LOW CONFIDENCE DETECTIONS
            if part.get("confidence", 0) < 0.5:
                continue

            crop_img = self.crop(image, part.get("bbox"))

            if crop_img is None:
                continue

            damage = predict_damage(crop_img)

            # SAFE FALLBACK
            if not damage:
                damage = {
                    "Damage_Type": "unknown",
                    "Confidence": 0,
                    "Severity": "Minor",
                    "Estimated_Cost": 0
                }

            conf = damage.get("Confidence", 0) or 0

            # FILTER LOW DAMAGE CONFIDENCE
            if conf < 60:
                damage = {
                    "Damage_Type": "unknown",
                    "Confidence": conf,
                    "Severity": "Minor",
                    "Estimated_Cost": 0
                }

            cost = damage.get("Estimated_Cost", 0)
            total_cost += cost

            results.append({
                "part": part.get("class_name", "unknown"),
                "part_id": part.get("class_id", -1),
                "part_confidence": round(part.get("confidence", 0), 3),

                "bbox": part.get("bbox", []),

                "damage_type": damage.get("Damage_Type", "unknown"),
                "damage_confidence": round(conf, 2),
                "severity": damage.get("Severity", "Minor"),
                "estimated_cost": round(cost, 2)
            })

        return {
            "status": "success",
            "parts": results,
            "total_estimated_cost": round(total_cost, 2)
        }