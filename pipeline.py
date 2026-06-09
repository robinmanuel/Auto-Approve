import cv2
from image_model import predict_damage
from detector import CarPartDetector


class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        self.detector = CarPartDetector(yolo_weights)

    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    def crop(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    # -----------------------------
    # REMOVE DUPLICATES (IMPORTANT FIX)
    # -----------------------------
    def _deduplicate(self, parts, iou_threshold=0.85):

        def iou(a, b):
            xA = max(a[0], b[0])
            yA = max(a[1], b[1])
            xB = min(a[2], b[2])
            yB = min(a[3], b[3])

            inter = max(0, xB - xA) * max(0, yB - yA)

            areaA = (a[2] - a[0]) * (a[3] - a[1])
            areaB = (b[2] - b[0]) * (b[3] - b[1])

            return inter / (areaA + areaB - inter + 1e-6)

        filtered = []

        for p in parts:
            keep = True
            for q in filtered:
                if iou(p["bbox"], q["bbox"]) > iou_threshold:
                    keep = False
                    break
            if keep:
                filtered.append(p)

        return filtered

    def analyze(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return {
                "status": "error",
                "message": "Invalid image"
            }

        parts = self.detect_parts(image_path)

        if not parts:
            return {
                "status": "no_damage_detected",
                "parts": [],
                "total_estimated_cost": 0
            }

        parts = self._deduplicate(parts)

        results = []
        total_cost = 0

        for part in parts:

            crop_img = self.crop(image, part["bbox"])

            if crop_img.size == 0:
                continue

            damage = predict_damage(crop_img)

            cost = float(damage.get("Estimated_Cost", 0))

            total_cost += cost

            results.append({
                "part": part.get("class_name", "unknown"),
                "part_id": part.get("class_id", -1),
                "part_confidence": round(part.get("confidence", 0), 3),
                "bbox": part.get("bbox"),
                "damage_type": damage.get("Damage_Type", "unknown"),
                "damage_confidence": damage.get("Confidence", 0),
                "severity": damage.get("Severity", "Unknown"),
                "estimated_cost": cost
            })

        return {
            "status": "success",
            "parts": results,
            "total_estimated_cost": round(total_cost, 2)
        }