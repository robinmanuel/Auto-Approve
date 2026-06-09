import cv2
from PIL import Image
from image_model import predict_damage
from detector import CarPartDetector


class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        self.detector = CarPartDetector(yolo_weights)

    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    def crop(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        return image[y1:y2, x1:x2]

    def analyze(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return {
                "status": "error",
                "parts": [],
                "total_estimated_cost": 0
            }

        parts = self.detect_parts(image_path)

        if not parts:
            return {
                "status": "no_damage_detected",
                "parts": [],
                "total_estimated_cost": 0
            }

        results = []
        total_cost = 0

        for part in parts:

            crop_img = self.crop(image, part["bbox"])

            if crop_img.size == 0:
                continue

            damage = predict_damage(crop_img)

            total_cost += damage["Estimated_Cost"]

            results.append({
                "part": part.get("class_name"),
                "part_id": part.get("class_id"),
                "part_confidence": part.get("confidence"),
                "bbox": part.get("bbox"),
                "damage_type": damage["Damage_Type"],
                "damage_confidence": damage["Confidence"],
                "severity": damage["Severity"],
                "estimated_cost": damage["Estimated_Cost"]
            })

        return {
            "status": "success",
            "parts": results,
            "total_estimated_cost": round(total_cost, 2)
        }