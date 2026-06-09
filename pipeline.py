from PIL import Image
import numpy as np

from image_model import predict_damage
from detector import CarPartDetector


class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        self.detector = CarPartDetector(yolo_weights)

    # -------------------------
    # Detect parts
    # -------------------------
    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    # -------------------------
    # Crop using PIL (NO cv2)
    # -------------------------
    def crop(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    # -------------------------
    # Main pipeline
    # -------------------------
    def analyze(self, image_path):

        image = Image.open(image_path).convert("RGB")

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