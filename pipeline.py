from PIL import Image
import numpy as np
from image_model import predict_damage
from detector import CarPartDetector


class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        self.detector = CarPartDetector(yolo_weights)

    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    def crop(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def analyze(self, image_path):

        # ✅ PIL replaces cv2 completely
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

            if crop_img is None:
                continue

            damage = predict_damage(crop_img)

            cost = float(damage.get("Estimated_Cost", 0))
            total_cost += cost

            results.append({
                "part": part.get("class_name", "unknown"),
                "part_id": part.get("class_id"),
                "confidence": part.get("confidence"),
                "bbox": part.get("bbox"),
                "damage": damage
            })

        return {
            "status": "success",
            "parts": results,
            "total_estimated_cost": round(total_cost, 2)
        }