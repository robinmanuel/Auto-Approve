class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        from detector import CarPartDetector
        self.detector = CarPartDetector(yolo_weights)

    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    def crop(self, image, bbox):
        import cv2   # 🔥 lazy import
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    def analyze(self, image_path):

        import cv2  # 🔥 lazy import (CRITICAL FIX)

        image = cv2.imread(image_path)

        if image is None:
            return {"status": "error"}

        parts = self.detect_parts(image_path)

        if not parts:
            return {
                "status": "no_damage_detected",
                "parts": [],
                "total_estimated_cost": 0
            }

        from image_model import predict_damage

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