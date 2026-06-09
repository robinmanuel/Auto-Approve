from PIL import Image
from image_model import predict_damage
from detector import CarPartDetector


class AutoApprovePipeline:

    def __init__(self, yolo_weights):
        self.detector = CarPartDetector(yolo_weights)

    # -------------------------
    # detect parts
    # -------------------------
    def detect_parts(self, image_path):
        return self.detector.predict(image_path)

    # -------------------------
    # IoU for duplicate removal
    # -------------------------
    def iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    # -------------------------
    # crop image
    # -------------------------
    def crop(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    # -------------------------
    # main pipeline
    # -------------------------
    def analyze(self, image_path):

        image = Image.open(image_path).convert("RGB")

        raw_parts = self.detect_parts(image_path)

        if not raw_parts:
            return {
                "status": "no_damage_detected",
                "parts": [],
                "total_estimated_cost": 0
            }

        # -------------------------
        # REMOVE DUPLICATES
        # -------------------------
        parts = []
        for p in raw_parts:
            duplicate = False

            for q in parts:
                if self.iou(p["bbox"], q["bbox"]) > 0.7:
                    duplicate = True
                    break

            if not duplicate:
                parts.append(p)

        # -------------------------
        # DAMAGE ANALYSIS
        # -------------------------
        results = []
        total_cost = 0

        for part in parts:

            crop_img = self.crop(image, part["bbox"])
            damage = predict_damage(crop_img)

            # fix wrong predictions (safety layer)
            if part["class_name"] in ["hood", "door", "bumper"] and damage["Damage_Type"] == "broken_lamp":
                damage["Damage_Type"] = "unknown"
                damage["Estimated_Cost"] = 0

            total_cost += damage["Estimated_Cost"]

            results.append({
                "part": part["class_name"],
                "part_id": part["class_id"],
                "part_confidence": part["confidence"],
                "bbox": part["bbox"],
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