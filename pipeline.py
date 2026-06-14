from detector import CarPartDetector
from damage_detector import DamageDetector


DAMAGE_COSTS = {
    "scratch": 2500,
    "dent": 4000,
    "crack": 6000,
    "broken_lamp": 7000,
    "flat_tire": 5000,
    "shattered_glass": 15000
}


class AutoApprovePipeline:

    def __init__(
        self,
        part_model_path,
        damage_model_path
    ):
        self.part_detector = CarPartDetector(part_model_path)
        self.damage_detector = DamageDetector(damage_model_path)

    # -------------------------
    # IoU
    # -------------------------
    def iou(self, b1, b2):

        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])

        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        if inter <= 0:
            return 0

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    # -------------------------
    # severity
    # -------------------------
    def get_severity(self, confidence):

        if confidence < 0.60:
            return "Minor"

        elif confidence < 0.85:
            return "Moderate"

        return "Severe"

    # -------------------------
    # duplicate removal
    # -------------------------
    def remove_duplicates(self, detections):

        filtered = []

        detections = sorted(
            detections,
            key=lambda x: x["confidence"],
            reverse=True
        )

        for det in detections:

            duplicate = False

            for kept in filtered:

                if (
                    det["class_name"] == kept["class_name"]
                    and self.iou(det["bbox"], kept["bbox"]) > 0.7
                ):
                    duplicate = True
                    break

            if not duplicate:
                filtered.append(det)

        return filtered

    # -------------------------
    # main pipeline
    # -------------------------
    def analyze(self, image_path):

        parts = self.part_detector.predict(image_path)

        damages = self.damage_detector.predict(image_path)

        if not parts:

            return {
                "status": "no_parts_detected",
                "parts": [],
                "total_estimated_cost": 0
            }

        parts = self.remove_duplicates(parts)

        total_cost = 0
        results = []

        for part in parts:

            best_damage = None
            best_overlap = 0

            for damage in damages:

                overlap = self.iou(
                    part["bbox"],
                    damage["bbox"]
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_damage = damage

            if best_damage and best_overlap > 0.05:

                damage_type = best_damage["damage_type"]

                damage_conf = round(
                    best_damage["confidence"] * 100,
                    2
                )

                severity = self.get_severity(
                    best_damage["confidence"]
                )

                estimated_cost = DAMAGE_COSTS.get(
                    damage_type,
                    0
                )

            else:

                damage_type = "no_damage"
                damage_conf = 0
                severity = "None"
                estimated_cost = 0

            total_cost += estimated_cost

            results.append({
                "part": part["class_name"],
                "part_id": part["class_id"],
                "part_confidence": round(
                    part["confidence"] * 100,
                    2
                ),
                "bbox": part["bbox"],
                "damage_type": damage_type,
                "damage_confidence": damage_conf,
                "severity": severity,
                "estimated_cost": estimated_cost
            })

        return {
            "status": "success",
            "parts": results,
            "total_estimated_cost": round(total_cost, 2)
        }