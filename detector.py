from ultralytics import YOLO

class CarPartDetector:

    def __init__(self, weights):
        self.model = YOLO(weights)

    def predict(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=0.45,   # 🔥 increased threshold (fix duplicates)
            iou=0.5,
            verbose=False
        )

        parts = []

        for r in results:
            if r.boxes is None:
                continue

            names = r.names

            for box in r.boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 🔥 skip weak detections
                if conf < 0.5:
                    continue

                parts.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 3),
                    "bbox": box.xyxy[0].tolist()
                })

        return parts