from ultralytics import YOLO


class CarPartDetector:

    def __init__(self, weights):
        self.model = YOLO(weights)

    def predict(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=0.4,          # higher threshold to remove noise
            verbose=False
        )

        parts = []

        for r in results:

            if r.boxes is None:
                continue

            names = r.names

            for box in r.boxes:

                conf = float(box.conf[0])

                # filter weak detections
                if conf < 0.5:
                    continue

                parts.append({
                    "class_id": int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": conf,
                    "bbox": box.xyxy[0].tolist()
                })

        return parts