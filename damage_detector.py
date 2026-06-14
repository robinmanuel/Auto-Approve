from ultralytics import YOLO


class DamageDetector:

    def __init__(self, weights):
        self.model = YOLO(weights)

    def predict(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=0.30,
            verbose=False
        )

        damages = []

        for r in results:

            if r.boxes is None:
                continue

            names = r.names

            for box in r.boxes:

                damages.append({
                    "damage_type": names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return damages