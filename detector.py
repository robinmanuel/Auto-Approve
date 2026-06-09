class CarPartDetector:

    def __init__(self, weights):
        self.weights = weights
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)

    def predict(self, image_path):

        self._load_model()

        results = self.model.predict(
            source=image_path,
            conf=0.25,
            verbose=False
        )

        parts = []

        for r in results:
            if r.boxes is None:
                continue

            names = r.names

            for box in r.boxes:
                parts.append({
                    "class_id": int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return parts