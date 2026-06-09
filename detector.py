class CarPartDetector:

    def __init__(self, weights):
        self.weights = weights
        self.model = None   # not loaded yet

    def _load_model(self):
        from ultralytics import YOLO
        if self.model is None:
            self.model = YOLO(self.weights)

    def predict(self, image_path):

        self._load_model()   # 🔥 only loads when needed

        results = self.model.predict(
            source=image_path,
            conf=0.45,
            verbose=False
        )

        parts = []

        for r in results:
            if r.boxes is None:
                continue

            names = r.names

            for box in r.boxes:

                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                parts.append({
                    "class_id": int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": conf,
                    "bbox": box.xyxy[0].tolist()
                })

        return parts