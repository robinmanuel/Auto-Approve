from ultralytics import YOLO


class CarPartDetector:

    def __init__(self, weights):
        self.model = YOLO(weights)

    def predict(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=0.25,
            verbose=False
        )

        parts = []

        for r in results:

            boxes = r.boxes
            names = r.names

            if boxes is None:
                continue

            for i in range(len(boxes)):

                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()

                parts.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "bbox": list(map(float, bbox))
                })

        return parts