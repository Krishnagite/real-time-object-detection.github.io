import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.class_names = list(self.model.names.values())
        np.random.seed(42)
        self.colors = {
            name: tuple(int(c) for c in np.random.randint(60, 255, size=3))
            for name in self.class_names
        }

    def process_frame(self, frame, allowed_classes, conf_threshold: float = 0.70):
        if frame is None or frame.size == 0:
            return frame, []

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / float(w)
            annotated = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            annotated = frame.copy()

        detections = []
        results = self.model.predict(source=annotated, conf=conf_threshold, verbose=False)
        allowed_set = set(allowed_classes) if allowed_classes else set(self.class_names)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                obj_cls = self.model.names.get(cls_id, "unknown")

                if obj_cls not in allowed_set or conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bx, by = max(0, x1), max(0, y1)
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)

                detections.append(
                    {
                        "object_class": obj_cls,
                        "confidence": round(conf, 4),
                        "bbox_x": bx,
                        "bbox_y": by,
                        "bbox_w": bw,
                        "bbox_h": bh,
                    }
                )

                color = self.colors.get(obj_cls, (0, 255, 0))
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
                label = f"{obj_cls}: {conf * 100:.0f}%"
                cv2.putText(
                    annotated,
                    label,
                    (bx, max(18, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        return annotated, detections
