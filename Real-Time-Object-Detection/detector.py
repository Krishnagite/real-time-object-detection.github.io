import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """
    Encapsulates the Ultralytics YOLOv8 object detection model,
    frame resizing for mobile streams, class/confidence filtering,
    and OpenCV bounding box annotation.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        # Extract the 80 COCO class names in order
        self.class_names = list(self.model.names.values())

        # Generate distinct, consistent BGR colors for each class
        np.random.seed(42)
        self.colors = {
            name: tuple(int(c) for c in np.random.randint(60, 255, size=3))
            for name in self.class_names
        }

    def process_frame(
        self,
        frame: np.ndarray,
        allowed_classes: list,
        conf_threshold: float = 0.70,
        resize_width: int = 640,
    ):
        """
        Runs YOLOv8 inference on a video frame, filters by allowed_classes
        and conf_threshold, draws bounding boxes, and returns:
        (annotated_frame, detections_list)
        """
        if frame is None or frame.size == 0:
            return frame, []

        # Optionally resize high-res mobile frames for smooth real-time FPS
        h, w = frame.shape[:2]
        if resize_width and w > resize_width:
            scale = resize_width / float(w)
            new_h = int(h * scale)
            annotated_frame = cv2.resize(
                frame, (resize_width, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            annotated_frame = frame.copy()

        detections_list = []

        try:
            # Run YOLOv8 inference with minimum confidence threshold
            results = self.model.predict(
                source=annotated_frame,
                conf=conf_threshold,
                verbose=False,
            )

            allowed_set = set(allowed_classes) if allowed_classes else set(self.class_names)

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    object_class = self.model.names.get(cls_id, "unknown")

                    # Filter by selected object classes and confidence threshold
                    if object_class not in allowed_set or conf < conf_threshold:
                        continue

                    # Extract top-left (x1, y1) and bottom-right (x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox_x = max(0, x1)
                    bbox_y = max(0, y1)
                    bbox_w = max(1, x2 - x1)
                    bbox_h = max(1, y2 - y1)

                    detections_list.append(
                        {
                            "object_class": object_class,
                            "confidence": round(conf, 4),
                            "bbox_x": bbox_x,
                            "bbox_y": bbox_y,
                            "bbox_w": bbox_w,
                            "bbox_h": bbox_h,
                        }
                    )

                    # Draw bounding box and label banner on the frame
                    color = self.colors.get(object_class, (0, 255, 0))
                    cv2.rectangle(
                        annotated_frame,
                        (bbox_x, bbox_y),
                        (bbox_x + bbox_w, bbox_y + bbox_h),
                        color,
                        2,
                    )

                    label = f"{object_class}: {conf * 100:.0f}%"
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    label_y1 = max(0, bbox_y - text_h - 10)
                    label_y2 = max(text_h + 10, bbox_y)

                    cv2.rectangle(
                        annotated_frame,
                        (bbox_x, label_y1),
                        (bbox_x + text_w + 8, label_y2),
                        color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (bbox_x + 4, label_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

        except Exception as e:
            print(f"[Detector Error] Frame processing failed: {e}")

        return annotated_frame, detections_list
