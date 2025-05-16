import math
from static.camera_index import CameraIndex
import cv2
from pathlib import Path
from ultralytics import YOLO
import cvzone


class EcoSortAI:
    def __init__(
        self,
        camera_source_index: CameraIndex | int = CameraIndex.BUILT_IN,
        model_path: str | Path = "/models/yolov8m.pt",
    ):
        """
        Initialize the EcoSortCamera with a camera source index.

        :param camera_source_index: The index of the camera source.
        """
        self.camera_source_index = (
            camera_source_index.value
            if isinstance(camera_source_index, CameraIndex)
            else camera_source_index
        )

        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        self.model_path = model_path

    def start_capture(self):
        """
        Start capturing video from the camera.
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_source_index)
            self.is_opened = self.capture.isOpened()
            model = YOLO(model=self.model_path, verbose=True)

            fps_counter = cv2.TickMeter()
            fps_counter.start()

            while self.is_opened:
                is_success, frame = self.capture.read()
                if not is_success:
                    raise Exception("Failed to read frame from camera.")

                fps_counter.start()

                results = model(source=frame, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(frame, (x1, y1, w, h))

                        conf = math.ceil((box.conf[0] * 100)) / 100

                        cls = box.cls[0]
                        name = self.class_names[int(cls)]

                        cvzone.putTextRect(
                            img=frame,
                            text=f"{name} " f"{conf}",
                            pos=(max(0, x1), max(35, y1)),
                            thickness=1,
                            scale=1,
                            colorB=(183, 243, 157),
                            colorR=(183, 243, 157),
                            colorT=(18, 24, 9),
                        )

                fps_counter.stop()
                fps = fps_counter.getFPS()
                fps_counter.reset()

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 255, 0),
                    2,
                )

                cv2.imshow("EcoSortAI Camera Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.release()

        except KeyboardInterrupt as ki:
            print(f"Program interrupted.")

        except Exception as e:
            print(f"Error: {e}")

    def release(self):
        """
        Release the camera capture.
        """
        if self.capture.isOpened():
            self.capture.release()
            cv2.destroyAllWindows()
