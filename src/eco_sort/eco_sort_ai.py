import math
import time
import cv2
from pathlib import Path
from ultralytics import YOLO
import cvzone

# Assuming these are in your project structure, e.g., in a 'service' folder
from service.blynk_service import BlynkService, BlynkPins

# Assuming this is in a 'static' folder
from static.cam.camera_index import CameraIndex
from service.servo_state import ServoState


class EcoSortAI:
    def __init__(
        self,
        blynk_service: BlynkService | None = None,
        camera_source_index: CameraIndex | int = CameraIndex.BUILT_IN,
        model_path: str | Path = "models/best.pt",
        *,
        confidence_threshold: float = 0.6,
        send_interval: float = 2.5,
    ):
        """
        Initializes the EcoSortAI object detector.

        Args:
            blynk_service: Instance of BlynkService for IoT mode.
            camera_source_index: The index of the camera source.
            model_path: Path to the YOLOv8 model file.
            confidence_threshold: Minimum confidence to consider a detection valid.
            send_interval: Interval in seconds to send data to Blynk.
        """
        self.blynk_service = blynk_service
        self._iot_mode = blynk_service is not None
        self._confidence_threshold = confidence_threshold
        self._send_interval = send_interval

        # Internal state for IoT communication logic
        self._last_label = None
        self._last_sent_time = 0
        self._last_blynk_message_display_time = 0

        # 🧠 Load the model once during initialization for efficiency
        self.model = YOLO(model=model_path)
        self.class_names = self.model.names

    def process_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Processes a single video frame to detect and annotate objects.

        This method is designed to be called in a loop by an external
        application like Streamlit.

        Args:
            frame: A single frame captured from cv2.VideoCapture.

        Returns:
            The frame with bounding boxes and labels drawn on it.
        """
        results = self.model(source=frame, stream=False, verbose=False)

        # The result from a single image is the first element of the list
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # --- Bounding Box Drawing ---
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(
                    frame, (x1, y1, w, h), l=15, rt=2, colorR=(255, 100, 50)
                )

                # --- Confidence and Class Name ---
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf < self._confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                name = self.class_names[class_id]

                self._drawBoundingBox(frame=frame, name=name, conf=conf, x1=x1, y1=y1)

                # --- IoT Logic (if enabled) ---
                if self._iot_mode and self.blynk_service:
                    self._handle_blynk_communication(name)
                    self._wait_servo()

        # Display "Sending data" message if needed
        if self._iot_mode and (
            time.time() - self._last_blynk_message_display_time < 1.0
        ):
            self._draw_blynk_status(frame)

        return frame

    def _handle_blynk_communication(self, label: str):
        """Sends data to Blynk based on detection and time interval."""
        now = time.time()
        is_label_different = label != self._last_label
        is_interval_passed = (now - self._last_sent_time) > self._send_interval

        if self._last_label is None or is_label_different or is_interval_passed:
            self.blynk_service.updateDatastreamValue(
                virtual_pin=BlynkPins.V0, value=label
            )
            self._last_label = label
            self._last_sent_time = now
            self._last_blynk_message_display_time = now

    def _wait_servo(self):
        servo_state = self.blynk_service.getDatastreamValue(BlynkPins.V7)
        is_servo_ready = ServoState.is_ready(servo_state)

        while not is_servo_ready:
            servo_state = self.blynk_service.getDatastreamValue(BlynkPins.V7)
            is_servo_ready = ServoState.is_ready(servo_state)

    def _drawBoundingBox(self, *, frame, name: str, conf: float, x1: int, y1: int):
        """Draws the label and confidence score above the bounding box."""
        cvzone.putTextRect(
            img=frame,
            text=f"{name} {conf}",
            pos=(max(0, x1), max(35, y1 - 10)),  # Positioned slightly above the box
            thickness=2,
            scale=1.5,
            colorB=(255, 100, 50),
            colorR=(255, 100, 50),
            colorT=(255, 255, 255),
        )

    def _draw_blynk_status(self, frame):
        """Draws 'Sending data to Blynk' on the frame."""
        h, w, _ = frame.shape
        cv2.putText(
            frame,
            "Sending data to Blynk...",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 100),
            2,
        )
