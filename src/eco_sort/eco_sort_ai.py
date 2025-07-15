import math
import traceback
import time

from static.camera_index import CameraIndex
import cv2
from pathlib import Path
from ultralytics import YOLO
import cvzone
from service.blynk_service import BlynkService, BlynkPins


class EcoSortAI:
    def __init__(
        self,
        blynk_service: BlynkService | None = None,
        camera_source_index: CameraIndex | int = CameraIndex.BUILT_IN,
        model_path: str | Path = "/models/yolov8m.pt",
        *,
        confidence_threshold: float = 0.6,
        send_interval: float = 2.5,
    ):
        """
        Initialize the EcoSortCamera with a camera source index.

        :param camera_source_index: The index of the camera source.
        :param model_path: Path to model file. Default is `./models/yolov8m.pt`
        :param blynk_service: Instance of BlynkService for IoT mode. Pass an instance to enable IoT mode.
        To disable IoT mode, pass `None` or just ignore this parameter.
        :param confidence_threshold: Threshold to send inference result to blynk platform. Default to 0.6.
        :param send_interval: Interval to send data to blynk in terms of seconds. Default to 2.5 second
        """

        self.camera_source_index = (
            camera_source_index.value
            if isinstance(camera_source_index, CameraIndex)
            else camera_source_index
        )
        self.model_path = model_path
        self.blynk_service = blynk_service

        self._iot_mode = blynk_service is not None
        self._confidence_threshold = confidence_threshold
        self._last_label = None
        self._last_sent_time = 0
        self._send_interval = send_interval
        self._last_blynk_message_display_time = 0

    def start_capture(self):
        """
        Start capturing video from the camera.
        """
        WINDOW_NAME = "EcoSortAI Camera Feed"
        try:
            self.capture = cv2.VideoCapture(self.camera_source_index)
            self.is_opened = self.capture.isOpened()
            model = YOLO(model=self.model_path, verbose=True)

            fps_counter = cv2.TickMeter()

            cv2.namedWindow(winname=WINDOW_NAME, flags=cv2.WINDOW_FULLSCREEN)

            while self.is_opened:
                is_success, frame = self.capture.read()
                if not is_success:
                    raise Exception("Failed to read frame from camera.")

                fps_counter.start()

                results = model(source=frame, stream=True)
                for r in results:
                    print("=== RESULT ===", r, "=== RESULT ===", sep="\n")
                    boxes = r.boxes
                    for box in boxes:
                        print("=== BOX ===", box, "=== BOX ===", sep="\n")
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(frame, (x1, y1, w, h))

                        conf = math.ceil((box.conf[0] * 100)) / 100
                        if conf >= self._confidence_threshold:
                            id = int(box.cls[0])
                            name = model.names[id]

                            self._drawBoundingBox(
                                frame=frame, name=name, conf=conf, x1=x1, y1=y1
                            )

                            if self._iot_mode:
                                now = time.time()
                                is_label_different = name != self._last_label
                                is_should_send = (
                                    now - self._last_sent_time > self._send_interval
                                )

                                if (
                                    self._last_label is None  # first run
                                    or is_label_different
                                    or is_should_send
                                ):
                                    self.blynk_service.updateDatastreamValue(
                                        virtual_pin=BlynkPins.V0, value=name
                                    )
                                    self._last_label = name
                                    self._last_sent_time = now

                            if self._iot_mode and (
                                time.time() - self._last_blynk_message_display_time
                                > self._send_interval
                            ):
                                height = frame.shape[0]
                                cv2.putText(
                                    frame,
                                    "Sending data to Blynk",
                                    (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 200, 100),
                                    2,
                                )

                fps_counter.stop()
                fps = fps_counter.getFPS()
                fps_counter.reset()

                self._drawFPScounter(frame, fps)

                cv2.imshow(WINDOW_NAME, frame)

                isEscPressed = cv2.waitKey(1) == 27  # wait for ESC button
                isWindowVisible = cv2.getWindowProperty(
                    winname=WINDOW_NAME, prop_id=cv2.WND_PROP_VISIBLE
                )
                isCloseWindow = isWindowVisible < 1
                if isEscPressed or isCloseWindow:
                    self.release()
                    break

        except KeyboardInterrupt as ki:
            print(f"\nProgram interrupted.")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

        finally:
            print("Program ended.")

    def release(self):
        """
        Release the camera capture.
        """
        if self.capture.isOpened():
            self.capture.release()
            cv2.destroyAllWindows()

    def _drawBoundingBox(self, *, frame, name: str, conf: float, x1: int, y1: int):
        cvzone.putTextRect(
            img=frame,
            text=f"{name} {conf}",
            pos=(max(0, x1), max(35, y1)),
            thickness=1,
            scale=1,
            colorB=(183, 243, 157),
            colorR=(183, 243, 157),
            colorT=(18, 24, 9),
        )

    def _drawFPScounter(self, frame, fps: float):
        cv2.putText(
            img=frame,
            text=f"FPS: {fps:.1f}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(100, 255, 0),
            thickness=2,
        )
