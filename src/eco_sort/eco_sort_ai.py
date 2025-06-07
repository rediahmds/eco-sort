import math
import traceback
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
    ):
        """
        Initialize the EcoSortCamera with a camera source index.

        :param camera_source_index: The index of the camera source.
        :param model_path: Path to model file
        :param blynk_service: Instance of BlynkService for IoT mode. Pass an instance to enable IoT mode.
        To disable IoT mode, pass `None` or just ignore this parameter.
        """

        self.camera_source_index = (
            camera_source_index.value
            if isinstance(camera_source_index, CameraIndex)
            else camera_source_index
        )
        self.model_path = model_path

        self.blynk_service = blynk_service
        self._iot_mode = blynk_service is not None

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
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(frame, (x1, y1, w, h))

                        conf = math.ceil((box.conf[0] * 100)) / 100

                        id = int(box.cls[0])
                        name = model.names[id]

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

                        if self._iot_mode:
                            self.blynk_service.updateDatastreamValue(
                                virtual_pin=BlynkPins.V0, value=name
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
            print(f"Program interrupted.")

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
