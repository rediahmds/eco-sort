from static.camera_index import CameraIndex
import cv2


class EcoSortAI:
    def __init__(self, camera_source_index: CameraIndex | int = 0):
        """
        Initialize the EcoSortCamera with a camera source index.

        :param camera_source_index: The index of the camera source.
        """
        self.camera_source_index = camera_source_index
        self.capture = cv2.VideoCapture(camera_source_index)
        self.is_opened = self.capture.isOpened()

    def start_capture(self):
        """
        Start capturing video from the camera.
        """
        try:
            while self.is_opened:
                is_success, frame = self.capture.read()
                if not is_success:
                    raise Exception("Failed to read frame from camera.")
                
                cv2.imshow("EcoSortAI Camera Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            self.release()

    def release(self):
        """
        Release the camera capture.
        """
        if self.capture.isOpened():
            self.capture.release()
            cv2.destroyAllWindows()