import cv2


class CamUtility:
    @staticmethod
    def get_available_cameras(max_index=5):
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
