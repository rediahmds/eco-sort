from eco_sort.eco_sort_ai import EcoSortAI
from static.camera_index import CameraIndex


from pathlib import Path


if __name__ == "__main__":
    model_path = Path("models/yolov5nu.pt").absolute()

    eco_ai = EcoSortAI(camera_source_index=CameraIndex.BUILT_IN, model_path=model_path)
    eco_ai.start_capture()
