from eco_sort_ai import EcoSortAI

# from static.camera_index import
from pathlib import Path

model_path = Path("models/yolov5n.pt").absolute()

eco_ai = EcoSortAI(camera_source_index=0, model_path=model_path)
eco_ai.start_capture()
