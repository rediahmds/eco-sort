from eco_sort_ai import EcoSortAI
from ultralytics import YOLO
from static.camera_index import CameraIndex

model = YOLO(model="../model/yolov8m.pt")

eco_ai = EcoSortAI(camera_source_index=2)
eco_ai.start_capture()
