from dotenv import load_dotenv

load_dotenv(verbose=True)

from eco_sort.eco_sort_ai import EcoSortAI
from static.camera_index import CameraIndex
from service.blynk_service import BlynkService


from pathlib import Path
import os


if __name__ == "__main__":
    model_path = Path("models/best.pt").absolute()

    BLYNK_TOKEN = os.getenv("BLYNK_AUTH_TOKEN")
    blynk_service = BlynkService(token=BLYNK_TOKEN)

    eco_ai = EcoSortAI(
        camera_source_index=CameraIndex.BUILT_IN,
        model_path=model_path,
        blynk_service=blynk_service,
    )
    eco_ai.start_capture()
