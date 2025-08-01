import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import streamlit as st
from collections import deque, Counter
from pathlib import Path
import os
from dotenv import load_dotenv

# --- Dependencies from the Object Detection script ---
# Note: You must ensure these modules are available in your project structure.
# For this example, it's assumed 'EcoSortAI' has a 'process_frame' method.
from eco_sort.eco_sort_ai import EcoSortAI
from service.blynk_service import BlynkService

# --- Load Environment Variables ---
load_dotenv(verbose=True)


# --- Original Waste Classifier Class (Unchanged) ---
class WasteClassifier:
    def __init__(
        self,
        model_path: str,
        class_names=[
            "background",
            "glass",
            "metal",
            "organic",
            "paper",
            "plastic",
            "styrofoam",
            "textiles",
        ],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier[3] = nn.Linear(
            in_features=self.model.classifier[3].in_features,
            out_features=len(class_names),
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def predict(self, image_pil):
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            label = self.class_names[pred_idx] if confidence > 0.6 else "Tidak yakin"
        return label, confidence


# --- Helper function to find cameras ---
def get_available_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


# --- Main Streamlit App Class (Integrated) ---
class WasteRecognizerApp:
    def __init__(self):
        st.set_page_config(page_title="Waste Recognizer", layout="centered")

        # --- Models and Services Initialization ---
        self.classifier = WasteClassifier("models/MobileNetV3_best_model.pt")

        # For Object Detection Mode
        BLYNK_TOKEN = os.getenv("BLYNK_AUTH_TOKEN")
        if BLYNK_TOKEN:
            self.blynk_service = BlynkService(token=BLYNK_TOKEN)
        else:
            self.blynk_service = None
            # Show a warning once if the token is missing and detection is planned
            st.warning(
                "Blynk token not found. Object detection may not fully function.",
                icon="⚠️",
            )

        self.object_detection_model_path = Path("models/best.pt").absolute()

        # --- UI Setup ---
        self.available_cameras = get_available_cameras()
        if not self.available_cameras:
            st.error("Tidak ada kamera yang tersedia.")

        self.mode = st.radio(
            "Pilih Mode", ["Klasifikasi Gambar", "Klasifikasi Video", "Deteksi Objek"]
        )

        self.camera_index = (
            st.selectbox(
                "Pilih Kamera",
                options=self.available_cameras,
                format_func=lambda x: f"Kamera {x}",
            )
            if self.available_cameras
            else 0
        )

        self.prediction_history = deque(maxlen=20)
        self.stable_prediction = "Menganalisis..."

    def run(self):
        st.title("♻️ Waste Recognizer")
        if not self.available_cameras:
            return

        if self.mode == "Klasifikasi Video":
            self.video_classification_mode()
        elif self.mode == "Klasifikasi Gambar":
            self.image_classification_mode()
        elif self.mode == "Deteksi Objek":
            self.object_detection_mode()

    def video_classification_mode(self):
        st.header("Klasifikasi Sampah via Video")
        run_camera = st.toggle("Aktifkan Kamera untuk Klasifikasi")
        frame_window = st.image([])
        prediction_text = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(self.camera_index)
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Tidak bisa membaca frame dari kamera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")

                img_pil = Image.fromarray(frame_rgb)
                label, confidence = self.classifier.predict(img_pil)

                self.prediction_history.append(label)
                if len(self.prediction_history) == self.prediction_history.maxlen:
                    most_common = Counter(self.prediction_history).most_common(1)[0]
                    if most_common[1] > (self.prediction_history.maxlen / 2):
                        self.stable_prediction = most_common[0]

                prediction_text.markdown(
                    f"**Prediksi Stabil:** `{self.stable_prediction}`"
                )
            cap.release()
        else:
            st.info("Aktifkan kamera untuk mulai klasifikasi sampah.")

    def image_classification_mode(self):
        st.header("Klasifikasi Sampah via Gambar")
        uploaded_file = st.file_uploader(
            "Upload gambar sampah", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar diunggah", use_column_width=True)
            with st.spinner("Menganalisis..."):
                label, confidence = self.classifier.predict(image)
                if label == "Tidak yakin":
                    st.warning(
                        f"Tidak dapat menentukan jenis sampah dengan pasti (Keyakinan: {confidence:.2f})"
                    )
                else:
                    st.success(f"**Prediksi:** {label} (Keyakinan: {confidence:.2f})")

    def object_detection_mode(self):
        st.header("Deteksi Objek Sampah")
        run_camera = st.toggle("Aktifkan Kamera untuk Deteksi")
        frame_window = st.image([])

        if run_camera:
            # Initialize EcoSortAI here, only when the mode is active
            eco_ai = EcoSortAI(
                camera_source_index=self.camera_index,
                model_path=self.object_detection_model_path,
                blynk_service=self.blynk_service,
            )

            st.info("Kamera deteksi objek aktif. Menginisialisasi model...")
            cap = cv2.VideoCapture(self.camera_index)

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Tidak bisa membaca frame dari kamera.")
                    break

                annotated_frame = eco_ai.process_frame(frame)

                # Convert back to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_window.image(annotated_frame_rgb, channels="RGB")

            cap.release()
        else:
            st.info("Aktifkan kamera untuk mulai deteksi objek.")


if __name__ == "__main__":
    app = WasteRecognizerApp()
    app.run()
