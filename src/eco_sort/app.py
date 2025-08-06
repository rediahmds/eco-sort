import cv2
from PIL import Image
import streamlit as st
from collections import deque, Counter
from pathlib import Path
import os
from dotenv import load_dotenv
import time
from eco_sort.eco_sort_ai import EcoSortAI
from service.blynk_service import BlynkService, BlynkPins
from eco_sort.eco_sort_cnn import EcoSortCNN
from static.cam.cam_util import CamUtility

load_dotenv(verbose=True)


class App:
    def __init__(self, model_path="models/MobileNetV3_best_model.pt", iot_mode=True):
        st.set_page_config(page_title="Waste Recognizer", layout="centered")

        self.classifier = EcoSortCNN(model_path)

        # For Object Detection Mode
        BLYNK_TOKEN = os.getenv("BLYNK_AUTH_TOKEN")
        if BLYNK_TOKEN and iot_mode:
            self.blynk_service = BlynkService(token=BLYNK_TOKEN)
            self.BLYNK_SEND_INTERVAL = 3
        else:
            self.blynk_service = None
            st.warning(
                "Blynk token not found. Object detection may not fully function.",
                icon="⚠️",
            )

        if not iot_mode:
            self.blynk_service = None

        self.object_detection_model_path = Path("models/best.pt").absolute()

        # --- UI Setup ---
        self.available_cameras = CamUtility.get_available_cameras(5)
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

        self._DESIRED_FPS = 10
        self._FRAME_DELAY = 1.0 / self._DESIRED_FPS

        self.prediction_history = deque(maxlen=20)
        self._CONFIDENCE_THRESHOLD = 0.75
        self.stable_prediction: str = "Menganalisis..."

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

        last_label = None
        last_send_time = 0
        last_frame_time = time.time()
        if run_camera:
            cap = cv2.VideoCapture(self.camera_index)
            while run_camera:
                ret, frame = cap.read()

                if not ret:
                    st.warning("Tidak bisa membaca frame dari kamera.")
                    break

                now = time.time()
                if now - last_frame_time > self._FRAME_DELAY:
                    last_frame_time = now

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB")

                    img_pil = Image.fromarray(frame_rgb)
                    label, confidence = self.classifier.predict(img_pil)

                    if confidence >= self._CONFIDENCE_THRESHOLD:
                        self.prediction_history.append(label)

                    if len(self.prediction_history) == self.prediction_history.maxlen:
                        most_common = Counter(self.prediction_history).most_common(1)[0]
                        if most_common[1] > (self.prediction_history.maxlen / 2):
                            self.stable_prediction = most_common[0]

                            if (
                                self.blynk_service
                                and self.stable_prediction != "background"
                            ):
                                now = time.time()
                                is_label_different = (
                                    self.stable_prediction != last_label
                                )
                                is_interval_passed = (
                                    now - last_send_time >= self.BLYNK_SEND_INTERVAL
                                )

                                if (
                                    last_label is None
                                    or is_label_different
                                    or is_interval_passed
                                ):
                                    last_label = self.stable_prediction
                                    last_send_time = now
                                    is_recyclable = (
                                        self.classifier.decide_recyclability(last_label)
                                    )
                                    st.toast("Menyortir...", icon=":material/cycle:")
                                    if is_recyclable:
                                        self.blynk_service.updateDatastreamValue(
                                            virtual_pin=BlynkPins.V0,
                                            value="Daur ulang",
                                        )
                                    else:
                                        self.blynk_service.updateDatastreamValue(
                                            virtual_pin=BlynkPins.V0,
                                            value="Organik",
                                        )

                                    self.blynk_service.waitServoReady()

                    prediction_text.markdown(
                        f"##### Prediksi Stabil: `{self.stable_prediction} ({confidence})`"
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
