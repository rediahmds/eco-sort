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
    def __init__(self, model_path="models/mobilenet_v2_best_model.pt", iot_mode=True):
        st.set_page_config(page_title="Waste Recognizer", layout="centered")

        self.classifier = EcoSortCNN(model_path, model_name="mobilenet_v2")

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

        # --- UI Setup ---
        self.available_cameras = CamUtility.get_available_cameras(5)
        if not self.available_cameras:
            st.error("Tidak ada kamera yang tersedia.")

        self.mode = st.radio("Pilih Mode", ["Klasifikasi Gambar", "Klasifikasi Video"])

        self.camera_index = (
            st.selectbox(
                "Pilih Kamera",
                options=self.available_cameras,
                format_func=lambda x: f"Kamera {x}",
            )
            if self.available_cameras
            else 0
        )

        self._DESIRED_FPS = 20
        self._FRAME_DELAY = 1.0 / self._DESIRED_FPS

        self.prediction_history = deque(maxlen=20)
        self._CONFIDENCE_THRESHOLD = 0.60
        self.stable_prediction: str = "Menganalisis..."

    def run(self):
        st.title("♻️ Waste Recognizer")
        if not self.available_cameras:
            return

        if self.mode == "Klasifikasi Video":
            self.video_classification_mode()
        elif self.mode == "Klasifikasi Gambar":
            self.image_classification_mode()

    def video_classification_mode(self):
        st.header("Klasifikasi Sampah via Video")
        run_camera = st.toggle("Aktifkan Kamera untuk Klasifikasi")
        frame_window = st.image([])
        prediction_text = st.empty()

        is_ready_to_sort = True
        avg_confidence = 0.0
        last_frame_time = time.time()
        valid_object_labels = [
            "glass",
            "metal",
            "organic",
            "paper",
            "plastic",
            "textiles",
            "styrofoam",
        ]

        if run_camera:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                st.error(f"Error: Tidak bisa membuka kamera {self.camera_index}")
                return

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Tidak bisa membaca frame dari kamera.")
                    break

                now = time.time()
                if now - last_frame_time < self._FRAME_DELAY:
                    continue
                last_frame_time = now

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")
                img_pil = Image.fromarray(frame_rgb)

                label, confidence = self.classifier.predict(img_pil)

                if confidence >= self._CONFIDENCE_THRESHOLD:
                    self.prediction_history.append((label, confidence))

                if len(self.prediction_history) == self.prediction_history.maxlen:
                    labels_only = [item[0] for item in self.prediction_history]
                    most_common = Counter(labels_only).most_common(1)[0]
                    stable_label = most_common[0]
                    stable_count = most_common[1]

                    if stable_count > (self.prediction_history.maxlen / 2):
                        self.stable_prediction = stable_label

                        confidences_for_stable_label = [
                            item[1]
                            for item in self.prediction_history
                            if item[0] == stable_label
                        ]
                        if confidences_for_stable_label:
                            avg_confidence = sum(confidences_for_stable_label) / len(
                                confidences_for_stable_label
                            )

                if (
                    self.blynk_service
                    and self.stable_prediction in valid_object_labels
                    and is_ready_to_sort
                ):
                    is_ready_to_sort = False
                    is_recyclable = self.classifier.decide_recyclability(
                        self.stable_prediction
                    )

                    st.toast(f"Menyortir {self.stable_prediction}...", icon="♻️")
                    print(
                        f"[AKSI] Menyortir: {self.stable_prediction}. Menunggu servo."
                    )

                    if is_recyclable:
                        self.blynk_service.updateDatastreamValue(
                            BlynkPins.V0, "Daur ulang"
                        )
                    else:
                        self.blynk_service.updateDatastreamValue(
                            BlynkPins.V0, "Organik"
                        )
                    self.blynk_service.waitServoReady()

                    time.sleep(3)
                    print("[INFO] Servo selesai. Mereset state.")

                    self.prediction_history.clear()
                    self.stable_prediction = "Menganalisis..."
                    avg_confidence = 0.0

                elif self.stable_prediction == "background" and not is_ready_to_sort:
                    print(
                        "[INFO] Area bersih. Sistem SIAP untuk penyortiran berikutnya."
                    )
                    is_ready_to_sort = True
                    self.prediction_history.clear()

                prediction_text.markdown(
                    f"##### Prediksi Stabil: `{self.stable_prediction} ({avg_confidence:.2f})`"
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
                label, confidence = self.classifier.predict(
                    image, confidence_threshold=0
                )
                if label == "Tidak yakin":
                    st.warning(
                        f"Tidak dapat menentukan jenis sampah dengan pasti (Keyakinan: {confidence:.2f})"
                    )
                else:
                    st.success(f"**Prediksi:** {label} (Keyakinan: {confidence:.2f})")
