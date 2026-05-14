import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import os
import gdown

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Deteksi Bahasa Isyarat",
    layout="wide"
)

# =========================================
# CSS
# =========================================
st.markdown("""
<style>

body {
    background-color: #06142E;
}

.main {
    background: linear-gradient(to right, #06142E, #0B1F4D);
    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.title {
    font-size: 48px;
    font-weight: bold;
    color: white;
}

.subtitle {
    font-size: 22px;
    color: #B8C7E0;
}

.card {
    background-color: #0B1736;
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# =========================================
MODEL_PATH = "model_bisindo_fixed.h5"

if not os.path.exists(MODEL_PATH):

    file_id = "1vAGRAqIy8lHRttvQS0uRHPxSFZIjxyXb"

    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Mengunduh model AI..."):
        gdown.download(
            url,
            MODEL_PATH,
            quiet=False
        )

# =========================================
# LOAD MODEL
# =========================================

#model = tf.keras.models.load_model(
 #   "model_bisindo_fixed.h5",
  #  compile=False
#)

MODEL_PATH = "model_bisindo_fixed.h5"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):

    file_id = "1vAGRAqIy8lHRttvQS0uRHPxSFZIjxyXb"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

# Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# =========================================
# LOAD LABEL
# =========================================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

IMG_SIZE = 128

# =========================================
# VIDEO TRANSFORMER
# =========================================
class SignDetector(VideoTransformerBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # MIRROR
        img = cv2.flip(img, 1)

        # ROI
        x1, y1 = 100, 100
        x2, y2 = 400, 400

        # DRAW ROI
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            3
        )

        cv2.putText(
            img,
            "ROI TANGAN",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # ROI IMAGE
        roi = img[y1:y2, x1:x2]

        # PREPROCESS
        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        resized = resized / 255.0
        resized = np.expand_dims(resized, axis=0)

        # PREDICT
        preds = model.predict(resized, verbose=0)

        class_id = np.argmax(preds)
        confidence = np.max(preds)

        label = labels[class_id]

        # =========================================
        # RESULT TEXT
        # =========================================
        cv2.putText(
            img,
            f"Huruf : {label}",
            (30, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        cv2.putText(
            img,
            f"Confidence : {confidence*100:.2f}%",
            (30, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================================
# HEADER
# =========================================
st.markdown(
    '<div class="title"> Deteksi Bahasa Isyarat</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-time Detection menggunakan Webcam</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================
# WEBCAM CARD
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📷 Live Webcam")

webrtc_streamer(
    key="sign-detection",
    video_transformer_factory=SignDetector,
    media_stream_constraints={
        "video": {
            "width": 640,
            "height": 480
        },
        "audio": False
    },
    video_html_attrs={
        "style": {
            "width": "700px",
            "margin": "0 auto",
            "display": "block",
            "border-radius": "15px"
        },
        "autoPlay": True,
        "controls": False
    },
    async_processing=True
)

st.markdown('</div>', unsafe_allow_html=True)
