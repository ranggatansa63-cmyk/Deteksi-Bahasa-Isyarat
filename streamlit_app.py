import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
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

camera = st.camera_input("📷 Ambil gambar tangan")

if camera is not None:

    # baca gambar
    file_bytes = np.asarray(
        bytearray(camera.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    # mirror
    img = cv2.flip(img, 1)

    # resize untuk model
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    resized = resized / 255.0
    resized = np.expand_dims(resized, axis=0)

    # prediksi
    preds = model.predict(resized, verbose=0)

    class_id = np.argmax(preds)
    confidence = np.max(preds)

    label = labels[class_id]

    # tampilkan hasil
    st.image(img, channels="BGR")

    st.success(f"Huruf Terdeteksi : {label}")

    st.info(f"Confidence : {confidence*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)
