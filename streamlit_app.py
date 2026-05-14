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

.result-box {
    background-color: #132850;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# DOWNLOAD MODEL
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
    '<div class="title">Deteksi Bahasa Isyarat</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deteksi Huruf BISINDO menggunakan Webcam</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================
# MAIN CARD
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)

# =========================================
# LAYOUT
# =========================================
col1, col2 = st.columns([2,1])

# =========================================
# WEBCAM
# =========================================
with col1:

    st.subheader("📷 Webcam")

    camera = st.camera_input(
        "Ambil gambar tangan"
    )

# =========================================
# HASIL
# =========================================
with col2:

    st.subheader("🖼 ROI Tangan")

    roi_placeholder = st.empty()

    st.subheader("📊 Hasil Deteksi")

    result_placeholder = st.empty()

# =========================================
# PROSES DETEKSI
# =========================================
if camera is not None:

    # baca gambar
    file_bytes = np.asarray(
        bytearray(camera.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    # mirror
    img = cv2.flip(img, 1)

    # =========================================
    # ROI
    # =========================================
    x1, y1 = 100, 100
    x2, y2 = 400, 400

    # gambar kotak ROI
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

    # crop ROI
    roi = img[y1:y2, x1:x2]

    # =========================================
    # PREPROCESS
    # =========================================
    resized = cv2.resize(
        roi,
        (IMG_SIZE, IMG_SIZE)
    )

    resized = resized / 255.0

    resized = np.expand_dims(
        resized,
        axis=0
    )

    # =========================================
    # PREDICT
    # =========================================
    preds = model.predict(
        resized,
        verbose=0
    )

    class_id = np.argmax(preds)

    confidence = np.max(preds)

    label = labels[class_id]

    # =========================================
    # TAMPILKAN WEBCAM
    # =========================================
    st.image(
        img,
        channels="BGR",
        caption="Hasil Webcam dengan ROI"
    )

    # =========================================
    # TAMPILKAN ROI
    # =========================================
    roi_placeholder.image(
        roi,
        channels="BGR",
        caption="ROI Tangan"
    )

    # =========================================
    # HASIL DETEKSI
    # =========================================
    result_placeholder.markdown(f"""
    <div class="result-box">

    <h2 style="color:white;">
    Huruf : {label}
    </h2>

    <h3 style="color:#B8C7E0;">
    Confidence : {confidence*100:.2f}%
    </h3>

    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)