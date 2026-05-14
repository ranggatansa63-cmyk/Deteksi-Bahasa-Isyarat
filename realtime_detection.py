import cv2
import numpy as np
import tensorflow as tf
import json

# LOAD MODEL 
model = tf.keras.models.load_model("model_bisindo.keras")

# LOAD LABEL 
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# PARAMETER 
IMG_SIZE = 128

# START WEBCAM 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip biar natural (mirror)
    frame = cv2.flip(frame, 1)

    # ROI (AREA TANGAN) 
    x1, y1 = 100, 100
    x2, y2 = 400, 400

    roi = frame[y1:y2, x1:x2]

    # PREPROCESS 
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # PREDICT 
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    label = labels[class_id]

    # TAMPILKAN 
    text = f"{label} ({confidence:.2f})"

    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Kotak ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  (255, 0, 0), 2)

    cv2.imshow("BISINDO Detection", frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()