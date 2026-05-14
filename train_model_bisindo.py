import os
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


# PARAMETER 
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

DATASET_PATH = "D:/SKRIPSI/Project/dataset_bisindo1"

train_dir = os.path.join(DATASET_PATH, "train")
val_dir   = os.path.join(DATASET_PATH, "val")
test_dir  = os.path.join(DATASET_PATH, "test")

# AUGMENTASI 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# LOAD DATA
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# CLASS WEIGHT 
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

# MODEL 
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

# FINE TUNING 
for layer in base_model.layers[-10:]:
    layer.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

# CLASSIFIER
x = base_model.output
x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# COMPILE 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# CALLBACK 
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "model_bisindo.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# TRAIN 
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# GRAFIK
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# SAVE MODEL
model.save("model_bisindo_fixed.h5")
print("Model berhasil disimpan!")

# SAVE LABEL 
import json

with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("Label berhasil disimpan!")

# EVALUASI 
print("\n===== CLASSIFICATION REPORT =====")

test_gen.reset()
preds = model.predict(test_gen)

y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

labels = list(test_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))

# CONFUSION MATRIX 
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===================== PROSES PREDIKSI + TRANSLATE =====================
print("\n" + "="*50)
print("PROSES PREDIKSI DAN TRANSLATE")
print("="*50)

from tensorflow.keras.preprocessing import image
import random

# Ambil label kelas
class_labels = list(train_gen.class_indices.keys())

# Ambil 1 gambar dari setiap kelas
sample_images = []

for label in class_labels:
    class_folder = os.path.join(test_dir, label)
    if os.path.exists(class_folder):
        files = os.listdir(class_folder)
        if len(files) > 0:
            chosen = random.choice(files)
            sample_images.append(os.path.join(class_folder, chosen))

# Tampilkan hasil
plt.figure(figsize=(15, 4))

for i, img_path in enumerate(sample_images):
    # Load gambar
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0   # karena kamu pakai rescale
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array, verbose=0)[0]

    print("\n" + "-"*50)
    print(f"Gambar: {os.path.basename(img_path)}")

    print("Probabilitas tiap kelas:")
    for j, label in enumerate(class_labels):
        print(f"{label}: {pred[j]:.4f}")

    # Ambil nilai tertinggi
    pred_index = np.argmax(pred)
    pred_label = class_labels[pred_index]
    confidence = pred[pred_index] * 100

    print(f"\nIndex tertinggi (argmax): {pred_index}")
    print(f"Hasil translate: {pred_label}")
    print(f"Confidence: {confidence:.2f}%")

    # Label asli
    true_label = os.path.basename(os.path.dirname(img_path))

    # Warna (benar/salah)
    color = 'green' if pred_label == true_label else 'red'

    # Tampilkan gambar
    plt.subplot(1, len(sample_images), i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.2f}%)",
              color=color)
    plt.axis("off")

plt.suptitle("Proses Prediksi dan Translasi Output Model")
plt.tight_layout()
plt.show()