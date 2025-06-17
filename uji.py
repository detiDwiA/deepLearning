import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, accuracy_score

# =====================
# Model yang Digunakan
# =====================
model_path = 'model_batik_densenet.h5'

# =====================
# Data Uji
# =====================
uji_dir = 'uji'
uji_images = [img for img in os.listdir(uji_dir) if img.endswith('.jpg')]

# Mapping nama file ke label jawaban
label_mapping = {
    'uji_1.jpg': 'batik-bali',
    'uji_2.jpg': 'batik-betawi',
    'uji_3.jpg': 'batik-celup',
    'uji_4.jpg': 'batik-cendrawasih',
    'uji_5.jpg': 'batik-ceplok',
    'uji_6.jpg': 'batik-ciamis',
    'uji_7.jpg': 'batik-garutan',
    'uji_8.jpg': 'batik-gentongan',
    'uji_9.jpg': 'batik-kawung',
    'uji_10.jpg': 'batik-keraton',
    'uji_11.jpg': 'batik-lasem',
    'uji_12.jpg': 'batik-megamendung',
    'uji_13.jpg': 'batik-parang',
    'uji_14.jpg': 'batik-pekalongan',
    'uji_15.jpg': 'batik-priangan',
    'uji_16.jpg': 'batik-sekar',
    'uji_17.jpg': 'batik-sidoluhur',
    'uji_18.jpg': 'batik-sidomukti',
    'uji_19.jpg': 'batik-sogan',
    'uji_20.jpg': 'batik-tambal'
}

# =====================
# Ambil Label dari Dataset
# =====================
labels = sorted(os.listdir('splittt_dataset/train'))

# =====================
# Fungsi Preprocessing
# =====================
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# =====================
# Evaluasi Model
# =====================
print(f"\nEvaluasi Model: {model_path}")
model = load_model(model_path)

y_true = []
y_pred = []
error_files = []

for img_name in uji_images:
    img_path = os.path.join(uji_dir, img_name)

    # Abaikan file yang tidak ada di label_mapping
    if img_name not in label_mapping:
        print(f"{img_name} tidak ada dalam mapping label, dilewati.")
        continue

    try:
        img_array = load_and_preprocess(img_path)

        pred = model.predict(img_array, verbose=0)
        pred_label_index = np.argmax(pred)
        pred_label = labels[pred_label_index]

        y_true.append(label_mapping[img_name])
        y_pred.append(pred_label)

        print(f"{img_name} --> Prediksi: {pred_label} | Jawaban: {label_mapping[img_name]}")

    except Exception as e:
        print(f"Error pada file {img_name}: {e}")
        error_files.append(img_name)

# Hitung Akurasi
if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAkurasi Model {model_path}: {acc * 100:.2f}%")
else:
    print("\nTidak ada gambar yang berhasil diproses.")

# Plot Confusion Matrix
if len(y_true) > 0:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_path}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Simpan File yang Error
if error_files:
    with open('error_files.txt', 'w') as f:
        for error_file in error_files:
            f.write(f'{error_file}\n')
    print('Daftar gambar error disimpan di error_files.txt')
