import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Baca gambar dalam grayscale
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    features = [np.mean(img), np.std(img)]
                    images.append(features)
                    labels.append(label_folder)  # Nama folder sebagai label
    return np.array(images), np.array(labels)

# Fungsi untuk memproses gambar tunggal dan memprediksi pemilik tanda tangan
def predict_signature(image_path, model, scaler, label_encoder):
    # Baca gambar dalam grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"File gambar tidak ditemukan di: {image_path}")
    
    # Resize gambar agar sesuai dengan ukuran training (128x128)
    img = cv2.resize(img, (128, 128))
    
    # Ekstrak fitur (rata-rata intensitas piksel dan standar deviasi)
    features = [np.mean(img), np.std(img)]
    features = np.array(features).reshape(1, -1)  # Ubah ke bentuk 2D untuk input model
    
    # Normalisasi fitur menggunakan scaler
    features_scaled = scaler.transform(features)
    
    # Prediksi menggunakan model
    prediction = model.predict(features_scaled)
    label = label_encoder.inverse_transform(prediction)  # Konversi label numerik ke label asli
    
    return label[0]  # Kembalikan label pemilik tanda tangan

# Path ke folder dataset
data_folder = "TTD_A/TTD"

# Load data gambar
X, y = load_images_from_folder(data_folder)

# Encode label menjadi numerik jika perlu
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training model
rf_model.fit(X_train, y_train)

# Testing model
test_predictions = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Akurasi testing:", test_accuracy)
print("Laporan Klasifikasi:\n", classification_report(y_test, test_predictions))

# Validasi model dengan confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)

# Visualisasi confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Grafik Akurasi
accuracy_scores = [test_accuracy]  # Untuk menambahkan akurasi testing ke dalam grafik

plt.figure(figsize=(8, 5))
plt.bar(['Akurasi'], accuracy_scores, color='skyblue')
plt.ylim(0, 1)
plt.ylabel('Akurasi')
plt.title('Tingkat Akurasi Model')
plt.show()

# Contoh penggunaan untuk prediksi gambar baru
# Path ke gambar yang ingin diprediksi
image_path = "Abi\aug_0.png"

# Prediksi tanda tangan
try:
    owner = predict_signature(image_path, rf_model, scaler, label_encoder)
    print(f"Tanda tangan pada gambar '{image_path}' adalah milik: {owner}")
except ValueError as e:
    print(e)