import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ===================
# SETUP APLIKASI
# ===================

st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10", layout="centered")
st.title("🎯 Prediksi Gambar CIFAR-10 dengan CNN")

# Sidebar Navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi", "Info Model", "Grafik Akurasi/Loss", "Uji Model"])

# Label kelas CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

label_info = {
    'airplane': 'Pesawat terbang di udara',
    'automobile': 'Mobil atau kendaraan roda empat',
    'bird': 'Burung, hewan bersayap yang bisa terbang',
    'cat': 'Kucing peliharaan atau liar',
    'deer': 'Rusa dengan tanduk dan hidup di hutan',
    'dog': 'Anjing, sahabat manusia',
    'frog': 'Katak yang hidup di air dan darat',
    'horse': 'Kuda, hewan berkaki empat',
    'ship': 'Kapal yang berlayar di laut',
    'truck': 'Truk pengangkut barang'
}

# ===================
# LOAD MODEL
# ===================
def load_model():
    model_path = "cnn_cifar10_model.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan!")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ====================
# HALAMAN: Prediksi
# ====================
if menu == "Prediksi":
    st.subheader("🖼️ Upload Gambar untuk Diprediksi")
    uploaded_file = st.file_uploader("Pilih file gambar (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diupload', use_container_width=True)

        # Preprocessing
        image_resized = image.resize((32, 32))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_name = class_names[predicted_class]
        confidence = predictions[0][predicted_class]

        st.success(f"Hasil Prediksi: **{class_name.upper()}**")
        st.write(f"Confidence Score: **{confidence:.2f}**")
        st.markdown(f"📌 **Deskripsi Label:** {label_info[class_name]}")

        # Grafik probabilitas
        st.subheader("📊 Probabilitas Tiap Kelas:")
        fig, ax = plt.subplots()
        ax.bar(class_names, predictions[0])
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ====================
# HALAMAN: Info Model
# ====================
elif menu == "Info Model":
    st.subheader("📄 Informasi Model CNN")
    st.markdown("""
    - Model: CNN (3 Conv layers, BatchNorm, Dropout)
    - Optimizer: Adam
    - Loss: Categorical Crossentropy
    - Akurasi Validasi Terakhir: **~84%**
    - Dataset: CIFAR-10
    - Input: Gambar ukuran 32x32 piksel
    """)

# ====================
# HALAMAN: Grafik Akurasi/Loss
# ====================
elif menu == "Grafik Akurasi/Loss":
    st.subheader("📈 Grafik Training Model")
    try:
        history = np.load("history_cifar10.npy", allow_pickle=True).item()
        acc = history["accuracy"]
        val_acc = history["val_accuracy"]
        loss = history["loss"]
        val_loss = history["val_loss"]

        # Akurasi
        st.write("**Grafik Akurasi:**")
        fig1, ax1 = plt.subplots()
        ax1.plot(acc, label="Train Acc")
        ax1.plot(val_acc, label="Val Acc")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Akurasi")
        ax1.legend()
        st.pyplot(fig1)

        # Loss
        st.write("**Grafik Loss:**")
        fig2, ax2 = plt.subplots()
        ax2.plot(loss, label="Train Loss")
        ax2.plot(val_loss, label="Val Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("❌ File `history_cifar10.npy` tidak ditemukan.")
    except Exception as e:
        st.error(f"❌ Gagal menampilkan grafik: {e}")

# ====================
# HALAMAN: Uji Model
# ====================
elif menu == "Uji Model":
    st.title("🧪 Uji Model - Upload Gambar Bebas")

    uploaded_files = st.file_uploader(
        "Upload satu atau lebih gambar (jpg/jpeg/png)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Gambar akan otomatis di-resize ke 32x32 piksel sebelum diprediksi"
    )

    if uploaded_files:
        st.write(f"📥 Total gambar di-upload: {len(uploaded_files)}")
        num_cols = 4
        cols = st.columns(num_cols)

        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            image_resized = image.resize((32, 32))
            img_array = np.array(image_resized) / 255.0
            img_input = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_input)
            pred_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            label_pred = class_names[pred_index]

            with cols[i % num_cols]:
                st.image(image, caption=f"{label_pred} ({confidence*100:.1f}%)", use_container_width=True)
