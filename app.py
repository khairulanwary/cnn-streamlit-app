import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===================
# SETUP APLIKASI
# ===================

st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10", layout="centered")
st.title("🎯 Prediksi Gambar CIFAR-10 dengan CNN")

# Sidebar Navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi", "Info Model", "Grafik Akurasi/Loss", "Uji Model"])

# Label CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Deskripsi kelas (untuk Info Model)
label_info = {
    'airplane': 'Pesawat terbang di udara',
    'automobile': 'Mobil atau kendaraan roda empat',
    'bird': 'Burung, hewan bersayap',
    'cat': 'Kucing peliharaan',
    'deer': 'Rusa dengan tanduk',
    'dog': 'Anjing, sahabat manusia',
    'frog': 'Katak, hewan amfibi',
    'horse': 'Kuda berkaki empat',
    'ship': 'Kapal laut',
    'truck': 'Truk pengangkut barang'
}

# ===================
# LOAD MODEL (cached)
# ===================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("cnn_cifar10_model.keras")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

# ====================
# HALAMAN: Prediksi
# ====================
if menu == "Prediksi":
    st.subheader("🖼️ Upload Gambar untuk Diprediksi")
    uploaded_file = st.file_uploader("Pilih file gambar (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and model:
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

        st.success(f"🎯 Prediksi: **{class_name.upper()}** ({confidence*100:.2f}% yakin)")
        st.caption(f"📘 Info: {label_info[class_name]}")
    elif uploaded_file is not None and not model:
        st.error("⚠ Model belum termuat. Cek apakah file cnn_cifar10_model.keras ada di direktori yang sama.")

# ====================
# HALAMAN: Info Model
# ====================
elif menu == "Info Model":
    st.subheader("📌 Informasi Label CIFAR-10")
    for cls in class_names:
        st.markdown(f"**{cls.title()}**: {label_info[cls]}")

# ====================
# HALAMAN: Grafik Akurasi / Loss
# ====================
elif menu == "Grafik Akurasi/Loss":
    st.subheader("📈 Grafik Akurasi dan Loss")

    try:
        history = np.load("history_cifar10.npy", allow_pickle=True).item()
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(epochs, acc, label='Train Accuracy')
        ax[0].plot(epochs, val_acc, label='Val Accuracy')
        ax[0].set_title('Akurasi Model')
        ax[0].legend()

        ax[1].plot(epochs, loss, label='Train Loss')
        ax[1].plot(epochs, val_loss, label='Val Loss')
        ax[1].set_title('Loss Model')
        ax[1].legend()

        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠ Gagal menampilkan grafik: {e}")

# ====================
# HALAMAN: Uji Model (bonus testing langsung CIFAR-10 sample)
# ====================
elif menu == "Uji Model":
    st.subheader("🧪 Uji Model dengan Gambar CIFAR-10")

    from keras.datasets import cifar10
    (x_test, y_test) = cifar10.load_data()[1]

    idx = np.random.randint(0, len(x_test))
    sample_image = x_test[idx]
    sample_label = class_names[int(y_test[idx])]

    st.image(sample_image, caption=f"Label Asli: {sample_label.upper()}", use_container_width=True)

    if model:
        sample_input = sample_image / 255.0
        sample_input = np.expand_dims(sample_input, axis=0)

        predictions = model.predict(sample_input)
        pred_class = class_names[np.argmax(predictions)]

        st.success(f"🎯 Prediksi Model: **{pred_class.upper()}**")
    else:
        st.warning("⚠ Model belum berhasil dimuat.")
