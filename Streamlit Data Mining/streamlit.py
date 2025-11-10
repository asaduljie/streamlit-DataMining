import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide", page_title="Prediksi Model")

st.title("Prediksi dengan Model Orange Neural Network")

# 1️⃣ Load model dari Orange
# PASTIKAN PATH FILE INI SUDAH BENAR
model_path = r'C:\Users\Asadul\Downloads\iris.pkcls'
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: File model tidak ditemukan di {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()


# 2️⃣ Input dari user
st.header("Masukkan nilai fitur")

# Gunakan kolom agar lebih rapi
col1, col2 = st.columns(2)

with col1:
    # Ganti "Feature 1" dst. dengan nama fitur Anda yang sebenarnya
    # (Misal: "Panjang Sepal", "Lebar Sepal", dll.)
    feature1 = st.number_input("Feature 1", value=0.0, format="%.2f")
    feature2 = st.number_input("Feature 2", value=0.0, format="%.2f")

with col2:
    feature3 = st.number_input("Feature 3", value=0.0, format="%.2f")
    feature4 = st.number_input("Feature 4", value=0.0, format="%.2f")


# 3️⃣ Konversi ke format DataFrame
# Nama kolom di sini hanya untuk referensi, yang penting adalah
# urutan [feature1, feature2, feature3, feature4] sudah benar.
data = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                    columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"])

st.divider()

# 4️⃣ Prediksi
if st.button("Prediksi", type="primary"):
    try:
        # --- PERBAIKANNYA ADA DI SINI ---
        # Kita gunakan 'data.values' untuk mengubah DataFrame
        # menjadi NumPy array, sesuai format yang diharapkan model.
        pred = model(data.values) 
        
        # Ambil hasil prediksi (biasanya berupa array, jadi kita ambil elemen pertama)
        hasil_prediksi = pred[0]

        st.success(f"Hasil prediksi: {hasil_prediksi}")
        
        # Tampilkan DataFrame yang digunakan untuk prediksi (opsional)
        st.write("Data yang diprediksi:")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")