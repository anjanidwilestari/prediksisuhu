from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import sys
from model.model import Backpropagation, NeuralNetwork, Sigmoid

app = Flask(__name__)

# Fungsi untuk mengonversi nilai Nh ke dalam label angka
def convert_to_label(input_value):
    if input_value == 0:
        return 8  # 'no clouds'
    elif input_value <= 10:
        return 0  # '10% or less, but not 0'
    elif input_value <= 30:
        return 2  # '20-30%'
    elif input_value <= 40:
        return 3  # '40%'
    elif input_value <= 50:
        return 4  # '50%'
    elif input_value <= 60:
        return 5  # '60%'
    elif input_value <= 80:
        return 6  # '70 - 80%'
    elif input_value < 100:
        return 7  # '90 or more, but not 100%'
    elif input_value == 100:
        return 1  # '100%'
    else:
        return None  # Menangani input yang tidak valid



# Memuat model dari file 'model.pkl'
with open('model/model.pkl', 'rb') as b:
    model = pickle.load(b)

# Memuat scaler dari file 'scaler.pkl'
with open('model/scaler.pkl', 'rb') as s:
    scaler = pickle.load(s)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Jika user mengirimkan input via text field
    if 'Nh' in request.form and 'T' in request.form:
        Nh = float(request.form['Nh'])  # Ubah ke float
        T = float(request.form['T'])  # Ubah ke float
        Nh_label = convert_to_label(Nh)  # Mengonversi Nh ke label angka
        output = make_prediction(Nh_label, T)  # Prediksi menggunakan Nh_label
        return render_template('index.html', prediction_text=f"Prediksi Suhu berdasarkan Jumlah Awan {Nh} ({Nh_label}) adalah {output} Celcius")

    # Jika user mengunggah file Excel
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="Mohon pilih file terlebih dahulu.")
        if file:
            filename = secure_filename(file.filename)  # Perbaiki nama file
            if filename.endswith('.xlsx'):
                # Pastikan direktori 'uploads' ada
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                df = pd.read_excel(file_path)
                
                # Konversi kolom 'Nh' ke label angka
                df['Nh_label'] = df['Nh'].apply(convert_to_label)
                df['Prediction'] = df.apply(lambda row: make_prediction(row['Nh_label'], row['T']), axis=1)
                
                # Buat DataFrame baru dengan format yang diinginkan
                prediction_df = df[['Nh', 'T', 'Nh_label', 'Prediction']]
                prediction_df.columns = ['Jumlah Awan', 'Target Suhu', 'Label Jumlah Awan', 'Prediksi Suhu']
                
                # Render tabel prediksi di halaman web
                prediction_table = prediction_df.to_html(classes="table table-striped", index=False)

                # Tambahkan tombol unduh untuk tabel prediksi
                download_link = f"/download/{filename}"  # Mengarahkan ke endpoint unduhan

                # Pastikan direktori 'downloads' ada
                if not os.path.exists('downloads'):
                    os.makedirs('downloads')
                output_path = os.path.join('downloads', filename)
                prediction_df.to_excel(output_path, index=False)  # Simpan prediksi ke file Excel
                
                return render_template('index.html', prediction_table=prediction_table, download_link=download_link)

    return render_template('index.html', prediction_text="Mohon inputkan nilai Jumlah Awan dan T, atau unggah file Excel.")


def make_prediction(Nh_label, T):
    # Normalisasi data input
    Nh_normalized = scaler.transform(np.array([[Nh_label]]))[:, 0]  # Menggunakan Nh_label
    T_normalized = scaler.transform(np.array([[T]]))[:, 0]

    # Lakukan prediksi dengan model yang dimuat
    y_pred_normalized = model.predict([Nh_normalized], [T_normalized])

    # Denormalisasi hasil prediksi
    y_pred_denormalized = scaler.inverse_transform(np.array(y_pred_normalized).reshape(-1, 1))[:, 0]

    return round(y_pred_denormalized[0], 8)

@app.route("/download/<filename>")
def download(filename):
    # Tentukan path file hasil prediksi
    output_path = os.path.join('downloads', filename)

    # Kembalikan file Excel sebagai respons unduhan
    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
