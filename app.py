from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
from sklearn.impute import KNNImputer  # Import KNNImputer
from model.model import Backpropagation, NeuralNetwork, Sigmoid

app = Flask(__name__)

# Fungsi untuk mengonversi nilai Nh ke dalam label angka
def convert_to_label(input_value):
    if isinstance(input_value, str):  # Periksa apakah nilai input adalah string
        # Buat dictionary untuk memetakan kategori ke label
        if input_value == "no clouds":
            return 8  # 'no clouds'
        elif input_value == "10%  or less, but not 0":
            return 0  # '10% or less, but not 0'
        elif input_value == "20–30%.":
            return 2  # '20-30%'
        elif input_value == "40%.":
            return 3  # '40%'
        elif input_value == "50%.":
            return 4  # '50%'
        elif input_value == "60%.":
            return 5  # '60%'
        elif input_value == "70 – 80%.":
            return 6  # '70 - 80%'
        elif input_value == "90  or more, but not 100%":
            return 7  # "90  or more, but not 100%"
        elif input_value == "100%.":
            return 1  # '100%'
        else:
            return "Data masukan tidak valid"  # Menangani input yang tidak valid
    elif isinstance(input_value, float):  # Jika nilai input adalah float
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
            return "Data masukan tidak valid"  # Menangani input yang tidak valid
    else:
        return "Data masukan tidak valid"  # Menangani input yang tidak valid

# Memuat model dari file 'model.pkl'
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Memuat scaler dari file 'scaler.pkl'
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Jika user mengirimkan input via text field
    if 'Nh' in request.form and 'T' in request.form:
        try:
            Nh = float(request.form['Nh'])
            T = float(request.form['T'])
            Nh_label = convert_to_label(Nh)
            output = make_prediction(Nh_label, T)
            prediction_df = pd.DataFrame({'Jumlah Awan': [Nh], 'Nh Label': [Nh_label], 'Prediksi Suhu': [output]})
        
        except Exception as e:
            return render_template('error_template.html', error_message=str(e))
        
        prediction_table = prediction_df.to_html(classes="table table-striped", index=False)
        prediction_table = prediction_table.replace('<th>', '<th style="text-align: center;">')
        return render_template('index.html', prediction_df=prediction_df, prediction_table=prediction_table,)

    # Jika user mengunggah file Excel
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="Mohon pilih file terlebih dahulu.")
        if file:
            filename = secure_filename(file.filename)
            if filename.endswith('.xlsx'):
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                df = pd.read_excel(file_path)

                try:
                    if 'Nh' not in df.columns or 'T' not in df.columns:
                        raise ValueError("Format file tidak sesuai. Pastikan kolom 'Nh' dan 'T' tersedia.")

                    # Buat salinan DataFrame agar data asli tidak terpengaruh
                    df_processed = df.copy()

                    # Preprocessing kolom 'Nh'
                    df_processed['Nh_label'] = df['Nh'].apply(lambda x: convert_to_label(x) if isinstance(x, str) else x)
                    df_processed['Prediction'] = df_processed.apply(lambda row: make_prediction(row['Nh_label'], row['T']), axis=1)

                    # Buat DataFrame baru dengan format yang diinginkan
                    prediction_df = df_processed[['Nh', 'Nh_label','Prediction']]
                    prediction_df.columns = ['Jumlah Awan', 'Nh Label','Prediksi Suhu']

                except Exception as e:
                    return render_template('error_template.html', error_message=str(e))

                # Menghasilkan tabel HTML dengan menambahkan properti style="text-align: center;" pada header
                prediction_table = prediction_df.to_html(classes="table table-striped", index=False)
                prediction_table = prediction_table.replace('<th>', '<th style="text-align: center;">')
                download_link = f"/download/{filename}"

                if not os.path.exists('downloads'):
                    os.makedirs('downloads')
                output_path = os.path.join('downloads', filename)
                prediction_df.to_excel(output_path, index=False)

                return render_template('index.html', prediction_df=prediction_df, prediction_table=prediction_table, download_link=download_link)

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

@app.route("/download_template")
def download_template():
    # Tentukan path ke template Excel
    template_path = "templates/template.xlsx"
    # Kembalikan template sebagai respons unduhan
    return send_file(template_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
