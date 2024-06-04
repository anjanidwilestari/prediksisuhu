from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_absolute_error
from model.model_class import Backpropagation, NeuralNetwork, Sigmoid

app = Flask(__name__)

# Fungsi untuk mengonversi nilai Nh ke dalam label angka
def convert_to_label(input_value):
    if input_value == "no clouds":
        return 8.0
    elif input_value == "10%  or less, but not 0":
        return 0.0
    elif input_value == "20–30%.":
        return 2.0
    elif input_value == "40%.":
        return 3.0
    elif input_value == "50%.":
        return 4.0
    elif input_value == "60%.":
        return 5.0
    elif input_value == "70 – 80%.":
        return 6.0
    elif input_value == "90  or more, but not 100%":
        return 7.0
    elif input_value == "100%.":
        return 1.0
    else:
        return None  # Mengembalikan nilai None untuk nilai yang tidak valid
    
def convert_to_informative_description(input_value):
    if input_value == "no clouds":
        return "Tidak ada awan"
    elif input_value == "10%  or less, but not 0":
        return "10% atau kurang, tapi bukan 0"
    elif input_value == "20–30%.":
        return "20 - 30%"
    elif input_value == "40%.":
        return "40%"
    elif input_value == "50%.":
        return "50%"
    elif input_value == "60%.":
        return "60%"
    elif input_value == "70 – 80%.":
        return "70 - 80%"
    elif input_value == "90  or more, but not 100%":
        return "90% atau lebih, tapi bukan 100%"
    elif input_value == "100%.":
        return "100%"
    else:
        return input_value  # Mengembalikan nilai asli jika tidak ada yang cocok


def calculate_mae_percentage(predictions, actual):
    mae = mean_absolute_error(predictions, actual)
    mean_observation = sum(actual) / len(actual)
    mae_percentage = (mae / mean_observation) * 100
    return mae, mae_percentage

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
    # Jika user mengirimkan input hanya via text field untuk Nh
    if 'Nh' in request.form:
        try:
            Nh = request.form['Nh']
            Nh_label = convert_to_label(Nh)
            output = make_prediction(Nh_label)
            Nh_informative = convert_to_informative_description(Nh)
            prediction_df = pd.DataFrame({'Jumlah Awan': [Nh_informative], 'Nh Label': [Nh_label], 'Prediksi Suhu': [output]})
        except Exception as e:
            return render_template('error_template.html', error_message=str(e))

        prediction_table = prediction_df.to_html(classes="table table-striped", index=False)
        prediction_table = prediction_table.replace('<th>', '<th style="text-align: center;">')
        download_link = f"/download/hasil-prediksi"
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
        output_path = os.path.join('downloads', 'hasil-prediksi.xlsx')
        prediction_df.to_excel(output_path, index=False)
        
        return render_template('index.html', prediction_df=prediction_df, prediction_table=prediction_table, download_link=download_link)

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
                    if 'Nh' not in df.columns:
                        raise ValueError("Format file tidak sesuai. Pastikan kolom 'Nh' tersedia.")
                    # Buat salinan DataFrame agar data asli tidak terpengaruh
                    df_processed = df.copy()
                    # Preprocessing kolom 'Nh'
                    df_processed['Nh_label'] = df['Nh'].apply(lambda x: convert_to_label(x) if isinstance(x, str) or isinstance(x, float) else x)
                    df_processed['Prediction'] = df_processed['Nh_label'].apply(make_prediction)
                    df_processed['Jumlah Awan'] = df['Nh'].apply(convert_to_informative_description)

                    # Buat DataFrame baru dengan format yang diinginkan
                    prediction_df = df_processed[['Jumlah Awan', 'Nh_label', 'Prediction']]
                    prediction_df.columns = ['Jumlah Awan', 'Nh Label', 'Prediksi Suhu']
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

def make_prediction(Nh_label):
    # Normalisasi data input
    Nh_normalized = scaler.transform(np.array([[Nh_label]]))[:, 0]  # Menggunakan Nh_label
    # Lakukan prediksi dengan model yang dimuat
    y_pred_normalized = model.predict_new_value([Nh_normalized])
    # Denormalisasi hasil prediksi
    y_pred_denormalized = scaler.inverse_transform(np.array(y_pred_normalized).reshape(-1, 1))[:, 0]

    return round(y_pred_denormalized[0], 8)


@app.route("/download/<filename>")
def download_file(filename):
    # Tentukan path file hasil prediksi
    output_path = os.path.join('downloads', filename)
    if os.path.exists(output_path):
        # Kembalikan file Excel sebagai respons unduhan
        return send_file(output_path, as_attachment=True)
    else:
        return render_template('error_template.html', error_message="File not found")
        
@app.route("/download/hasil-prediksi", methods=['GET'])
def download_field():
    output_path = os.path.join('downloads', 'hasil-prediksi.xlsx')
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        return render_template('error_template.html', error_message="File not found")
        
@app.route("/download_template")
def download_template():
    # Tentukan path ke template Excel
    template_path = "templates/template.xlsx"
    # Kembalikan template sebagai respons unduhan
    return send_file(template_path, as_attachment=True)

@app.route("/about")
def about():
    return render_template('about.html')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)