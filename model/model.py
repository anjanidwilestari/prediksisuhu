import pandas as pd
import numpy as np
import regex as re
import openpyxl
import os

# Get the current working directory
current_dir = os.getcwd()
# Define the file path based on the operating system
if os.name == 'posix':  # Unix-based systems (like macOS and Linux)
    file_path = os.path.join(current_dir, 'model', 'weather.csv')
elif os.name == 'nt':   # Windows
    file_path = os.path.join(current_dir, 'model', 'weather.csv')
else:
    # Handle other operating systems if needed
    raise OSError("Unsupported operating system")

# Import data
df=pd.read_csv(file_path, encoding='latin1')

#### Preprocessing Data ####

# 1. MEMBUANG KOLOM TIDAK DIGUNAKAN
column_names_to_drop = ["Local time in Surabaya", "ff10", "ff3", "WW", "W1", "W2", "Tn", "Tx", "RRR", "tR", "E", "Tg", "E'", "sss" ] # column_names_to_drop adalah daftar kolom yang ingin dibuang
df = df.drop(column_names_to_drop, axis=1) # Menggunakan metode drop untuk menghapus kolom-kolom yang tidak digunakan
# # Menampilkan DataFrame setelah menghapus kolom-kolom
# print("\nDataFrame 15 Kolom:")
# print(df)

# 2. MEMPERBAIKI KARAKTER ANEH 
columns_to_replace = ["T", "Po", "P", "Pa", "U", "N", "H", "Nh", "Cm", "Ch"] # Kolom-kolom yang ingin diperbaiki
# Loop melalui kolom-kolom dan ganti karakter "" dengan strip ("-")
for col in columns_to_replace:
    df[col] = df[col].replace('', '-', regex=True)
# # Menampilkan DataFrame setelah perbaikan
# print("\nDataFrame Setelah Perbaikan :")
# print(df)

# 3. MEMPERBAIKI DATA VV
#PENGECEKAN NILAI NON-NUMERIK
column_to_check = "VV"
# Konversi nilai kolom ke numerik
numeric_values = pd.to_numeric(df[column_to_check], errors='coerce')
# Temukan baris yang memiliki nilai non-numerik
non_numeric_rows = df[numeric_values.isna()]
# # Tampilkan baris yang memiliki nilai non-numerik
# print("\nBaris dengan nilai non-numerik:")
# print(non_numeric_rows)
# Mengganti data 'less than 0.1'
df[column_to_check] = df[column_to_check].replace('less than 0.1', 1.0)
# # Menampilkan Baris data setelah perbaikan
# print("\nBaris VV Setelah Penggantian:")
# print(df.loc[[5651, 6663]])

#4. ENCODE FITUR KATEGORIKAL
# Menangani fitur kategorikal dengan mengonversinya menjadi bilangan bulat menggunakan Label Encoder/Ordinal Coding
from sklearn.preprocessing import LabelEncoder
# Kolom kategorikal yang ingin di-label encode
columns_categorical = ['DD', 'Ff', 'N', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV']
# Label Encoder dengan mempertahankan nilai NaN
for column in columns_categorical:
    le = LabelEncoder()
    non_nan_indices = df[column].notna()
    df.loc[non_nan_indices, column] = le.fit_transform(df.loc[non_nan_indices, column].astype(str))
# # Menampilkan DataFrame setelah Label Encoder
# print("\nDataFrame Setelah Label Encoder:")
# print(df)

#5. MEMPERBAIKI MISSING VALUE
from sklearn.impute import KNNImputer
columns_numeric = df.columns
# Melakukan imputasi dengan KNNImputer pada kolom numerik
imputer_numeric = KNNImputer(n_neighbors=5)
df[columns_numeric] = imputer_numeric.fit_transform(df[columns_numeric])
# # Menampilkan dataset setelah imputasi
# print("\nDataFrame Setelah Imputasi:")
# print(df)

#6. NORMALISASI LINIER MAX MIN
from sklearn.preprocessing import MinMaxScaler
# Membuat objek MinMaxScaler
scaler = MinMaxScaler()
# Normalisasi semua kolom di DataFrame
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# # Tampilkan DataFrame setelah normalisasi
# print("DataFrame setelah normalisasi:")
# print(df_normalized)

#7a. TRAIN TEST SPLIT DATAFRAME ASLI
from sklearn.model_selection import train_test_split
# Assuming 'target_column' is the name of your target variable = 'T'
X_asli = df.drop('T', axis=1)  # Features
y_asli = df['T']  # Target variable
#Train=70% Test=30%
X_asli_train, X_asli_test, y_asli_train, y_asli_test = train_test_split(X_asli, y_asli, test_size = 0.3)

# # Mencetak ukuran dari masing-masing set
# print('[DATA SEBELUM NORMALISASI] \n')
# print("Ukuran X_train:", X_asli_train.shape)
# print("Ukuran y_train:", y_asli_train.shape)
# print("Ukuran X_test:", X_asli_test.shape)
# print("Ukuran y_test:", y_asli_test.shape)

#7b. TRAIN TEST SPLIT DATAFRAME SETELAH NORMALISASI
from sklearn.model_selection import train_test_split
# Assuming 'target_column' is the name of your target variable = 'T'
X = df_normalized.drop('T', axis=1)  # Features
y = df_normalized['T']  # Target variable
#Train=70% Test=30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# # Mencetak ukuran dari masing-masing set
# print('[DATA SETELAH NORMALISASI] \n')
# print("Ukuran X_train:", X_train.shape)
# print("Ukuran y_train:", y_train.shape)
# print("Ukuran X_test:", X_test.shape)
# print("Ukuran y_test:", y_test.shape)

#### LIBRARY BACKPROPAGATION ####

class NeuralNetwork:
    def __init__(self,input,hidden,output):
        self.input = input
        self.hidden = hidden
        self.output = output
        
    def initialize_weights(self, bias=False):
        self.hidden_weights=np.random.normal(size=(self.input,self.hidden))
        self.output_weights=np.random.normal(size=(self.hidden,self.output))
        self.bias = False
        if bias:
            self.hidden_bias_weights=np.random.normal(size=(1,self.hidden))
            self.output_bias_weights=np.random.normal(size=(1,self.output))
            self.bias = True
            
    def print_weights(self):
        print("Hidden Weights:")
        print(self.hidden_weights)
        print("\nOutput Weights:")
        print(self.output_weights)
        if self.bias:
            print("\nHidden Bias Weights:")
            print(self.hidden_bias_weights)
            print("\nOutput Bias Weights:")
            print(self.output_bias_weights)

class Sigmoid:
    def activate(self, x):
        return 1/(1 + np.exp(-x))
    def derivative(self, x):
        return x * (1 - x)
    
class Backpropagation:
    def __init__(self, neuralnet, epochs=2000, lr=0.1, activation_function=Sigmoid()):
        self.neuralnet = neuralnet
        self.epochs = epochs
        self.lr = lr
        self.activation_function = activation_function
    
    def feedForward(self, input):
        hidden_layer = np.dot(input, self.neuralnet.hidden_weights)
        if self.neuralnet.bias:
            hidden_layer += self.neuralnet.hidden_bias_weights
        hidden_layer = self.activation_function.activate(hidden_layer)
        output_layer = np.dot(hidden_layer, self.neuralnet.output_weights)
        if self.neuralnet.bias:
            output_layer += self.neuralnet.output_bias_weights
        output_layer = self.activation_function.activate(output_layer)
        return hidden_layer, output_layer
    
    def train(self, input, target):
        for _ in range(self.epochs):
            # Feed Forward
            hidden_layer, output_layer = self.feedForward(input)
            # Error term for each output unit k
            derivative_output = self.activation_function.derivative(output_layer)
            del_k = output_layer * derivative_output * (target - output_layer)
            # Error term for each hidden unit h
            sum_del_h = del_k.dot(self.neuralnet.output_weights.T)
            derivative_hidden = self.activation_function.derivative(hidden_layer)
            del_h = hidden_layer * derivative_hidden * sum_del_h
            # Weight Update
            self.neuralnet.output_weights += hidden_layer.T.dot(del_k) * self.lr
            self.neuralnet.hidden_weights += input.T.dot(del_h) * self.lr
    
    def predict(self, input, actual_output):
        hidden_layer, output_layer = self.feedForward(input)
        predicted_values = [] 
        for i in range(len(input)):
          for j in range(len(actual_output)):
            if i==j:
              predicted_value = output_layer[i][j]
              actual_value = actual_output[i][0]
              # print(f"For input {input[i]}, the predicted output is {predicted_value} and the actual output is {actual_value}")
              predicted_values.append(predicted_value)
        return predicted_values  # Mengembalikan list nilai predicted_value
    
    def predict_new_value(self, input):
        hidden_layer, output_layer = self.feedForward(input)
        predicted_values = []  # List untuk menyimpan nilai output_layer[i][0]
        for i in range(len(input)):
          for j in range(len(input)):
            if i==j:
              predicted_value = output_layer[i][j]
              # print(f"For input {input[i]}, the predicted output is {predicted_value} and the actual output is {actual_value}")
              # Simpan nilai predicted_value ke dalam list
              predicted_values.append(predicted_value)
        return predicted_values  # Mengembalikan list nilai predicted_value


#### PENERAPAN BEST SOLUTION ####

best_solution = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

# Fungsi seleksi fitur
def select_features(solution, df):
    selected_columns = [col for col, value in zip(df.columns, solution[0]) if value == 1]
    selected_df = df[selected_columns]
    return selected_df

# Pilih fitur-fitur yang sesuai dengan best solution
selected_features_test = select_features(best_solution, X_test)
# print('\nselected_features_test')
# print(selected_features_test)

#PREDIKSI DATA UJI
import time
t1 = time.perf_counter()

# Inisialisasi objek NeuralNetwork
input_size = selected_features_test.shape[1]
hidden_size=2
output_size=len(y_test)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Inisialisasi bobot dengan atau tanpa bias
nn.initialize_weights(bias=False)

# Inisialisasi objek Backpropagation dengan objek NeuralNetwork yang telah dibuat
epochs=2
learning_rate=0.001
activation_function=Sigmoid()
bp = Backpropagation(nn, epochs, learning_rate, activation_function)

# Lakukan prediksi dengan data uji
input_data = selected_features_test.values
actual_data = np.array(y_test)
y_pred = bp.predict(input_data, actual_data.reshape(-1, 1))

#Denormalisasi y_pred
scaler_T = MinMaxScaler() # Membuat scaler baru untuk kolom 'T'
scaler_T.fit(df['T'].values.reshape(-1, 1)) # Sesuaikan scaler baru dengan nilai asli 'T'
y_pred_denorm = scaler_T.inverse_transform(np.array(y_pred).reshape(-1, 1))[:, 0] # Denormalisasi kolom 'T'
y_test_denorm = scaler_T.inverse_transform(np.array(y_test).reshape(-1, 1))[:, 0] # Denormalisasi kolom 'T'

# Membuat DataFrame untuk menampung hasil
df_results_test = pd.DataFrame({
    'Nilai Prediksi': y_pred,
    'Nilai Target': y_test,
    'Denormalisasi Nilai Prediksi': y_pred_denorm,
    'Denormalisasi Nilai Target': y_test_denorm
})
# print('\nPerbandingan Hasil Normalisasi & Denormalisasi')
# print(df_results_test)
t2 = time.perf_counter()
# print('Waktu yang dibutuhkan untuk eksekusi', t2-t1, 'detik')

#EVALUASI MAE
from sklearn.metrics import mean_absolute_error

# Menghitung MAE data uji denormalisasi
mae_test = mean_absolute_error(y_test_denorm, y_pred_denorm)
# print("Mean Absolute Error:", mae_test)

# #SAVE MODEL
# # Simpan model ke file menggunakan pickle
# import pickle

# with open('model/model.pkl', 'wb') as f:
#     pickle.dump(bp, f)
    
# with open('model/scaler.pkl', 'wb') as f:
#     pickle.dump(scaler_T, f)

# print("Model telah disimpan dalam file 'model.pkl'")