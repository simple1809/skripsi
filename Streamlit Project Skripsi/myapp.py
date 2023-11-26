import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

a = np.array([0, 0, 0, 0])
b = np.array([(0, 0), (0,0), (0,0), (0,0), (0,0)],dtype=float)

def home():
    st.title("Home Page")
    st.write("Selamat Datang Di Aplikasi Web Hasil Klastering UMKM Madura")

def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        st.write("Data from Excel file:")
        st.write(df)
        st.write("Simple Statistics:")
        st.write(df.describe())
    except Exception as e:
        st.error(f"Error: {e}")

def loaddata_normalisasi(file_pathnorm):
    try:
        df = pd.read_excel(file_pathnorm)
        st.write("Data from Excel file:")
        st.write(df)
        st.write("Simple Statistics:")
        st.write(df.describe())

        st.write("Line Chart based on the dataset:")
        fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Line Chart")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {e}")

def datasetnorm():
    file_pathnorm = "C:\Azhar\SKRIPSI AZHAR\SKRIPSI\codingan\Streamlit Project Skripsi\hasilnormalisasidata.xlsx"

    if file_pathnorm:
        load_data(file_pathnorm)

def dataset():
    file_path = "C:\Azhar\SKRIPSI AZHAR\SKRIPSI\codingan\Streamlit Project Skripsi\dataset.xlsx"

    if file_path:
        load_data(file_path)

def normalisasidata():
    uploaded_files = st.file_uploader("Choose a Xlsx file", accept_multiple_files=True)
    df = pd.DataFrame()

    if uploaded_files:
        dfs = [pd.read_excel(file) for file in uploaded_files if file is not None]

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            for uploaded_file in uploaded_files:
                st.write("filename:", uploaded_file.name)
        else:
            st.write("No valid DataFrames.")
    else:
        st.write("No files uploaded.")

    st.write("Data Asli :")
    st.write(df)

    st.write("Kolom Yang Terdapat Pada Data :")
    st.write(df.columns)

    st.write("Tipe Data Kolom :")
    st.write(df.dtypes)

    columns_to_drop = ["NO", "NAMA USAHA"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    if 'KAB KOTA' in df.columns:
        label_encoder = LabelEncoder()
        df['KAB KOTA'] = label_encoder.fit_transform(df['KAB KOTA'])

    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

    if len(df) > 0:
        scaler = MinMaxScaler()
        daftar_col = df.columns
        df[daftar_col] = scaler.fit_transform(df[daftar_col])
        st.write("Hasil Normalisasi Data Menggunakan Min-Max Scaling :")
        st.dataframe(df)
    else:
        st.write("DataFrame is empty.")
        
def kmeans():
    df = pd.DataFrame()

    uploaded_file = st.file_uploader("Input dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file, error_bad_lines=False)
        elif uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)

    if not df.empty:
        st.write("Preview of the dataset:")
        st.write(df.head())

        minmax = MinMaxScaler().fit_transform(df[['OMSET RATA', 'BIAYA PRODUKSI RATA', 'JUMLAH TENAGA KERJA']])
        df_scaled = pd.DataFrame(minmax, columns=['OMSET RATA', 'BIAYA PRODUKSI RATA', 'JUMLAH TENAGA KERJA'])

        st.write("Scaled Data:")
        st.write(df_scaled.head(10))

        # K Means Clustering
        k_value = st.slider("Select the number of clusters (K)", min_value=2, max_value=10, value=4)

        # Inisialisasi pusat klaster secara acak untuk iterasi pertama
        kmeans_iter1 = KMeans(n_clusters=k_value, random_state=123)
        kmeans_iter1.fit(minmax)

        # Salin pusat klaster iterasi pertama
        centers_previous = kmeans_iter1.cluster_centers_

        # Hitung jarak antara titik data dan pusat klaster (iterasi pertama)
        distances_previous = np.linalg.norm(minmax - centers_previous[kmeans_iter1.labels_], axis=1)
        st.write("Hasil Klastering Pada Iteration 1:")
        labels_iteration1 = kmeans_iter1.labels_ + 1
        clustered_data_iter1 = pd.concat([df, pd.DataFrame({'Cluster': labels_iteration1})], axis=1)
        st.write(clustered_data_iter1)

        silhouette_scores_per_iteration = []
        sse_per_iteration = []  

        # Iterasi selanjutnya
        iteration = 2
        distances_current = distances_previous  # Simpan distances iterasi sebelumnya

        while True:
            # Inisialisasi pusat klaster secara acak untuk iterasi sekarang
            kmeans = KMeans(n_clusters=k_value, init='random', random_state=iteration * 123)
            kmeans.fit(minmax)

            # Salin pusat klaster iterasi sekarang
            centers_current = kmeans.cluster_centers_

            # Hitung jarak antara titik data dan pusat klaster (iterasi sekarang)
            distances_current_new = np.linalg.norm(minmax - centers_previous[kmeans.labels_], axis=1)

            # Hitung nilai SSE
            sse_cluster = np.sum((minmax - centers_previous[kmeans.labels_]) ** 2, axis=1)
            sse_per_iteration.append(np.sum(sse_cluster))

            # Menampilkan nilai Silhouette Coefficient untuk setiap klaster
            silhouette_avg = silhouette_score(minmax, kmeans.labels_)
            silhouette_scores_per_iteration.append(silhouette_avg)

            # Periksa apakah hasil klastering berubah dari iterasi sebelumnya
            if np.array_equal(distances_current, distances_current_new):
                st.write(f" Pada iterasi ke - {iteration} - Sudah tidak ada perubahan perhitungan jarak.")
                break

            # Simpan distances iterasi sekarang
            distances_current = distances_current_new

            # Perbarui pusat klaster untuk iterasi berikutnya
            centers_previous = centers_current.copy()
            iteration += 1

        # Menampilkan jumlah iterasi
        st.subheader(f"Jumlah Iterasi: {iteration}")
        labels = kmeans.labels_
        labels_shifted = labels + 1
        # Menampilkan hasil klastering pada iterasi terakhir
        st.subheader("Hasil Klastering pada Iterasi Terakhir:")
        df['Cluster'] = labels_shifted
        st.write(df)

        st.subheader("K Value")
        st.write(k_value)

        # Menampilkan nilai SSE untuk setiap klaster pada hasil klastering terakhir
        st.subheader("Nilai SSE untuk Hasil Klastering Terakhir")
        sse_values = []
        for cluster in range(k_value):
            sse_cluster = np.sum(sse_per_iteration[cluster::k_value])
            st.write(f"Klaster {cluster + 1}: {sse_cluster}")
            sse_values.append(sse_cluster)

        # Menampilkan rata-rata SSE
        avg_sse = np.mean(sse_cluster)
        st.subheader(f"Rata-rata SSE: {avg_sse}")

        # Menampilkan rata-rata Silhouette Score
        avg_silhouette = np.mean(silhouette_scores_per_iteration)
        st.subheader(f"Rata-rata Silhouette Score: {avg_silhouette}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k_value + 1), sse_values, marker='o')
        plt.title('Penentuan K Terbaik Berdasarkan Nilai SSE')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        st.pyplot(plt)

    else:
        st.write("DataFrame is empty.")
    
def kharmonicmeans():
    st.subheader("Klastering UMKM K - Harmonic Means")

def dashboard():
    st.subheader("Dashboard")
  
def analisapage():
    st.subheader("Hasil Analisa Klastering UMKM")

def main():
    st.sidebar.title("Menu Aplikasi")
    app_mode = st.sidebar.selectbox("Pilihan", ["Home", "Dashboard", "Dataset", "Normalisasi Data", "K - Means", "K - Harmonic Means", "Hasil Analisa Klastering"])

    if app_mode == "Home":
        home()
    elif app_mode == "Dashboard":
        dashboard()
    elif app_mode == "Dataset":
        dataset()
    elif app_mode == "Normalisasi Data":
        normalisasidata()
    elif app_mode == "K - Means":
        kmeans()
    elif app_mode == "K - Harmonic Means":
        kharmonicmeans()
    elif app_mode == "Hasil Analisa Klastering":
        analisapage()

if __name__ == "__main__":
    main()
