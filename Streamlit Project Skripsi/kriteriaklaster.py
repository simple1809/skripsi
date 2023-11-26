import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("Aplikasi Klastering Omset")

    # Input data omset
    st.header("Input Data Omset")
    omset_data = st.text_area("Masukkan omset (pisahkan dengan koma)", "100000,150000,80000,120000,200000,250000,30000,50000,70000")
    omset_list = [float(x.strip()) for x in omset_data.split(',')]

    # Menentukan jumlah klaster
    num_clusters = st.slider("Pilih Jumlah Klaster", min_value=2, max_value=10, value=3)

    # Membuat DataFrame
    df = pd.DataFrame({'Omset': omset_list})

    # Menggunakan K-Means untuk klastering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df[['Omset']])
    df['Klaster'] = kmeans.labels_

    # Menampilkan hasil klastering
    st.subheader("Hasil Klastering")
    st.write(df)

    # Visualisasi hasil klastering
    st.subheader("Visualisasi Hasil Klastering")
    plt.scatter(df['Omset'], np.zeros_like(df['Omset']), c=df['Klaster'], cmap='viridis')
    st.pyplot()

if __name__ == "__main__":
    main()
