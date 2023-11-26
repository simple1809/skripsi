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

a = np.array([0.0, 0.0, 0.0, 0.0])


def kharmonicmeans():
    st.subheader("Klastering UMKM K - Harmonic Means")
    df = pd.DataFrame()
    # Proses untuk mengoupload data / memangil data dari path library 
    uploaded_file = st.file_uploader("Silahkan Input Dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file, eror_bad_lines=False)
        elif uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)

    if not df.empty:
        st.write("Dataset Preview:")
        st.write(df.head(10))

    st.subheader("Inisialisasi Centroid")
    st.subheader("Perhitungan Jarak")
    st.subheader("Nilai Keanggotaan")
    st.subheader("Nilai Bobot")
    st.subheader("Hasil Bobot dan Keanggotaan pada Iterasi Terakhir")

if __name__ == "__main__":
    kharmonicmeans()



    