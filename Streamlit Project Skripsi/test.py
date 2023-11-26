import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def home():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

def dashboard():
    st.title("Dashboard Page")
    st.write("This is the Dashboard Page. Add your dashboard content here.")

def dataset():
    st.title("Welcome to Dataset Visualization Page")
    st.write("Dataset")
    file = st.file_uploader("SILAHKAN INPUT FILE DENGAN FORMAT (.xlsx)", type=["xlsx"])
    if file is not None:
        df = pd.read_excel(file)
        st.write("Data dari file Excel:")
        st.write(df)
        st.write("Statistik Sederhana:")
        st.write(df.describe())

def klastering_kmeans():
    st.title("Klastering UMKM K-Means Method")

def klastering_kharmonicmeans():
    st.title("Klastering UMKM K-Harmonic Means")

def settings():
    st.title("Settings Page")
    st.write("You can configure your settings on this page.")

def main():
    st.sidebar.title("Menu Aplikasi")
    app_mode = st.sidebar.radio("Pilihan", ["Home", "Dashboard", "Dataset", "K-Means", "K-Harmonic Means", "Settings"])

    if app_mode == "Home":
        home()
    elif app_mode == "Dashboard":
        dashboard()
    elif app_mode == "Dataset":
        dataset()
    elif app_mode == "K-Means":
        klastering_kmeans()
    elif app_mode == "K-Harmonic Means":
        klastering_kharmonicmeans()
    elif app_mode == "Settings":
        settings()

if __name__ == "__main__":
    main()
