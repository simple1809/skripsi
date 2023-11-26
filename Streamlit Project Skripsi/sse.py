import streamlit as st
import numpy as np

def calculate_sse(true_values, predicted_values):
    """
    Menghitung nilai SSE (Sum of Squared Errors).
    """
    sse = np.sum((true_values - predicted_values) ** 2)
    return sse

def main():
    st.title("Hitung SSE dengan Streamlit")
    true_values = st.text_input("Masukkan true values (pisahkan dengan koma):")
    true_values = np.array([float(x.strip()) for x in true_values.split(',')])

    predicted_values = st.text_input("Masukkan predicted values (pisahkan dengan koma):")
    predicted_values = np.array([float(x.strip()) for x in predicted_values.split(',')])
    sse = calculate_sse(true_values, predicted_values)

    st.write(f"Nilai SSE: {sse}")

if __name__ == "__main__":
    main()
