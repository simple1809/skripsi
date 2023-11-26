import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

st.write("Contoh DATA CHART GRAFIK")

st.write("Data Simple Line Chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.area_chart(chart_data)

st.write("Data Chart Data Bar")
chart_data_bar = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.bar_chart(chart_data_bar)

st.write("Data Line Chart")
chart_data_line = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data_line)

st.write("Data Plot")
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.pyplot(fig)