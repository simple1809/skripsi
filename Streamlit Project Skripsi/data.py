import streamlit as st
import pandas as pd
import numpy as np

st.write("TEST")

a = np.array([1, 2, 3, 4])
b = np.array([(1, 2), (1,2), (1,3), (1,4), (1,5)],dtype=float)


data1 = {
    'nama' : 'azhar',
    'umur' : '18',
    'hobi' : ['futsal', 'makan', 'minum'],
    'favorit' : {
        'makanan' : 'bakso',
        'minuman' : 'jus',
     } 
}

st.text("Tampilan Data Frame")
st.dataframe(b)

st.text("Data Table")
st.table(a)

st.text("Data Json")
st.json(data1)

st.text("Data Metric")
st.metric(label="Harga Saham", value=120000, delta=-10,
    delta_color="inverse")

st.metric(label="Active developers", value=123, delta=123,
    delta_color="off")