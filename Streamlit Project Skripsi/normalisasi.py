import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# File upload
uploaded_files = st.file_uploader("Choose a Xlsx file", accept_multiple_files=True)
dfs = [pd.read_excel(file) for file in uploaded_files]
df = pd.concat(dfs, ignore_index=True)

# Display filenames
for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)

# Columns to drop
columns_to_drop = ["NO", "NAMA USAHA"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Columns to transform
columns_to_transform = df.columns.difference(["JUMLAH TENAGA KERJA", "OMSET RATA", "BIAYA PRODUKSI RATA"])

# Dictionary of replacement values
nilai = {
    "KAB KOTA": {
        "[26] BANGKALAN" : 1
    },
    "STATUS USAHA" : {
        "CV" : 1,
        "IUMKM" : 2,
        "KOPERASI" : 3,
        "PERSEORANGAN" : 4,
        "UD" :5,
        "YAYASAN": 6
    }, "JENIS LAPANGAN USAHA" :{
        "[A] PERTANIAN, KEHUTANAN & PERIKANAN" : 1,
        "[C] INDUSTRI PENGOLAHAN" : 2,
        "[D] PENGADAAN LISTRIK, GAS, UAP/AIR PANAS DAN UDARA DINGIN" : 3,
        "[E] TREATMENT AIR, TREATMENT AIR LIMBAH, TREATMENT DAN PEMULIHAN MATERIAL SAMPAH, DAN AKTIVITAS REMEDIASI" : 4,
        "[G] PERDAGANGAN BESAR & ECERAN; REPARASI MOBIL & MOTOR" : 5,
        "[H] PENGANGKUTAN & PERGUDANGAN" : 6,
        "[I] PENYEDIAAN AKOMODASI DAN MAKAN MINUM" : 7,
        "[K] AKTIVITAS KEUANGAN DAN ASURANSI" : 8,
        "[M,N] JASA PERUSAHAAN" : 9,
        "[R,S] AKTIVITAS JASA LAINNYA" : 10,
    },  "JENIS KESULITAN USAHA" : {
        "TIDAK" : 1,
        "BAHAN BAKU" : 2,
        "DISTRIBUSI/ TRANSPORTASI" : 3,
        "LEGALITAS/ PERIJINAN" : 4,
        "PEMASARAN" : 5,
        "PENGGUNAAN ENERGI LISTRIK" : 6,
        "PERMODALAN" : 7,
    }, "DAPAT KREDIT" : {
        "TIDAK" : 1,
        "YA" : 2
    }, "PERLU PINJAMAN PIHAK LUAR" : {
        "YA" : 1,
        "TIDAK" : 2
    }
}

# Replace values in specified columns
df[columns_to_transform] = df[columns_to_transform].replace(nilai)

# Check and replace values in the 'NPWP USAHA' column
df['NPWP USAHA'] = df['NPWP USAHA'].replace({0: 2, 1: 1})

# Min-Max Scaling for selected features
scaler = MinMaxScaler()
df[columns_to_transform] = scaler.fit_transform(df[columns_to_transform])

# Display the transformed DataFrame
st.write(df)
