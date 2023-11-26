import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

# Upload dataset
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

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

        st.write("Running K-Means...")

        # Inisialisasi pusat klaster secara acak untuk iterasi pertama
        kmeans_iter1 = KMeans(n_clusters=k_value, random_state=123)
        kmeans_iter1.fit(minmax)

        # Salin pusat klaster iterasi pertama
        centers_previous = kmeans_iter1.cluster_centers_

        # Hitung jarak antara titik data dan pusat klaster (iterasi pertama)
        distances_previous = np.linalg.norm(minmax - centers_previous[kmeans_iter1.labels_], axis=1)

        silhouette_scores_per_iteration = []
        sse_per_iteration = []

        sse_cluster_iter1 = np.sum((minmax - centers_previous[kmeans_iter1.labels_]) ** 2, axis=1)
        sse_per_iteration.append(np.sum(sse_cluster_iter1))

        # Menampilkan nilai SSE pada iterasi pertama
        st.write(f"SSE (Iteration 1): {np.sum(sse_cluster_iter1)}")

        # Menampilkan nilai Silhouette Coefficient untuk iterasi pertama
        silhouette_avg = silhouette_score(minmax, kmeans_iter1.labels_)
        st.write(f"Silhouette Score (Iteration 1): {silhouette_avg}")

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
                st.write(f"Converged at Iteration {iteration} - No further changes expected.")
                break

            # Simpan distances iterasi sekarang
            distances_current = distances_current_new

            # Perbarui pusat klaster untuk iterasi berikutnya
            centers_previous = centers_current.copy()
            iteration += 1

        # Menampilkan jumlah iterasi
        st.subheader(f"Jumlah Iterasi: {iteration}")

        # Menampilkan hasil klastering pada iterasi terakhir
        st.subheader("Hasil Klastering pada Iterasi Terakhir:")
        df['Cluster'] = kmeans.labels_
        st.write(df)

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
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        st.pyplot(plt)

    else:
        st.write("DataFrame is empty.")

if __name__ == "__main__":
    kmeans()