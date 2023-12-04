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
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
import random

a = np.array([0, 0, 0, 0])
b = np.array([(0, 0), (0,0), (0,0), (0,0), (0,0)],dtype=float)
k_value = 3
    
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
class KMeanss:
    def __init__(self, n_clusters, max_iters=100, random_state=42):
        self.n_clusters = k_value
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.iteration_data = []

    def get_sse_per_cluster(self, X):
        sse_per_cluster = []
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            centroid = self.centroids[k]
            sse = np.sum((cluster_points - centroid) ** 2)
            sse_per_cluster.append(sse)
        return sse_per_cluster

    def get_silhouette_per_cluster(self, X):
        silhouette_per_cluster = []
        
        silhouette_values = silhouette_samples(X, self.labels)
  
        for k in range(self.n_clusters):
            cluster_indices = np.where(self.labels == k)[0]
            if len(cluster_indices) > 1:
                avg_silhouette = np.mean(silhouette_values[cluster_indices])
                silhouette_per_cluster.append(avg_silhouette)
            else:
                silhouette_per_cluster.append(np.nan)
            
        return silhouette_per_cluster
        
    def get_cluster_means(self, X):
        cluster_means = [X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)]
        return cluster_means
    
    def get_cluster_centers(self):
        return self.centroids

    def fit(self, X):
            np.random.seed(self.random_state)
            self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]

            prev_inertia = None  # Menyimpan hasil inertia iterasi sebelumnya

            for iteration in range(self.max_iters):
                self.labels = self._assign_clusters(X)
                new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
                self.inertia = np.sum((X - new_centroids[self.labels]) ** 2)
                if prev_inertia is not None and np.allclose(self.inertia, prev_inertia):
                    print(f"tidak ada perubahan pada {iteration + 1} iterations.")
                    break

                prev_inertia = self.inertia

                self.centroids = new_centroids

                iteration_info = {
                    'Iteration': iteration + 1,
                    'Centroids': [centroid.tolist() for centroid in self.centroids],
                    'Inertia': self.inertia,
                    'Labels': self.labels.tolist(),
                    'Distances': self._calculate_distances(X)
                }
                self.iteration_data.append(iteration_info)

            columns = ['Iteration', 'Centroids', 'Inertia', 'Labels', 'Distances']
            self.iteration_df = pd.DataFrame(self.iteration_data, columns=columns)

            print("\nHasil Iterasi Terakhir:")
            print(self.iteration_df.iloc[-1])

            sse_per_cluster = self.get_sse_per_cluster(X)
            print("\nSSE Per Klaster pada Iterasi Terakhir:")
            for k, sse in enumerate(sse_per_cluster):
                print(f"Klaster {k + 1}: {sse}")

            silhouette_per_cluster = self.get_silhouette_per_cluster(X)
            print("\nSilhouette Per Klaster pada Iterasi Terakhir:")
            for k, silhouette in enumerate(silhouette_per_cluster):
                print(f"Klaster {k + 1}: {silhouette}")

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_distances(self, X):
        distances = np.min(np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2), axis=1)
        return distances
    
    def plot_clusters(data, features, hue):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=features[0], y=features[1], hue=hue, palette='viridis', s=100, alpha=0.8)
        plt.title('Clustering Results')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend(title='Cluster Labels')
        st.pyplot(plt)
        
def kmeanscluster():
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

        st.write("K - Means")
    
        minmax = MinMaxScaler().fit_transform(df[['JUMLAH TENAGA KERJA','OMSET RATA', 'BIAYA PRODUKSI RATA']])
        kmeans = KMeanss(n_clusters=k_value, max_iters=100, random_state=42)
        kmeans.fit(minmax)

        # Combine the last iteration results with the original dataset
        data = pd.DataFrame(df, columns=['JUMLAH TENAGA KERJA','OMSET RATA','BIAYA PRODUKSI RATA'])
        data['Labels'] = kmeans.labels
        st.write(data)
        
        # Display results from the last iteration as DataFrame
        iteration_result = kmeans.iteration_df.iloc[-1]
        st.write("\nHasil Iterasi Terakhir:")
        st.table(iteration_result)
        
        # Display SSE Per Cluster
        sse_per_cluster = kmeans.get_sse_per_cluster(minmax)
        st.write("\nSSE Per Klaster pada Iterasi Terakhir:")
        for i, sse_value in enumerate(sse_per_cluster, start=1):
            st.write(f"Klaster {i}: {sse_value}")

        # Display Silhouette Per Cluster
        silhouette_per_cluster = kmeans.get_silhouette_per_cluster(minmax)
        st.write("\nSilhouette Per Klaster pada Iterasi Terakhir:")
        for i, silhouette_value in enumerate(silhouette_per_cluster, start=1):
            st.write(f"Klaster {i}: {silhouette_value}")
            
        cluster_means = kmeans.get_cluster_means(data)
        for k, mean_values in enumerate(cluster_means):
            st.write(f"Rata-rata Fitur Klaster {k + 1}:", mean_values)
        
        st.write("\nElbow Method:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Plot SSE values
        ax1.plot(range(1, len(sse_per_cluster) + 1), sse_per_cluster, marker='o')
        ax1.set_title('Elbow Method For Optimal k (SSE)')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within-Cluster Sum of Squares (SSE)')

        # Plot Silhouette scores
        ax2.plot(range(1, len(silhouette_per_cluster) + 1), silhouette_per_cluster, marker='o')
        ax2.set_title('Silhouette Scores For Optimal k')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')

        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.write("DataFrame is empty.")

class KMeansInitializer:
    def __init__(self, n_clusters, random_state=42):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def initialize_centers(self, data):
        self.kmeans.fit(data)
        return self.kmeans.cluster_centers_
class KHarmonicMeans:
    def __init__(self, n_clusters, m, max_iterations=200, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iterations = max_iterations
        self.tol = tol
        self.centers = None
        self.membership_degree = None
        self.weights = None
        self.df_results = None
        self.df_jarak = None
        self.best_sse = float('inf')

    def initialize_membership_degree(self, num_data):
        membership_degree = np.random.rand(num_data, self.n_clusters)
        return membership_degree / np.sum(membership_degree, axis=1, keepdims=True)

    def kharmonic_means_distance(self, data):
        distances = np.zeros((data.shape[0], self.centers.shape[0]))

        for i in range(data.shape[0]):
            for j in range(self.centers.shape[0]):
                # Distance formula for K-Harmonic Means
                distance_ij = np.sqrt(np.sum(self.membership_degree[i, j]**2 * (data[i] - self.centers[j])**2))
                distances[i, j] = distance_ij

        return distances

    def update_centers(self, data):
        num_features = data.shape[1]
        self.centers = np.zeros((self.n_clusters, num_features))

        for j in range(self.n_clusters):
            # Update cluster centers
            numerator = np.sum((self.membership_degree[:, j]**self.m).reshape(-1, 1) * data, axis=0)
            denominator = np.sum(self.membership_degree[:, j]**self.m)
            self.centers[j] = numerator / denominator

    def update_membership_degree(self, data, distances):
        num_data = data.shape[0]
        membership_degree = np.zeros((num_data, self.n_clusters))

        for i in range(num_data):
            for j in range(self.n_clusters):
                # Calculate distance for K-Harmonic Means
                distance_ij = np.sqrt(np.sum((data[i] - self.centers[j])**2))
                # Update membership degree formula to avoid division by zero
                membership_degree[i, j] = 1 / np.maximum(1e-10, np.sum((distance_ij / distances[i, :])**(-2/(self.m-1))))

        self.membership_degree = membership_degree / np.sum(membership_degree, axis=1, keepdims=True)

    def update_weights(self):
        num_data = self.membership_degree.shape[0]
        self.weights = np.zeros((num_data, self.n_clusters))

        for i in range(num_data):
            for j in range(self.n_clusters):
                # Calculate weights or new cluster centers
                self.weights[i, j] = self.membership_degree[i, j]**2 / np.sum(self.membership_degree[:, j]**2)

    def sort_centers(self):
        # Sort new cluster centers columns
        sorted_centers = self.centers[:, np.argsort(-self.centers.max(axis=0))]
        return sorted_centers

    def calculate_sse_per_cluster(self, data, labels):
        num_clusters = len(set(labels))
        sse_per_cluster = np.zeros(num_clusters)

        for i in range(num_clusters):
            cluster_points = data[labels == i]
            sse_per_cluster[i] = np.sum(np.linalg.norm(cluster_points - self.centers[i], axis=1) ** 2)

        return sse_per_cluster

    def run_kharmonic_means(self, data, initializer):
        # Check for NaN in the input data
        if np.isnan(np.sum(data)) or np.isinf(np.sum(data)):
            print("Input data contains NaN or infinity. Check and resolve.")
            return

        for _ in range(5):  # Perform multiple times to try different initializations
            # Initialize cluster centers and membership degree using KMeans
            self.centers = initializer.initialize_centers(data)
            self.membership_degree = self.initialize_membership_degree(data.shape[0])

            # Save results in DataFrame
            result_list = []

            for iteration in range(self.max_iterations):
                # Calculate distances
                distances = self.kharmonic_means_distance(data)

                # Calculate Objective Function values
                objective_values = np.sum((self.membership_degree ** self.m) * distances ** 2, axis=1)

                # Update cluster centers
                old_centers = self.centers.copy()
                self.update_centers(data)

                # Update membership degree
                old_membership_degree = self.membership_degree.copy()
                self.update_membership_degree(data, distances)

                # Update weights or new cluster centers
                self.update_weights()

                # Check convergence
                center_change = np.sum(np.abs(self.centers - old_centers))
                membership_change = np.sum(np.abs(self.membership_degree - old_membership_degree))

                # Add iteration results to DataFrame
                result_list.append({
                    "Iteration": iteration + 1,
                    "Centers": self.centers.tolist(),
                    "Distances": np.diagonal(distances).tolist(),
                    "ObjectiveFunction": objective_values.tolist(),
                    "MembershipDegree": self.membership_degree.tolist(),
                    "Weights": self.weights.tolist()
                    })
                print(f"Iteration {iteration + 1}: Distances = {np.sum(np.diagonal(distances))}, Objective Function = {np.sum(objective_values)}")

                if np.sum(np.abs(self.centers - old_centers)) < self.tol and np.sum(np.abs(self.membership_degree - old_membership_degree)) < self.tol:
                    print(f"Convergence achieved at iteration {iteration + 1}.")
                    break
            self.df_results = pd.DataFrame(result_list)
            # Use the best parameters to calculate SSE
            distances = self.kharmonic_means_distance(data)
            sse = np.sum(np.min(distances, axis=1)**2)

            # Select parameters that give the lowest SSE
            if sse < self.best_sse:
                self.best_sse = sse
                self.df_jarak = pd.DataFrame(self.df_results["Distances"].tolist(), columns=[f"Distance_{i}" for i in range(len(self.df_results["Distances"].iloc[0]))])
    
def kharmonicmeans():
    st.subheader("Klastering UMKM K - Harmonic Means")
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

    st.header("K-Harmonic Means Clustering")

    num_clusters = 2
    m_value = 2

    st.write(f"Running K-Harmonic Means with {num_clusters} clusters and m={m_value}...")

    # Prepare data
    selected_columns = ['JUMLAH TENAGA KERJA', 'OMSET RATA', 'BIAYA PRODUKSI RATA']
    selected_data = df[selected_columns]
    data = selected_data.to_numpy()

    # Instantiate KMeansInitializer and KHarmonicMeans
    kmeans_initializer = KMeansInitializer(n_clusters=num_clusters, random_state=42)
    kharmonic_means_instance = KHarmonicMeans(n_clusters=num_clusters, m=m_value)

    # Run K-Harmonic Means
    kharmonic_means_instance.run_kharmonic_means(data, kmeans_initializer)

    # Using the best parameters to create the final results DataFrame
    df_final_centers = pd.DataFrame(kharmonic_means_instance.centers, columns=[f'Center_{i}' for i in range(kharmonic_means_instance.centers.shape[1])])
    df_membership_degree = pd.DataFrame(kharmonic_means_instance.membership_degree, columns=[f'Membership_Degree_{i}' for i in range(kharmonic_means_instance.membership_degree.shape[1])])
    df_weights = pd.DataFrame(kharmonic_means_instance.weights, columns=[f'Weight_{i}' for i in range(kharmonic_means_instance.weights.shape[1])])

    # Sorting the final centers
    df_sorted_final_centers = pd.DataFrame(kharmonic_means_instance.sort_centers(), columns=[f'Sorted_Center_{i}' for i in range(kharmonic_means_instance.centers.shape[1])])

    # Creating labels
    labels = np.argmax(kharmonic_means_instance.membership_degree, axis=1)

    # Adjust labels to start from 1
    adjusted_labels = labels + 1

    # Create a DataFrame for cluster labels
    df_labels = pd.DataFrame(adjusted_labels, columns=['Cluster_Label'])

    # Combine results into a single DataFrame
    df_combined = pd.concat([selected_data, kharmonic_means_instance.df_jarak, kharmonic_means_instance.df_results[['ObjectiveFunction']], df_membership_degree, df_weights, df_labels], axis=1)

    # Display the combined DataFrame
    st.write("Combined Results DataFrame:")
    st.write(df_combined)

    # Additional visualizations
    st.write("Additional Visualizations:")

    # Example: Silhouette Score Plot
    silhouette_avg = silhouette_score(data, np.argmax(kharmonic_means_instance.membership_degree, axis=1))
    st.write(f"Silhouette Score: {silhouette_avg}")

    silhouette_values = silhouette_samples(data, labels)
    cluster_silhouette_avg = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_silhouette_avg[i] = np.mean(silhouette_values[labels == i])

    fig, ax = plt.subplots()
    ax.bar(range(1, num_clusters + 1), cluster_silhouette_avg)
    ax.set_title('Silhouette Score per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Silhouette Score Average')
    ax.set_xticks(range(1, num_clusters + 1))
    ax.set_xticklabels([f'Cluster {i}' for i in range(1, num_clusters + 1)])

    st.pyplot(fig)
    
def main():
    st.sidebar.title("Menu Aplikasi")
    app_mode = st.sidebar.selectbox("Pilihan", ["Home","Dataset", "Normalisasi Data", "K - Means", "K - Harmonic Means"])

    if app_mode == "Home":
        home()
    elif app_mode == "Dataset":
        dataset()
    elif app_mode == "Normalisasi Data":
        normalisasidata()
    elif app_mode == "K - Means":
        kmeanscluster()
    elif app_mode == "K - Harmonic Means":
        kharmonicmeans()

if __name__ == "__main__":
    main()
