import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import streamlit as st
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
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
