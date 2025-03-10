import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("/mnt/data/StudentsPerformance.csv")

# Step 2: Data Overview
print(df.info())
print(df.describe())

# Step 3: Display PairPlot
sns.pairplot(df, hue="gender")
plt.show()

# Step 4: Preprocessing - Selecting numerical features for clustering
features = ["math score", "reading score", "writing score"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 6: Visualizing Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["math score"], y=df["reading score"], hue=df["Cluster"], palette="viridis")
plt.title("K-Means Clustering of Student Performance")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.legend()
plt.show()

# Step 7: Display Cluster Centers
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))

# Step 8: Evaluate Clustering Performance
silhouette_avg = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# Step 9: Determine Optimal Number of Clusters using Elbow Method
distortions = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    distortions.append(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])

plt.figure(figsize=(8, 6))
plt.plot(k_range, distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 10: Apply K-Medoids Clustering
from sklearn_extra.cluster import KMedoids
kmedoids = KMedoids(n_clusters=3, random_state=42)
df["KMedoids_Cluster"] = kmedoids.fit_predict(X_scaled)

# Step 11: Visualizing K-Medoids Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["math score"], y=df["reading score"], hue=df["KMedoids_Cluster"], palette="coolwarm")
plt.title("K-Medoids Clustering of Student Performance")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.legend()
plt.show()

# Step 12: Compare Cluster Assignments
comparison = df[["Cluster", "KMedoids_Cluster"]].value_counts()
print("Comparison of K-Means and K-Medoids Clusters:")
print(comparison)

# Step 13: Evaluate K-Medoids Clustering Performance
silhouette_avg_medoid = silhouette_score(X_scaled, df["KMedoids_Cluster"])
print(f"Silhouette Score for K-Medoids: {silhouette_avg_medoid}")

# Step 14: Analyze Cluster Characteristics
cluster_means = df.groupby("Cluster")[features].mean()
print("Cluster Characteristics for K-Means:")
print(cluster_means)

medoid_cluster_means = df.groupby("KMedoids_Cluster")[features].mean()
print("Cluster Characteristics for K-Medoids:")
print(medoid_cluster_means)

# Step 15: Conclusion
print("Conclusion: The clustering results from K-Means and K-Medoids have been analyzed, and the best approach depends on the dataset characteristics and business context.")
