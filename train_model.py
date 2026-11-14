import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
dataset = pd.read_csv(BASE_DIR / "data" / r'C:\Users\ybali\Downloads\Mall_Customers.csv')

print("Dataset loaded successfully!")
print(dataset.head())

# Select features
X = dataset.iloc[:, [3, 4]].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dendrogram
plt.figure(figsize=(12, 6))

sch.dendrogram(
    sch.linkage(X_scaled, method="ward")
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig(BASE_DIR / "dendrogram.png")
plt.show()

# Agglomerative Clustering
hc = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

y_hc = hc.fit_predict(X_scaled)

# Add cluster labels
dataset["Cluster"] = y_hc

# Save results
dataset.to_csv(
    BASE_DIR / "Mall_Customers_Clustered.csv",
    index=False
)

print("\nClustering completed successfully!")

print(dataset.head())

print("\nCluster distribution:")
print(dataset["Cluster"].value_counts())

print("\nResults saved successfully!")