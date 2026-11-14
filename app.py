import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Title
st.title("🛍️ Mall Customer Segmentation")
st.subheader("Hierarchical Clustering using Machine Learning")

st.write(
    "This application segments mall customers "
    "based on their annual income and spending score."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / r'C:\Users\ybali\Downloads\Mall_Customers.csv')

try:
    dataset = load_data()

except FileNotFoundError:
    st.error("Dataset not found. Please add Mall_Customers.csv inside the data folder.")
    st.stop()

# Sidebar
st.sidebar.header("Clustering Settings")

n_clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)

# Features
X = dataset.iloc[:, [3, 4]].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="euclidean",
    linkage="ward"
)

dataset["Cluster"] = hc.fit_predict(X_scaled)

# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(dataset))

with col2:
    st.metric("Total Clusters", n_clusters)

with col3:
    st.metric("Features Used", 2)

st.divider()

# Customer dataset
st.header("📊 Customer Dataset")

st.dataframe(dataset, use_container_width=True)

# Cluster visualization
st.header("🎯 Customer Segmentation")

fig, ax = plt.subplots(figsize=(10, 6))

colors = [
    "red", "blue", "green", "cyan", "magenta",
    "orange", "purple", "brown", "pink", "gray"
]

for cluster in range(n_clusters):

    cluster_data = dataset[dataset["Cluster"] == cluster]

    ax.scatter(
        cluster_data.iloc[:, 3],
        cluster_data.iloc[:, 4],
        s=80,
        color=colors[cluster % len(colors)],
        label=f"Cluster {cluster}"
    )

ax.set_title("Mall Customer Segments")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()

st.pyplot(fig)
plt.close(fig)

# Dendrogram
st.header("🌳 Hierarchical Clustering Dendrogram")

fig, ax = plt.subplots(figsize=(12, 5))

sch.dendrogram(
    sch.linkage(X_scaled, method="ward"),
    ax=ax
)

ax.set_title("Customer Hierarchical Dendrogram")
ax.set_xlabel("Customers")
ax.set_ylabel("Euclidean Distance")

st.pyplot(fig)
plt.close(fig)

# Cluster distribution
st.header("📈 Cluster Distribution")

cluster_counts = dataset["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_counts)

# Download results
st.header("📥 Download Results")

csv = dataset.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Clustered Dataset",
    data=csv,
    file_name="Mall_Customers_Clustered.csv",
    mime="text/csv"
)

st.success("Customer segmentation completed successfully!")