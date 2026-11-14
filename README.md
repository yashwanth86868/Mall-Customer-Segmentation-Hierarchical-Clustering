
# Mall Customer Segmentation Using Hierarchical Clustering

## Project Overview

This project implements an unsupervised machine learning algorithm to segment mall customers based on their annual income and spending scores.

The objective is to identify different customer groups and help businesses understand customer purchasing behavior.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- Streamlit

## Machine Learning Algorithm

Agglomerative Hierarchical Clustering

### Techniques Used

- Data preprocessing
- Feature selection
- Standardization
- Dendrogram visualization
- Ward's linkage method
- Euclidean distance

## Features

- Interactive Streamlit dashboard
- Customer segmentation visualization
- Adjustable number of clusters
- Hierarchical dendrogram
- Cluster distribution analysis
- Downloadable clustering results

## Project Structure

```text
Mall-Customer-Segmentation/
│
├── data/
│   └── Mall_Customers.csv
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd Mall-Customer-Segmentation-Hierarchical-Clustering
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Machine Learning Workflow

1. Load the customer dataset.
2. Select annual income and spending score.
3. Standardize the selected features.
4. Generate the hierarchical dendrogram.
5. Apply Agglomerative Clustering.
6. Visualize customer segments.
7. Download the clustered dataset.

## Business Applications

- Customer behavior analysis
- Targeted marketing campaigns
- Customer segmentation
- Personalized product recommendations
- Business strategy development

## Author

Yashwanth Balija

Aspiring AI/ML Engineer
