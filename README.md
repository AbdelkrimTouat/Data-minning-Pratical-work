# Data Mining Practical Work

This repository contains a series of three practical works (TPs) focused on data mining techniques. Each TP is presented as a Jupyter Notebook and covers different stages of the data mining process, from data preprocessing to clustering and evaluation.

## Description

These practical works aim to provide hands-on experience with fundamental data mining concepts and algorithms. They guide the user through data manipulation, exploration, normalization, application of various clustering methods, and comparison of their performance.

## Practical Work (TP) Overview

### 📂 TP1: Data Exploration and Preprocessing (`Data_Mining_TP1.ipynb`)

* **Objective:** This notebook focuses on the initial stages of data mining, including loading, understanding, and preparing a dataset for analysis.
* **Key Concepts Covered:**
    * Importing necessary libraries (Pandas, NumPy, Matplotlib, Seaborn).
    * Loading and inspecting the dataset.
    * Exploratory Data Analysis (EDA): understanding data types, distributions, and identifying missing values.
    * Data cleaning and handling missing values.
    * Data normalization and scaling techniques (e.g., Min-Max normalization, Z-score normalization).

### 📂 TP2: Introduction to Clustering (`Data_Mining_TP2.ipynb`)

* **Objective:** This notebook introduces fundamental clustering algorithms and techniques for grouping similar data points.
* **Key Concepts Covered:**
    * K-Means clustering: algorithm implementation, determining the optimal number of clusters (e.g., Elbow method, Silhouette score).
    * K-Medoids clustering: algorithm implementation and comparison with K-Means.
    * Principal Component Analysis (PCA) for dimensionality reduction and visualization of clusters.
    * Visualization of clustering results.

### 📂 TP3: Advanced Clustering Techniques and Evaluation (`Data_Mining_TP3.ipynb`)

* **Objective:** This notebook expands on clustering by exploring additional algorithms and focusing on the comparative evaluation of their performance.
* **Key Concepts Covered:**
    * Application of various clustering algorithms:
        * K-Means
        * K-Medoids
        * Agglomerative Hierarchical Clustering (AGNES)
        * Divisive Hierarchical Clustering (DIANA)
        * Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    * Determining the optimal number of clusters for different methods.
    * Comparative analysis of clustering results using metrics such as inertia and silhouette score.
    * Visualization and interpretation of the performance of different clustering techniques.

## Getting Started

To run these notebooks, you will need a Python environment with Jupyter Notebook or JupyterLab installed.

### Prerequisites

Ensure you have the following libraries installed:

* Python (3.x recommended)
* Jupyter Notebook or JupyterLab
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

You can typically install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
