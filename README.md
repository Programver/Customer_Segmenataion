# Customer Segmentation using Unsupervised Learning

This project demonstrates how to apply **unsupervised machine learning** techniques to segment customers based on their demographic and behavioral data. By clustering similar customers, businesses can target each group more effectively with personalized marketing strategies.

## Project Overview

Customer segmentation is a key task in customer relationship management. This project uses **K-Means Clustering** and **t-SNE** (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction and visualization of high-dimensional customer data.

## Dataset

The dataset (`new.csv`) contains customer information including:

- Demographic data (Age, Income, Education, Marital Status, etc.)
- Spending data across multiple product categories
- Customer registration date (`Dt_Customer`)
- Campaign response indicators

## Technologies & Libraries Used

- **Python**
- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — data visualization
- `scikit-learn` — preprocessing, clustering, t-SNE

## Project Steps

### 1. Data Cleaning & Preparation
- Removed irrelevant and null-value columns (`Z_CostContact`, `Z_Revenue`, `Dt_Customer`)
- Extracted `day`, `month`, `year` from customer registration date
- Encoded categorical features using `LabelEncoder`

### 2. Exploratory Data Analysis (EDA)
- Count plots of all categorical features
- Correlation heatmap to inspect highly correlated numerical features

### 3. Feature Scaling
- Applied `StandardScaler` to normalize all numerical values for clustering

### 4. Dimensionality Reduction (t-SNE)
- Reduced data to 2D for visualization using t-SNE
- Visualized customers in reduced feature space

### 5. Clustering (K-Means)
- Used **Elbow Method** to find the optimal number of clusters
- Applied K-Means clustering with chosen cluster count
- Visualized clustered segments in 2D space using t-SNE results
