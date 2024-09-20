## K-Means Clustering for Customer Segmentation

This project demonstrates the use of the K-Means clustering algorithm to segment customers based on their annual income and spending score. Below is a step-by-step explanation of how the code works:

### 1. Importing the Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```
We start by importing the necessary libraries:
- `numpy`: For numerical operations.
- `pandas`: To handle data in DataFrame format.
- `matplotlib.pyplot` and `seaborn`: For data visualization.
- `KMeans` from `sklearn`: To perform clustering.

### 2. Data Collection & Analysis
```python
customer_data = pd.read_csv('/content/Mall_Customers.csv')
customer_data.head()
customer_data.shape
customer_data.info()
customer_data.isnull().sum()
```
The dataset is loaded using `pd.read_csv()`. We perform basic exploratory data analysis:
- `head()` displays the first five rows of the data.
- `shape` shows the dimensions of the dataset (200 rows and 5 columns).
- `info()` provides details about the columns, data types, and non-null entries.
- `isnull().sum()` checks for any missing values (there are none in this dataset).

### 3. Choosing the Features for Clustering
```python
X = customer_data.iloc[:,[3,4]].values
print(X)
```
We choose the **Annual Income (k$)** and **Spending Score (1-100)** columns as the features for clustering, and store them in the variable `X`. This two-dimensional dataset will be used for the K-Means algorithm.

### 4. Choosing the Number of Clusters using the Elbow Method
```python
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```
We calculate the **Within-Cluster Sum of Squares (WCSS)** for different values of `n_clusters` (from 1 to 10) to identify the optimum number of clusters using the **Elbow Method**. The goal is to minimize the WCSS while keeping the number of clusters small.

```python
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```
We then plot the **Elbow Graph**. The 'elbow' point in the graph is where the WCSS stops decreasing significantly, indicating the optimal number of clusters. In this case, the optimum number of clusters is **5**.

### 5. Training the K-Means Clustering Model
```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
print(Y)
```
We train the K-Means model with `n_clusters=5` and fit it to our dataset `X`. The `fit_predict()` method assigns each data point to one of the clusters (represented by labels `0` to `4`).

### 6. Visualizing the Clusters
```python
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```
Finally, we plot the clusters and their centroids. Each cluster is represented by a different color:
- Cluster 1: Green
- Cluster 2: Red
- Cluster 3: Yellow
- Cluster 4: Violet
- Cluster 5: Blue

The centroids of the clusters are plotted in cyan. This visualization shows how the customers are grouped based on their annual income and spending score.
