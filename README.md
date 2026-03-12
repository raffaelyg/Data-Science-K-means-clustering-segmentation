# Unsupervised Learning Framework: K-Means Clustering for High-Dimensional Behavioural Segmentation

![Project Type](https://img.shields.io/badge/Project-Unsupervised_Learning-blue)
![Course](https://img.shields.io/badge/University-Cambridge_C101-gold)
![Algorithm](https://img.shields.io/badge/Algorithm-K--Means_Clustering-green)

## 📌 Project Overview
Completed as part of the **University of Cambridge** Data Science curriculum, this project implements an unsupervised learning pipeline to decode complex consumer patterns. By transitioning from traditional demographic snapshots to behavioural clustering, this framework allows for precision "Segmented Activation" in retail environments—identifying the latent structures in how different cohorts value and spend.

## 💡 Analytical Objective
The goal was to apply **K-Means Clustering** to segment a retail database based on **Annual Income** and **Spending Score**. The project focuses on mathematical stability (cluster cohesion) and strategic interpretability to drive personalised marketing and loyalty optimisation.

---

## 🛠️ Technical Workflow & Implementation

### 1. Feature Engineering & Scaling
K-Means is a distance-based algorithm (Euclidean distance); therefore, feature scaling is a critical prerequisite to ensure that features with larger ranges (like Annual Income) do not bias the model.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encoding categorical gender data
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Standardising features to mean=0 and variance=1
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
```


## 2. Optimising K (The Elbow Method)

To find the optimal number of clusters, I performed an iterative evaluation of the **Within-Cluster Sum of Squares (WCSS)**.

```python
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)
```

## 📊 Result Analysis & Visual Gallery
The analysis successfully converged on 5 distinct clusters, providing a clear map of the consumer landscape.

### I. Model Stability: The Elbow Plot

![Elbow_test.png](/images/Elbow_test.png)

* **Interpretation**: The "Elbow" point at K=5 signifies the optimal balance between cluster granularity and model simplicity. Beyond this point, the reduction in WCSS becomes marginal, indicating diminishing returns in segment clarity.

### II. Silhouette Analysis

| *n=* | Silhouette Analysis | *n=* | Silhouette Analysis | 
| :---: | :---: | :---: | :---: | 
| 1 | n/a | 6 | ![Cluster n=6.png](/images/Unknown-29.png) |
| 2 | ![Cluster n=2.png](/images/Unknown-26.png) | 7 | ![Cluster n=7.png](/images/Unknown-30.png) |
| 3 | ![Cluster n=3.png](/images/Unknown-26.png) | 8 | ![Cluster n=8.png](/images/Unknown-31.png) |
| 4 | ![Cluster n=4.png](/images/Unknown-27.png) | 9 | ![Cluster n=9.png](/images/Unknown-32.png) |
| 5 | ![Cluster n=5.png](/images/Unknown-28.png) | 10 | ![Cluster n=10.png](/images/Unknown-33.png) |

![Dendrogram](/images/Heriarchical%20Dendrogram.png)

* **Interpretation**: This metric validated the intra-cluster density. A high silhouette score across the 5 clusters confirmed that the segments are well-separated and internally consistent.

### III. Behavioural Cluster Mapping

![PCA](/images/PCA.png) 

![t-SNE](/images/t-sne.png)


* **Interpretation**: Both PCA and t-SNE confirmed that the 3-cluster model is the most stable. The visual separation between the three groups indicates that the model has successfully captured the primary "Behavioural Drivers" without over-fitting the data.
* The 3-cluster projection reveals three core personas, each representing a specific "Job to be Done" for the brand.

![distribution](/images/distribution.png)


| Cluster	| Archetype	| Behavioural Insight | 
|:---:|:---:|:---|
| Cluster 0	| The Prime Target	| High Spending Score + Mid-to-High Income. This group represents the "Value Engine" of the brand. | 
| Cluster 1	| The Mature/Conservative	| Higher Age + Lower Spending. Wealthy but cautious; requires "Pull" marketing to re-engage. | 
| Cluster 2	| The Emerging/Active	| Lower Age + High Spending. Reactive to trends and social commerce (Douyin/Kuaishou) triggers. | 



## 🔑 Key Takeaways & Strategic Applications
**Precision Activation**: Instead of broad-spectrum discounting, we can now apply Value-Based Marketing. Cluster 1 (Target) receives exclusive VIP early access, while Cluster 4 (High-Potential) receives education-based content to justify premiumisation.

**Mitigating the "Premium Sandwich"**: By identifying Cluster 4, brands can design specific interventions to increase their spending score without relying on mass-market promotions that erode margins.

**Synergy with Market Research**: This unsupervised approach complements my work in APC Strategic Intelligence, providing the quantitative "backbone" for qualitative consumer deep-dives.


---
2026 © Raffael Yuliang Gao | [LinkedIn](http://www.linkedin.com/in/raffaelyg/)
