# CS201L – Lab 12 Viva Questions & Answers
## Clustering: K-Means and K-Medoid

---

## Section 1: Conceptual Foundations

**Q1. What is clustering? How does it differ from classification?**

Clustering is an *unsupervised* learning technique that groups data points into clusters based on similarity, without using any pre-existing class labels. Classification, on the other hand, is a *supervised* task that trains a model on labeled data to predict the class of unseen examples. In clustering, the algorithm discovers the structure on its own; in classification, that structure is explicitly given by labels.

---

**Q2. What is K-Means clustering? Briefly describe its algorithm.**

K-Means partitions N data points into K clusters by minimizing the within-cluster sum of squared distances to the cluster centroid.

**Algorithm:**
1. Initialize K centroids randomly (or with K-Means++).
2. **Assign** each point to the nearest centroid.
3. **Update** each centroid to the mean of all points assigned to it.
4. Repeat steps 2–3 until convergence (centroids stop moving or labels stop changing).

---

**Q3. What is the objective function minimized by K-Means?**

K-Means minimizes the **inertia** (within-cluster sum of squares):

```
J = Σ_k Σ_{x ∈ C_k} ||x - μ_k||²
```

where `μ_k` is the centroid of cluster `C_k` and `||·||²` is the squared Euclidean distance.

---

**Q4. What is K-Medoid clustering? How does it differ from K-Means?**

| Property | K-Means | K-Medoid |
|---|---|---|
| Center | Mean of cluster (can be a non-existent point) | Actual data point (medoid) |
| Distance metric | Euclidean (squared) | Any metric (more flexible) |
| Outlier sensitivity | High – mean is pulled by outliers | Low – medoid is a real point |
| Computational cost | Lower – O(NKT) | Higher – O(N²KT) for PAM |
| Use case | Numeric, spherical data | Non-Euclidean or noisy data |

---

**Q5. What is the PAM algorithm used in K-Medoids?**

PAM (**Partitioning Around Medoids**) works as follows:
1. Initialize K medoids (randomly or heuristically).
2. Assign each non-medoid point to its nearest medoid.
3. For each medoid `m` and each non-medoid `o`, compute the cost change if `o` replaces `m`.
4. Perform the swap that gives the greatest cost reduction.
5. Repeat until no swap improves the total cost.

---

## Section 2: Evaluation Metrics

**Q6. What is the Purity Score? How is it computed?**

Purity measures how homogeneous each cluster is with respect to the true labels.

```
Purity = (1/N) × Σ_k max_j |C_k ∩ T_j|
```

- `C_k` = set of points in cluster k  
- `T_j` = set of points with true label j  
- For each cluster, we find the majority true class and count those points. Sum over all clusters, then divide by N.

**Range:** [0, 1]. Higher is better. A value of 1 means every cluster contains only one class.

---

**Q7. What is Normalized Mutual Information (NMI)?**

NMI measures the mutual information between the clustering assignment and the true labels, normalized to lie in [0, 1]:

```
NMI(U, V) = 2 × I(U; V) / [H(U) + H(V)]
```

- `I(U; V)` = mutual information between cluster labels U and true labels V  
- `H(·)` = entropy  

**Range:** [0, 1]. A score of 1 indicates perfect agreement between clusters and true labels; 0 indicates no mutual information.

---

**Q8. What is the Silhouette Score? What does it measure?**

The Silhouette Score measures how well each point fits into its own cluster compared to neighboring clusters.

For a single point `i`:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- `a(i)` = mean intra-cluster distance (to other points in the same cluster)  
- `b(i)` = mean distance to the nearest different cluster  

**Range:** [-1, 1]  
- Close to **+1**: point is well-matched to its own cluster  
- Close to **0**: point is on the boundary  
- Close to **-1**: point may be misclassified  

The overall score is the average `s(i)` across all points.

---

**Q9. What is inertia/distortion in K-Means? What does a lower value indicate?**

Inertia is the total within-cluster sum of squared Euclidean distances from each point to its cluster centroid. A lower inertia means more compact, tightly-knit clusters. However, inertia always decreases as K increases (K = N gives inertia = 0), so it cannot be used alone to choose K.

---

## Section 3: Elbow Method & Model Selection

**Q10. What is the Elbow Method for choosing K?**

The Elbow Method plots inertia (distortion) against different values of K. As K increases, inertia decreases. At some point the decrease slows dramatically, forming an "elbow" shape. The K at the elbow is taken as the heuristic optimal number of clusters — beyond it, adding more clusters yields diminishing returns.

**Limitation:** The elbow is not always well-defined; the method is heuristic and subjective.

---

**Q11. Why is K=6 expected to be near-optimal for this dataset?**

The dataset contains speech audio embeddings from exactly **6 languages** (English, Hindi, Kannada, Marathi, Bengali, Manipuri). Since t-SNE was applied to separate these embeddings in 2D, we expect the elbow in the distortion curve to appear around K=6, matching the true number of classes.

---

**Q12. Besides the elbow method, what other methods exist for choosing K?**

- **Silhouette Analysis:** Choose K that maximizes average silhouette score.
- **Gap Statistic:** Compares within-cluster dispersion to expected dispersion under a null reference distribution.
- **Bayesian Information Criterion (BIC) / AIC:** For Gaussian Mixture Models.
- **Davies-Bouldin Index:** Lower is better; measures ratio of within-cluster scatter to between-cluster separation.
- **Calinski-Harabasz Index:** Higher is better; ratio of between-cluster to within-cluster dispersion.

---

## Section 4: t-SNE and the Dataset

**Q13. What is t-SNE and why is it used here?**

t-SNE (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that maps high-dimensional data to 2D (or 3D) while preserving local neighborhood structure. It is used here to visualize high-dimensional speech audio embeddings in 2D so that clustering algorithms can be applied and visualized easily.

---

**Q14. Why should the language column be excluded during clustering?**

Clustering is unsupervised — it must discover groupings from feature data alone without access to labels. Including the language column would be data leakage and would trivially give perfect clusters. Labels are used **only after** clustering, to compute evaluation metrics (Purity, NMI).

---

## Section 5: Practical & Code Questions

**Q15. What does `KMeans.labels_` return?**

It returns a NumPy array of integers of length N, where each integer is the cluster index (0 to K-1) assigned to the corresponding data point after fitting.

---

**Q16. What does `KMeans.cluster_centers_` return?**

A 2D NumPy array of shape (K, n_features) containing the coordinates of the K cluster centroids in the feature space.

---

**Q17. What is `random_state` in KMeans and why is it important?**

`random_state` seeds the random number generator used for centroid initialization. Setting it ensures **reproducibility** — running the same code twice gives the same result. Without it, results may differ across runs due to random initialization.

---

**Q18. What is K-Means++ initialization? Why is it preferred over random initialization?**

K-Means++ spreads the initial centroids apart by selecting each new centroid with probability proportional to its squared distance from the nearest already-chosen centroid. This leads to:
- Faster convergence
- Lower inertia on average
- Avoiding degenerate initializations

Scikit-learn uses K-Means++ by default (`init='k-means++'`).

---

**Q19. Can K-Means handle non-spherical clusters? What is a better alternative?**

K-Means assumes clusters are **convex and isotropic (spherical)**. It performs poorly on elongated, ring-shaped, or irregular clusters. Better alternatives include:
- **DBSCAN** (density-based, handles arbitrary shapes)
- **Gaussian Mixture Models** (soft assignments, handles elliptical clusters)
- **Spectral Clustering** (graph-based, handles complex shapes)

---

**Q20. How does K-Medoid handle outliers better than K-Means?**

In K-Means, the centroid is the arithmetic mean of all points in the cluster. A single extreme outlier can shift the centroid far from the cluster's bulk. In K-Medoid, the center must be an **actual data point** (the medoid), so it cannot be pulled arbitrarily far by outliers. The medoid is the point that minimizes the total distance to all other points in the cluster, making it more robust.

---

*End of Viva Q&A — Lab 12*
