# Lab 12 – Viva Questions & Answers
## Clustering: K-Means and K-Medoid | CS201L: AI Laboratory | IIT Dharwad

---

## Section 1 – Conceptual Foundations

**Q1. What is clustering, and how does it differ from classification?**

Clustering is an **unsupervised** learning technique that groups data points such that points within a group (cluster) are more similar to each other than to points in other groups. Unlike classification, there are no pre-defined labels; the algorithm discovers structure on its own. Classification is supervised — it learns from labelled training data and predicts labels for new data.

---

**Q2. What is the K-Means algorithm? Describe its steps.**

K-Means partitions `n` data points into `K` clusters by minimising the within-cluster sum of squared distances (inertia).

**Steps:**
1. Choose `K` and randomly initialise `K` centroids.
2. **Assignment step** – assign each point to the nearest centroid (Euclidean distance).
3. **Update step** – recompute each centroid as the mean of its assigned points.
4. Repeat steps 2–3 until centroids do not change (convergence) or a maximum iteration count is reached.

---

**Q3. What is the K-Medoid algorithm, and how does it differ from K-Means?**

K-Medoid (PAM – Partitioning Around Medoids) also partitions data into `K` clusters, but the **cluster representative is always an actual data point (medoid)**, not a computed mean.

| Property | K-Means | K-Medoid |
|---|---|---|
| Centre | Mean of cluster | Actual data point |
| Sensitivity to outliers | High | Low |
| Distance metric | Euclidean (typically) | Any distance |
| Computational cost | Lower | Higher |
| Works on non-numeric data | No | Yes (with suitable distance) |

---

**Q4. Why do we apply PCA before clustering in this lab?**

- **Dimensionality reduction** – Iris has 4 features; PCA compresses them to 2 principal components while retaining most variance (~97%).
- **Visualisation** – 2D data can be plotted directly as scatter plots.
- **Noise reduction** – PCA can remove low-variance dimensions that may act as noise.
- **Speed** – Fewer dimensions mean faster distance computations.

---

**Q5. Why is normalisation/standardisation necessary before PCA and clustering?**

Features measured on different scales (e.g., sepal length in cm vs. petal width in cm) will dominate distance calculations if not standardised. StandardScaler transforms each feature to zero mean and unit variance, ensuring all features contribute equally to PCA and clustering.

---

## Section 2 – Metrics

**Q6. What is the Purity Score? How is it computed?**

Purity measures how well clusters align with true class labels.

**Formula:**
$$\text{Purity} = \frac{1}{N} \sum_{k=1}^{K} \max_{j} |C_k \cap T_j|$$

Where:
- `N` = total number of data points
- `C_k` = set of points in cluster `k`
- `T_j` = set of points with true label `j`

For each cluster, find the true class that is most represented and count those points. Sum across clusters, then divide by `N`. Ranges from 0 to 1; higher is better.

---

**Q7. What is the NMI (Normalized Mutual Information) Score?**

NMI measures the mutual information between the cluster assignments and the true labels, normalised to lie in [0, 1].

$$\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

- `I(U;V)` = mutual information between cluster labels `U` and true labels `V`
- `H(U)`, `H(V)` = entropies of `U` and `V`

NMI = 1 → perfect clustering; NMI = 0 → no mutual information (random). It is more robust than purity because it penalises trivial solutions (e.g., one point per cluster gives purity = 1 but NMI ≈ 0).

---

**Q8. What is the Silhouette Score?**

The Silhouette Score measures how similar a point is to its own cluster compared to other clusters, without using true labels (unsupervised metric).

For each point `i`:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\ b(i))}$$

- `a(i)` = mean intra-cluster distance (distance to other points in same cluster)
- `b(i)` = mean nearest-cluster distance (distance to points in the closest different cluster)

Range: [-1, 1].  
- Close to +1 → well-clustered  
- Close to 0 → on cluster boundary  
- Close to -1 → possibly mis-assigned

---

**Q9. What is the distortion/inertia measure?**

Inertia is the **within-cluster sum of squared distances** from each point to its cluster centroid/medoid.

$$\text{Inertia} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$

Lower inertia means tighter (better) clusters. It always decreases as `K` increases, so it cannot be used alone to choose `K`.

---

**Q10. What is the Elbow Method, and how does it help choose K?**

Plot inertia vs. number of clusters `K`. As `K` increases, inertia decreases sharply at first, then levels off. The **elbow point** — where the rate of decrease sharply slows — is the heuristic optimal `K`.

For the Iris 2D dataset, inertia drops significantly from K=2 to K=4 and then the decrease becomes gradual, suggesting K=3 or K=4 as the elbow.

---

## Section 3 – Implementation Details

**Q11. What does `KMeans.labels_` return?**

It returns a NumPy array of integers of length `n_samples`, where each integer is the cluster index (0 to K-1) assigned to the corresponding data point.

---

**Q12. What does `KMeans.cluster_centers_` return?**

A 2D array of shape `(K, n_features)` containing the coordinates of the `K` cluster centroids in the feature space.

---

**Q13. What does `KMedoids.cluster_centers_` return vs K-Means?**

Both return shape `(K, n_features)`, but K-Medoid centres are **actual data points** (medoids) from the dataset, while K-Means centres are computed means that may not correspond to any real point.

---

**Q14. Why do we set `random_state=42` in KMeans and KMedoids?**

Clustering algorithms initialise centroids/medoids randomly. Setting `random_state` makes results **reproducible** — the same random seed guarantees the same initialisation and therefore the same output across runs.

---

**Q15. What is the `n_init` parameter in KMeans?**

`n_init` specifies how many times KMeans is run with different random centroid initialisations. The best result (lowest inertia) across all runs is returned. Default is 10. This mitigates the risk of converging to a local minimum.

---

## Section 4 – PCA Deep Dive

**Q16. What are Principal Components?**

Principal Components are the eigenvectors of the covariance matrix of the data, ordered by their corresponding eigenvalues (variance). The first PC explains the most variance, the second the next most (orthogonal to the first), and so on.

---

**Q17. How much variance is retained when Iris is reduced to 2D via PCA?**

Approximately **97%** of the total variance is captured by the first two principal components of the standardised Iris dataset, making 2D reduction very effective for this dataset.

---

**Q18. Can PCA be applied directly on the raw (un-normalised) Iris data? Why or why not?**

Technically yes, but it is incorrect practice. Without normalisation, features with larger ranges dominate the covariance matrix and the resulting PCs will mainly reflect those features' variance rather than the true structure of the data. Always standardise before PCA.

---

## Section 5 – Comparative Analysis

**Q19. Which clustering method performed better on the Iris dataset — K-Means or K-Medoid? Why?**

Both perform similarly on the Iris dataset since it is well-structured and approximately spherical after PCA. K-Medoid may show marginally more robustness if any outlier points exist, as medoids resist being pulled by extreme values. Compare purity, NMI, and silhouette scores to determine which is better for this specific run.

---

**Q20. In what scenarios would K-Medoid be preferred over K-Means?**

- When data contains **outliers** (medoids are robust to them).
- When the **mean is not meaningful** (e.g., categorical, ordinal, or non-Euclidean data).
- When the cluster representative must be an **interpretable, real data point**.
- When using **non-Euclidean distances** (e.g., cosine, Manhattan, edit distance).

---

**Q21. What are the limitations of K-Means clustering?**

1. Requires specifying `K` in advance.
2. Sensitive to outliers (outliers pull centroids).
3. Assumes spherical, equally-sized clusters — struggles with elongated or irregular shapes.
4. May converge to local minima (mitigated by multiple initialisations).
5. Assumes Euclidean distance.

---

**Q22. What is the time complexity of K-Means?**

O(n · K · d · I) where:
- `n` = number of data points
- `K` = number of clusters
- `d` = number of dimensions
- `I` = number of iterations

K-Medoid (PAM) is O(K · (n − K)² · I), which is significantly more expensive for large `n`.

---

## Section 6 – Tricky/Advanced Questions

**Q23. Can purity score be 1 by using a trivial clustering? How does NMI handle this?**

Yes — if each point is its own cluster (K = n), every cluster is perfectly pure. Purity = 1, but this is meaningless. NMI penalises this because the entropy of cluster assignments `H(U)` becomes very high, reducing the NMI score. This is why NMI is a more reliable metric than purity alone.

---

**Q24. What happens if K equals the number of data points in K-Means?**

Inertia becomes 0 (every point is its own centroid). This is the trivial overfitting case — useless for generalisation or insight.

---

**Q25. How would you choose K if the elbow method gives an ambiguous result?**

- Use the **Silhouette Score** — maximise it over a range of K values.
- Use the **Gap Statistic** — compare inertia to a null reference distribution.
- Apply **domain knowledge** — e.g., for Iris, we know there are 3 species.
- Use **BIC/AIC** with Gaussian Mixture Models as an alternative.

---

*End of Viva Q&A — Lab 12 | Good luck!*
