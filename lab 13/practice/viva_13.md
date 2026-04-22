# CS201L — Lab 13 Viva Questions & Answers
## Hierarchical and DBSCAN Clustering
**IIT Dharwad | Roll No: IS24BM003**

---

## Section 1: Conceptual Foundations

**Q1. What is hierarchical clustering and how does it differ from partition-based clustering like K-means?**

Hierarchical clustering builds a tree of clusters (dendrogram) by successively merging (agglomerative) or splitting (divisive) data points without requiring the number of clusters `k` upfront. K-means partitions data into exactly `k` clusters in a flat structure and requires `k` as input. Hierarchical clustering is deterministic and produces a full merge history, while K-means is iterative and sensitive to initialisation.

---

**Q2. What is agglomerative clustering? Describe its algorithm step by step.**

Agglomerative clustering is a bottom-up approach:
1. Start: each data point is its own cluster (N clusters).
2. Compute pairwise distances between all clusters.
3. Merge the two closest clusters into one.
4. Update the distance matrix using the chosen linkage criterion.
5. Repeat steps 3–4 until only one cluster remains.

The result is a dendrogram representing the full merge history.

---

**Q3. What is a dendrogram and what information does it convey?**

A dendrogram is a tree diagram where:
- Leaf nodes represent individual data points.
- The height (y-axis) at which two branches merge indicates the distance (dissimilarity) between those clusters at the point of merging.
- Cutting the dendrogram at a given height gives a flat clustering with the number of clusters equal to the number of branches crossed by the horizontal cut.
- Tall vertical lines indicate large gaps between merges — good cut points for natural clusters.

---

**Q4. Explain the four linkage methods: Ward, Complete, Average, and Single.**

| Linkage  | Distance between clusters A and B |
|----------|-----------------------------------|
| **Ward** | Increase in total within-cluster variance after merging; minimises intra-cluster spread. |
| **Complete** | Maximum pairwise distance between any point in A and any point in B (diameter). |
| **Average** | Mean of all pairwise distances between points in A and points in B (UPGMA). |
| **Single** | Minimum pairwise distance — nearest neighbour. Prone to chaining. |

---

**Q5. What is the "chaining effect" in single linkage clustering?**

Single linkage measures only the minimum distance between clusters. If two clusters are connected through a sequence of close intermediate points (like a chain), they get merged even if the clusters themselves are far apart overall. This causes elongated, non-compact clusters and poor performance on well-separated globular data.

---

**Q6. Why is Ward linkage generally preferred for compact, spherical clusters?**

Ward linkage minimises the total increase in within-cluster sum of squares (ESS/variance) at each merge step. This means it favours merging clusters that result in the smallest loss of intra-cluster compactness, producing clusters of roughly equal size and variance — ideal for globular structures.

---

**Q7. How do you choose the number of clusters from a dendrogram?**

- Look for the longest vertical line that is not crossed by any horizontal cut — this represents the largest gap in merge distances and indicates the most natural number of clusters.
- Set a threshold (horizontal line) and count the number of branches it crosses — each branch corresponds to one cluster.
- Apply domain knowledge or use evaluation metrics (silhouette score) to validate different cut levels.

---

**Q8. What is the 60% of maximum linkage distance rule used in this lab?**

It is a heuristic threshold: compute `max_dist = max(linked[:, 2])`, then set `threshold = 0.60 × max_dist`. Cutting the dendrogram at this height often yields a reasonable number of meaningful clusters. It is a rule of thumb, not a theoretical guarantee, and should be validated against evaluation metrics and domain expectations.

---

## Section 2: DBSCAN

**Q9. What is DBSCAN and what does the acronym stand for?**

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**. It groups together points that are closely packed (high density) and marks points in low-density regions as outliers (noise). It does not require specifying the number of clusters.

---

**Q10. Define the three types of points in DBSCAN: core, border, and noise.**

- **Core point:** A point with at least `min_samples` neighbours within radius `eps` (including itself).
- **Border point:** A point within `eps` of a core point but with fewer than `min_samples` neighbours in its own `eps`-ball.
- **Noise point (outlier):** A point that is neither core nor border — too far from any dense region. Labelled as `-1` in sklearn.

---

**Q11. What are the two hyperparameters of DBSCAN and how do they affect clustering?**

- **`eps` (ε):** The radius of the neighbourhood around each point. Larger `eps` → more points become neighbours → fewer, larger clusters and less noise. Smaller `eps` → tighter neighbourhoods → more clusters, more noise.
- **`min_samples` (MinPts):** Minimum number of points required to form a dense region (core point). Larger `min_samples` → stricter density requirement → more noise points.

---

**Q12. How do you choose appropriate values for `eps` in DBSCAN?**

Use a **k-distance plot** (also called elbow plot):
1. Compute the distance from each point to its k-th nearest neighbour (k = `min_samples`).
2. Sort distances in ascending order and plot them.
3. Look for the "elbow" — a sharp increase in distance — and use that value as `eps`.

This is implemented using `sklearn.neighbors.NearestNeighbors`.

---

**Q13. What are the advantages of DBSCAN over K-means and hierarchical clustering?**

- Does not require specifying `k` in advance.
- Can discover clusters of arbitrary shape (not just spherical).
- Explicitly identifies and handles noise/outliers.
- Robust to outliers since they are excluded from cluster membership.

---

**Q14. What are the limitations of DBSCAN?**

- Struggles with clusters of varying density — a single global `eps` may not suit all clusters.
- Performance degrades in high-dimensional data (curse of dimensionality makes distance metrics less meaningful).
- Parameter sensitivity: results are very sensitive to `eps` and `min_samples`.
- Does not scale well to very large datasets (O(n²) naive complexity, though with indexing it can be O(n log n)).

---

**Q15. What does a label of `-1` from `DBSCAN.fit_predict()` mean?**

It indicates a noise point — a point that does not belong to any cluster because it is not within `eps` distance of enough other points to form or join a dense region.

---

## Section 3: Evaluation Metrics

**Q16. What is Purity Score and how is it computed?**

Purity measures how well clusters align with true class labels:

```
Purity = (1/N) × Σ_k max_j |C_k ∩ T_j|
```

For each cluster `k`, find the majority true class `j` and count how many points in that cluster belong to it. Sum these counts and divide by total points `N`. Range: [0, 1]; higher is better.

**Limitation:** Purity always increases with more clusters (trivially 1 if each point is its own cluster).

---

**Q17. What is Normalised Mutual Information (NMI) and why is it preferred over accuracy for clustering evaluation?**

NMI measures the mutual dependence between cluster assignments and true labels, normalised to [0, 1]:

```
NMI(Y, C) = I(Y; C) / sqrt(H(Y) × H(C))
```

where `I` is mutual information and `H` is entropy. NMI = 1 means perfect correlation; NMI = 0 means independence. Unlike accuracy, NMI is symmetric, does not require label alignment, and is not inflated by trivially many clusters.

---

**Q18. What is the Silhouette Score and what does it measure?**

Silhouette score measures how well each point fits its assigned cluster relative to other clusters:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- `a(i)`: Mean intra-cluster distance (distance to all points in same cluster).
- `b(i)`: Mean nearest-cluster distance (distance to all points in the closest different cluster).

Range: [-1, 1]. Score near +1 → well separated; near 0 → on cluster boundary; negative → possibly misassigned. Averaged over all points to get a single score.

---

**Q19. Can you compute Silhouette Score for DBSCAN output? What is the challenge?**

Yes, but only when more than one cluster is formed. The challenge is that DBSCAN assigns noise points a label of `-1`, and Silhouette Score requires every point to belong to a valid cluster. In practice, noise points are either excluded or kept (sklearn's `silhouette_score` excludes them automatically by not assigning them to a cluster). You must ensure at least 2 valid clusters exist before calling it.

---

**Q20. Why is NMI used instead of Adjusted Rand Index (ARI) here?**

Both are valid. NMI is symmetric and bounded [0,1] with a clear probabilistic interpretation. ARI is corrected for chance and can be negative. NMI is typically preferred when comparing many methods at once because it is easy to interpret — 0 means no correlation, 1 means perfect. ARI is preferred when you want a baseline correction for random assignments.

---

## Section 4: Implementation & Analysis

**Q21. What does `scipy.cluster.hierarchy.linkage()` return?**

It returns a linkage matrix of shape `(n-1, 4)` where each row `[idx1, idx2, dist, count]` represents one merge:
- `idx1`, `idx2`: indices of the two clusters being merged (indices ≥ N refer to previously formed clusters).
- `dist`: distance between the two merged clusters.
- `count`: number of original observations in the new cluster.

---

**Q22. What does `truncate_mode='level'` mean in the dendrogram plot?**

It controls how the dendrogram is simplified for visualisation. `truncate_mode='level'` shows only the top `p` levels of the hierarchy (leaf nodes are collapsed into labelled triangles showing the count). This is necessary for large datasets where showing every leaf would make the plot unreadable.

---

**Q23. How does `scipy.cluster.hierarchy.fcluster()` work?**

`fcluster(Z, t, criterion='distance')` cuts the dendrogram `Z` at threshold height `t` and returns flat cluster labels for each observation. All merges above height `t` are not performed, giving a flat clustering corresponding to the number of branches below the cut.

---

**Q24. In this lab, what effect did increasing `eps` have on DBSCAN clustering?**

Increasing `eps` expands the neighbourhood radius, making it easier for points to be neighbours. This generally reduces the number of clusters (as previously separate clusters merge) and also reduces the number of noise points. Very large `eps` can collapse all points into a single cluster. Very small `eps` produces many tiny clusters and large amounts of noise.

---

**Q25. Which clustering method performed best on the t-SNE language dataset, and why?**

Ward linkage hierarchical clustering typically performs best on this dataset because:
- The t-SNE projection tends to produce compact, roughly spherical blobs per language.
- Ward linkage is designed for compact cluster formation (minimises variance increase).
- DBSCAN performance depends heavily on parameter tuning; with proper `eps`/`min_samples`, it can also do well.
- Single linkage is worst due to chaining on the scattered points between language clusters.

The specific ranking depends on the actual metric values computed from the dataset.

---

**Q26. Why do we encode the language labels using `LabelEncoder` before computing metrics?**

sklearn's `normalized_mutual_info_score` and `silhouette_score` accept both string and integer labels, but using integers avoids any ambiguity. `LabelEncoder` also allows consistent mapping so that metric computations across different clustering methods are comparable.

---

**Q27. What is the time complexity of agglomerative hierarchical clustering?**

- Naive implementation: O(n³) — recomputing all pairwise distances at each merge step.
- With priority queue (min-heap): O(n² log n).
- Ward linkage with Lance-Williams update: O(n² log n) in sklearn's implementation.

For large datasets (n > 10,000), this makes hierarchical clustering slow compared to K-means (O(nkt) per iteration) or DBSCAN with spatial indexing (O(n log n)).

---

**Q28. What happens if `min_samples` is set very high in DBSCAN?**

Very high `min_samples` means a point needs many neighbours to be a core point. This results in very strict density requirements — most points fail to qualify as core points and get labelled as noise. The result is a large noise set and few (or zero) clusters. This is analogous to being too conservative about what counts as a "dense region."

---

**Q29. Can hierarchical clustering be applied to non-Euclidean data?**

Yes. Agglomerative clustering in sklearn supports a `metric` parameter (e.g., `manhattan`, `cosine`, `precomputed`) and any linkage that works with that metric. However, Ward linkage is defined only for Euclidean distance. For non-Euclidean data, `complete`, `average`, or `single` linkage must be used.

---

**Q30. What is the difference between `fit()` and `fit_predict()` in sklearn clustering models?**

- `fit(X)`: Fits the model to data and stores cluster labels internally (accessible via `.labels_` attribute).
- `fit_predict(X)`: Fits the model and immediately returns the cluster label array — equivalent to `fit(X).labels_`. For `AgglomerativeClustering` and `DBSCAN`, `fit_predict` is the standard shorthand since there is no separate `predict()` step for new data in these models.

---

*End of Viva Q&A — Lab 13*
