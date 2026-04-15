# CS201L: Artificial Intelligence Laboratory
## Lab 13 - Viva Questions and Answers
### Hierarchical and DBSCAN Clustering

---

## Section 1: Dimensionality Reduction and PCA

### Q1: What is PCA and why is it used?
**Answer:** PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It works by identifying the principal components (directions of maximum variance) in the data.

**Why use PCA:**
- Reduces computational complexity
- Removes noise and redundant features
- Helps in visualization (2D/3D)
- Mitigates the curse of dimensionality
- Can improve clustering performance by removing noise

### Q2: How does PCA select principal components?
**Answer:** PCA selects principal components by:
1. Computing the covariance matrix of the data
2. Finding eigenvalues and eigenvectors of the covariance matrix
3. Sorting eigenvectors by their corresponding eigenvalues in descending order
4. Selecting the top k eigenvectors (principal components) that capture the most variance

The eigenvalues represent the amount of variance explained by each principal component.

### Q3: What is explained variance ratio?
**Answer:** Explained variance ratio indicates the proportion of total variance in the data captured by each principal component. For example, if PC1 has an explained variance ratio of 0.73, it means PC1 captures 73% of the total variance in the original data.

### Q4: Why do we exclude the species column when performing PCA?
**Answer:** We exclude the species column because:
- PCA is an unsupervised technique that works on features only
- Labels (species) are not features and should not influence the transformation
- Labels are used only for evaluation after clustering
- Including labels would leak information and make the analysis supervised

---

## Section 2: Hierarchical Clustering

### Q5: What is Hierarchical Clustering?
**Answer:** Hierarchical clustering is a clustering method that builds a hierarchy of clusters. It can be:
- **Agglomerative (bottom-up):** Starts with each point as a cluster and merges closest clusters iteratively
- **Divisive (top-down):** Starts with all points in one cluster and splits recursively

Agglomerative is more commonly used due to computational efficiency.

### Q6: Explain the different linkage methods.
**Answer:**

**Ward Linkage:**
- Minimizes the variance within clusters
- Merges clusters that result in minimum increase in total within-cluster variance
- Tends to create compact, spherical clusters
- Best for: Well-separated, similar-sized clusters

**Complete Linkage (Maximum):**
- Distance between clusters = maximum distance between any two points in different clusters
- Creates compact clusters
- Sensitive to outliers
- Best for: Avoiding elongated clusters

**Average Linkage:**
- Distance = average of all pairwise distances between points in different clusters
- Balanced approach between single and complete
- Less sensitive to outliers than complete linkage
- Best for: General-purpose clustering

**Single Linkage (Minimum):**
- Distance = minimum distance between any two points in different clusters
- Can form elongated, chain-like clusters (chaining effect)
- Sensitive to noise
- Best for: Non-elliptical shapes, but generally avoided

### Q7: What is a dendrogram?
**Answer:** A dendrogram is a tree-like diagram that shows the hierarchical relationship between clusters. It displays:
- How clusters are merged at each step (agglomerative)
- The distance/dissimilarity at which merges occur (y-axis)
- All data points or cluster representatives (x-axis)

**Uses:**
- Visualize the clustering hierarchy
- Determine optimal number of clusters by cutting at appropriate height
- Understand cluster relationships

### Q8: How do you determine the optimal number of clusters from a dendrogram?
**Answer:** 
1. **Visual inspection:** Look for large vertical distances (gaps) between merges
2. **Horizontal cut:** Draw a horizontal line (threshold) that intersects the dendrogram at a point where vertical lines are longest
3. **Number of clusters:** Count the number of vertical lines the horizontal line crosses
4. **Elbow-like approach:** Look for the point where merging clusters starts to happen at significantly larger distances

### Q9: What are the advantages and disadvantages of Hierarchical Clustering?

**Advantages:**
- No need to specify number of clusters beforehand
- Provides a dendrogram for visualization
- Can discover hierarchy in data
- Deterministic (no random initialization)
- Works with any distance metric

**Disadvantages:**
- High computational complexity: O(n²log n) for best implementations
- Not scalable to very large datasets
- Sensitive to noise and outliers
- Cannot undo previous merges (greedy approach)
- Difficult to handle clusters of different sizes and densities

---

## Section 3: DBSCAN Clustering

### Q10: What is DBSCAN?
**Answer:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that:
- Groups together points that are closely packed (high-density regions)
- Marks points in low-density regions as outliers/noise
- Can discover clusters of arbitrary shape
- Doesn't require pre-specifying the number of clusters

### Q11: Explain the key parameters of DBSCAN.
**Answer:**

**eps (ε - Epsilon):**
- Maximum distance between two points to be considered neighbors
- Defines the radius of the neighborhood around each point
- Smaller eps → more clusters, more noise
- Larger eps → fewer clusters, less noise

**min_samples:**
- Minimum number of points required to form a dense region (cluster)
- Includes the point itself
- Higher values → denser clusters, more noise points
- Lower values → more clusters, fewer noise points

### Q12: What are core points, border points, and noise points in DBSCAN?

**Answer:**

**Core Point:**
- Has at least `min_samples` points within its eps-neighborhood (including itself)
- Forms the interior of clusters
- Can start new clusters

**Border Point:**
- Has fewer than `min_samples` points in its eps-neighborhood
- Is within eps distance of a core point
- Belongs to a cluster but is on the boundary

**Noise Point (Outlier):**
- Not a core point
- Not within eps distance of any core point
- Not assigned to any cluster (labeled as -1)

### Q13: How does DBSCAN algorithm work?
**Answer:**

**Algorithm steps:**
1. For each unvisited point P:
   - Mark P as visited
   - Find all points within eps distance (neighbors)
   - If neighbors < min_samples: mark P as noise (temporarily)
   - If neighbors ≥ min_samples: P is a core point
     - Create new cluster
     - Add P and all reachable points to cluster
     - For each neighbor: if unvisited, repeat process

2. Points can be moved from noise to border if they're within eps of a core point

### Q14: What are the advantages and disadvantages of DBSCAN?

**Advantages:**
- Can find arbitrarily shaped clusters
- Robust to outliers (identifies them as noise)
- No need to specify number of clusters
- Deterministic (given same parameters)
- Can handle clusters of different sizes and densities

**Disadvantages:**
- Difficult to choose eps and min_samples parameters
- Struggles with varying density clusters
- Sensitive to parameter settings
- Not suitable for high-dimensional data (curse of dimensionality)
- Cannot cluster data with widely differing densities

### Q15: How do you choose eps and min_samples for DBSCAN?

**Answer:**

**For eps:**
- **K-distance graph:** Plot k-nearest neighbor distances (sorted), look for "elbow"
- **Domain knowledge:** Based on understanding of data scale
- **Trial and error:** Test multiple values and evaluate results
- General guideline: Start with k-distance of min_samples-th nearest neighbor

**For min_samples:**
- **Rule of thumb:** min_samples ≥ dimensions + 1
- **For 2D data:** Often 4-5 works well
- **Larger values:** For noisy data or when you want denser clusters
- **Smaller values:** For cleaner data or to capture smaller clusters

### Q16: What happens if eps is too small or too large?

**Answer:**

**eps too small:**
- Most points become noise
- Many small clusters or no clusters
- Underclustering
- High fragmentation

**eps too large:**
- All points might merge into one cluster
- Overclustering (clustering non-similar points)
- Loss of cluster structure
- Cannot distinguish between natural groups

---

## Section 4: Evaluation Metrics

### Q17: What is Purity Score and how is it calculated?
**Answer:** Purity measures how "pure" each cluster is in terms of class labels.

**Calculation:**
```
Purity = (1/N) × Σ(max_j |cluster_k ∩ class_j|)
```

Where:
- N = total number of points
- For each cluster k, find the majority class
- Sum the counts of majority class in each cluster

**Range:** [0, 1], higher is better
**Interpretation:** 1.0 means perfect clustering (each cluster contains only one class)

### Q18: What is Normalized Mutual Information (NMI)?
**Answer:** NMI measures the mutual information between cluster assignments and true labels, normalized to [0, 1].

**Properties:**
- 0: No mutual information (random clustering)
- 1: Perfect correlation (perfect clustering)
- Adjusts for chance agreement
- Symmetric measure

**Advantages over Purity:**
- Considers the overall cluster distribution
- Not biased by number of clusters
- More robust evaluation metric

### Q19: What is Silhouette Score?
**Answer:** Silhouette score measures how similar a point is to its own cluster compared to other clusters.

**For each point i:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- a(i) = average distance to points in same cluster
- b(i) = average distance to points in nearest neighboring cluster

**Range:** [-1, 1]
- 1: Point is well-clustered
- 0: Point is on the boundary between clusters
- -1: Point is likely in wrong cluster

**Average silhouette score** across all points indicates overall clustering quality.

### Q20: What is Inertia (Within-Cluster Sum of Squares)?
**Answer:** Inertia measures the compactness of clusters by calculating the sum of squared distances from each point to its cluster center.

**Formula:**
```
Inertia = Σ Σ ||x - μ_k||²
```

Where:
- x = data point in cluster k
- μ_k = centroid of cluster k

**Properties:**
- Lower inertia = more compact clusters
- Used in K-Means optimization
- Can be used for elbow method
- Decreases as K increases (always)

---

## Section 5: Comparison and Analysis

### Q21: Compare K-Means, Hierarchical, and DBSCAN clustering.

**Answer:**

| Aspect | K-Means | Hierarchical | DBSCAN |
|--------|---------|-------------|---------|
| **K required?** | Yes | Can be determined from dendrogram | No |
| **Shape** | Spherical | Depends on linkage | Arbitrary |
| **Outliers** | Sensitive | Sensitive | Robust |
| **Scalability** | Good (O(nkt)) | Poor (O(n²)) | Moderate (O(n log n)) |
| **Deterministic** | No (random init) | Yes | Yes |
| **Density** | Uniform | Uniform | Can handle varying |

### Q22: When should you use Hierarchical Clustering vs DBSCAN?

**Answer:**

**Use Hierarchical Clustering when:**
- Dataset is small to medium size (< 10,000 points)
- You want to see the hierarchical structure
- Clusters are roughly spherical and similar in size
- You want deterministic results
- You need to explore different numbers of clusters

**Use DBSCAN when:**
- Clusters have arbitrary shapes
- Data contains significant noise/outliers
- Clusters have similar density
- You don't know the number of clusters
- Dataset is moderate size
- You can tune eps and min_samples effectively

### Q23: Why might Ward linkage give better results than other linkages?
**Answer:** Ward linkage often performs better because:
- Minimizes within-cluster variance (similar to K-Means objective)
- Creates compact, balanced clusters
- Less sensitive to outliers than single linkage
- More robust than complete linkage
- Works well with Euclidean distance
- Natural for many real-world datasets with spherical clusters

However, it may perform poorly with:
- Non-spherical clusters
- Widely varying cluster sizes
- Non-Euclidean distance metrics

### Q24: How does PCA affect clustering results?
**Answer:** PCA affects clustering in several ways:

**Positive effects:**
- **Noise reduction:** Removes dimensions with low variance (often noise)
- **Visualization:** Enables 2D/3D visualization
- **Computational efficiency:** Fewer dimensions = faster clustering
- **Curse of dimensionality:** Reduces distance concentration problems

**Potential negatives:**
- **Information loss:** May lose variance that's important for clustering
- **Assumption:** Assumes variance = importance (not always true)
- **Linear combinations:** Only captures linear relationships

**Best practice:** Check explained variance ratio; if too low, clustering on original features might be better.

---

## Section 6: Practical Implementation

### Q25: Why do we compute silhouette score only for non-noise points in DBSCAN?
**Answer:** 
- Noise points (label -1) are not assigned to any cluster
- Silhouette score requires cluster membership
- Including noise points would be undefined (no "own cluster")
- Noise points are intentionally excluded as outliers
- We want to evaluate the quality of actual clusters, not outliers

### Q26: What does it mean if DBSCAN creates only one cluster?
**Answer:** If DBSCAN creates one cluster:
- **eps is too large:** Neighborhood includes too many points
- **min_samples is too small:** Easy to form dense regions
- **Data is actually one dense cluster:** Natural result
- **Need to decrease eps** or **increase min_samples** to separate clusters

### Q27: What does it mean if DBSCAN marks most points as noise?
**Answer:** If most points are noise:
- **eps is too small:** Neighborhoods don't include enough points
- **min_samples is too large:** Cannot form dense enough regions
- **Data is sparse:** No natural dense clusters
- **Need to increase eps** or **decrease min_samples**

### Q28: How do you interpret a dendrogram threshold?
**Answer:** The threshold (horizontal line) on a dendrogram:
- Represents the distance/dissimilarity level at which to cut
- **Above threshold:** Clusters are separate
- **Below threshold:** Points belong to same cluster
- Number of vertical lines crossed = number of clusters
- **Higher threshold:** Fewer, larger clusters
- **Lower threshold:** More, smaller clusters

### Q29: What is the computational complexity of different clustering algorithms?

**Answer:**

**K-Means:**
- Time: O(n × k × d × iterations)
- Space: O(n × d)
- Where n = points, k = clusters, d = dimensions

**Hierarchical (Agglomerative):**
- Time: O(n² log n) to O(n³) depending on implementation
- Space: O(n²) for distance matrix

**DBSCAN:**
- Time: O(n log n) with spatial indexing (e.g., KD-tree)
- Time: O(n²) without indexing
- Space: O(n)

### Q30: How do you handle imbalanced clusters?
**Answer:**

**In K-Means/K-Medoids:**
- May create unequal cluster sizes naturally
- Tries to minimize overall variance

**In Hierarchical:**
- Ward linkage tends to create balanced clusters
- Single linkage can create very imbalanced clusters (chaining)
- Complete linkage prefers balanced clusters

**In DBSCAN:**
- Naturally handles different cluster sizes
- Density-based approach doesn't assume equal sizes
- Best choice for highly imbalanced data

---

## Section 7: Advanced Concepts

### Q31: What is the chaining effect in hierarchical clustering?
**Answer:** The chaining effect occurs with single linkage where:
- Clusters form elongated chains
- Points are added one at a time based on nearest neighbor
- Can connect two distinct clusters through intermediate points
- Results in poor clustering quality for compact clusters
- Similar to "following a bridge" between clusters

**Solution:** Use complete, average, or Ward linkage instead.

### Q32: Can DBSCAN find clusters of different densities?
**Answer:** **Not effectively.** DBSCAN uses global eps and min_samples:
- Same density threshold for entire dataset
- Clusters with lower density may be marked as noise
- Clusters with higher density may merge inappropriately

**Solutions:**
- **OPTICS:** Variant that handles varying densities
- **HDBSCAN:** Hierarchical DBSCAN with varying densities
- Multiple DBSCAN runs with different parameters

### Q33: What is the curse of dimensionality in clustering?
**Answer:** As dimensions increase:
- **Distance concentration:** All points become equidistant
- **Sparsity:** Data becomes sparse in high-dimensional space
- **Volume increase:** Exponential increase in space volume
- **Distance metrics:** Euclidean distance becomes less meaningful

**Effects on clustering:**
- K-Means struggles to find meaningful clusters
- DBSCAN: All points may become noise or one cluster
- Hierarchical: Distance-based linkages become unreliable

**Solutions:**
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Feature selection
- Use appropriate distance metrics (cosine, Manhattan)

### Q34: What are the stopping criteria for hierarchical clustering?
**Answer:**

**Distance-based:**
- Stop when distance between clusters exceeds threshold
- Large jump in merge distance indicates natural separation

**Number of clusters:**
- Pre-specify desired number of clusters
- Cut dendrogram to get k clusters

**Dendrogram analysis:**
- Identify the longest vertical line (largest distance increase)
- Cut just before that merge

**Evaluation metrics:**
- Monitor silhouette score or other metrics at each merge
- Stop when metric stops improving

### Q35: How does DBSCAN handle border points that could belong to multiple clusters?
**Answer:** 
- Border points are assigned to the first cluster that reaches them
- The assignment can depend on the order of processing
- This introduces some non-determinism in border assignments
- However, the core structure of clusters remains deterministic
- In practice, most implementations use consistent ordering to maintain determinism

---

## Section 8: Lab-Specific Questions

### Q36: Why do we use the Iris dataset in this lab?
**Answer:**
- Well-studied benchmark dataset
- 150 samples, 4 features, 3 classes
- Classes have some overlap (not perfectly separable)
- Good for evaluating clustering algorithms
- 2D visualization possible after PCA
- Small enough for all algorithms to run quickly
- Ground truth labels available for evaluation

### Q37: What insights can you gain from comparing all clustering methods?
**Answer:**

**Performance:**
- Which algorithm achieves highest purity/NMI
- Which handles the Iris structure best

**Characteristics:**
- K-Means/Ward: Good for spherical, balanced clusters
- DBSCAN: Can identify outliers
- Single linkage: May create chained clusters

**Trade-offs:**
- Computational cost vs. quality
- Need for parameter tuning vs. automation
- Interpretability vs. flexibility

### Q38: What does it mean if different linkage methods give different results?
**Answer:** Different linkages give different results because:
- They use different distance definitions between clusters
- Data structure may not favor one particular linkage
- Presence of noise or outliers affects different linkages differently
- Cluster shapes may not be spherical

**Interpretation:**
- If Ward performs best: Clusters are relatively spherical
- If Complete is similar to Ward: Compact, well-separated clusters
- If Single performs poorly: Likely some chaining effect
- Large differences suggest complex cluster structure

### Q39: How do you report clustering results in practice?
**Answer:** A complete clustering report should include:

1. **Dataset description:** Size, features, pre-processing
2. **Method used:** Algorithm, parameters, reasoning
3. **Evaluation metrics:** Purity, NMI, Silhouette
4. **Visualizations:** 
   - Scatter plots of clusters
   - Dendrogram (for hierarchical)
   - Parameter sensitivity analysis
5. **Cluster characteristics:**
   - Size of each cluster
   - Dominant class in each cluster
   - Quality metrics per cluster
6. **Comparison:** With baseline or alternative methods
7. **Insights:** What the clusters represent

### Q40: What would you do if clustering results are poor?
**Answer:**

**Diagnostic steps:**
1. **Check data quality:**
   - Look for outliers or errors
   - Check feature scales (normalize if needed)
   - Examine data distribution

2. **Try different algorithms:**
   - Test K-Means, Hierarchical, DBSCAN
   - Each has different assumptions

3. **Parameter tuning:**
   - K-Means: Try different K values
   - Hierarchical: Try different linkages
   - DBSCAN: Adjust eps and min_samples

4. **Feature engineering:**
   - Try different dimensionality reduction
   - Select relevant features
   - Create new features

5. **Re-evaluate assumptions:**
   - Maybe data doesn't have natural clusters
   - Consider if clustering is appropriate

6. **Ensemble methods:**
   - Use consensus clustering
   - Combine results from multiple algorithms

---

## Bonus Questions

### Q41: What is the difference between partitional and hierarchical clustering?
**Answer:**

**Partitional (e.g., K-Means):**
- Creates flat structure (single level)
- Requires pre-specified K
- Generally faster
- Points assigned to exactly one cluster
- Iterative optimization

**Hierarchical:**
- Creates nested structure (multiple levels)
- Doesn't require K
- Provides dendrogram
- More expensive computationally
- Deterministic (for agglomerative)

### Q42: Can you apply DBSCAN to high-dimensional data?
**Answer:** **Not recommended** because:
- Distance metrics become unreliable in high dimensions
- All points tend to be equidistant
- Difficult to define meaningful eps
- Computational cost increases significantly

**Alternatives:**
- Reduce dimensions first (PCA, UMAP)
- Use subspace clustering algorithms
- Consider other methods designed for high dimensions (e.g., CLIQUE)

### Q43: What is the relationship between K-Means and Ward linkage?
**Answer:** Both minimize within-cluster variance:
- **K-Means:** Iteratively minimizes SSE (sum of squared errors)
- **Ward linkage:** Hierarchically minimizes increase in SSE at each merge

They often produce similar results for:
- Spherical, compact clusters
- Balanced cluster sizes
- Euclidean distance metric

Ward can be seen as a hierarchical version of K-Means.

---

## Common Viva Pitfalls to Avoid

1. **Don't confuse:** Core/border/noise points in DBSCAN
2. **Don't say:** DBSCAN always finds better clusters than K-Means
3. **Don't forget:** PCA loses information (check explained variance)
4. **Don't assume:** Higher silhouette score always means better clustering
5. **Don't ignore:** The importance of parameter selection in DBSCAN
6. **Don't overlook:** Computational complexity differences between algorithms

---

## Quick Reference

**Key Formulas:**
- Purity: (1/N) × Σ max_count_per_cluster
- Silhouette: (b - a) / max(a, b)
- Explained Variance: eigenvalue / sum(eigenvalues)

**Best Practices:**
- Always visualize data before clustering
- Try multiple algorithms and compare
- Use multiple evaluation metrics
- Document parameter choices
- Consider computational constraints

**Red Flags:**
- All points in one cluster (overclustering)
- Almost all points as noise (underclustering)
- Very low purity (<0.5)
- Negative silhouette scores
- Extremely imbalanced clusters (unless expected)

---

**End of Viva Questions**
