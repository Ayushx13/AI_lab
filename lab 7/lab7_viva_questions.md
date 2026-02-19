# Lab 7 Viva Questions: Logistic Regression and SVM
### CS201L - Artificial Intelligence Laboratory

---

## Section 1: Logistic Regression

**Q1. What is Logistic Regression and why is it called "regression" if it's used for classification?**
> Logistic Regression is a classification algorithm that models the probability of a class using the logistic (sigmoid) function. It's called regression because it fits a linear regression internally, but then applies a sigmoid to squash the output between 0 and 1, which is interpreted as a probability.

**Q2. What is the sigmoid function and what is its output range?**
> The sigmoid function is σ(x) = 1 / (1 + e^(-x)). Its output always lies between 0 and 1, making it suitable for probability estimation.

**Q3. Why did we use `solver='liblinear'` in this lab?**
> `liblinear` is a library designed for large linear classification. It naturally supports One-vs-Rest (OvR) multiclass strategy, which trains a separate binary classifier for each class against all others. It is efficient for high-dimensional datasets like the HAR dataset with 561 features.

**Q4. What is the One-vs-Rest (OvR) strategy?**
> In OvR, for a problem with K classes, K separate binary classifiers are trained. Each classifier learns to distinguish one class from all remaining classes. During prediction, the class whose classifier gives the highest confidence score is chosen.

**Q5. What is `max_iter` in Logistic Regression and why did we set it to 1000?**
> `max_iter` is the maximum number of iterations for the solver to converge. The default (100) may not be enough for complex datasets. We set it to 1000 to give the solver enough iterations to find a good solution without a convergence warning.

**Q6. What is the cost/loss function used in Logistic Regression?**
> Logistic Regression uses Binary Cross-Entropy (Log Loss): L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]. The goal is to minimize this loss over all training samples.

---

## Section 2: Support Vector Machines (SVM)

**Q7. What is the core idea behind SVM?**
> SVM finds the optimal hyperplane that maximally separates two classes. The "maximum margin" hyperplane is the one that is farthest from the nearest data points of each class. These nearest points are called Support Vectors.

**Q8. What are Support Vectors?**
> Support Vectors are the data points that lie closest to the decision boundary (hyperplane). They are the most critical points — removing any other point does not change the hyperplane, but removing a support vector does.

**Q9. What is the margin in SVM and why do we want to maximize it?**
> The margin is the distance between the decision boundary and the nearest support vectors from each class. Maximizing it leads to better generalization — a wider margin means the model is less likely to misclassify unseen data points that are close to the boundary.

**Q10. What is the role of the regularization parameter C in SVM?**
> C controls the trade-off between maximizing the margin and minimizing classification errors on training data.
> - **Large C**: Smaller margin, fewer training errors (risk of overfitting).
> - **Small C**: Larger margin, more training errors allowed (risk of underfitting).
> In this lab, we used the default C = 1.

**Q11. What is the Kernel Trick? Why is it useful?**
> The Kernel Trick allows SVM to operate in a high-dimensional feature space without explicitly computing the transformation. It replaces the dot product in the SVM formulation with a kernel function K(x, z), enabling the classifier to learn non-linear decision boundaries efficiently.

**Q12. What are the three kernels we used in this lab?**
> - **Linear**: K(x, z) = x·z — used when data is linearly separable.
> - **Polynomial**: K(x, z) = (γ·x·z + r)^d — captures polynomial interactions between features.
> - **RBF (Gaussian)**: K(x, z) = exp(-γ·||x-z||²) — maps data into infinite-dimensional space; good for complex boundaries.

**Q13. What does the `gamma` parameter control in RBF and Polynomial kernels?**
> Gamma (γ) controls the influence of a single training example.
> - **High γ**: Each point has a small radius of influence → complex, wiggly boundary → overfitting risk.
> - **Low γ**: Each point has a large radius of influence → smoother boundary → underfitting risk.
> We used `gamma='scale'` which automatically sets γ = 1 / (n_features × X.var()).

**Q14. Why did we use the validation set to choose the best polynomial degree?**
> We cannot use the test set for model selection because that would lead to data leakage — the test set must remain unseen until final evaluation. The validation set acts as a proxy to tune hyperparameters (like degree), and then the test set gives an unbiased estimate of final performance.

**Q15. What happens as the polynomial degree increases?**
> Higher degrees create more complex decision boundaries. Very high degrees can overfit the training data, performing well on training but poorly on unseen data. Lower degrees might underfit. The best degree is selected using validation accuracy.

---

## Section 3: Evaluation Metrics

**Q16. What is a Confusion Matrix?**
> A confusion matrix is a table that shows the counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) for each class. It gives a detailed view of where the model is making errors.

**Q17. Define Accuracy, Precision, Recall, and F1-Score.**
> - **Accuracy** = (TP + TN) / Total — overall correctness.
> - **Precision** = TP / (TP + FP) — of all predicted positives, how many are actually positive.
> - **Recall** = TP / (TP + FN) — of all actual positives, how many did we correctly identify.
> - **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean of precision and recall.

**Q18. When would you prefer Recall over Precision?**
> When the cost of a False Negative is high. For example, in disease detection, missing a sick patient (FN) is more dangerous than a false alarm (FP), so we prioritize Recall.

**Q19. Why did we use `average='weighted'` for precision, recall, and F1?**
> Because this is a multiclass problem with class imbalance. `weighted` averaging computes the metric for each class and takes a weighted average based on the number of samples in each class. This gives more importance to classes with more samples.

**Q20. What does a diagonal confusion matrix mean?**
> A perfect diagonal means all predictions are correct — every sample was assigned its true label. Off-diagonal elements represent misclassifications.

---

## Section 4: Dataset and Preprocessing

**Q21. What is the HAR dataset and what does it contain?**
> The Human Activity Recognition dataset contains sensor data (accelerometer and gyroscope) from 30 people performing 6 activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying. It has 10,299 samples and 561 features extracted from time and frequency domains.

**Q22. Why do we need feature scaling for SVMs?**
> SVMs compute distances between data points using kernel functions. If features have very different scales (e.g., one feature ranges 0–1 and another 0–1000), the larger-scale feature will dominate the distance computation, biasing the model. Scaling ensures all features contribute equally.

**Q23. Does Logistic Regression require feature scaling?**
> Not strictly required, but it helps. Scaling leads to faster convergence of the solver and more stable gradient updates. Without scaling, the algorithm may take many more iterations.

**Q24. What is PCA and why was it used?**
> Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms features into a new set of uncorrelated components (principal components) ordered by the variance they explain. We used it to reduce the 561 features while retaining most of the information, which can speed up training and reduce noise.

**Q25. What is the difference between "PCA All Components" and "PCA 99% Variance"?**
> - **PCA All**: Keeps all principal components (same number as original features but rotated). No dimensionality reduction, just decorrelation.
> - **PCA 99%**: Keeps only enough components to explain 99% of the variance, discarding the rest. This actually reduces the number of features, speeding up training with minimal accuracy loss.

---

## Section 5: Comparison and Analysis

**Q26. Which classifier generally performs best on this type of dataset and why?**
> SVM with RBF kernel typically performs best on the scaled HAR dataset because the data likely has a non-linear structure that the RBF kernel can capture. Linear SVM is also competitive since the 561-feature space is high-dimensional, where classes may already be nearly linearly separable.

**Q27. Why might SVM Linear perform comparably to SVM RBF on high-dimensional data?**
> In very high-dimensional spaces (like 561 features), data tends to be more linearly separable due to the "curse of dimensionality." The extra flexibility of the RBF kernel may not provide much benefit, and both kernels achieve similar performance.

**Q28. What is the difference between training accuracy and test accuracy? Why does the gap matter?**
> Training accuracy measures performance on data the model was trained on. Test accuracy measures performance on unseen data. A large gap (high train accuracy, low test accuracy) indicates overfitting — the model memorized training data instead of learning general patterns.

**Q29. Why is SVM with a polynomial kernel the slowest to train in this lab?**
> Training multiple models (one per degree) and the inherently higher computational cost of the polynomial kernel (especially at higher degrees) makes it slower. Each SVM training is O(n²) to O(n³) in the number of samples.

**Q30. If you were to improve the models further, what would you try?**
> - Tune the regularization parameter C using cross-validation.
> - Try different gamma values for RBF and polynomial kernels.
> - Use GridSearchCV or RandomizedSearchCV for systematic hyperparameter tuning.
> - Try ensemble methods like Random Forest or Gradient Boosting for comparison.
> - Apply feature selection to remove irrelevant features.

---

*Good luck with your viva!*
