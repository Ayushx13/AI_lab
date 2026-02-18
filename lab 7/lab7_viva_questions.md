# Viva Questions: Logistic Regression and Support Vector Machines
### CS201L – Artificial Intelligence Laboratory | IIT Dharwad

---

## Section 1: Logistic Regression

**Q1. What is Logistic Regression, and why is it called "regression" if it is used for classification?**

Logistic Regression is a classification algorithm that models the probability of a sample belonging to a class using the sigmoid (logistic) function. It is called "regression" because internally it fits a linear regression model to the log-odds of the class probability. The final output is squashed to [0,1] using sigmoid, and a threshold (usually 0.5) is used to assign class labels.

---

**Q2. Write the sigmoid function and explain its role in Logistic Regression.**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = w^T x + b$. The sigmoid maps any real-valued input to a value between 0 and 1, which can be interpreted as a probability.

---

**Q3. What is the loss function used in Logistic Regression? Write it.**

Binary Cross-Entropy (Log Loss):

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

For multiclass (like this lab), we use **Softmax + Categorical Cross-Entropy**.

---

**Q4. What does the weight vector `w` in Logistic Regression represent?**

Each weight $w_j$ represents the contribution (importance) of feature $j$ to the model's decision. A large absolute value of $w_j$ means feature $j$ strongly influences the output. This is why we can plot the absolute values of the weight vector to understand feature importance.

---

**Q5. Why do we use `max_iter=1000` in `LogisticRegression()`?**

Logistic Regression is solved iteratively using gradient descent or solvers like LBFGS. The default `max_iter=100` may be insufficient for complex datasets with many features, causing the solver to not converge. Increasing it to 1000 gives the optimizer more iterations to find the optimal weights.

---

**Q6. How does Logistic Regression handle multiclass classification (as in this dataset)?**

sklearn's `LogisticRegression` handles multiclass using:
- **One-vs-Rest (OvR)**: Trains one binary classifier per class.
- **Multinomial (Softmax)**: Models all classes jointly using the softmax function.

By default, sklearn uses OvR for most solvers, or multinomial when `multi_class='multinomial'` and a suitable solver (like `lbfgs`) is specified.

---

## Section 2: Support Vector Machines

**Q7. What is the core idea behind a Support Vector Machine (SVM)?**

SVM tries to find the **optimal hyperplane** that separates classes with the **maximum margin**. The margin is the distance between the hyperplane and the nearest data points from each class (called **support vectors**). Maximizing the margin generally leads to better generalization.

---

**Q8. What are support vectors?**

Support vectors are the data points that lie closest to the decision boundary (hyperplane). They are the most difficult to classify and directly influence the position and orientation of the hyperplane. If you remove non-support vector points, the hyperplane stays the same.

---

**Q9. What is the role of the regularization parameter C in SVM?**

C controls the trade-off between maximizing the margin and minimizing training errors:

- **Small C** → Larger margin, more misclassifications allowed → can underfit
- **Large C** → Smaller margin, fewer misclassifications → can overfit

It is the most important hyperparameter to tune in SVM.

---

**Q10. Why can't we directly access the weight vector for RBF and Polynomial kernels?**

The Kernel Trick maps data to a high-dimensional (even infinite-dimensional for RBF) feature space implicitly — without ever computing the coordinates in that space. Since we never explicitly compute the feature map $\phi(x)$, we cannot retrieve the weight vector $w$ in the original space. Only for the **linear kernel** do we work in the original feature space, so `coef_` is accessible.

---

**Q11. Explain the Kernel Trick in simple terms.**

The Kernel Trick allows SVM to operate in a high-dimensional space without ever explicitly transforming the data. Instead of computing $\phi(x_i)^T \phi(x_j)$, we compute a kernel function $K(x_i, x_j)$ that gives the same result. This is computationally efficient and allows us to model non-linear decision boundaries.

---

**Q12. What is the RBF (Radial Basis Function) kernel? Write its formula.**

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

It measures similarity between two points based on distance. Points closer together get a higher kernel value (closer to 1). The parameter $\gamma$ controls how quickly the similarity drops with distance.

---

**Q13. What does the `gamma` parameter control in RBF/Polynomial kernels?**

- **Small gamma** → Each training point has a far reach → smoother, simpler boundary → risk of underfitting
- **Large gamma** → Each training point has a short reach → complex, wiggly boundary → risk of overfitting

`gamma='scale'` sets it to $1 / (\text{n\_features} \times \text{Var}(X))$, which is a good default.

---

**Q14. What is the Polynomial kernel? Write its formula.**

$$K(x_i, x_j) = (\gamma \cdot x_i^T x_j + r)^d$$

Where $d$ is the degree of the polynomial. It captures interactions between features. A higher degree allows more complex boundaries but increases risk of overfitting.

---

**Q15. How do you choose the best kernel for a given problem?**

- **Linear kernel**: Use when data is linearly separable or when you have many features (e.g., text classification).
- **RBF kernel**: Good general-purpose choice for non-linear problems. Works well in most cases.
- **Polynomial kernel**: Useful when the relationship between features involves polynomial interactions.

In practice, try all kernels and use validation accuracy to pick the best one.

---

## Section 3: Data Preprocessing & Metrics

**Q16. Why is feature scaling important for Logistic Regression and SVM?**

Both algorithms depend on distances or dot products between feature vectors. If one feature has values in the range [0, 10000] and another in [0, 1], the first will dominate the learning process. `StandardScaler` scales each feature to zero mean and unit variance, ensuring all features contribute equally.

---

**Q17. Why should we fit the scaler only on training data and not on validation/test data?**

Fitting the scaler on val/test data would cause **data leakage** — the model would gain information about the test distribution during training. We must simulate a real-world scenario where test data is completely unseen. So we `fit_transform` on training data and only `transform` on val/test data using the same scaler.

---

**Q18. What is a Confusion Matrix? What information does it give?**

A confusion matrix is a table that shows the number of:
- **True Positives (TP)**: Correctly predicted as positive
- **True Negatives (TN)**: Correctly predicted as negative
- **False Positives (FP)**: Incorrectly predicted as positive (Type I error)
- **False Negatives (FN)**: Incorrectly predicted as negative (Type II error)

For multiclass, it is an $N \times N$ matrix where entry $(i, j)$ is the number of samples of class $i$ predicted as class $j$.

---

**Q19. What is the difference between Macro and Micro averaging?**

- **Macro averaging**: Compute the metric (e.g., precision) independently for each class and then take the unweighted average. Treats all classes equally regardless of their size.
- **Micro averaging**: Aggregate the contributions (TP, FP, FN) of all classes first, then compute the metric. For multiclass classification, micro accuracy equals overall accuracy.

Macro is better when you care equally about all classes (even rare ones). Micro is better when you care about overall performance.

---

**Q20. Define Precision, Recall, and F1-score.**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Precision**: Out of all predicted positives, how many were actually positive?
- **Recall**: Out of all actual positives, how many did we correctly predict?
- **F1-Score**: Harmonic mean of Precision and Recall. Useful when you need a balance between the two.

---

## Section 4: Conceptual & Comparative

**Q21. What is the difference between Logistic Regression and SVM?**

| Aspect | Logistic Regression | SVM |
|--------|--------------------|----|
| Objective | Maximize likelihood / minimize log-loss | Maximize margin between classes |
| Output | Probability estimates | Class labels (no direct probability) |
| Boundary | Linear only (without basis expansion) | Linear and Non-linear (via kernels) |
| Sensitivity to outliers | Moderate | High (support vectors can shift) |
| Scaling needed | Yes | Yes |

---

**Q22. Can Logistic Regression handle non-linear data? How?**

By default, Logistic Regression is a linear classifier. However, we can make it handle non-linear data by:
1. Adding polynomial or interaction features manually before training.
2. Applying kernel trick (Kernel Logistic Regression).

SVM with non-linear kernels is typically a more natural choice for non-linear problems.

---

**Q23. What happens when C is very large in SVM?**

A very large C makes the SVM try to correctly classify every training point. This reduces the margin and can lead to a very complex decision boundary that fits the training data too closely — this is **overfitting**. The model may perform well on training data but poorly on unseen data.

---

**Q24. How do you decide the optimal C using validation data?**

Train the SVM with different values of C (e.g., [0.001, 0.01, 0.1, 1, 10, 100, 1000]) and evaluate each model on the validation set. Choose the C that gives the **highest validation accuracy**. Then, report the final performance on the test set using this optimal C.

---

**Q25. Why do we use a separate validation set instead of testing directly?**

The test set should represent unseen, real-world data. If we tune hyperparameters (like C or degree) using the test set, we are essentially "peeking" at the test data, which gives us an overly optimistic estimate of performance. The validation set is used for tuning, and the test set is used only for **final, unbiased evaluation**.

---

*End of Viva Questions*
