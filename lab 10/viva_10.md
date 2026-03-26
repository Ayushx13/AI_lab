# CS201L – Lab 10: Regression with Multi-Input Variables
## Viva Questions & Answers

---

### Section 1: Data & Problem Understanding

**Q1. What is the Energy Efficiency dataset and what is the prediction goal?**

The Energy Efficiency dataset contains building design parameters (compactness, surface area, wall area, roof area, height, orientation, glazing area, glazing area distribution, and cooling load) simulated using Ecotect. The goal is to predict **Heating Load** — the energy required to heat a residential building — using these structural features.

---

**Q2. Why is the dataset split into 60/20/20 instead of a simple 80/20 split?**

A three-way split provides an **independent validation set** separate from the test set. The training set is used to fit the model; the validation set is used to tune hyperparameters (e.g., polynomial degree, number of hidden neurons, regularisation strength) without touching the test set; and the test set provides a final, unbiased estimate of generalisation performance. Using the test set for tuning would cause data leakage and overly optimistic results.

---

**Q3. What does RMSE measure and why is it preferred over MSE for reporting?**

RMSE (Root Mean Squared Error) = √(Σ(yᵢ − ŷᵢ)² / n). It measures the average magnitude of prediction error in the **same units as the target variable** (in this case, kWh/m²), making it more interpretable than MSE. Larger errors are penalised more heavily than smaller ones due to the squaring operation.

---

### Section 2: Polynomial Regression

**Q4. What is multivariate polynomial regression?**

It extends linear regression by creating new features that are **polynomial combinations** of the original inputs. For two features x₁, x₂ and degree 2, the feature vector becomes: [1, x₁, x₂, x₁², x₁x₂, x₂²]. A linear regression model is then fit on these expanded features. This allows the model to capture non-linear relationships.

---

**Q5. What happens to the number of polynomial features as the degree increases for 9 input features?**

The number of features is C(n+d, d) where n is the number of original features and d is the degree. For n=9 features:
- Degree 1: 10 features
- Degree 2: 55 features
- Degree 3: 220 features
- Degree 4: 715 features
- Degree 5: 2002 features
- Degree 6: 5005 features

This combinatorial explosion causes extreme overfitting at higher degrees.

---

**Q6. What is overfitting and how does it appear in the RMSE vs Degree plot?**

Overfitting occurs when the model memorises training data instead of learning the underlying pattern. In the RMSE vs Degree plot, overfitting appears as a large **gap between training RMSE (very low) and validation RMSE (much higher)**. Training RMSE keeps decreasing with degree while validation RMSE starts increasing — this is the classical bias-variance trade-off in action.

---

**Q7. Why do we use `PolynomialFeatures` from scikit-learn instead of manually creating features?**

`PolynomialFeatures` automatically and correctly generates all polynomial and interaction terms up to the specified degree, handles the fit/transform split properly (fitting only on training data to avoid data leakage), and is compatible with scikit-learn pipelines. Manual computation is error-prone and doesn't scale.

---

**Q8. How is the best polynomial degree selected?**

The best degree is chosen as the one with the **lowest validation RMSE**. The validation set acts as a proxy for generalisation performance. After selecting the degree, the model is evaluated once on the test set to report the final performance.

---

### Section 3: Ridge Regression

**Q9. What is Ridge Regression and how does it differ from ordinary linear regression?**

Ridge Regression adds an **L2 regularisation penalty** to the least squares objective:

> Minimize: ||y − Xw||² + α||w||²

The penalty term α||w||² shrinks the coefficient weights towards zero, which reduces model complexity and variance. Ordinary linear regression (α=0) has no such constraint, making it prone to overfitting with high-degree polynomial features.

---

**Q10. What is the role of the hyperparameter α in Ridge Regression?**

- **α = 0**: No regularisation; equivalent to ordinary least squares.
- **Small α (e.g., 1e-2)**: Mild regularisation; allows some large coefficients.
- **Large α**: Strong regularisation; all coefficients are heavily shrunk towards zero, which can cause underfitting.

In this lab, α = 1e-2 is applied at degree 6 to reduce the overfitting observed with the unregularised polynomial model.

---

**Q11. Why is Ridge Regression applied specifically at degree 6?**

Degree 6 produces the most overfitting (widest train-validation RMSE gap) due to the huge number of polynomial features (~5005 for 9 inputs). Ridge regularisation is most useful exactly in such high-variance scenarios — it penalises large weights that arise from overfitting complex polynomial surfaces.

---

**Q12. Why is Ridge preferred over Lasso for this task?**

Ridge (L2) is preferred when all features are expected to contribute and we want **coefficient shrinkage without sparsity**. Lasso (L1) drives some coefficients exactly to zero (feature selection). Since polynomial features are all derived from the same 9 physical inputs, it is reasonable to keep all of them but shrink their magnitudes, making Ridge the more natural choice.

---

### Section 4: Neural Networks (PyTorch)

**Q13. Describe the neural network architecture used in this lab.**

- **Input layer**: Linear layer with neurons equal to the number of input features (9 for all features, 2 for top-2)
- **Hidden layer**: Single fully-connected layer with {4, 8, 16, 32, 64} neurons and **tanh activation**
- **Output layer**: Single linear neuron (regression output — no activation)
- **Loss function**: Mean Squared Error (MSE)
- **Optimiser**: Stochastic Gradient Descent (SGD), learning rate 0.01
- **Epochs**: 700

---

**Q14. Why is the tanh activation function used instead of ReLU?**

The `tanh` function maps inputs to (-1, 1) and is **smooth and differentiable everywhere**, which helps with gradient flow in shallow networks. For regression on normalised data, tanh is a reasonable choice. ReLU can suffer from the dying ReLU problem in certain configurations. Since the lab specifies tanh, it is also a pedagogical choice to demonstrate non-linear activation behaviour.

---

**Q15. Why is the output layer linear (no activation) for regression?**

In regression, we predict a **continuous unbounded value**. Applying an activation like sigmoid or tanh to the output would artificially limit the range of predictions. A linear output layer lets the network predict any real number, which is appropriate for heating load values.

---

**Q16. Why is feature normalisation (StandardScaler) important before training a neural network?**

Neural networks use gradient descent optimisation. If features are on vastly different scales (e.g., surface area in hundreds vs. orientation as 2–5), gradients will be dominated by large-scale features, causing slow or unstable training. StandardScaler transforms features to have **zero mean and unit variance**, ensuring balanced gradient updates across all inputs.

---

**Q17. What is SGD and why might it converge slower than Adam?**

Stochastic Gradient Descent updates parameters using the gradient computed on the entire training batch (or mini-batches). It has a fixed learning rate and no adaptive momentum. **Adam** maintains per-parameter adaptive learning rates and uses momentum, which typically leads to faster convergence and better handling of sparse gradients. SGD is simpler and more predictable but may require more epochs or careful learning rate tuning.

---

**Q18. How does increasing hidden neurons affect model capacity and risk of overfitting?**

More hidden neurons increase the number of learnable parameters, giving the model more capacity to fit complex patterns. With very few neurons (e.g., 4), the model may underfit. With many neurons (e.g., 64) and limited data, the model may overfit, evidenced by low training RMSE but higher validation RMSE. The optimal hidden size is chosen by comparing validation RMSE across all architectures.

---

**Q19. What does the loss vs. epochs plot tell you?**

- If both training and validation loss decrease together and converge: the model is learning and generalising well.
- If training loss decreases but validation loss plateaus or increases: **overfitting** — training continued too long or the model is too complex.
- If both losses remain high: **underfitting** — the model lacks capacity or the learning rate is too small.

---

### Section 5: Top-2 Features

**Q20. How are the top-2 features selected?**

Pearson correlation coefficients are computed between each input feature and the target variable (Heating Load) **on the training set only**. The two features with the highest **absolute correlation** values are selected. Computing correlation only on training data prevents information leakage from the validation and test sets.

---

**Q21. Why might models using only the top-2 features perform worse than models using all 9 features?**

Using only 2 features discards information carried by the remaining 7 features. Even features with lower correlation may contribute to prediction jointly with other features (interaction effects). Therefore, models using all 9 features generally achieve lower RMSE because they have access to the complete information in the dataset.

---

**Q22. What is the purpose of the 3D best-fit surface plot?**

The 3D surface visualises the learned regression function in the space of the two selected features (x-axis, y-axis) against the predicted heating load (z-axis). Overlaying training data points shows how well the surface fits the observed data. At low polynomial degrees, the surface is flat/smooth; at high degrees, it may exhibit extreme oscillations (overfitting), which is visually apparent.

---

### Section 6: Comparisons and Concepts

**Q23. What is the bias-variance trade-off?**

- **Bias**: Error from wrong assumptions in the model (e.g., a linear model for non-linear data). High bias → underfitting.
- **Variance**: Error from sensitivity to fluctuations in training data. High variance → overfitting.
- **Trade-off**: Increasing model complexity (e.g., higher polynomial degree, more hidden neurons) reduces bias but increases variance. The optimal model minimises total error on unseen data.

---

**Q24. Compare polynomial regression and neural networks for this task.**

| Aspect | Polynomial Regression | Neural Network |
|---|---|---|
| Feature transformation | Explicit (PolynomialFeatures) | Implicit (learned by hidden layer) |
| Interpretability | Moderate (coefficients exist) | Low (black box) |
| Scalability with features | Poor (feature explosion) | Good |
| Overfitting control | Regularisation (Ridge/Lasso) | Dropout, early stopping, architecture choice |
| Flexibility | Limited to polynomial shapes | Can approximate any function |

---

**Q25. Why do we evaluate the final model on the test set only once?**

Evaluating on the test set multiple times (once per hyperparameter choice) would cause the test set to indirectly influence model selection, making the final reported performance **optimistically biased**. The test set must remain a held-out proxy for real-world data the model has never seen. This is why hyperparameter tuning uses the validation set.

---

**Q26. If validation RMSE at degree 3 is lower than at degree 6 with Ridge, which model should you deploy?**

Deploy the degree 3 polynomial model. Despite Ridge reducing overfitting at degree 6, if the degree 3 model achieves lower validation RMSE, it is simpler, faster to compute, and likely generalises better. Occam's razor: prefer the simpler model that achieves equivalent or better performance.

---

**Q27. What does it mean if the scatter plot of actual vs. predicted values shows points spread far from the diagonal?**

Points close to the diagonal (y = x line) indicate accurate predictions. Points spread far from the diagonal indicate large prediction errors. A fan-shaped spread (errors increasing with prediction magnitude) suggests **heteroscedasticity** — the model's error variance is not constant across the range of outputs, which may suggest the need for log-transformation of the target or a more expressive model.

---
*Lab 10 — CS201L AI Laboratory, IIT Dharwad*
