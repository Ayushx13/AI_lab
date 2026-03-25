# Lab 10 – Viva Questions & Answers
### CS201L: Artificial Intelligence Laboratory — IIT Dharwad
**Topic:** Regression with Multi-Input Variables (AirQuality Dataset)

---

## Section 1: Data Preprocessing & Splitting

**Q1. Why do we split data into train, validation, and test sets? What would happen if we used only train/test?**

Training data fits the model parameters. The validation set is used to tune hyperparameters (like regularization strength or hidden layer size) without touching the test set. If we tuned on the test set, we'd be optimizing for it specifically — giving an overly optimistic and unreliable estimate of real-world performance. The test set must remain completely unseen until final evaluation.

---

**Q2. In this lab you used a 60/20/20 split. What factors influence the choice of split ratio?**

Dataset size is the main factor. Large datasets (100k+ samples) can afford smaller validation/test proportions. For small datasets, techniques like k-fold cross-validation are preferable to fixed splits. The 60/20/20 ratio is a common default that balances having enough training data while still having representative validation and test sets.

---

**Q3. Why did we replace -200 values in AirQuality.csv with NaN and drop them?**

`-200` is a sentinel value used in the dataset to indicate missing or erroneous sensor readings. Including them as real numeric values would severely distort the model — a regression model would try to learn from these fake extreme negatives as if they were real CO concentrations. Dropping rows with missing values is a simple strategy; imputation (filling with mean/median) is an alternative but may introduce noise.

---

**Q4. What is the Pearson correlation coefficient and what are its limitations?**

Pearson correlation (`r`) measures the linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive).

Limitations:
- It only captures **linear** relationships. Two variables can have a strong non-linear relationship (e.g., quadratic) with `r ≈ 0`.
- It is sensitive to **outliers**.
- Therefore, the "best feature" identified by correlation may not always be the most predictive in a non-linear model.

---

## Section 2: Ridge Regression

**Q5. What is Ridge Regression and why is it preferred over ordinary least squares (OLS) in this lab?**

Ridge Regression adds an L2 penalty term to the OLS loss function:

```
Minimize: ||y - Xw||² + α·||w||²
```

This shrinks the coefficients toward zero, reducing model variance at the cost of a small bias. With degree-4 polynomial features, OLS can overfit badly (many parameters, potentially correlated features). Ridge regularization controls this overfitting, making the model generalize better to unseen data.

---

**Q6. What does the regularization parameter α control in Ridge Regression? What happens as α → 0 and α → ∞?**

`α` balances the data-fitting term vs. the penalty term:

| α value | Effect |
|---------|--------|
| α → 0 | Approaches OLS — no regularization, risk of overfitting |
| α = 1 (used here) | Moderate regularization — balanced bias-variance |
| α → ∞ | All coefficients → 0, model predicts near the mean of y — underfitting |

In practice, `α` is tuned via the validation set using techniques like grid search or cross-validation.

---

**Q7. Why do we use `PolynomialFeatures` before Ridge Regression? What does a degree-4 transformation do?**

`PolynomialFeatures` transforms a single input `x` into:

```
[1, x, x², x³, x⁴]
```

This gives the model 5 features to fit a more flexible curve, letting a linear model (Ridge) approximate non-linear relationships between the feature and CO concentration. The risk is that higher degrees can overfit, which is why Ridge regularization is especially important here.

---

**Q8. Why must `PolynomialFeatures` be fit only on training data and only transformed on validation/test data?**

Using `fit_transform` on training data computes the transformation structure. Using `transform` (not `fit_transform`) on val/test ensures they are processed identically to training data — no information from val/test leaks into the transformation. This is the same principle as fitting `StandardScaler` only on training data. Violating this causes **data leakage**, giving falsely optimistic results.

---

**Q9. How would you choose the optimal polynomial degree for Ridge Regression?**

Train Ridge models with degrees 1 through N, and for each compute the validation RMSE. Plot the train and validation RMSE vs. degree. The optimal degree is the one with the lowest validation RMSE before the validation error starts rising (indicating overfitting). This is an example of hyperparameter tuning using the validation set.

---

## Section 3: Neural Network Regression (PyTorch)

**Q10. Why did we normalize the input features before training the neural network?**

SGD is sensitive to feature scale. If one feature ranges 0–2000 and another 0–1, the gradient steps will be dominated by the large-scale feature, causing slow or unstable convergence. `StandardScaler` brings all features to zero mean and unit variance, making the loss landscape more isotropic and allowing SGD to converge faster and more reliably.

---

**Q11. Explain the neural network architecture used: input → tanh hidden layer → linear output. Why tanh and not ReLU?**

The network maps 2 input features to a hidden layer of size `h` using tanh activation, then to a single output neuron with **linear** activation (no activation = regression output).

- `tanh` is bounded `[-1, 1]` and **zero-centered**, which can help with gradient flow in shallow networks.
- ReLU could also work but risks dying neurons (neurons stuck at zero for negative inputs).
- For regression, the output layer **must be linear** — applying an activation would constrain the range of predictions.

---

**Q12. What is the role of `MSELoss` in training? Why MSE and not MAE?**

`MSELoss` computes the mean of squared differences between predictions and targets. It is preferred because:
- It is **differentiable everywhere** (unlike MAE which has a non-differentiable kink at 0), making gradient computation clean.
- It **penalizes large errors more heavily** due to squaring, which can be desirable.
- For reporting results we take the square root (RMSE) to restore original units (mg/m³).

---

**Q13. Why is SGD used instead of Adam? What are the trade-offs?**

| Optimizer | Pros | Cons |
|-----------|------|------|
| SGD | Simple, good generalization, foundational | Slow convergence, sensitive to LR |
| Adam | Fast convergence, adaptive LR | Can overfit, harder to tune in some cases |

The lab specifies SGD as it is the foundational optimizer. With `lr=0.001` and 500 epochs, SGD converges adequately. Adam would reach lower loss faster but the lab focuses on understanding the basics.

---

**Q14. How do you select the best neural network configuration among hidden sizes {16, 32, 64}?**

We select the model with the **lowest validation RMSE**. The validation set was not used in training, so a lower validation RMSE indicates better generalization. We do not use test RMSE for selection — that is reserved for final unbiased evaluation after all hyperparameter decisions are finalized.

---

**Q15. What does the Training Loss vs. Epochs plot tell you? What patterns indicate overfitting or underfitting?**

| Pattern | Interpretation |
|---------|---------------|
| Both losses decrease and converge | Good fit — model is generalizing |
| Train loss low, val loss rising | **Overfitting** — model memorized training data |
| Both losses remain high and plateau | **Underfitting** — try more neurons, more epochs, or lower LR |
| Val loss spiky/noisy | Small validation set or high learning rate |

---

## Section 4: Evaluation & Metrics

**Q16. What is RMSE and why is it preferred over MSE for reporting results?**

```
RMSE = √(mean of squared prediction errors)
```

RMSE is in the **same units as the target variable** (mg/m³ for CO concentration), making it directly interpretable — e.g., RMSE = 0.5 means predictions are off by ~0.5 mg/m³ on average. MSE is in squared units, which is harder to interpret. We use MSE during training (smoother for optimization) but report RMSE for human readability.

---

**Q17. If Train RMSE is very low but Test RMSE is high, what does that indicate and how would you fix it?**

This is a classic sign of **overfitting** — the model has memorized training data but fails to generalize.

Fixes:
- Increase regularization `α` in Ridge Regression
- Reduce neural network capacity (fewer hidden neurons)
- Use **dropout** regularization in the neural network
- Collect more training data
- Use **early stopping** based on validation loss
- Reduce polynomial degree

---

**Q18. What is the difference between validation RMSE and test RMSE conceptually? Can they be used interchangeably?**

- **Validation RMSE** guides model selection and hyperparameter tuning — it is seen during development, so the model implicitly adapts to it.
- **Test RMSE** is the final, unbiased measure of generalization — seen only once after all decisions are made.

Using test RMSE for tuning converts it effectively into a second validation set, making the final test RMSE **optimistically biased**. They should never be used interchangeably in a rigorous evaluation pipeline.

---

## Section 5: Theory & Concepts

**Q19. What is the bias-variance trade-off? How does it relate to Ridge Regression and the choice of α?**

- **Bias**: Error from wrong/oversimplified assumptions → underfitting
- **Variance**: Sensitivity to training data fluctuations → overfitting

| α value | Bias | Variance |
|---------|------|----------|
| Low α | Low | High (overfits) |
| High α | High | Low (underfits) |
| Optimal α | Balanced | Balanced |

Ridge finds a sweet spot. The degree-4 polynomial introduces potential high variance, and `α=1` regularizes it — tuning `α` on the validation set finds the optimal bias-variance trade-off.

---

**Q20. Why is the 3D surface plot useful? What additional insight does it give over a 2D scatter plot?**

The 3D surface visualizes how the model uses **two features simultaneously** to predict CO. It shows the interaction effect — how the predicted CO changes across the joint space of both features. A 2D plot can only show one feature's relationship with CO at a time. The 3D surface also confirms whether a linear plane is a good fit or if the data has curvature requiring polynomial or non-linear models.

---

**Q21. Could you use all 12 features for Ridge Regression instead of just `x_best`? What are the trade-offs?**

Yes. Using more features can reduce bias (model has more information). But it also:
- Increases variance, especially with polynomial expansion of all features (feature space explodes)
- Makes the model harder to interpret
- Increases computation time
- Risks **multicollinearity** — many sensor features in AirQuality.csv are highly correlated with each other, which can destabilize coefficient estimates even in Ridge

Feature selection using correlation is a practical way to balance expressiveness with simplicity.

---

**Q22. What is the difference between L1 (Lasso) and L2 (Ridge) regularization? When would you prefer each?**

| Property | Ridge (L2) | Lasso (L1) |
|----------|-----------|-----------|
| Penalty term | α·||w||² | α·||w||₁ |
| Effect on coefficients | Shrinks all toward zero | Can set some exactly to zero |
| Feature selection | No (keeps all features) | Yes (automatic feature selection) |
| Solution | Closed-form | Requires iterative solver |
| Preferred when | All features are relevant | Many irrelevant features exist |

In this lab, Ridge is used because polynomial features are all derived from the same physical feature and all contribute to the fit.

---

**Q23. What would happen if you did not use `torch.no_grad()` during validation?**

Without `torch.no_grad()`, PyTorch would build a computation graph for the validation forward pass, consuming unnecessary memory and slowing down the loop. More importantly, gradients from validation would accumulate in the model parameters if `backward()` was accidentally called. `torch.no_grad()` is a context manager that disables gradient tracking — it is a best practice for any inference or evaluation step.

---

**Q24. What is the purpose of `optimizer.zero_grad()` before each training step?**

PyTorch **accumulates gradients** by default — each call to `loss.backward()` adds gradients to the existing `.grad` buffers. Without calling `zero_grad()` before each step, gradients from previous batches would add up, causing incorrect weight updates and unstable training. `zero_grad()` resets all gradients to zero before computing fresh ones for the current batch.

---

*End of Viva Questions — Lab 10*