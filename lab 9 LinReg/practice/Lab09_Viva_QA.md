# CS201L – Lab 9: Regression | Viva Preparation Q&A

---

## Section 1: Fundamentals of Regression

**Q1. What is regression, and how does it differ from classification?**

Regression is a supervised learning task where the model predicts a **continuous numeric output** (e.g., CO concentration in mg/m³). Classification, by contrast, predicts a **discrete class label** (e.g., cat vs. dog). The loss functions differ too — regression typically uses MSE/RMSE while classification uses cross-entropy.

---

**Q2. What is the objective function minimized in Linear Regression?**

Linear regression minimizes the **Residual Sum of Squares (RSS)**, also called the **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

The closed-form solution (Normal Equation) is:

$$\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

scikit-learn's `LinearRegression` uses this (or a numerically stable variant via SVD) internally.

---

**Q3. What is RMSE and why is it preferred over MSE in reporting?**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$

RMSE is preferred because its **units match the target variable** (here, mg/m³), making it directly interpretable. MSE squares the errors, so its unit is mg²/m⁶, which is harder to reason about.

---

**Q4. What assumptions does linear regression make?**

1. **Linearity** – relationship between X and y is linear.
2. **Independence** – observations are independent of each other.
3. **Homoscedasticity** – constant variance of residuals.
4. **Normality** – residuals are normally distributed (important for inference, not prediction).
5. **No multicollinearity** – features are not highly correlated with each other (for multiple regression).

---

## Section 2: Data Splitting

**Q5. Why do we split the data into train, validation, and test sets?**

- **Train set (60%):** The model learns its parameters from this.
- **Validation set (20%):** Used to tune hyperparameters (e.g., polynomial degree) and detect overfitting *without* touching the test set.
- **Test set (20%):** The final, held-out evaluation of model performance — a proxy for real-world generalization. It must **never be used during model selection**.

Using only train/test (no validation) risks tuning your model to the test set, leading to overly optimistic estimates.

---

**Q6. Why should you compute correlations only on the training set?**

Computing statistics (like correlation) on the full dataset and *then* splitting is a form of **data leakage** — the model or feature selection process has indirectly "seen" the test data. All preprocessing decisions must be made using training data only, and the same transformation applied to val/test.

---

## Section 3: Correlation Analysis

**Q7. What is Pearson correlation and what does it measure?**

Pearson correlation coefficient measures the **linear relationship** between two variables:

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \cdot \sum(y_i - \bar{y})^2}}$$

- Range: **[-1, 1]**
- `r = 1` → perfect positive linear relationship
- `r = -1` → perfect negative linear relationship
- `r = 0` → no linear relationship (but could still have non-linear relationship)

We use **absolute value** for feature selection because a strong negative correlation is equally informative as a strong positive one.

---

**Q8. Why might the best correlated feature still not be the best predictor?**

Pearson correlation only captures **linear** relationships. A feature could have a strong non-linear relationship with the target (e.g., quadratic) but a low Pearson r. Additionally, a feature may be correlated but contain noise, or may be highly correlated with another feature already in the model (multicollinearity). For simple regression here, we pick the best linear correlate as a starting point.

---

**Q9. In this dataset, which feature is typically selected as x_best and why?**

`PT08.S1(CO)` (the tin oxide sensor nominally targeting CO) typically has the highest correlation with `CO(GT)` because it is specifically designed to detect CO — the same quantity we're trying to predict. It essentially measures a chemical proxy of the target directly.

---

## Section 4: Simple Linear Regression

**Q10. Write out the mathematical form of simple linear regression.**

$$\hat{y} = b_0 + b_1 \cdot x_{\text{best}}$$

Where:
- $b_0$ = intercept (value of $\hat{y}$ when $x = 0$)
- $b_1$ = slope (change in $\hat{y}$ per unit change in $x$)

These are estimated by minimizing SSR using the Normal Equation or gradient descent.

---

**Q11. How do you interpret the intercept and slope in the context of this problem?**

- **Slope ($b_1$):** For every one-unit increase in `PT08.S1(CO)` sensor reading, the predicted CO concentration increases by $b_1$ mg/m³.
- **Intercept ($b_0$):** The predicted CO concentration when the sensor reading is 0 (may not be physically meaningful but is mathematically necessary).

---

**Q12. Can Train RMSE be lower than Test RMSE? Why?**

Yes, almost always. The model is **directly optimized** on the training set, so it fits training data best. The test set is unseen, so errors are generally higher. A large gap between Train RMSE and Test RMSE signals **overfitting**.

---

## Section 5: Polynomial Regression

**Q13. What is polynomial regression? Is it still a "linear" model?**

Polynomial regression extends linear regression by adding **polynomial terms** of the feature:

$$\hat{y} = b_0 + b_1 x + b_2 x^2 + \cdots + b_p x^p$$

Yes — it is still a **linear model** because it is linear in the *parameters* ($b_0, b_1, \ldots, b_p$), even though it is non-linear in $x$. We use `PolynomialFeatures` to engineer the new columns $[x, x^2, \ldots, x^p]$, then fit `LinearRegression` on them.

---

**Q14. What does `PolynomialFeatures(degree=p)` do exactly?**

It transforms the input $x$ into a feature matrix:

$$[1, x, x^2, x^3, \ldots, x^p]$$

For example, with `degree=3` and input $x = 5$:
$$[1, 5, 25, 125]$$

`fit_transform` is called on training data; `transform` (without fitting) is called on val/test — this prevents data leakage.

---

**Q15. Why do we use `poly.fit_transform(X_train)` but `poly.transform(X_val)`?**

`fit_transform` **learns** statistics from the data (e.g., input ranges) and then transforms. `transform` only applies the **already-learned** transformation. Using `fit_transform` on val/test would mean fitting the transformer separately on each split, which is data leakage and leads to inconsistent feature spaces.

---

**Q16. What is the bias-variance tradeoff? How does polynomial degree relate to it?**

- **Bias:** Error from overly simplistic model assumptions (underfitting). High bias → high train *and* test error.
- **Variance:** Error from model being too sensitive to training data (overfitting). High variance → low train error but high test error.

As polynomial degree increases:
- Degree too low → high bias (underfitting, both train and val RMSE high)
- Degree too high → high variance (overfitting, train RMSE drops but val RMSE rises)
- **Optimal degree** = sweet spot where val RMSE is minimized.

---

**Q17. How do you select the best polynomial degree?**

By plotting **Validation RMSE vs. Degree**. The degree where validation RMSE is **minimum** is selected. We do NOT use test RMSE for this decision — using test RMSE to choose a model violates evaluation integrity and leads to optimistic results.

---

**Q18. Why might a degree-6 polynomial overfit even with only one input feature?**

Because with degree 6, the model has 7 parameters ($b_0$ through $b_6$). With sufficient degrees of freedom, the polynomial can bend to fit noise in the training data — memorizing outliers instead of learning the underlying pattern. This shows up as very low train RMSE but higher val/test RMSE.

---

## Section 6: Practical / Code Questions

**Q19. Why do we replace -200 with NaN in the AirQuality dataset?**

The AirQuality dataset uses **-200 as a sentinel value** for missing/faulty sensor readings (documented in the UCI repository). CO concentrations cannot be negative in reality. If we include -200 as a real value, the model would learn incorrect patterns. Replacing with NaN allows us to then drop or impute those rows correctly.

---

**Q20. Why split with `test_size=0.25` in the second `train_test_split` call?**

After the first split, 80% of data remains in `train_val`. To get a 60/20 split out of the original data:
$$0.80 \times 0.75 = 0.60 \quad \text{(train)} \qquad 0.80 \times 0.25 = 0.20 \quad \text{(val)}$$

So using `test_size=0.25` on the 80% remainder gives exactly 60% train and 20% val of the original dataset.

---

**Q21. What would happen if you fit `PolynomialFeatures` on the full dataset before splitting?**

This is **data leakage**. The polynomial transformer would "see" val/test data during fitting. While `PolynomialFeatures` doesn't learn statistics the way `StandardScaler` does, the habit is still bad practice and especially harmful when combined with other preprocessing steps. Always fit preprocessors on training data only.

---

**Q22. What is the difference between underfitting and overfitting? How would you detect each from the RMSE plot?**

| Situation | Train RMSE | Val RMSE | Diagnosis |
|-----------|------------|----------|-----------|
| Underfitting | High | High | Model too simple (bias) |
| Good fit | Low | Low (≈ Train) | Generalizing well |
| Overfitting | Very Low | High | Model too complex (variance) |

In the polynomial RMSE plot: if Train RMSE keeps falling but Val RMSE starts rising with degree, that crossover region marks the onset of overfitting.

---

**Q23. Can you use multiple features for polynomial regression? How?**

Yes. `PolynomialFeatures` with multiple inputs generates **all interaction terms** too. For example, with features $x_1, x_2$ and degree 2:
$$[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$$

This is called **multivariate polynomial regression**. For this lab, we restrict to one feature (x_best) for simplicity and interpretability.

---

## Section 7: Quick-Fire Conceptual Questions

**Q24. Is RMSE sensitive to outliers?**  
Yes. Since RMSE squares errors, large errors (outliers) are penalized disproportionately. Metrics like MAE (Mean Absolute Error) are more robust to outliers.

**Q25. Can the test RMSE be better (lower) than validation RMSE?**  
Yes, by chance, if the test set happened to be easier to predict. This is why evaluation over multiple folds (cross-validation) gives more reliable estimates.

**Q26. What is R² and how does it relate to RMSE?**  
$R^2$ (coefficient of determination) measures the proportion of variance in $y$ explained by the model:
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
$R^2 = 1$ means perfect fit; $R^2 = 0$ means the model is no better than predicting the mean. Unlike RMSE, $R^2$ is scale-independent, making it useful for comparing across datasets.

**Q27. What regularization techniques could reduce overfitting in polynomial regression?**  
- **Ridge Regression (L2):** Penalizes large coefficients, shrinks them toward zero.
- **Lasso Regression (L1):** Can zero out coefficients entirely, performing implicit feature selection.
- Both add a penalty term $\lambda \|\mathbf{w}\|^2$ (or $\lambda \|\mathbf{w}\|_1$) to the loss, controlled by hyperparameter $\lambda$.

---

*Good luck with your viva! Focus on being able to explain the why behind each step, not just the how.*
