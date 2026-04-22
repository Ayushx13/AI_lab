# CS201L: Artificial Intelligence Laboratory
## Lab 9 – Regression: Linear & Polynomial Curve Fitting
### Viva Questions and Answers | IIT Dharwad

---

## Section 1: Fundamentals of Regression

**Q1. What is regression in machine learning?**

> Regression is a type of supervised learning where the model learns to predict a continuous numerical output (like Heating Load) from one or more input features. Unlike classification which predicts categories, regression predicts values on a continuous scale.

---

**Q2. What is the difference between Simple Linear Regression and Multiple Linear Regression?**

> Simple Linear Regression uses only ONE input feature to predict the output (e.g., using x1 alone to predict Heating Load). Multiple Linear Regression uses TWO or MORE input features simultaneously.
> - Simple LR equation: `y = w0 + w1*x1`
> - Multiple LR equation: `y = w0 + w1*x1 + w2*x2 + ... + wn*xn`

---

**Q3. What does the best-fit line mean in linear regression?**

> The best-fit line is the straight line that minimizes the total prediction error across all training samples. Mathematically, it minimizes the Sum of Squared Errors (SSE) between actual and predicted values. The slope and intercept are found during training using the Ordinary Least Squares (OLS) method.

---

**Q4. What is RMSE and why do we use it as an evaluation metric?**

> RMSE stands for Root Mean Squared Error:
> ```
> RMSE = sqrt( mean( (y_actual - y_predicted)^2 ) )
> ```
> We use it because it is in the same unit as the target variable (Heating Load in kWh/m²), making it easy to interpret. A lower RMSE means better model performance. Squaring penalizes large errors more than small ones.

---

**Q5. What is the purpose of splitting data into Train, Validation, and Test sets?**

> - **Training set (60%)** — used to fit/learn the model parameters
> - **Validation set (20%)** — used to tune hyperparameters and select the best model (e.g., best polynomial degree) without touching test data
> - **Test set (20%)** — used only at the very end to report the final, unbiased performance
>
> This 3-way split prevents overfitting to the test data during model selection.

---

## Section 2: Polynomial Regression & Overfitting

**Q6. What is polynomial regression and how does it differ from linear regression?**

> Polynomial regression extends linear regression by adding higher-degree powers of the input feature (x², x³, etc.) as new features. While linear regression fits a straight line, polynomial regression can fit curves. Internally it still uses `LinearRegression` — the trick is transforming the input using `PolynomialFeatures` first.

---

**Q7. What happens as we increase the polynomial degree from 2 to 11?**

> - **Low degree (2-3):** Model may underfit — both train and val RMSE are high
> - **Optimal degree:** Model generalizes well — val RMSE is at its lowest
> - **High degree (9-11):** Model overfits — training RMSE keeps falling but val RMSE starts rising, meaning it memorizes noise instead of learning the true pattern

---

**Q8. How do we select the best polynomial degree? Why not use the test set for this?**

> We train models for each degree (2 to 11) and pick the degree with the **lowest validation RMSE**. We never use the test set for this selection — doing so would cause data leakage, making our final test evaluation overly optimistic and unreliable.

---

**Q9. What is overfitting? How do you spot it in the RMSE vs Degree plot?**

> Overfitting is when the model learns noise in training data and performs poorly on new data. In the RMSE vs Degree plot, you can identify it when:
> - Training RMSE keeps decreasing as degree increases
> - Validation RMSE starts **increasing** after a certain point
> - The **gap between training and validation RMSE grows** — this is the classic overfitting signature

---

**Q10. What is the role of `PolynomialFeatures` in scikit-learn?**

> `PolynomialFeatures(degree=p)` transforms a single input `x` into a feature matrix:
> ```
> [1, x, x², x³, ..., x^p]
> ```
> This lets `LinearRegression` fit polynomial curves. Important: always call `fit_transform()` on training data and only `transform()` on val/test to avoid data leakage.

---

## Section 3: Correlation & Feature Selection

**Q11. What is Pearson correlation coefficient and what does its value mean?**

> Pearson r measures the linear relationship between two variables, ranging from -1 to +1:
> - `r = +1` → perfect positive linear relationship
> - `r = -1` → perfect negative linear relationship
> - `r = 0` → no linear relationship
>
> We compute r between each feature (x1..x9) and Heating Load (y) to find which features are most informative.

---

**Q12. Why do we use the absolute value of correlation to select the best feature?**

> Both strong positive (r ≈ +1) and strong negative (r ≈ -1) correlations indicate a strong linear relationship with the target. The sign just tells us the direction, not the strength. For feature selection we care about **strength**, so we use `|r|`. The feature with the highest `|r|` is the most useful predictor.

---

**Q13. Why does a model using the highest-correlated feature perform better than x1?**

> x1 (Relative Compactness) may not be the feature most linearly related to Heating Load. If another feature like x5 (Overall Height) has a higher absolute correlation, it provides more predictive signal — the regression line fits better and RMSE drops. Correlation analysis helps us pick the feature the model can learn from most effectively.

---

**Q14. Why does adding a second feature (top-2 MLR) improve the model over single-feature models?**

> Adding a second correlated feature gives the model additional information not already captured by the first feature. If the two features are somewhat independent (low inter-feature correlation) but both correlated with y, the model can exploit both signals and reduce prediction error. This is why MLR generally outperforms simple LR.

---

**Q15. What is multicollinearity and is it a concern here?**

> Multicollinearity is when two or more input features are highly correlated with each other (e.g., x1 and x2 are geometrically related since compactness and surface area are inversely linked). This makes regression coefficients unstable and hard to interpret. For **prediction** purposes it doesn't hurt RMSE much, but for interpreting coefficients it's a problem. Ridge/Lasso regression can handle it.

---

## Section 4: Multiple Linear Regression & 3D Visualization

**Q16. How do you interpret the 3D best-fit plane for the top-2 features model?**

> In the 3D plot:
> - x-axis and y-axis → the two selected features
> - z-axis → predicted Heating Load
> - The **red plane** is the model's prediction surface
> - **Blue dots** are actual training samples
>
> Points close to the plane = accurate predictions. Points far from it = prediction errors (residuals).

---

**Q17. Why does using all 9 features generally give the lowest test RMSE?**

> More relevant features give the model more information to learn from. Each feature that correlates with the target reduces unexplained variance. With all 9 features the model captures more signal in the data, leading to better generalization. With 768 samples (enough data), overfitting is not a significant concern for linear models.

---

**Q18. What do the coefficients in a Multiple Linear Regression model represent?**

> Each coefficient `w_i` represents the expected change in Heating Load for a **one-unit increase in that feature, while all other features are held constant**. A large positive coefficient means the feature increases Heating Load; a large negative one means it decreases it. The intercept `w_0` is the predicted value when all features are zero.

---

## Section 5: Practical & Code-Based Questions

**Q19. Why do we use `fit_transform()` on training data but only `transform()` on val/test?**

> `fit_transform()` computes transformation parameters **from** the training data and then transforms it. `transform()` applies those **same parameters** to new data. If we called `fit_transform()` on val/test, it would compute different parameters from that data — causing data leakage and giving an unfairly optimistic evaluation.

---

**Q20. What does the diagonal line in Actual vs Predicted scatter plots represent?**

> The diagonal dashed line (`y = x`) represents **perfect prediction** — where actual equals predicted. Points **on** the line are predicted exactly right. Points **above** the line are underpredicted; points **below** are overpredicted. A better model has points clustered tightly around this diagonal. Spread away from it corresponds to higher RMSE.

---

**Q21. What is the formula for RMSE and why take the square root of MSE?**

> ```
> MSE  = (1/n) * Σ(y_i - ŷ_i)²
> RMSE = sqrt(MSE)
> ```
> We take the square root to bring the error back to the **same units as the target variable**. MSE is in squared units (e.g., kWh²/m⁴) which is hard to interpret. RMSE is in kWh/m² — the same as Heating Load — making it intuitive to understand the average prediction error.

---

**Q22. How would you improve the models in this lab if RMSE is still high?**

> Several approaches:
> 1. **Feature engineering** — create interaction terms or ratios between features
> 2. **Regularization** — use Ridge or Lasso regression to prevent overfitting
> 3. **Non-linear models** — try Decision Trees, Random Forest, or SVR
> 4. **Polynomial MLR** — apply polynomial features to all inputs, not just x1
> 5. **Remove outliers** — outliers can skew regression coefficients significantly

---

## Quick Reference: Model Comparison

| Model | Features Used | Expected Test RMSE |
|---|---|---|
| Simple LR (x1) | 1 | Highest — x1 alone is a weak predictor |
| Polynomial LR (best degree, x1) | 1 | Better than simple LR |
| Simple LR (Best Correlated Feature) | 1 | Good — most informative single feature |
| Multiple LR (Top-2 Features) | 2 | Better — two informative features |
| Multiple LR (All Features) | 9 | Lowest — full information used |

---

*Good luck with your viva! — CS201L, IIT Dharwad*
