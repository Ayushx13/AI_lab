# CS214: AI Laboratory — Lab 11 Practice Viva
## Topic: Autoregression & Neural Networks for Time Series (COVID-19 Cases)

---

## Section 1 — Autocorrelation & ACF

**Q1. What is autocorrelation in a time series?**

Autocorrelation (serial correlation) measures the linear relationship between a time series and a lagged version of itself. For lag $k$, it is:
$$\rho_k = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}$$
A high autocorrelation at lag 1 means today's value is strongly predicted by yesterday's value.

---

**Q2. Why is the lag-1 autocorrelation of the COVID-19 series so high (close to 1)?**

Epidemic spread is a smooth, continuous process — daily case counts change gradually rather than randomly. The rolling 7-day average further smooths the series, artificially inflating autocorrelation. Adjacent days share almost the same epidemiological state, making the correlation coefficient near 1.

---

**Q3. What is the ACF plot (correlogram) and what does it tell you?**

The AutoCorrelation Function (ACF) plot shows the correlation coefficient $\rho_k$ between the series and its $k$-step lag, for $k = 0, 1, \ldots, K$. It tells you:
- How many past values are statistically significant predictors.
- Whether the series is stationary (ACF decays rapidly) or non-stationary (ACF decays slowly/stays high).
- A good upper bound for the AR lag order $p$.

---

**Q4. What does it mean when the ACF decays very slowly?**

Slow ACF decay (remaining significant across many lags) indicates **non-stationarity** — the series has a trend or persistent structure. For the COVID series, all 60 lags remain significant because the series passes through two distinct epidemic waves with very different mean levels. This violates the stationarity assumption of basic AR models.

---

**Q5. What is the significance band in the ACF plot?**

The 95% confidence band is $\pm \frac{2}{\sqrt{T}}$, where $T$ is the number of observations. Lags whose ACF values fall outside this band are statistically significant at the 5% level, meaning their correlation is unlikely due to chance.

---

## Section 2 — Autoregression (AR) Model

**Q6. What is an Autoregressive (AR) model?**

An AR model of order $p$ — written AR($p$) — predicts the current value $X_t$ as a linear combination of the $p$ most recent past values plus a noise term:
$$X_t = w_0 + w_1 X_{t-1} + w_2 X_{t-2} + \cdots + w_p X_{t-p} + \epsilon_t$$
The coefficients $w_0, w_1, \ldots, w_p$ are estimated by Ordinary Least Squares (OLS) on the training data.

---

**Q7. Why is the train-test split done without shuffling for time series?**

Time series data is temporally ordered. Shuffling would cause data leakage — future values would appear in the training set and past values in the test set. This inflates performance metrics and makes evaluation unrealistic. The temporal order must be preserved to simulate real forecasting conditions.

---

**Q8. What does `model.predict(start=len(train), end=len(df)-1)` do in the AR model?**

It generates 1-step-ahead predictions for each time step in the test range. At each test index $t$, it uses the **actual** past $p$ values (not its own earlier predictions) as inputs. This is called **1-step-ahead prediction** and gives more accurate results than multi-step-ahead prediction.

---

**Q9. What is RMSE and how is it interpreted?**

Root Mean Squared Error:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
It is in the same units as the target variable (number of cases). It penalises large errors more heavily than small ones due to squaring. A lower RMSE means better predictive accuracy.

---

**Q10. What is MAPE and when can it be misleading?**

Mean Absolute Percentage Error:
$$\text{MAPE} = \frac{1}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$
MAPE is scale-independent (expressed as %). It is misleading when actual values $y_i$ are near zero — division by near-zero inflates MAPE dramatically. In the early COVID period (near-zero cases), MAPE values can be unreliable.

---

**Q11. What happens to RMSE and MAPE as the AR lag increases from 1 to 60?**

- At very low lags (1–5), RMSE and MAPE are typically lowest because recent values are the most predictive.
- As lag increases, the model includes older, less relevant information, and OLS must estimate more parameters on the same training data, increasing variance.
- Beyond an optimal lag, performance stabilises or slightly degrades — a classic bias-variance tradeoff.

---

## Section 3 — Optimal Lag Selection

**Q12. How is the optimal lag determined using the ACF threshold condition?**

The condition $|ACF(k)| > \frac{2}{\sqrt{T}}$ identifies lags where correlation is statistically significant at the 5% level. The optimal lag is set to the **largest lag** still satisfying this condition. This is a data-driven heuristic that avoids manual lag selection.

---

**Q13. Why might the heuristic optimal lag differ from the empirically best lag in Q3?**

The heuristic is based on linear statistical significance — it does not minimise RMSE or MAPE directly. It captures all statistically significant autocorrelation structure, which may over-estimate the useful lag for forecasting. The empirically best lag minimises test-set error, but risks overfitting to the specific test period. Neither method is universally superior; they serve different goals.

---

## Section 4 — Neural Network Model

**Q14. Why is normalisation applied before training the neural network?**

Neural networks using gradient descent converge faster and more stably when input features have similar scales. Without normalisation, features with large magnitudes dominate gradient updates, causing slow convergence or divergence. `StandardScaler` transforms each feature to zero mean and unit variance.

---

**Q15. Why is the sigmoid activation function used here, and what are its limitations?**

Sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ is smooth and differentiable, making it compatible with backpropagation. It was the classic choice for hidden layers. Its limitations include:
- **Vanishing gradient**: Derivatives near 0 or 1 are extremely small, causing gradients to vanish in deeper layers, slowing training.
- **Not zero-centred**: Outputs are always positive (0 to 1), which can slow SGD convergence.
- **Saturation**: Very large or small inputs get mapped to near-constant outputs.

ReLU is typically preferred in modern networks for these reasons.

---

**Q16. Why is SGD with momentum used rather than Adam?**

The lab specification requires SGD. However, conceptually: SGD with momentum accumulates a velocity vector in the direction of gradients, which dampens oscillations and accelerates convergence compared to vanilla SGD. Adam adapts the learning rate per parameter and usually converges faster, but SGD with momentum often generalises better and is the standard baseline for such experiments.

---

**Q17. How does increasing the number of hidden neurons affect performance?**

- More neurons increase model **capacity** — the ability to approximate complex non-linear functions.
- For this relatively small time series (~400 training points), very wide layers (256 neurons) may **overfit** the training data without better test performance.
- The optimal width balances capacity with generalisation.

---

**Q18. How does adding a second hidden layer affect the model?**

A second hidden layer allows the network to learn hierarchical representations — the second layer can compose features learned by the first. For smooth univariate time series, one hidden layer is often sufficient. Two layers introduce more parameters, increasing risk of overfitting and vanishing gradient when using sigmoid activations.

---

**Q19. Why might an AR(5) model outperform a neural network on this dataset?**

- The COVID series is **smooth and locally linear** — consecutive values are nearly identical (lag-1 correlation ≈ 1). AR exploits this directly.
- AR has very few parameters (~6 for AR(5)), making it robust on small datasets.
- Neural networks need more data to learn non-linear patterns; with ~400 training points and sigmoid activations, they may underfit or converge to suboptimal solutions.
- AR is a specialised model designed exactly for this type of problem.

---

**Q20. What is a lagged input representation for a neural network in time series forecasting?**

Instead of feeding raw time indices, the input $\mathbf{x}_t = [X_{t-1}, X_{t-2}, \ldots, X_{t-p}]$ is constructed from the $p$ most recent past values. This converts the time series into a supervised learning problem: predict $X_t$ from $\mathbf{x}_t$. The lag $p$ controls how much history the network can use, analogous to the order in an AR model.

---

## Section 5 — Conceptual / Deeper Questions

**Q21. Is the COVID-19 case series stationary? How do you check?**

No, it is non-stationary — it has two clear waves (trend-like behaviour) with different mean levels. Methods to check:
1. **Visual inspection** — the mean and variance clearly change over time.
2. **ACF plot** — slow decay indicates non-stationarity.
3. **Augmented Dickey-Fuller (ADF) test** — a statistical test for unit roots; failure to reject the null means non-stationary.

For strict AR modelling, differencing ($\Delta X_t = X_t - X_{t-1}$) is often applied to achieve stationarity.

---

**Q22. What is the difference between AR, MA, and ARMA models?**

| Model | Definition |
|-------|-----------|
| AR($p$) | Current value depends on $p$ past values |
| MA($q$) | Current value depends on $q$ past error (noise) terms |
| ARMA($p$, $q$) | Combination of AR and MA components |
| ARIMA($p$,$d$,$q$) | ARMA on $d$-times differenced series (handles non-stationarity) |

For the COVID series, ARIMA would be more theoretically appropriate than plain AR.

---

**Q23. What is multi-step-ahead forecasting and how does it differ from 1-step-ahead?**

- **1-step-ahead**: At each step, actual observed values are used as inputs to predict the next value. This is the setting in the lab.
- **Multi-step-ahead**: Predictions from previous steps are fed back as inputs. Errors accumulate over the forecast horizon, making this harder and generally less accurate.
- `model.predict()` in `statsmodels` AutoReg by default does 1-step-ahead when given the full data range.

---

**Q24. How would you improve the neural network model for this task?**

1. Use **ReLU** instead of sigmoid to avoid vanishing gradients.
2. Apply **LSTM or GRU** layers — designed specifically for sequential data.
3. Use **Adam** optimiser for faster convergence.
4. Apply **dropout regularisation** to reduce overfitting.
5. Use **more training epochs** with **learning rate scheduling**.
6. Difference the series before feeding to the NN to handle non-stationarity.
7. Feature engineering — add day-of-week, lockdown indicators, etc.

---

**Q25. Why do we use `torch.manual_seed(42)` before training?**

Neural network weights are initialised randomly. Without fixing the seed, results differ between runs due to different initialisations and stochastic gradient updates. Setting `manual_seed(42)` ensures **reproducibility** — the same random sequence is used every run, making experiments comparable.

---

*End of Practice Viva Q&A — Lab 11*
