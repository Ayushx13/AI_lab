# CS214: AI Lab — Lab 11 Viva Q&A
## Autoregression: AAPL Stock Price Prediction

---

## Section 1: Autocorrelation & Time Series Basics

**Q1. What is autocorrelation and why is it important in time series analysis?**

Autocorrelation measures the correlation of a time series with a lagged version of itself. It reveals how strongly past values influence future values. High autocorrelation means the series has strong temporal memory, which is exploitable by models like AR. It is important because it helps determine the appropriate lag order for AR models and reveals whether the series is stationary.

---

**Q2. What did the scatter plot between Close and Lag-1 Close tell you?**

The scatter plot showed a near-perfect positive linear relationship — points lie close to a diagonal line. This visually confirms the extremely high Pearson correlation coefficient (~0.9998), indicating that today's closing price is almost entirely predictable from yesterday's price. This is characteristic of a random walk process.

---

**Q3. What is the ACF (Autocorrelation Function) plot, and what did it reveal here?**

The ACF plot (correlogram) shows autocorrelation values at different lags. For AAPL closing prices, the ACF remains very close to 1.0 even at lags up to 600 days, decaying very slowly. This indicates:
- The series is **non-stationary** (has a long-term trend).
- There is strong **long-range dependence** — past prices remain correlated over very long horizons.
- The series likely needs differencing to become stationary.

---

**Q4. What does the Pearson correlation coefficient measure, and what value did you get for lag-1?**

Pearson correlation measures linear dependency between two variables, ranging from -1 to +1. For AAPL's Close vs Lag-1 Close, the coefficient is approximately **0.9998**, indicating an almost perfect positive linear relationship. This aligns with the visual scatter plot observation.

---

**Q5. What is a random walk? Is AAPL price a random walk?**

A random walk is a process where the next value equals the current value plus a random noise term: `y(t) = y(t-1) + ε(t)`. Since AAPL shows ACF near 1.0 that decays extremely slowly and lag-1 correlation ≈ 0.9998, it closely resembles a random walk. AR models on such series essentially learn to predict `y(t) ≈ y(t-1)`.

---

## Section 2: Autoregression Model

**Q6. What is an Autoregressive (AR) model? Write its mathematical formulation.**

An AR(p) model predicts the current value as a linear combination of the previous p values:

```
y(t) = w0 + w1*y(t-1) + w2*y(t-2) + ... + wp*y(t-p) + ε(t)
```

Where:
- `w0` is the intercept (bias)
- `w1, ..., wp` are the autoregressive coefficients
- `p` is the lag order
- `ε(t)` is white noise

---

**Q7. Why was the data NOT shuffled before splitting into train and test sets?**

Time series data has temporal ordering — each data point depends on its past values. Shuffling would break this temporal dependency, causing data leakage (future data in training set) and making the model learn spurious patterns. The sequential 10000/remaining split preserves the time order essential for valid autoregressive modeling.

---

**Q8. What is 1-step ahead prediction in the context of AR models?**

At each time step `t` in the test set, we predict only the next single value `y(t)` using the most recent `p` actual values from history (not previously predicted values). After prediction, we append the actual observed value to history before predicting the next step. This prevents error propagation and gives more accurate predictions compared to multi-step forecasting.

---

**Q9. What are RMSE and MAPE? Why are both used?**

- **RMSE (Root Mean Square Error):** `sqrt(mean((y_actual - y_pred)^2))` — penalizes large errors heavily, in the same unit as the target.
- **MAPE (Mean Absolute Percentage Error):** `mean(|y_actual - y_pred| / y_actual) * 100` — scale-independent percentage error, useful for comparing across different price ranges.

RMSE highlights large deviations; MAPE gives a relative measure. Using both together provides a complete picture of model performance.

---

**Q10. What are the AR(5) coefficients and what do they represent?**

The `params` from a fitted AutoReg model give `[w0, w1, w2, w3, w4, w5]`:
- `w0` = intercept
- `w1` = coefficient for lag-1 (yesterday's price)
- `w2..w5` = coefficients for lag-2 through lag-5

For stock prices, `w1` is typically very close to 1.0 and other coefficients are small, reflecting the near-random-walk behavior.

---

## Section 3: AR Models with Different Lags

**Q11. How does RMSE/MAPE change with increasing lag in the AR model?**

Generally:
- Very small lags (e.g., lag=1) capture only immediate autocorrelation.
- Moderate lags show marginal improvement since the series has strong long-range autocorrelation.
- Very large lags (500+) may slightly increase error due to estimation noise in fitting many coefficients on finite data.
- The improvement plateau is expected because AAPL prices have near-constant autocorrelation across all lags.

---

**Q12. What is the significance of choosing the right lag in an AR model?**

The lag order determines:
1. **Model complexity** — higher lag = more parameters = risk of overfitting.
2. **Information captured** — too small a lag misses important temporal patterns.
3. **Computational cost** — larger lags are more expensive to fit.

The optimal lag balances bias (underfitting with too few lags) and variance (overfitting with too many).

---

## Section 4: Optimal Lag Selection

**Q13. What is the heuristic for optimal lag selection used in this lab?**

The heuristic selects the largest lag `k` such that the absolute autocorrelation exceeds the significance threshold:

```
|ACF(k)| > sqrt(2 / T)
```

Where `T` is the number of training observations. This threshold approximates the 95% confidence band for testing whether autocorrelation is statistically significantly different from zero.

---

**Q14. Why is this threshold `sqrt(2/T)` used specifically?**

The approximate 95% confidence interval for ACF under the null hypothesis of no autocorrelation is `±1.96/sqrt(T)`. The `sqrt(2/T)` is a simplified version (since `1.96 ≈ sqrt(2) * 1.38` — roughly similar scale). It's a heuristic to identify the number of statistically significant lags without using strict hypothesis testing.

---

**Q15. For AAPL data, what happens with this heuristic and why?**

Since AAPL's ACF remains very high (near 1.0) even at lag 600, the heuristic will select a lag at or near 600 (the maximum tested). This reflects the fact that AAPL prices have statistically significant autocorrelation across all tested lags due to the strong underlying trend.

---

## Section 5: Neural Network Model

**Q16. How are lagged inputs constructed for the neural network?**

For a lag of `p`, a sliding window approach creates input-output pairs:
```
X[i] = [y(i-p), y(i-p+1), ..., y(i-1)]
y[i] = y(i)
```
Each row of X contains the previous `p` closing prices, and the target is the next price. This converts the time series into a supervised regression problem.

---

**Q17. Why is normalization (StandardScaler) applied before training the NN?**

Neural networks are sensitive to the scale of inputs:
- Large, unscaled values cause gradient explosion or very slow convergence.
- StandardScaler (zero mean, unit variance) puts all features on the same scale.
- Critically, the scaler is **fit only on training data** and applied to test data to prevent data leakage.
- The inverse transform is applied at evaluation to recover actual price predictions.

---

**Q18. Why is Sigmoid used as the activation function here?**

The lab specifies Sigmoid as the activation function. In general, Sigmoid outputs values between 0 and 1, which maps well to normalized inputs. However, Sigmoid can suffer from vanishing gradients in deep networks. For shallow networks (single hidden layer), it works adequately. In practice, ReLU is preferred for deeper architectures.

---

**Q19. What is SGD with momentum, and why is it used here?**

SGD (Stochastic Gradient Descent) updates parameters using the gradient of the loss on each batch. Momentum (0.9 here) adds a fraction of the previous update direction, which:
- Accelerates convergence in consistent gradient directions.
- Dampens oscillations in directions with high curvature.
- Helps escape shallow local minima.

---

**Q20. Why is a lower learning rate (0.001) used for lag=600 compared to lag=5 (0.01)?**

With lag=600 inputs, the input layer has 600 features and the hidden layer has 512–2048 neurons, creating a much larger parameter space. A lower learning rate:
- Prevents overshooting minima in a high-dimensional loss landscape.
- Stabilizes training with larger networks.
- Is standard practice: larger models require more careful optimization.

---

**Q21. Compare NN performance for lag=5 vs lag=600. What do you expect?**

- **lag=5:** Small input, fast training, may underfit if long-range patterns matter.
- **lag=600:** Large input, captures 2+ years of history, potentially better at identifying long-term trends.

Expected: lag=600 NN should achieve lower RMSE/MAPE as it sees more context, though this depends on training convergence. In practice, for near-random-walk stock data, the improvement is often marginal since yesterday's price dominates prediction.

---

**Q22. How do you compare performance across neural network architectures?**

By comparing RMSE and MAPE on the test set:
- Lower RMSE = smaller average deviation from actual prices.
- Lower MAPE = more accurate relative predictions.
- Training loss curves show convergence speed and stability.
- Architectures with more neurons generally have more capacity but may overfit with limited training data.

---

**Q23. How does the best NN model compare to the AR model?**

For stock price prediction with strong autocorrelation:
- AR models with 1-step ahead prediction using actual history are very competitive because the series is essentially a random walk.
- NN models trained on fixed lagged windows may slightly underperform AR because they don't use actual rolling history for prediction in the same way.
- AR(1) is asymptotically equivalent to: `y(t) ≈ y(t-1)` for random walks — a hard baseline to beat.
- NNs may outperform AR in non-linear regimes (e.g., during crashes/bubbles) but AR wins on smooth, trending segments.

---

**Q24. What is the difference between the AR model prediction approach and the NN model prediction approach in this lab?**

| | AR Model | Neural Network |
|---|---|---|
| **Prediction** | 1-step ahead using actual rolling history | Fixed window of normalized inputs |
| **History update** | Actual value appended after each step | No rolling update — batch inference |
| **Normalization** | No normalization needed | StandardScaler on train, applied to test |
| **Coefficients** | Interpretable linear weights | Black-box nonlinear weights |
| **Training** | Closed-form / OLS | Iterative gradient descent (SGD) |

---

**Q25. What improvements would you suggest to make the NN model better?**

1. Use **LSTM or GRU** instead of a fully connected network — designed for temporal sequences.
2. Use **ReLU or Leaky ReLU** instead of Sigmoid to avoid vanishing gradients.
3. Apply **Adam optimizer** instead of SGD for faster convergence.
4. Add **dropout layers** to regularize and reduce overfitting.
5. Use **rolling 1-step prediction** with actual history updates (same as AR approach).
6. Apply **differencing** to make the series stationary before modeling.
7. Increase epochs and use **learning rate scheduling**.

---

*End of Viva Q&A — Lab 11*
