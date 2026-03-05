# Lab 08 - Neural Networks: Viva Questions & Answers
### CS201L: Artificial Intelligence Laboratory | IIT Dharwad

---

## 1. Conceptual / Theory

**Q1. What is the role of an activation function in a neural network? Why can't we just use linear neurons throughout?**

An activation function introduces **non-linearity** into the network. Without it, no matter how many layers you stack, the entire network collapses into a single linear transformation — because the composition of linear functions is still linear. Non-linearity allows the network to learn complex, curved decision boundaries needed for tasks like classifying human activities. In this assignment, `tanh` is used as the hidden layer activation to introduce this non-linearity.

---

**Q2. Why is `tanh` used as the hidden layer activation function in this assignment? What are its properties?**

`tanh` (hyperbolic tangent) was specified in the assignment instructions. Its properties are:
- **Output range:** −1 to +1 (zero-centered, unlike sigmoid which outputs 0 to 1)
- **Formula:** tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)
- **Zero-centered output** means gradients during backpropagation are more balanced, leading to faster convergence than sigmoid
- It is **smooth and differentiable** everywhere, which is necessary for gradient-based optimization
- The saturation at ±1 can still cause vanishing gradients in very deep networks

---

**Q3. What is the vanishing gradient problem? How does `tanh` relate to it, and why is ReLU often preferred in deeper networks?**

The vanishing gradient problem occurs when gradients become extremely small as they are backpropagated through many layers. Since the gradient is multiplied by the derivative of the activation at each layer, and `tanh`'s derivative (1 − tanh²(x)) is at most 1 and approaches 0 at saturation (large |x|), repeated multiplication shrinks the gradient — making early layers learn very slowly or not at all.

ReLU (Rectified Linear Unit) avoids this because its derivative is either 0 or 1, so gradients don't shrink as they pass through unsaturated neurons. This is why ReLU is preferred in deeper networks. However, for shallow networks (1–2 hidden layers) like in this assignment, `tanh` works reasonably well.

---

**Q4. Why do we use Cross Entropy loss for multiclass classification instead of Mean Squared Error (MSE)?**

Cross Entropy is better suited for classification for two reasons:
1. **Probabilistic interpretation:** Cross Entropy directly measures the difference between the predicted probability distribution (from Softmax) and the true distribution (one-hot label). It penalizes confident wrong predictions very heavily.
2. **Gradient behavior:** MSE combined with Softmax leads to flat gradients when predictions are very wrong (saturation problem), slowing learning significantly. Cross Entropy avoids this — gradients remain strong even when predictions are far from the target.

Formula: L = −Σ yᵢ · log(ŷᵢ), where yᵢ is the true label and ŷᵢ is the predicted probability.

---

**Q5. What is Softmax, and why is it suitable for the output layer in a multiclass classification problem?**

Softmax converts a vector of raw scores (logits) into a **probability distribution** — all values are between 0 and 1 and they sum to exactly 1. This makes it ideal for multiclass classification because each output neuron's value can be interpreted as the probability that the input belongs to that class. The class with the highest probability is taken as the prediction.

Formula: Softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ

In this assignment, the output layer has 6 neurons (one per activity class), and Softmax turns their raw scores into class probabilities.

---

**Q6. Why does PyTorch's `CrossEntropyLoss` not require you to explicitly apply Softmax in the output layer?**

PyTorch's `nn.CrossEntropyLoss` internally combines **LogSoftmax + Negative Log Likelihood Loss (NLLLoss)** in a single numerically stable operation. So if you apply Softmax in the model and then pass it to `CrossEntropyLoss`, you are applying Softmax twice — which gives wrong results. That's why our model's output layer is a plain `nn.Linear` with no activation, and we let `CrossEntropyLoss` handle the Softmax internally.

---

**Q7. What is the difference between a single hidden layer and a two hidden layer neural network in terms of representational capacity?**

- A **single hidden layer** network (with enough neurons) is a universal approximator — it can theoretically approximate any continuous function. However, it may require an exponentially large number of neurons to do so.
- A **two hidden layer** network can represent the same functions with fewer total neurons by learning hierarchical features — the first hidden layer learns simple patterns and the second layer combines them into more complex ones.
- In practice, depth often helps more than width for complex problems. For this HAR dataset with pre-engineered features, both 1 and 2 hidden layer networks tend to perform well.

---

**Q8. What does it mean for a neural network to "converge"? How is convergence detected in this assignment?**

Convergence means the model's weights have reached a stable state where further training produces negligible improvement in the loss. In this assignment, convergence is detected using **early stopping**:

```python
if abs(train_loss - prev_loss) < threshold:   # threshold = 1e-3
    counter += 1
else:
    counter = 0
if counter >= patience:   # patience = 10
    break
```

If the change in training loss between consecutive epochs is less than 0.001 for 10 consecutive epochs, training stops. This prevents wasting time on unnecessary epochs.

---

**Q9. What is the role of the learning rate in SGD? What happens if it is too high or too low?**

The learning rate (lr) controls **how large a step** the optimizer takes in the direction of the negative gradient when updating weights:

**weight = weight − lr × gradient**

- **Too high (e.g., 0.9):** The updates are too large. The loss may oscillate wildly or even diverge (increase instead of decrease). The model never settles at a minimum.
- **Too low (e.g., 0.00001):** Training is extremely slow. The model will eventually converge but takes many more epochs. Can also get stuck in a local minimum.
- **Good range for SGD:** 0.001 to 0.01, which is why `lr=0.01` was used in this assignment.

---

**Q10. What is the difference between Stochastic Gradient Descent (SGD) and full-batch Gradient Descent? Which one is used here, and why?**

| | Full-batch GD | SGD | Mini-batch SGD |
|---|---|---|---|
| Data used per update | All samples | 1 sample | A small batch (e.g. 32) |
| Gradient quality | Exact | Noisy | Approximate |
| Speed per epoch | Slow | Fast | Medium |
| Memory | High | Low | Medium |

In this assignment, **full-batch gradient descent** is actually used (all training samples are passed at once to `model(X_train)`), but the optimizer is called `optim.SGD`. This is a common naming convention in PyTorch — `optim.SGD` is the SGD optimizer algorithm; whether it's stochastic depends on how you feed the data (batched or all-at-once). True mini-batch SGD would require a `DataLoader`.

---

## 2. Dataset & Preprocessing

**Q11. What is the Human Activity Recognition (HAR) dataset? What are the six activity classes?**

The HAR dataset was built from recordings of **30 participants** performing daily activities while wearing a Samsung Galaxy S II smartphone on their waist. The smartphone's accelerometer and gyroscope captured 3-axial signals at 50Hz. From these signals, a 561-feature vector was extracted using time and frequency domain analysis.

The six activity classes are:
1. WALKING
2. WALKING UPSTAIRS
3. WALKING DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

---

**Q12. Why was the `subject` column removed from the dataset before training?**

The `subject` column is an **identifier** — it just tells us which participant (1 to 30) performed the activity. It carries no predictive information about what activity was performed. Including it would cause the model to memorize which participant did what, rather than learning the actual activity patterns from sensor data. This would lead to poor generalization on new, unseen participants.

---

**Q13. What is the purpose of standardizing (scaling) the dataset before feeding it to a neural network?**

Standardization (z-score scaling) transforms each feature to have **mean = 0 and standard deviation = 1**:

**x_scaled = (x − mean) / std**

Benefits for neural networks:
- All features are on the same scale, so no single feature dominates the gradient updates
- Gradients are more balanced, leading to **faster and more stable convergence**
- Weight initialization (which assumes inputs are roughly zero-centered) works better
- Prevents numerical issues like exploding/vanishing gradients

Without scaling, a feature with values in the thousands would dominate over a feature with values between 0 and 1.

---

**Q14. What is PCA (Principal Component Analysis)? Why is it used here?**

PCA is a **dimensionality reduction** technique that transforms the original features into a new set of uncorrelated features called **principal components**, ordered by the amount of variance they explain. The first principal component explains the most variance, the second the next most, and so on.

In this assignment, PCA is used to:
- Reduce the 561 features to a smaller set while retaining most of the information
- Remove redundant/correlated features (many of the 561 HAR features are correlated)
- Speed up training by reducing input dimensionality
- Potentially improve generalization by reducing noise

---

**Q15. What is the difference between the PCA-All dataset and the PCA-99% dataset? How many features does each have?**

- **PCA-All:** All 561 principal components are kept. The input size is still 561, but the features are now the principal components (rotated, uncorrelated). This is essentially the same information as the scaled data but in a different coordinate system.
- **PCA-99%:** Only the principal components that together explain **99% of the total variance** are kept. This reduces the input from 561 to **156 features**, discarding the remaining components that carry only 1% of the information.

The key difference is that PCA-99 actually **reduces dimensionality**, while PCA-All is just a rotation of the original scaled feature space.

---

**Q16. Why does retaining 99% variance in PCA reduce the input from 561 to 156 features? What do the remaining components represent?**

Many of the 561 original features are highly correlated (e.g., different statistical measures of the same sensor signal). PCA finds that just 156 principal components capture 99% of the variance in the data — the other 405 components mostly capture noise and redundant information. Discarding them barely loses any meaningful signal.

The remaining 405 components (the discarded ones) represent:
- Noise in the sensor measurements
- Extremely subtle patterns that contribute minimally to distinguishing activities
- Redundant information already captured by the first 156 components

---

**Q17. How were the training, validation, and test sets split in this assignment (ratio)?**

The dataset was split as:
- **Training set: 60%** — used to train the model (update weights)
- **Validation set: 20%** — used to evaluate during training and select the best architecture
- **Test set: 20%** — used only for final evaluation, never seen during training or model selection

With 10,299 total samples: ~6,180 train, ~2,060 validation, ~2,059 test.

---

**Q18. Why do we use a validation set separate from the test set during training?**

- The **validation set** is used during training to compare different architectures and select hyperparameters (e.g., which architecture to choose as "best"). Since we look at validation results to make decisions, the model indirectly "sees" this data.
- The **test set** is held out completely and only used once at the very end to report the final, unbiased performance of the selected model.

If we used the test set for model selection, we would be "cheating" — the reported accuracy would be overly optimistic because we picked the model that happened to work best on that specific test data, rather than measuring true generalization.

---

## 3. Architecture & Implementation

**Q19. How do you define a neural network in PyTorch? What is the role of `nn.Module` and the `forward()` method?**

In PyTorch, a neural network is defined as a class that **inherits from `nn.Module`**:

```python
class MyNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # define layers here
    
    def forward(self, x):
        # define how data flows through layers
        return output
```

- **`nn.Module`** is the base class for all neural networks in PyTorch. It provides parameter tracking (for optimization), the ability to save/load models, and moving models to GPU.
- **`forward()`** defines the computation performed at every call — i.e., how the input tensor flows through the layers to produce the output. PyTorch calls this automatically when you do `model(x)`.

---

**Q20. What does `nn.Sequential` do? How is it used in this assignment to stack layers?**

`nn.Sequential` is a container that chains layers together in order — the output of one layer becomes the input of the next. It simplifies the `forward()` method since you don't need to manually pass data through each layer.

In this assignment:
```python
self.layers = nn.Sequential(
    nn.Linear(input_size, hidden_size),  # Layer 1
    nn.Tanh(),                           # Activation
    nn.Linear(hidden_size, output_size)  # Layer 2
)

def forward(self, x):
    return self.layers(x)   # data flows through all layers automatically
```

---

**Q21. What is `LabelEncoder` used for? Why do we need to convert string class labels to integers?**

`LabelEncoder` from scikit-learn converts string class labels (like "WALKING", "SITTING") into **integer indices** (0, 1, 2, 3, 4, 5). This is necessary because:
- PyTorch tensors can only store numerical data — strings are not supported
- `CrossEntropyLoss` expects class indices as `torch.long` (integer) tensors
- The output layer has 6 neurons indexed 0–5, so each label must map to one of these indices

In this assignment, we fit the `LabelEncoder` once on the training data and use the same mapping for validation and test sets to ensure consistency.

---

**Q22. Why do we call `model.train()` before training and `model.eval()` before evaluation?**

These methods switch the model between two modes:

- **`model.train()`:** Enables training-specific layers like Dropout (randomly drops neurons) and BatchNorm (uses batch statistics). Gradients are computed normally.
- **`model.eval()`:** Disables Dropout (all neurons active) and BatchNorm uses running statistics instead of batch statistics. This ensures deterministic, reproducible predictions.

Even though this assignment doesn't use Dropout or BatchNorm, it is good practice to always call these methods. It also signals intent clearly to anyone reading the code.

---

**Q23. What does `torch.no_grad()` do, and why is it used during evaluation?**

`torch.no_grad()` is a context manager that **disables gradient computation** for all operations inside it. During evaluation (validation/test), we only need forward pass predictions — we don't need to compute gradients or update weights. Using `no_grad()`:
- **Saves memory** — no gradient tensors are stored
- **Speeds up computation** — gradient tracking has overhead
- **Prevents accidental weight updates** during evaluation

```python
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)
```

---

**Q24. What does `optimizer.zero_grad()` do? What would happen if we skipped this step?**

`optimizer.zero_grad()` **resets all parameter gradients to zero** before computing new gradients for the current batch. 

PyTorch accumulates (adds) gradients by default — this is useful in some advanced scenarios. But in standard training, we want fresh gradients at each step. If we skip `zero_grad()`, gradients from the previous iteration get added to the current iteration's gradients, causing **incorrect and inflated gradient values**, leading to wrong weight updates and unstable training.

---

**Q25. What does `loss.backward()` compute? What is the mathematical operation happening under the hood?**

`loss.backward()` computes the **gradient of the loss with respect to every learnable parameter** in the network using the **chain rule of calculus (backpropagation)**. 

Starting from the loss, it works backwards through the computation graph:
- ∂L/∂output_weights
- ∂L/∂hidden_weights = ∂L/∂output × ∂output/∂hidden

These partial derivatives are stored in each parameter's `.grad` attribute and are used by the optimizer to update the weights. The automatic differentiation engine (autograd) in PyTorch builds a dynamic computation graph during the forward pass, which is then traversed backwards.

---

**Q26. What does `optimizer.step()` do after `loss.backward()`?**

`optimizer.step()` **updates the model parameters** using the gradients computed by `loss.backward()`. For SGD, the update rule is:

**w = w − lr × ∂L/∂w**

It reads the `.grad` attribute of each parameter and subtracts the learning-rate-scaled gradient from the parameter value. Without calling `step()`, the gradients are computed but the weights never actually change — the model would not learn anything.

---

**Q27. How does `torch.argmax(outputs, dim=1)` give us the predicted class label?**

The model's output has shape `(N, 6)` — N samples, 6 class scores each. `torch.argmax(outputs, dim=1)` returns the **index of the maximum value along dimension 1** (the class axis) for each sample.

For example, if outputs for one sample are `[0.1, 0.05, 0.7, 0.05, 0.05, 0.05]`, then `argmax` returns `2`, meaning the model predicts class 2. Since our `LabelEncoder` maps class labels to indices 0–5, index 2 would correspond to the 3rd activity class.

---

**Q28. How are the trained models saved and loaded in PyTorch? What does `torch.save(model.state_dict(), path)` store?**

**Saving:**
```python
torch.save(model.state_dict(), 'model.pth')
```
`state_dict()` returns a Python dictionary containing all **learnable parameters** (weights and biases) of the model as tensors. Only the parameters are saved, not the model architecture itself.

**Loading:**
```python
model = MyNet(...)            # must recreate the architecture first
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

You must define the same architecture before loading, because `load_state_dict()` just fills in the weights — it doesn't reconstruct the structure.

---

## 4. Training & Hyperparameters

**Q29. What is the `patience` parameter in early stopping? How was it set in this assignment?**

`patience` is the number of **consecutive epochs** with no significant improvement in training loss before training is stopped early. It prevents the model from training for more epochs than needed.

In this assignment: `patience = 10` — if the training loss changes by less than `threshold` (0.001) for 10 consecutive epochs, training stops. The PDF recommends setting patience to 1% of max_epochs (1% of 1000 = 10) as the minimum, or 10% for stronger convergence confidence.

---

**Q30. What is the `threshold` parameter used for in the convergence check? What value was used here?**

`threshold = 1e-3` (0.001) defines the **minimum meaningful change** in training loss between consecutive epochs. If the change is smaller than this, we consider the loss to have "plateaued" and increment the patience counter.

```python
if abs(train_loss - prev_loss) < threshold:
    counter += 1
```

The PDF also suggests experimenting with smaller thresholds like `5e-4` and `1e-4` for stronger convergence checks. A smaller threshold means you require the loss to truly stop moving before stopping, which can lead to more training epochs.

---

**Q31. Why was a learning rate of `0.01` chosen? What range of values is generally considered good for SGD?**

`lr = 0.01` is a standard, commonly used starting value for SGD. The PDF explicitly states this value and notes that **0.001 ≤ lr ≤ 0.01** is a good range.

- lr = 0.01 is large enough to make meaningful updates each epoch
- Small enough to not overshoot the minimum
- For this dataset, it provides a good balance between convergence speed and stability

In practice, learning rate is one of the most important hyperparameters and is often tuned using techniques like learning rate schedulers or grid search.

---

**Q32. What would happen if `max_epochs` is set too low? And too high?**

- **Too low (e.g., 10 epochs):** The model doesn't have enough iterations to learn — weights remain near their initial values, resulting in poor accuracy (underfitting). The loss may still be high when training stops.
- **Too high (e.g., 100,000 epochs):** Without early stopping, the model might overfit — it memorizes the training data and loses the ability to generalize. With early stopping (as used here), a high `max_epochs` is fine because training will stop automatically when the loss converges. It just acts as a safety cap.

In this assignment, `max_epochs=1000` is used with early stopping, so the actual number of epochs run is typically much less.

---

**Q33. How does the training loss curve (epoch vs loss plot) help you understand the training process?**

The loss curve reveals several things:
- **Steadily decreasing loss:** The model is learning well
- **Flat/plateau early on:** The learning rate may be too small, or the model is stuck
- **Sharp initial drop then plateau:** Normal convergence — the model learned quickly then settled
- **Oscillating loss:** Learning rate is too high
- **Loss decreasing very slowly:** Learning rate is too low or the architecture is too simple
- **Diverging loss (increasing):** Learning rate is too high or there's a bug

Comparing curves across architectures shows which ones converge faster and to a lower loss.

---

**Q34. What does it mean if the training loss curve is oscillating and not decreasing smoothly?**

Oscillation usually means the **learning rate is too high**. The optimizer is taking steps that are too large — it overshoots the minimum, then overshoots back the other way. The weights bounce around the optimal values without settling.

Other possible causes:
- Noisy gradients (more common in true mini-batch SGD)
- Numerical instability (e.g., very large or very small features — solved by scaling)

Solution: Reduce the learning rate or use a learning rate scheduler that reduces lr over time.

---

**Q35. If two architectures have similar training loss but very different validation accuracy, what might be happening?**

This is a classic sign of **overfitting in one of the models**. The model with higher training loss but better validation accuracy is generalizing better. The model with lower training loss but worse validation accuracy has memorized the training data — it fits the training set well but fails on unseen data.

Possible causes of overfitting:
- Too many parameters (too wide/deep) relative to the amount of training data
- Training for too many epochs
- Lack of regularization (no dropout, no weight decay)

Solution: Use regularization (dropout, L2 weight decay), early stopping, or choose a simpler architecture.

---

## 5. Evaluation Metrics

**Q36. What is a confusion matrix? How do you read it for a multiclass problem?**

A confusion matrix is a square matrix of size N×N (where N = number of classes) that shows how many samples of each true class were predicted as each other class.

```
              Predicted
              W  WU  WD  Si  St  L
Actual  W  [ 170   2   3   0   0   0 ]
        WU [   1 148   4   0   0   0 ]
        WD [   2   3 132   0   0   0 ]
        Si [   0   0   0 155  20   0 ]
        St [   0   0   0  15 165   0 ]
        L  [   0   0   0   0   0 190 ]
```

- **Diagonal entries** = correct predictions (true positives for each class)
- **Off-diagonal entries** = misclassifications (where the model confused two classes)
- Large off-diagonal values indicate which classes the model confuses most (e.g., SITTING vs STANDING)

---

**Q37. What is accuracy, and when can it be a misleading metric?**

Accuracy = (Number of correct predictions) / (Total predictions)

It is misleading when the **dataset is imbalanced**. For example, if 90% of samples belong to class A, a model that always predicts class A achieves 90% accuracy without learning anything useful. In such cases, you should look at per-class metrics (precision, recall, F1) instead.

For this HAR dataset, the classes are reasonably balanced (13.65% to 18.88%), so accuracy is a fairly reliable metric, but it's still good practice to also check precision, recall, and F1.

---

**Q38. What is the difference between Precision and Recall? Give an intuitive explanation.**

- **Precision** = Of all the samples the model predicted as class X, how many actually are class X?
  - "When I say it's WALKING, how often am I right?"
  - Formula: TP / (TP + FP)

- **Recall** = Of all the samples that actually are class X, how many did the model correctly identify?
  - "Of all actual WALKING cases, how many did I catch?"
  - Formula: TP / (TP + FN)

**Trade-off:** Increasing recall often decreases precision and vice versa. For example, if you predict everything as "WALKING", recall for WALKING is 100% but precision is very low.

---

**Q39. What is the F1-score, and when is it more useful than accuracy alone?**

F1-score is the **harmonic mean of Precision and Recall**:

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

It balances both metrics in a single number. A high F1 means both precision and recall are high — the model is both correct when it predicts a class AND catches most instances of that class.

F1 is more useful than accuracy when:
- The dataset is **imbalanced** (some classes have far fewer samples)
- You care equally about false positives and false negatives
- Accuracy hides poor performance on minority classes

---

**Q40. What is the difference between micro-average and macro-average for precision, recall, and F1-score?**

- **Micro-average:** Aggregates the contributions of all classes by summing TP, FP, FN across all classes first, then computing the metric. **Gives more weight to larger classes.** For balanced datasets, micro-average equals accuracy.

- **Macro-average:** Computes the metric independently for each class, then takes the **unweighted average**. **Treats all classes equally regardless of size.**

Example with 2 classes (100 samples class A, 10 samples class B):
- Micro-average is dominated by class A's performance
- Macro-average equally weights A and B, so poor performance on class B is more visible

---

**Q41. When would you prefer macro-average over micro-average for an imbalanced dataset?**

You would prefer **macro-average** when you care equally about performance on all classes, regardless of how many samples each has. In an imbalanced dataset, micro-average is dominated by the majority class and can make the model look good even if it performs poorly on minority classes.

For example, in a medical diagnosis dataset where rare disease detection is critical, macro-average would highlight poor recall on the rare disease class, whereas micro-average might hide it.

For the HAR dataset, both are reasonable since the dataset is fairly balanced.

---

**Q42. Looking at the class distribution (LAYING: 18.88%, WALKING DOWNSTAIRS: 13.65%), is this dataset balanced? How might imbalance affect evaluation?**

The dataset is **mildly imbalanced** — the most common class (LAYING: 18.88%) has about 1.38× more samples than the least common (WALKING DOWNSTAIRS: 13.65%). This is relatively mild and generally not a major concern.

With more severe imbalance:
- The model might bias towards predicting majority classes
- Accuracy would be misleadingly high (majority class dominates)
- Minority class recall would be low
- Solutions: class weighting in the loss function, oversampling minority classes (SMOTE), or undersampling majority classes

For this assignment, the mild imbalance means accuracy is a reliable enough metric, but reporting macro-F1 provides additional confidence.

---

## 6. Comparison Across Datasets & Architectures

**Q43. Which dataset gave the best test accuracy and why?**

Generally, the **Scaled (standardized) dataset** or **PCA-All dataset** tend to give the best results for neural networks because:
- Scaling normalizes all 561 features to the same range, making gradient updates more balanced
- PCA-All also decorrelates features (removes multicollinearity), which can help
- The original unscaled dataset likely has features with very different magnitudes, causing uneven gradient updates and slower/unstable convergence

PCA-99 sometimes performs comparably despite fewer features, showing that the removed 405 components were mostly noise/redundancy.

---

**Q44. Did adding a second hidden layer consistently improve performance? Why or why not?**

Not necessarily. For the HAR dataset, which uses hand-crafted, well-engineered features, a single hidden layer often captures the necessary non-linear relationships. Adding a second layer:
- Can help if the relationship between features and classes is very complex
- May hurt if the dataset is not large enough relative to the extra parameters (overfitting)
- Increases training time

In practice, you might find Arch1(561) single layer performs similarly to or better than some two-layer architectures, while two-layer architectures with good widths (like 1024→128) might slightly outperform. It depends on the specific run.

---

**Q45. Did increasing the number of neurons in the hidden layer always lead to better accuracy? What tradeoff does this introduce?**

Not always. Wider layers:
- ✅ Have more parameters, so can potentially model more complex patterns
- ❌ Take longer to train (more computations per epoch)
- ❌ Risk overfitting if the dataset is not large enough
- ❌ May converge to a similar or worse solution than a smaller network

For this HAR dataset with ~6,000 training samples and 561 features, going from 128 to 1024 neurons might not always help and could hurt if the extra capacity causes overfitting. The best architecture varies per dataset and requires empirical testing.

---

**Q46. Why might the PCA-99 dataset (156 features) perform comparably or better than the full 561-feature dataset?**

Several reasons:
1. **Noise removal:** The 405 discarded components capture only 1% of variance — they're mostly noise. Removing them prevents the model from fitting noise.
2. **Reduced overfitting:** Fewer input features → fewer parameters → less chance of overfitting
3. **Faster convergence:** Smaller input means fewer computations, and the model can converge in fewer epochs
4. **Decorrelated features:** PCA components are orthogonal (uncorrelated), which is mathematically cleaner for gradient descent

---

**Q47. What are the advantages of using PCA before training a neural network?**

1. **Dimensionality reduction:** Reduces computational cost and training time
2. **Removes redundancy:** Correlated features become uncorrelated principal components
3. **Noise reduction:** Low-variance components (often noise) can be discarded
4. **Numerical stability:** PCA-transformed features are often better scaled
5. **Can improve generalization:** By removing noise and redundant features, the model focuses on the most informative directions in the data

Disadvantage: PCA components lose interpretability — you can no longer say "this feature is the mean acceleration of the x-axis."

---

**Q48. Compare the convergence speed across different datasets. Which converged fastest and why?**

Expected convergence speed (fastest to slowest):
1. **Scaled data** — All features are zero-centered with unit variance → gradients are well-behaved → fastest convergence
2. **PCA-All** — Similar to scaled (also normalized) but with decorrelated features → similar speed
3. **PCA-99** — Fewer dimensions → fewer computations per epoch, similar gradient quality
4. **Original (unscaled) data** — Features have wildly different scales → uneven gradients → slowest convergence, possibly unstable

Scaled data converges fastest because the loss surface is more "spherical" (isotropic), meaning gradient descent moves more directly towards the minimum rather than zigzagging.

---

## 7. Practical / Code-Based

**Q49. If you got `RuntimeError: mat1 and mat2 shapes cannot be multiplied`, what does it mean?**

This error means there is a **shape mismatch** in a matrix multiplication — the number of columns in the input tensor doesn't match the number of rows in the weight matrix of the next layer.

Common causes in this assignment:
- You set the wrong `input_size` when creating the model (e.g., used 561 for Part D which has 156 features)
- A mismatch between the output size of one layer and the input size of the next layer in `nn.Sequential`

Fix: Check `X_train.shape[1]` and make sure the model's `input_size` parameter matches.

---

**Q50. What change would you make to the code if you wanted to use ReLU instead of tanh?**

Simply replace `nn.Tanh()` with `nn.ReLU()` in the `nn.Sequential` block:

```python
# Before (tanh):
self.layers = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.Tanh(),
    nn.Linear(hidden_size, output_size)
)

# After (ReLU):
self.layers = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
```

No other changes needed. ReLU would likely converge faster (avoids vanishing gradients) and might give slightly better or similar accuracy.

---

**Q51. How would you modify the `SingleHiddenLayerNN` class to add dropout regularization?**

Add `nn.Dropout(p)` after the activation function, where `p` is the dropout probability (fraction of neurons randomly set to 0 during training):

```python
class SingleHiddenLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=6, dropout_p=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),   # <-- added dropout
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)
```

Dropout is automatically disabled during `model.eval()`, so no other changes are needed for evaluation.

---

**Q52. How would you add a learning rate scheduler to reduce the learning rate over time?**

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)

# StepLR reduces lr by factor 'gamma' every 'step_size' epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()   # <-- update the learning rate
```

This gradually reduces the learning rate, allowing the model to take smaller, more precise steps as it gets closer to the optimum — often leads to better final accuracy.

---

**Q53. How would you modify the code to use mini-batches with `DataLoader`?**

```python
from torch.utils.data import TensorDataset, DataLoader

# Create dataset and dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop with mini-batches
for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:   # iterate over batches
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
```

Mini-batches introduce noise into gradient updates, which can actually help escape local minima and often leads to better generalization.

---

**Q54. What would you change in the code to use the Adam optimizer instead of SGD?**

Just change one line — replace `optim.SGD` with `optim.Adam`. Adam typically uses a smaller default learning rate:

```python
# Before:
optimizer = optim.SGD(model.parameters(), lr=0.01)

# After:
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Adam adapts the learning rate for each parameter individually using estimates of first and second moments of gradients. It typically converges faster than plain SGD and is less sensitive to the choice of learning rate.

---

**Q55. If you wanted to load a saved model and run predictions on new data, what steps would you follow?**

```python
# Step 1: Recreate the exact same architecture
model = SingleHiddenLayerNN(input_size=561, hidden_size=561, output_size=6)

# Step 2: Load the saved weights
model.load_state_dict(torch.load('saved_models/partA/single_Arch1_561.pth'))

# Step 3: Set to evaluation mode
model.eval()

# Step 4: Prepare new data as a tensor
X_new = torch.tensor(new_data.values, dtype=torch.float32)

# Step 5: Get predictions
with torch.no_grad():
    outputs = model(X_new)
    preds   = torch.argmax(outputs, dim=1)

# Step 6: Decode back to class labels
predicted_labels = le.inverse_transform(preds.numpy())
print(predicted_labels)
```

---

## 8. Tricky / Insight Questions

**Q56. The output layer has no explicit Softmax. Does this mean the model outputs probabilities? Explain.**

No — the raw output of `nn.Linear` (the final layer) is **logits**, not probabilities. They are unbounded real numbers and do not sum to 1.

However, PyTorch's `nn.CrossEntropyLoss` internally applies LogSoftmax before computing the loss, so training is mathematically equivalent to using Softmax + NLLLoss. During prediction, we use `torch.argmax()` which finds the maximum logit — since Softmax is a monotonic transformation, the argmax of logits equals the argmax of the Softmax probabilities. So we get the correct predicted class without ever explicitly computing probabilities.

If you needed actual probabilities (e.g., for confidence estimation), you would apply `torch.softmax(outputs, dim=1)` manually after the model's output.

---

**Q57. If training accuracy is very high but test accuracy is low, what is the likely problem? How would you fix it?**

This is **overfitting** — the model has memorized the training data but fails to generalize to unseen data. The model has learned the noise and specific quirks of the training set rather than the underlying patterns.

Fixes:
1. **Dropout:** Randomly disable neurons during training to prevent co-adaptation
2. **L2 regularization (weight decay):** Add penalty for large weights — `optim.SGD(lr=0.01, weight_decay=1e-4)`
3. **Early stopping:** Stop training before the model starts overfitting (already implemented here)
4. **Reduce model complexity:** Use fewer neurons or fewer layers
5. **More training data:** If possible, get more data or use data augmentation
6. **Batch normalization:** Adds a regularization effect

---

**Q58. Why might a very deep or very wide network perform worse than a simpler one on this dataset?**

For the HAR dataset, the 561 features are already **hand-crafted domain-specific features** (mean, std, energy, correlation of sensor signals). The classification boundaries are relatively simple — the activities differ clearly in their sensor patterns.

A very large/deep network:
- Has far more parameters than needed → overfits the training data
- Is harder to optimize → may get stuck in poor local minima with SGD
- Takes much longer to converge
- May suffer from vanishing gradients (for very deep networks with tanh)

A simpler network (e.g., 128 hidden neurons) that captures the key patterns without overfitting often generalizes better. This is the **bias-variance tradeoff** — very complex models have low bias but high variance.

---

**Q59. Is it possible for a model with lower training loss to have lower test accuracy than a model with higher training loss? Explain.**

Yes, absolutely. This happens when the model with lower training loss is **overfitting**. It has fit the training data so precisely (including its noise) that it fails on the test set. The model with slightly higher training loss may have found a smoother, more generalizable decision boundary.

Think of it like memorizing answers vs. understanding concepts: a student who memorizes all exam answers (low training loss) might fail a new exam (low test accuracy), while a student who truly understands (slightly higher training loss from not memorizing) does better on new questions.

This is why we evaluate on a separate validation/test set and not just trust the training loss.

---

**Q60. If you were to improve this assignment further, what would you try?**

Several improvements, in order of expected impact:

1. **Mini-batch SGD with DataLoader:** True stochastic training with batches of 32–128 samples introduces beneficial noise, improves generalization, and makes the training more scalable

2. **Adam optimizer:** Adapts learning rates per parameter, typically converges faster and to better solutions than plain SGD, especially with default settings

3. **Dropout regularization (p=0.3–0.5):** Prevents overfitting, especially for the larger architectures (1024 neurons)

4. **Batch Normalization:** Normalizes layer inputs, accelerates training, acts as implicit regularization, and allows higher learning rates

5. **Learning rate scheduler:** Reduce lr by 0.5 every 100 epochs — helps fine-tune weights in later training stages

6. **ReLU activation:** Avoids vanishing gradients better than tanh, typically trains faster

7. **Deeper architectures (3+ layers):** With dropout and batch norm to prevent overfitting, deeper networks might extract better hierarchical features

The biggest practical gains would likely come from (1) mini-batching + (2) Adam + (3) dropout, which are standard practices in modern deep learning.

---

*All the best for your viva!*
